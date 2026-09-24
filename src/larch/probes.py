import torch
import MinkowskiEngine as ME
import torch.distributed as dist
import torch.nn.functional as F
from larch.classification import ClassificationMetrics, SupervisedHead, supervised_loss
from larch.distributed import print0
from larch.training.logging import log_scalar

KNN_METRICS = ("cosine", "euclidean")

@torch.no_grad()
def extract_features(encoder,
                     loader,
                     device,
                     label_names,
                     label_groups=("particle_truth", "particle_visible")):
    
    was_training = encoder.training
    encoder.eval()
    model = encoder.module if hasattr(encoder, "module") else encoder
    
    feats = []
    labels = {
        group: {name: [] for name in label_names}
        for group in label_groups
    }
    
    for bcoords, bfeats, blabels, bs in loader:
        bcoords = bcoords.to(device, non_blocking=True)
        bfeats  = bfeats.to(device,  non_blocking=True)
        batch   = ME.SparseTensor(bfeats, bcoords, device=device)
        feats.append(model(batch, bs).float())

        for group in label_groups:
            for name in label_names:
                labels[group][name].append(
                    blabels[group][name].to(device, non_blocking=True).long()
                )
    if was_training:
        encoder.train()

    feats = torch.cat(feats)

    def gather_equal_size(tensor):
        gathered = [
            torch.empty_like(tensor)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    feats = gather_equal_size(feats)

    ## Concatenate local batches
    labels = {
        group: {name: torch.cat(chunks, dim=0)
                for name, chunks in group_labels.items()}
        for group, group_labels in labels.items()
    }

    ## Now gather across ranks
    labels = {
        group: {name: gather_equal_size(values)
                for name, values in group_labels.items()}
        for group, group_labels in labels.items()
    }

    return feats, labels


@torch.no_grad()
def pca_project(bank, query, n_components):

    if n_components is None:
        return bank, query

    max_components = min(bank.shape[0], bank.shape[1])

    mean = bank.mean(dim=0, keepdim=True)
    bank_centered = bank - mean
    query_centered = query - mean

    ## Vh rows are principal directions, ordered by singular value
    _, _, vh = torch.linalg.svd(bank_centered, full_matrices=False)
    components = vh[:n_components].T

    return (bank_centered @ components, query_centered @ components)


@torch.no_grad()
def knn_neighbors(query,
                  bank,
                  *,
                  k=20,
                  chunk=2048,
                  metric="euclidean"):
    
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"Unknown kNN metric {metric!r}")

    if bank.shape[0] == 0:
        raise ValueError("The kNN bank is empty")

    k = min(k, bank.shape[0])
    indices = []

    if metric == "cosine":
        bank_work = F.normalize(bank, dim=1)

        for start in range(0, query.shape[0], chunk):
            query_chunk = F.normalize(query[start:start + chunk], dim=1)
            similarities = query_chunk @ bank_work.T
            indices.append(similarities.topk(k, dim=1, largest=True).indices)

    else:
        ## Squared Euclidean distance (sqrt doesn't affect kNN, so neglect)
        bank_norm2 = (bank * bank).sum(dim=1, keepdim=False).unsqueeze(0)

        for start in range(0, query.shape[0], chunk):
            query_chunk = query[start:start + chunk]
            
            distance2 = (
                (query_chunk * query_chunk).sum(dim=1, keepdim=True)
                + bank_norm2
                - 2.0 * query_chunk @ bank.T
            ).clamp_min_(0.0)

            indices.append(distance2.topk(k, dim=1, largest=False).indices)

    return torch.cat(indices, dim=0)


@torch.no_grad()
def knn_votes_from_neighbors(
    indices,
    bank_labels,
    n_classes,
):
    neighbor_labels = bank_labels[indices]

    one_hot = F.one_hot(neighbor_labels,
                        num_classes=n_classes)

    return one_hot.sum(dim=1).float()


def evaluate_knn(bank_features,
                 bank_labels,
                 query_features,
                 query_labels,
                 *,
                 classifier_config,
                 device,
                 k,
                 metric="euclidean",
                 pca_dims=None,
                 chunk=2048):

    ## If pca_dims == 0, this just returns
    bank_features, query_features = pca_project(
        bank_features,
        query_features,
        pca_dims,
    )

    indices = knn_neighbors(
        query_features,
        bank_features,
        k=k,
        chunk=chunk,
        metric=metric,
    )

    votes = {
        name: knn_votes_from_neighbors(
            indices,
            bank_labels[name],
            cfg["n_classes"],
        )
        for name, cfg in classifier_config.items()
    }

    metrics = ClassificationMetrics(
        classifier_config,
        device=device,
    )
    metrics.update(
        votes,
        query_labels,
        outputs_are_logits=False,
    )

    return metrics.compute()


def fit_linear_probe(
    bank_features,
    bank_labels,
    query_features,
    query_labels,
    *,
    classifier_config,
    device,
    epochs=20,
    batch_size=1024,
    lr=1e-2,
    seed=12345,
):
    # Keep the extracted feature dataset on CPU.
    bank_features = bank_features.detach().float().cpu()
    query_features = query_features.detach().float().cpu()

    bank_labels = {
        name: labels.detach().long().cpu()
        for name, labels in bank_labels.items()
        if name in classifier_config
    }
    query_labels = {
        name: labels.detach().long().cpu()
        for name, labels in query_labels.items()
        if name in classifier_config
    }

    # Fit feature preprocessing using the bank only.
    mean = bank_features.mean(dim=0, keepdim=True)
    std = bank_features.std(
        dim=0,
        unbiased=False,
        keepdim=True,
    )
    
    # Prevent very low-variance dimensions from receiving huge amplification.
    std_floor = 0.01 * std.median()
    std_safe = std.clamp_min(std_floor)
    
    bank_features = (bank_features - mean) / std_safe
    query_features = (query_features - mean) / std_safe

    # Avoid probe initialization/training changing the main training RNG.
    cuda_devices = (
        [device.index]
        if device.type == "cuda"
        else []
    )

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)

        probe = SupervisedHead(
            encoder_dim=bank_features.shape[1],
            classifier_config=classifier_config,
        ).to(device)

        optimizer = torch.optim.AdamW(
            probe.parameters(),
            lr=lr,
            weight_decay=0.0,
        )

        generator = torch.Generator()
        generator.manual_seed(seed)

        probe.train()

        for i in range(epochs):
            
            #sum_loss = torch.zeros((), device=device)
            #num_samples = 0
            
            permutation = torch.randperm(
                bank_features.shape[0],
                generator=generator,
            )

            for start in range(
                0,
                bank_features.shape[0],
                batch_size,
            ):
                indices = permutation[start:start + batch_size]

                features = bank_features[indices].to(device, non_blocking=True)
                labels = {
                    name: values[indices].to(device, non_blocking=True)
                    for name, values in bank_labels.items()
                }

                outputs = probe(features)

                loss, _ = supervised_loss(outputs, labels, classifier_config)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                #this_batch_size = features.shape[0]
                #sum_loss += loss.detach().double()*this_batch_size
                #num_samples += this_batch_size

            ## Report the average loss for this epoch
            #av_loss = sum_loss/max(num_samples, 1)
            #print0(f"{i}: loss = {av_loss.item()}")
                
        # Evaluate on the query split.
        probe.eval()

        probe_metrics = ClassificationMetrics(
            classifier_config,
            device=device,
        )

        with torch.no_grad():
            for start in range(
                0,
                query_features.shape[0],
                batch_size,
            ):
                end = start + batch_size

                features = query_features[start:end].to(device, non_blocking=True)
                labels = {
                    name: values[start:end].to(device, non_blocking=True)
                    for name, values in query_labels.items()
                }

                outputs = probe(features)
                probe_metrics.update(outputs, labels, outputs_are_logits=True)

        # Do not call reduce(): this probe runs on rank 0 using globally
        # gathered bank and query features.
        results = probe_metrics.compute()

    del probe, optimizer
    return results

def run_probes(encoder,
               bank_loader,
               query_loader,
               device,
               targets,
               label_groups,
               *,
               rank,
               run_knn,
               run_linear,
               knn_k,
               knn_pca,
               linear_epochs,
               linear_batch_size,
               linear_lr,
               seed,
               knn_metrics=KNN_METRICS):

    if not (run_knn or run_linear):
        return None, None

    tstart = time.time()
    bank_f, bank_l = extract_features(encoder, bank_loader, device, targets.keys())
    qry_f, qry_l = extract_features(encoder, query_loader, device, targets.keys())

    knn_results = None
    linear_results = None

    if rank == 0 and run_knn:
        knn_results = {}
        for metric in knn_metrics:
            knn_results[metric] = {}
            for group in label_groups:
                print0(f"Running {metric} kNN for {group}")
                knn_results[metric][group] = evaluate_knn(
                    bank_f, bank_l[group],
                    qry_f, qry_l[group],
                    classifier_config=targets,
                    device=device,
                    k=knn_k,
                    metric=metric,
                    pca_dims=knn_pca,
                )

    if rank == 0 and run_linear:
        linear_results = {}
        for group in label_groups:
            print0(f"Running linear probe for {group}")
            linear_results[group] = fit_linear_probe(
                bank_f, bank_l[group],
                qry_f, qry_l[group],
                classifier_config=targets,
                device=device,
                epochs=linear_epochs,
                batch_size=linear_batch_size,
                lr=linear_lr,
                seed=seed,
            )

    dist.barrier()
    print0(f"Monitoring time taken: {time.time() - tstart:.2f}")
    return knn_results, linear_results


def log_probe_results(writer, metrics, knn_results, linear_results, iteration):
    if knn_results is not None:
        for dist_metric, group_results in knn_results.items():
            for group, target_results in group_results.items():
                for part, result in target_results.items():
                    for name, value in result.items():
                        log_scalar(writer, metrics,
                                   f"knn_{group}/{dist_metric}/{part}_{name}",
                                   value, iteration)

    if linear_results is not None:
        for group, target_results in linear_results.items():
            for part, result in target_results.items():
                for name, value in result.items():
                    log_scalar(writer, metrics,
                               f"linear_{group}/{part}_{name}",
                               value, iteration)
