import math
import torch
from larch.classification import ClassificationMetrics, SupervisedHead, supervised_loss
from larch.training.logging import log_scalar
from larch.distributed import print0

def fit_linear_probe(bank_features,
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
                     print_debug=False):
    
    # Move everything to the GPU for speed... imposes an implicit limit on the bank and query size
    bank_features = bank_features.detach().float().to(device)
    query_features = query_features.detach().float().to(device)

    bank_labels = {
        name: labels.detach().long().to(device)
        for name, labels in bank_labels.items()
        if name in classifier_config
    }
    query_labels = {
        name: labels.detach().long().to(device)
        for name, labels in query_labels.items()
        if name in classifier_config
    }

    # Fit feature preprocessing using the bank only.
    mean = bank_features.mean(dim=0, keepdim=True)
    std = bank_features.std(dim=0, unbiased=False, keepdim=True)
    
    # Prevent very low-variance dimensions from receiving huge amplification.
    std_floor = 0.01 * std.median()
    std_safe = std.clamp_min(std_floor)
    
    bank_features = (bank_features - mean) / std_safe
    query_features = (query_features - mean) / std_safe

    # Avoid probe initialization/training changing the main training RNG.
    cuda_devices = ([device.index] if device.type == "cuda" else [])

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)

        probe = SupervisedHead(encoder_dim=bank_features.shape[1],
                               classifier_config=classifier_config).to(device)

        ## Use Adam with a cosine scheduler
        optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0, fused=(device.type=="cuda"))
        nbank = bank_features.shape[0]
        steps_per_epoch = math.ceil(nbank / batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)
        
        generator = torch.Generator()
        generator.manual_seed(seed)

        probe.train()

        for i in range(epochs):
            
            sum_loss = torch.zeros((), device=device)
            num_samples = 0
            
            permutation = torch.randperm(
                bank_features.shape[0],
                generator=generator,
            ).to(device)

            for start in range(0, bank_features.shape[0], batch_size):
                indices = permutation[start:start + batch_size]

                features = bank_features[indices]
                labels = {name: values[indices] for name, values in bank_labels.items()}

                outputs = probe(features)
                loss, _ = supervised_loss(outputs, labels, classifier_config)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

                this_batch_size = features.shape[0]
                sum_loss += loss.detach()*this_batch_size
                num_samples += this_batch_size

            ## Report the average loss for this epoch
            if print_debug: print0(f"{i}: loss = {(sum_loss/max(num_samples, 1)).item():.4f}")

        # Evaluate on the query split.
        probe.eval()

        probe_metrics = ClassificationMetrics(classifier_config, device=device)

        with torch.no_grad():
            for start in range(0, query_features.shape[0], batch_size):
                end = start + batch_size

                features = query_features[start:end]
                labels = {name: values[start:end] for name, values in query_labels.items()}

                outputs = probe(features)
                probe_metrics.update(outputs, labels, outputs_are_logits=True)

        # Do not call reduce(): this probe runs on rank 0 using globally
        # gathered bank and query features.
        results = probe_metrics.compute()

    return results


def run_linear_probes(bank,
                      query,
                      device,
                      targets,
                      label_groups,
                      *,
                      linear_epochs,
                      linear_batch_size,
                      linear_lr,
                      seed):

    print0(f"Running linear probes")
    linear_results = {}
    for group in label_groups:
        linear_results[group] = fit_linear_probe(
            bank["feats"], bank["labels"][group],
            query["feats"], query["labels"][group],
            classifier_config=targets,
            device=device,
            epochs=linear_epochs,
            batch_size=linear_batch_size,
            lr=linear_lr,
            seed=seed,
        )
        
    return linear_results


def log_linear_results(writer, metrics, linear_results, iteration):

    if linear_results is not None:
        for group, target_results in linear_results.items():
            for part, result in target_results.items():
                for name, value in result.items():
                    log_scalar(writer, metrics,
                               f"linear_{group}/{part}_{name}",
                               value, iteration)
