import torch
import torch.nn.functional as F
from larch.classification import ClassificationMetrics
from larch.distributed import print0
from larch.training.logging import log_scalar

KNN_METRICS = ("cosine", "euclidean")

@torch.no_grad()
def pca_project(bank, query, n_components):

    if not n_components:
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
def knn_votes_from_neighbors(indices,
                             bank_labels,
                             n_classes):

    neighbor_labels = bank_labels[indices]
    
    one_hot = F.one_hot(neighbor_labels.long(),
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


def run_knn_probes(bank,
                   query,
                   device,
                   targets,
                   label_groups,
                   *,
                   knn_k,
                   knn_pca,
                   knn_metrics=KNN_METRICS):

    print0("Running kNN probes")
    knn_results = {}
    for metric in knn_metrics:
        knn_results[metric] = {}
        for group in label_groups:
            knn_results[metric][group] = evaluate_knn(
                bank["feats"], bank["labels"][group],
                query["feats"], query["labels"][group],
                classifier_config=targets,
                device=device,
                k=knn_k,
                metric=metric,
                pca_dims=knn_pca,
            )

    return knn_results

def log_knn_results(writer, metrics, knn_results, iteration):
    if knn_results is not None:
        for dist_metric, group_results in knn_results.items():
            for group, target_results in group_results.items():
                for part, result in target_results.items():
                    for name, value in result.items():
                        log_scalar(writer, metrics,
                                   f"knn_{group}/{dist_metric}/{part}_{name}",
                                   value, iteration)
