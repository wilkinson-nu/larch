import math
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score


def _entropy(p):
    p = p[p > 0]
    return -(p * p.log()).sum()


@torch.no_grad()
def encoder_neighbours(feats, k, metric="cosine", chunk=4096):
    x = feats.float()
    if metric == "cosine":
        x = F.normalize(x, dim=1)
    elif metric == "euclidean":
        sq_norm = (x * x).sum(dim=1)
    else:
        raise ValueError(f"unknown metric {metric!r}")

    n = x.shape[0]
    out = torch.empty(n, k, dtype=torch.long, device=x.device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sim = x[start:stop] @ x.T
        if metric == "euclidean": sim = 2 * sim - sq_norm.unsqueeze(0)
        rows = torch.arange(stop - start, device=x.device)
        sim[rows, rows + start] = -float("inf")
        out[start:stop] = sim.topk(k, dim=1).indices
    return out


@torch.no_grad()
def cluster_probabilities(head, feats, chunk=8192, outputs_are_probs=True):
    was_training = head.training
    head.eval()
    out = torch.cat([head(feats[s:s + chunk]) for s in range(0, feats.shape[0], chunk)])
    head.train(was_training)
    return out if outputs_are_probs else out.softmax(dim=1)


def cluster_health(probs):
    n, nclusters = probs.shape
    assign = probs.argmax(dim=1)
    counts = torch.bincount(assign, minlength=nclusters).float()
    frac = counts / n
    results = {
        "eff_clusters_hard": _entropy(frac).exp().item(),
        "eff_clusters_soft": _entropy(probs.mean(dim=0)).exp().item(),
        "frac_empty": (counts == 0).float().mean().item(),
        "largest_frac": frac.max().item(),
        "mean_confidence": probs.max(dim=1).values.mean().item(),
    }
    return assign, counts, results


def neighbour_agreement(assign, neighbours, counts):
    frac = counts / counts.sum()
    chance = (frac ** 2).sum().item()
    results = {"nbr_chance": chance}
    for metric, nbrs in neighbours.items():
        agree = (assign[nbrs] == assign.unsqueeze(1)).float().mean().item()
        results[f"nbr_{metric}_agreement"] = agree
        results[f"nbr_{metric}_lift"] = agree / chance if chance > 0 else math.nan
    return results

def assignment_stability(assign, prev_assign):

    ## Do something sensible for the zero-th iteration
    if prev_assign is None: return {}

    ## Check difference with last iteration
    a = assign.cpu().numpy()
    b = prev_assign.cpu().numpy()
    return {"frac_changed": float((a != b).mean()),
            "ari_prev": adjusted_rand_score(b, a)}


def run_cluster_monitoring(head, feats, neighbours, prev_assign, outputs_are_probs=True):
    probs = cluster_probabilities(head, feats, outputs_are_probs=outputs_are_probs)
    assign, counts, results = cluster_health(probs)
    results.update(neighbour_agreement(assign, neighbours, counts))
    results.update(assignment_stability(assign, prev_assign))
    return results, assign

