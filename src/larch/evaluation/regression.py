import itertools
import math

import torch

from larch.distributed import print0
from larch.training.logging import log_scalar

## Ridge penalties relative to the mean eigenvalue of the (standardized) gram matrix,
## so the grid means the same thing regardless of bank size or feature dimension.
REL_ALPHAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _log10_or_nan(y):
    ## Non-positive values become NaN and are masked out, rather than clamped into outliers
    return torch.where(y > 0, torch.log10(y.clamp_min(1e-30)), torch.full_like(y, float("nan")))


TRANSFORMS = {
    "linear": lambda y: y,
    "log10": _log10_or_nan,
}


def _standardize(Xb, Xq):
    ## Bank statistics only, with the same low-variance floor as the linear probe
    mu = Xb.mean(dim=0, keepdim=True)
    sd = Xb.std(dim=0, unbiased=False, keepdim=True)
    sd = sd.clamp_min(0.01 * sd.median().clamp_min(1e-12))
    return (Xb - mu) / sd, (Xq - mu) / sd


def _ridge_solutions(X, y, rel_alphas):
    """Closed-form ridge for several penalties from one eigendecomposition.
    Centres on the rows given, so the intercept is unpenalized."""
    xm, ym = X.mean(dim=0), y.mean()
    Xc, yc = X - xm, y - ym
    s, V = torch.linalg.eigh(Xc.T @ Xc)
    s = s.clamp_min(0.0)
    b = V.T @ (Xc.T @ yc)
    scale = s.mean()
    sols = []
    for a in rel_alphas:
        w = V @ (b / (s + a * scale))
        sols.append((w, ym - xm @ w))
    return sols


def ridge_fit_eval(Xb, yb, Xq, yq, *, rel_alphas=REL_ALPHAS, val_frac=0.2, seed=0):
    """Fit ridge on the bank (alpha chosen on a held-out bank split), evaluate on the query."""
    Xb, Xq = _standardize(Xb, Xq)

    ## Alpha selection on a held-out split of the bank
    n = Xb.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).to(Xb.device)
    nval = int(val_frac * n)
    va, tr = perm[:nval], perm[nval:]

    sols = _ridge_solutions(Xb[tr], yb[tr], rel_alphas)
    val_mse = torch.stack([((Xb[va] @ w + c - yb[va]) ** 2).mean() for w, c in sols])
    best = int(val_mse.argmin())

    ## Refit on the full bank with the chosen alpha
    (w, c), = _ridge_solutions(Xb, yb, [rel_alphas[best]])
    resid = Xq @ w + c - yq
    r2 = 1.0 - resid.pow(2).sum() / (yq - yq.mean()).pow(2).sum()

    return {
        "r2": r2.item(),
        "rmse": resid.pow(2).mean().sqrt().item(),   # in transformed units
        "log10_rel_alpha": math.log10(rel_alphas[best]),
        #"alpha_at_edge": float(best in (0, len(rel_alphas) - 1)),
    }


def _poly_basis(cols_b, cols_q, degree=3):
    """Full polynomial basis (with cross terms) up to `degree`, standardized on valid bank rows."""
    Zb, Zq = torch.stack(cols_b, dim=1), torch.stack(cols_q, dim=1)
    valid = torch.isfinite(Zb).all(dim=1)
    mu = Zb[valid].mean(dim=0)
    sd = Zb[valid].std(dim=0).clamp_min(1e-12)
    Zb, Zq = (Zb - mu) / sd, (Zq - mu) / sd

    combos = [c for d in range(1, degree + 1)
              for c in itertools.combinations_with_replacement(range(Zb.shape[1]), d)]

    def expand(Z):
        return torch.stack([Z[:, list(c)].prod(dim=1) for c in combos], dim=1)

    return expand(Zb), expand(Zq)


@torch.no_grad()
def run_regression_probes(bank,
                          query,
                          targets,
                          *,
                          occ_basis=None,
                          basis_degree=3,
                          seed=0,
                          min_events=1000):
    """
    targets:   {group: {name: transform}}, e.g. {"event": {"enu": "log10"}}
    occ_basis: optional tuple of (group, name) used to build an occupancy-only baseline
               (always log10-transformed). Enables partial R^2 for non-basis targets.
    """
    print0("Running regression probes")

    Fb = bank["feats"].double()
    Fq = query["feats"].double()

    if occ_basis:
        occ_basis = tuple(tuple(x) for x in occ_basis)
        cols_b = [_log10_or_nan(bank["labels"][g][n].double()) for g, n in occ_basis]
        cols_q = [_log10_or_nan(query["labels"][g][n].double()) for g, n in occ_basis]
        Bb, Bq = _poly_basis(cols_b, cols_q, basis_degree)

    results = {}
    for group, names in targets.items():
        for name, tf in names.items():
            yb = TRANSFORMS[tf](bank["labels"][group][name].double())
            yq = TRANSFORMS[tf](query["labels"][group][name].double())

            mb, mq = torch.isfinite(yb), torch.isfinite(yq)
            if occ_basis:
                ## Same rows for all three fits, so the R^2 values are comparable
                mb &= torch.isfinite(Bb).all(dim=1)
                mq &= torch.isfinite(Bq).all(dim=1)

            if mb.sum() < min_events or mq.sum() < min_events:
                print0(f"Skipping regression target {group}/{name}: too few valid events")
                continue

            res = ridge_fit_eval(Fb[mb], yb[mb], Fq[mq], yq[mq], seed=seed)
            # res["frac_valid"] = mq.double().mean().item()

            ## Partial R^2: what h adds beyond the occupancy basis
            if occ_basis and (group, name) not in occ_basis:
                r_b = ridge_fit_eval(Bb[mb], yb[mb], Bq[mq], yq[mq], seed=seed)
                r_hb = ridge_fit_eval(torch.cat([Fb[mb], Bb[mb]], dim=1), yb[mb],
                                      torch.cat([Fq[mq], Bq[mq]], dim=1), yq[mq],
                                      seed=seed)
                # res["r2_basis"] = r_b["r2"]
                # res["r2_with_basis"] = r_hb["r2"]
                res["partial_r2"] = (r_hb["r2"] - r_b["r2"]) / max(1.0 - r_b["r2"], 1e-12)

            results[f"{group}_{name}"] = res

    return results


def log_regression_results(writer, metrics, regression_results, iteration):
    if not regression_results: return
    for target, result in regression_results.items():
        for name, value in result.items():
            log_scalar(writer, metrics, f"regress/{target}_{name}", value, iteration)
