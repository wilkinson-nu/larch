from collections import defaultdict
import math

import torch
import torch.nn as nn

## A simply logging utility
def log_scalar(writer, metrics, name, value, step):
    if writer is not None:
        writer.add_scalar(name, value, step)
    metrics[name].append(value)

## Some diagnostic functions
@torch.no_grad()
def log_grad_norm(module, tag, writer, iteration):
    if writer is None: return
    device = next(module.parameters()).device
    total = torch.zeros((), device=device)
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        total += p.grad.pow(2).sum()

    grad_norm = total.sqrt().item()
    writer.add_scalar(f'grads/{tag}/sqrt_sum_grad_l2', grad_norm, iteration)
    return

@torch.no_grad()
def log_grad_rms(module, tag, writer, iteration):
    if writer is None: return

    device = next(module.parameters()).device
    total = torch.zeros((), device=device)
    count = 0
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        total += p.grad.pow(2).sum()
        count += p.grad.numel()

    if count == 0: return
    mean_rms = (total / count).sqrt().item()
    writer.add_scalar(f'grads/{tag}/mean_grad_rms', mean_rms, iteration)
    return

@torch.no_grad()
def log_grad_over_wgt(module, tag, writer, iteration, eps=1e-12):
    if writer is None: return
    device = next(module.parameters()).device
    g2 = torch.zeros((), device=device)
    w2 = torch.zeros((), device=device)
    
    for name, p in module.named_parameters():
        if p.grad is None:
            continue
        g2 += p.grad.pow(2).sum()
        w2 += p.data.pow(2).sum()
        

    ratio = (g2.sqrt() / (w2.sqrt() + eps)).item()
    writer.add_scalar(f'grads/{tag}/sum_grad_over_wgt', ratio, iteration)
    return

@torch.no_grad()
def log_weight_norm(module, tag, writer, iteration):
    if writer is None: return
    device = next(module.parameters()).device
    total = torch.zeros((), device=device)
    count = 0
    for name, p in module.named_parameters():
        if not p.requires_grad:
            continue
        total += p.pow(2).sum()
        count += p.numel()

    if count == 0: return

    # global weight L2 norm and element-weighted RMS
    weight_l2 = total.sqrt().item()
    weight_rms = (total / count).sqrt().item()
    writer.add_scalar(f'weights/{tag}/l2', weight_l2, iteration)
    writer.add_scalar(f'weights/{tag}/rms', weight_rms, iteration)


## Magic function to deal with the uninitialised gammas in BNs at the end of a residual  block
def _get_residual_gamma_ids(module):
    get = getattr(module, "residual_gammas", None)
    return {id(p) for p in get()} if callable(get) else set()

def param_fields(optimizer, modules, split=("encoder",)):
    """Return {id(p): fields} for every trainable parameter in `modules`.

    group:    optimizer param-group name, e.g. "lars", "omit"
    layer:    top-level child for tags in `split` (e.g. "layer2"), otherwise the tag (e.g. "proj")
    category: "<kind>/<role>", e.g. "conv/weight", "norm/gamma", "norm/gamma_residual"
    """
    group_of = {id(p): g.get("group_name", str(i))
                for i, g in enumerate(optimizer.param_groups) for p in g["params"]}
    fields = {}
    for tag, module in modules.items():
        residual = _get_residual_gamma_ids(module)
        for module_name, owner in module.named_modules():
            cls = type(owner).__name__
            if isinstance(owner, nn.modules.batchnorm._BatchNorm) or "Norm" in cls:
                kind = "norm"
            elif isinstance(owner, nn.modules.conv._ConvNd) or "Convolution" in cls:
                kind = "conv"
            elif isinstance(owner, nn.Linear):
                kind = "linear"
            else:
                kind = cls.lower()
            layer = (module_name.split(".")[0] or tag) if tag in split else tag

            for param_name, p in owner.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if kind == "norm":
                    role = ("gamma_residual" if id(p) in residual
                            else {"weight": "gamma", "bias": "beta"}.get(param_name, param_name))
                elif param_name == "kernel":  # MinkowskiEngine conv weights
                    role = "weight"
                else:
                    role = param_name
                fields[id(p)] = dict(param=p, group=group_of.get(id(p), "none"),
                                     layer=layer, category=f"{kind}/{role}")
    return fields

VIEWS = (
    ("by_group", ("group", "layer", "category")),
    ("by_group", ("group", "category")),
    ("by_layer", ("layer", "group", "category")),
    ("by_layer", ("layer", "group")),
)

def view_keys(f):
    return [f"{view}/" + "/".join(f[k] for k in keys) for view, keys in VIEWS]

@torch.no_grad()
def log_weight_norms(optimize, modules, writer, iteration, near_zero_gamma=0.05):
    if writer is None: return

    ## Assemble a list of groupings that each parameter belongs to
    fields = list(param_fields(optimizer, modules).values())

    ## Get weight information for each parameter
    rows = []
    for f in fields:
        p = f["param"].detach().float()
        rows.append(torch.stack([p.pow(2).sum(), (p < 0).sum().float(),
                                 (p.abs() < near_zero_gamma).sum().float()]))
    rows = torch.stack(rows).double().cpu().tolist()

    ## Sum over parameters for each group
    sq = defaultdict(float)
    gamma = defaultdict(lambda: [0, 0, 0])  # negative, near zero, total
    for f, (sq_norm, neg, near) in zip(fields, rows):
        for key in view_keys(f):
            sq[key] += sq_norm
            if f["category"] == "norm/gamma":  # residual γ has its own category, so it's excluded
                c = gamma[key]
                c[0] += neg
                c[1] += near
                c[2] += f["param"].numel()

    ## Save information
    for key, v in sq.items():
        writer.add_scalar(f"wgt_{key}", math.sqrt(v), iteration)
    for key, (neg, near, n) in gamma.items():
        writer.add_scalar(f"bn_{key}/frac_gamma_negative", neg / n, iteration)
        writer.add_scalar(f"bn_{key}/frac_gamma_near_zero", near / n, iteration)


@torch.no_grad()
def log_lars_diagnostics(optimizer, modules, writer, global_iter, steps_per_epoch=None):
    """Log LARS diagnostics aggregated under each view in VIEWS.

    Call on rank 0 after optimizer.step() and before scheduler.step(), with
    optimizer.collect_stats set to True for that step. Interval maxima cover
    every step since the previous call; snapshot stats cover the latest step only.
    """
    
    if writer is None: return
    
    ## Assemble a list of groupings that each parameter belongs to
    fields = param_fields(optimizer, modules)
    unlabelled = dict(group="unknown", layer="unlabelled", category="unlabelled")

    # Interval maxima: one GPU->CPU transfer for everything.
    interval = optimizer.pop_interval_max()
    if interval:
        pids = list(interval)
        values = torch.stack([interval[pid] for pid in pids]).float().cpu()
        interval_max = defaultdict(lambda: torch.zeros(2))
        for pid, v in zip(pids, values):
            for key in keys_for(pid):
                interval_max[key] = torch.maximum(interval_max[key], v)
        for key, (rel_max, grad_max) in interval_max.items():
            writer.add_scalar(f"lars_{key}/relative_step_interval_max", rel_max.item(), global_iter)
            writer.add_scalar(f"lars_{key}/grad_norm_interval_max", grad_max.item(), global_iter)

    # Snapshot medians from the most recent step.
    snapshot = defaultdict(lambda: defaultdict(list))
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # sq, init_sq, pre_sq, fallback count
    for stats in optimizer.last_stats.values():
        for s in stats:
            for key in keys_for(s["pid"]):
                for q in ("relative_step", "trust_ratio", "decay_ratio", "radial_cos", "log_r"):
                    snapshot[key][q].append(s[q])
                t = totals[key]
                t[0] += s["sq"]
                t[1] += s["init_sq"]
                t[2] += s["pre_sq"]
                t[3] += s["fallback"]

    # Aggregate quantities: sums over every tensor in the view.
    for key, (sq, init_sq, pre_sq, n_fallback) in totals.items():
        if init_sq > 0:  # all-zero-init views (e.g. residual gamma, biases) have no ratio
            writer.add_scalar(f"lars_{key}/l2_over_init", math.sqrt(sq / init_sq), global_iter)
        if pre_sq > 0:
            log_rate = 0.5 * math.log(sq / pre_sq)
            writer.add_scalar(f"lars_{key}/agg_log_rate_per_step", log_rate, global_iter)
            if steps_per_epoch:
                writer.add_scalar(f"lars_{key}/agg_rate_per_epoch", per_epoch(log_rate), global_iter)
        writer.add_scalar(f"lars_{key}/fallback_count", n_fallback, global_iter)

    # Medians: every tensor counts equally.
    for key, quantities in snapshot.items():
        for q, vals in quantities.items():
            v = torch.tensor(vals, dtype=torch.float64)
            v = v[torch.isfinite(v)]
            if not v.numel():
                continue
            med = v.median().item()
            if q == "log_r":
                writer.add_scalar(f"lars_{key}/implied_r_log_per_step", med, global_iter)
                if steps_per_epoch:
                    writer.add_scalar(f"lars_{key}/implied_r_per_epoch", per_epoch(med), global_iter)
            else:
                writer.add_scalar(f"lars_{key}/{q}_median", med, global_iter)

    for gi, g in enumerate(optimizer.param_groups):
        writer.add_scalar(f"lars_by_group/{g.get('group_name', gi)}/lr", g["lr"], global_iter)
        
