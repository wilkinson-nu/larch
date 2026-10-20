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


def param_labels(module, stage_depth=1, residual_ids=frozenset()):
    """Yield (param, labels) for each trainable parameter, labelled by its owning module."""
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

        for param_name, param in owner.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if kind == "norm":
                role = {"weight": "gamma", "bias": "beta"}.get(param_name, param_name)
                if id(param) in residual_ids:
                    role = "gamma_residual"
            elif param_name == "kernel":          # MinkowskiEngine conv weights
                role = "weight"
            else:
                role = param_name
            stage = ".".join(module_name.split(".")[:stage_depth]) or "root"
            yield param, {"kind": kind, "role": role, "stage": stage}


@torch.no_grad()
def log_weight_norms(module, tag, writer, iteration, near_zero_gamma=0.05):
    if writer is None: return

    by_type = defaultdict(float)                    # e.g. "norm/gamma_residual"
    by_stage = defaultdict(float)                   # e.g. "layer4"
    gamma_counts = defaultdict(lambda: [0, 0, 0])   # negative, near zero, total

    residual_ids = {id(g) for g in _get_residual_gamma_ids(module)}
    for param, labels in param_labels(module, residual_ids=residual_ids):
        sq_norm = param.detach().float().pow(2).sum().item()
        by_type[f"{labels['kind']}/{labels['role']}"] += sq_norm
        by_stage[f"{labels['stage']}"] += sq_norm

        # Residual gammas are excluded here
        if labels["kind"] == "norm" and labels["role"] == "gamma":
            gamma = param.detach()
            counts = gamma_counts[f"{labels['stage']}"]
            counts[0] += int((gamma < 0).sum())
            counts[1] += int((gamma.abs() < near_zero_gamma).sum())
            counts[2] += gamma.numel()

    for category, sq_norm in by_type.items():
        writer.add_scalar(f"weights/{tag}/{category}", math.sqrt(sq_norm), iteration)
    for category, sq_norm in by_stage.items():
        writer.add_scalar(f"weights/{category}", math.sqrt(sq_norm), iteration)
    for category, (negative, near_zero, total) in gamma_counts.items():
        writer.add_scalar(f"bn/{category}/frac_gamma_negative", negative / total, iteration)
        writer.add_scalar(f"bn/{category}/frac_gamma_near_zero", near_zero / total, iteration)
