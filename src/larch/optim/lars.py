import torch
from math import cos, pi

import math
import torch
from larch.training.logging import log_scalar

class LARS(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr,
        momentum=0.9,
        weight_decay=0.0,
        trust_coef=0.001,
        eps=1e-8,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            trust_coef=trust_coef,
            eps=eps,
            lars_exclude=False,
        )
        super().__init__(params, defaults)

        # Diagnostics from the most recent step.
        self.last_stats = {}

    @torch.no_grad()
    def step(self):
        
        # Clear statistics from the previous step
        collect_stats = getattr(self, "collect_stats", False)
        self.last_stats = {}

        for group_idx, group in enumerate(self.param_groups):
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            trust_coef = group["trust_coef"]
            eps = group["eps"]
            lars_exclude = group.get("lars_exclude", False)
            names = group.get("names")

            ## Statistics for each parameter group
            group_stats = []

            for param_idx, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                update = p.grad

                ## Weight decay, BEFORE getting the update norm
                if not lars_exclude and weight_decay != 0.0:
                    update = update.add(p, alpha=weight_decay)
                                                            
                update_norm = torch.norm(update)
                weight_norm = torch.norm(p)
                
                ## Only apply to LARS scaling to requested layers
                trust_ratio = torch.ones_like(weight_norm)
                if not lars_exclude:
                    
                    # A trust ratio of one is the fallback for zero-norm
                    # weights or updates, allowing zero-initialized tensors
                    # to begin learning.
                    if weight_norm > 0 and update_norm > 0:
                        trust_ratio = (
                            trust_coef
                            * weight_norm
                            / (update_norm + eps)
                        )
                        update = update.mul(trust_ratio)

                # Momentum.
                state = self.state[p]

                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = update.detach().clone()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(update)

                # Apply update
                p.add_(buf, alpha=-lr)

                if collect_stats:

                    name = (
                        names[param_idx]
                        if names is not None
                        else f"group{group_idx}_param{param_idx}"
                    )
                    
                    # Pre-update diagnostics
                    grad_norm = torch.norm(update)
                    grad_rms = grad_norm / math.sqrt(p.numel())

                    buf_norm = torch.norm(buf)

                    # Actual parameter change for this optimizer step
                    step_norm = lr * buf_norm
                    step_rms = step_norm / math.sqrt(p.numel())

                    if weight_norm > 0:
                        relative_step = step_norm / weight_norm
                    else:
                        relative_step = torch.full_like(step_norm, float("nan"))
                        
                    if update_norm > 0:
                        momentum_amplification = buf_norm / update_norm
                    else:
                        momentum_amplification = torch.full_like(buf_norm, float("nan"))
                        
                    # Post-update diagnostics
                    post_weight_rms = (
                        torch.norm(p)
                        / math.sqrt(p.numel())
                    )
                    post_abs_max = p.abs().max()
                    
                    group_stats.append({
                        "name": name,
                        "weight_norm": weight_norm.item(),
                        "grad_rms": grad_rms.item(),
                        "trust_ratio": trust_ratio.item(),
                        "step_rms": step_rms.item(),
                        "relative_step": relative_step.item(),
                        "momentum_amplification": momentum_amplification.item(),
                        "post_weight_rms": post_weight_rms.item(),
                        "post_abs_max": post_abs_max.item(),
                    })

            self.last_stats[group_idx] = group_stats
            

class LARS_LRScheduler:
    """Linear warmup from ~0 to lr_max over warmup_steps, then cosine decay
    to lr_min over the remaining (total_steps - warmup_steps) steps.

    Convention: call scheduler.step() AFTER optimizer.step().
    lr_max is the ALREADY-SCALED peak LR (e.g. ~4.8), not the base 0.3.
    """

    def __init__(self, optimizer, warmup_steps, total_steps, lr_max, lr_min=0.0):
        assert 0 <= warmup_steps < total_steps, "need 0 <= warmup < total"
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.step_count = 0
        # Set the LR for the very first optimizer step (step index 0).
        self._set_lr(self._lr_at(0))

    def _lr_at(self, step):
        if step < self.warmup_steps:
            # Linear warmup: reaches lr_max at step == warmup_steps.
            return self.lr_max * (step + 1) / self.warmup_steps
        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + cos(pi * progress))
        return self.lr_min + (self.lr_max - self.lr_min) * cosine

    def _set_lr(self, lr):
        for group in self.optimizer.param_groups:
            if group.get('lars_exclude', False):
                group['lr'] = lr * group.get('lr_scale', 1.0)
            else:
                group['lr'] = lr

    def step(self):
        self.step_count += 1
        self._set_lr(self._lr_at(self.step_count))

    def get_last_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def state_dict(self):
        return {'step_count': self.step_count}

    def load_state_dict(self, sd):
        self.step_count = sd['step_count']
        self._set_lr(self._lr_at(self.step_count))


def log_lars_diagnostics(
    optimizer,
    writer,
    metrics,
    global_iter,
    weight_norm_min=1e-6,
):
    """
    Log diagnostics stored by the most recent LARS.step().

    Call immediately after optimizer.step() and before scheduler.step(),
    on rank 0 only.
    """
    print(f"\nLARS diagnostics at step {global_iter}")

    for group_idx, stats in optimizer.last_stats.items():
        if not stats:
            continue

        group = optimizer.param_groups[group_idx]
        group_name = group.get("group_name", f"group{group_idx}")
        prefix = f"lars/{group_name}"

        lr = group["lr"]
        lars_exclude = group.get("lars_exclude", False)

        values = torch.tensor(
            [
                [
                    s["weight_norm"],
                    s["grad_rms"],
                    s["trust_ratio"],
                    s["step_rms"],
                    s["relative_step"],
                    s["momentum_amplification"],
                    s["post_weight_rms"],
                    s["post_abs_max"],
                ]
                for s in stats
            ],
            dtype=torch.float32,
        )

        (
            weight_norm,
            grad_rms,
            trust_ratio,
            step_rms,
            relative_step,
            momentum_amp,
            post_weight_rms,
            post_abs_max,
        ) = values.unbind(dim=1)

        # Metrics useful for every group.
        log_scalar(
            writer, metrics, f"{prefix}/lr",
            lr, global_iter,
        )
        log_scalar(
            writer, metrics, f"{prefix}/grad_rms_median",
            grad_rms.median().item(), global_iter,
        )
        log_scalar(
            writer, metrics, f"{prefix}/grad_rms_max",
            grad_rms.max().item(), global_iter,
        )
        log_scalar(
            writer, metrics, f"{prefix}/step_rms_median",
            step_rms.median().item(), global_iter,
        )
        log_scalar(
            writer, metrics, f"{prefix}/step_rms_max",
            step_rms.max().item(), global_iter,
        )
        log_scalar(
            writer, metrics, f"{prefix}/post_weight_rms_max",
            post_weight_rms.max().item(), global_iter,
        )
        log_scalar(
            writer, metrics, f"{prefix}/post_abs_max",
            post_abs_max.max().item(), global_iter,
        )

        writer.add_histogram(
            f"{prefix}/log10_step_rms",
            step_rms.clamp_min(1e-12).log10(),
            global_iter,
        )

        # Relative step is undefined for zero-initialized parameters.
        valid_relative = (
            (weight_norm > weight_norm_min)
            & torch.isfinite(relative_step)
        )

        if valid_relative.any():
            relative = relative_step[valid_relative]

            log_scalar(
                writer, metrics, f"{prefix}/relative_step_median",
                relative.median().item(), global_iter,
            )
            log_scalar(
                writer, metrics, f"{prefix}/relative_step_max",
                relative.max().item(), global_iter,
            )

            writer.add_histogram(
                f"{prefix}/log10_relative_step",
                relative.clamp_min(1e-12).log10(),
                global_iter,
            )

        # Trust ratio is meaningful only for LARS-managed groups.
        if not lars_exclude:
            valid_trust = torch.isfinite(trust_ratio)

            if valid_trust.any():
                trust = trust_ratio[valid_trust]

                log_scalar(
                    writer, metrics, f"{prefix}/trust_median",
                    trust.median().item(), global_iter,
                )
                log_scalar(
                    writer, metrics, f"{prefix}/trust_min",
                    trust.min().item(), global_iter,
                )
                log_scalar(
                    writer, metrics, f"{prefix}/trust_max",
                    trust.max().item(), global_iter,
                )

                writer.add_histogram(
                    f"{prefix}/log10_trust",
                    trust.clamp_min(1e-12).log10(),
                    global_iter,
                )

        valid_amp = torch.isfinite(momentum_amp)

        if valid_amp.any():
            amp = momentum_amp[valid_amp]

            log_scalar(
                writer, metrics, f"{prefix}/momentum_amp_median",
                amp.median().item(), global_iter,
            )
            log_scalar(
                writer, metrics, f"{prefix}/momentum_amp_max",
                amp.max().item(), global_iter,
            )

        largest_idx = step_rms.argmax().item()
        largest = stats[largest_idx]

        print(
            f"  {group_name}: "
            f"lr={lr:.4e}, "
            f"step_rms_max={largest['step_rms']:.4e}, "
            f"parameter={largest['name']}"
        )
