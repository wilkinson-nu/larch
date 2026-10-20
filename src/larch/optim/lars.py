import torch
from math import cos, pi

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
        # id(p) -> tensor([relative_step, grad_norm])
        self.interval_max = {}

    def pop_interval_max(self):
        out, self.interval_max = self.interval_max, {}
        return out        
        
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

                ## Get the gradient norm before scaling
                grad_norm = torch.norm(p.grad)
                weight_norm = torch.norm(p)

                ## Weight decay, BEFORE getting the update norm
                update = p.grad
                if weight_decay != 0.0:
                    update = update.add(p, alpha=weight_decay)
                                                            
                update_norm = torch.norm(update)
                
                ## Only apply to LARS scaling to requested layers
                trust_ratio = torch.ones_like(weight_norm)
                if not lars_exclude:
                    
                    # A trust ratio of one is the fallback for zero-norm
                    # weights or updates, allowing zero-initialized tensors
                    # to begin learning.
                    valid = (weight_norm > 0) & (update_norm > 0)
                    trust_ratio = torch.where(valid, trust_coef * weight_norm / (update_norm + eps), trust_ratio)
                    update = update.mul(trust_ratio)

                # Momentum
                state = self.state[p]
                if "momentum_buffer" not in state:
                    buf = state["momentum_buffer"] = update.detach().clone()
                else:
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(update)
                buf_norm = torch.norm(buf)

                # Needs the pre-update weights. The applied update is -lr * buf,
                # so a positive value means the step increases ||p||.
                if collect_stats:
                    radial_cos = -torch.dot(buf.flatten(), p.flatten()) / (
                        buf_norm * weight_norm).clamp_min(1e-12)
                
                # Apply update
                p.add_(buf, alpha=-lr)

                if collect_stats:
                    name = names[param_idx] if names is not None else f"group{group_idx}_param{param_idx}"
                    scaled_update_norm = torch.norm(update)
                    group_stats.append({
                        "pid": id(p),
                        "name": name,
                        "weight_norm": weight_norm.item(),
                        "grad_norm": grad_norm.item(),
                        "trust_ratio": trust_ratio.item(),
                        "relative_step": relative_step.item() if weight_norm > 0 else float("nan"),
                        "momentum_amplification": (buf_norm / scaled_update_norm.clamp_min(1e-12)).item(),
                        "decay_ratio": (weight_decay * weight_norm / grad_norm.clamp_min(1e-12)).item(),
                        "radial_cos": radial_cos.item(),
                        "fallback": bool(not lars_exclude and not valid),
                        "post_abs_max": p.abs().max().item(),
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
