import torch
import torch.nn as nn
import torch.nn.functional as F
from larch.distributed import GatherLayer
import torch.distributed as dist

class VICRegLossDistributed(nn.Module):
    def __init__(
        self,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
        eps=1e-4,
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps

    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _std_hinge(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(1.0 - std))

    def _cov_penalty(self, x):
        n, d = x.shape
        xc = x - x.mean(dim=0)
        cov = xc.T @ xc / max(n - 1, 1)
        return self.off_diagonal(cov).pow(2).sum() / d
    
    def forward(self, emb_cat):
        assert emb_cat.shape[0] % 2 == 0

        batch_size = emb_cat.shape[0] // 2
        z1 = emb_cat[:batch_size]
        z2 = emb_cat[batch_size:]

        # Paired-view invariance. DDP averaging gives the global mean
        # when local batch sizes are equal.
        invariance_loss = F.mse_loss(z1, z2)

        # Variance and covariance should use the global distributed batch.
        z1_all = torch.cat(GatherLayer.apply(z1), dim=0)
        z2_all = torch.cat(GatherLayer.apply(z2), dim=0)

        variance_loss = self._std_hinge(z1_all) + self._std_hinge(z2_all)
        covariance_loss = self._cov_penalty(z1_all) + self._cov_penalty(z2_all)

        loss = (self.sim_coeff * invariance_loss
                + self.std_coeff * variance_loss 
                + self.cov_coeff * covariance_loss)

        return loss, {
            "invariance": invariance_loss.detach(),
            "variance": variance_loss.detach(),
            "covariance": covariance_loss.detach(),
        }
    

## This slightly speculative, an attempt to fix the shortcut on event number by adding a term which deliberately splits local batches by the number of hits
class VICRegLossModDistributed(nn.Module):
    def __init__(
        self,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
        eps=1e-4,
        strat_lambda=0.0,       # 0 = standard VICReg; 1 = fully chunked variance
        strat_chunks=8,         # K: number of occupancy-quantile chunks per global batch
    ):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.eps = eps
        self.strat_lambda = strat_lambda
        self.strat_chunks = strat_chunks
        
    @staticmethod
    def off_diagonal(x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _std_hinge(self, x):
        std = torch.sqrt(x.var(dim=0, unbiased=False) + self.eps)
        return torch.mean(F.relu(1.0 - std))

    def _cov_penalty(self, x):
        n, d = x.shape
        xc = x - x.mean(dim=0)
        cov = xc.T @ xc / max(n - 1, 1)
        return self.off_diagonal(cov).pow(2).sum() / d

    def _gather_no_grad(self, x):
        if not (dist.is_available() and dist.is_initialized()): return x
        out = [torch.empty_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x.contiguous())
        return torch.cat(out, dim=0)
    
    def _chunked_terms(self, z1_all, z2_all, key_all):
        ## Stable sort on gathered inputs keeps things consistent between ranks
        order = torch.argsort(key_all, stable=True)

        ## Loop over chunks, and calculate the variance term for each chunk
        var_per_chunk, key_std = [], []

        chunks = torch.tensor_split(order, self.strat_chunks)
        for idx in chunks:
            a, b = z1_all[idx], z2_all[idx]
            var_per_chunk.append(self._std_hinge(a) + self._std_hinge(b))
            key_std.append(key_all[idx].std(unbiased=False))

        var_per_chunk = torch.stack(var_per_chunk)
        var_chunk = var_per_chunk.mean()

        return var_chunk, var_per_chunk, torch.stack(key_std)

    ## This now requires a key to be given for stratifying the number of events into sub-batches
    def forward(self, emb_cat, strat_key):
        assert emb_cat.shape[0] % 2 == 0

        batch_size = emb_cat.shape[0] // 2
        z1 = emb_cat[:batch_size]
        z2 = emb_cat[batch_size:]

        # Paired-view invariance. DDP averaging gives the global mean
        # when local batch sizes are equal.
        invariance_loss = F.mse_loss(z1, z2)

        # Variance and covariance should use the global distributed batch.
        z1_all = torch.cat(GatherLayer.apply(z1), dim=0)
        z2_all = torch.cat(GatherLayer.apply(z2), dim=0)

        var_global = self._std_hinge(z1_all) + self._std_hinge(z2_all)
        covariance_loss = self._cov_penalty(z1_all) + self._cov_penalty(z2_all)

        ## Now deal with the stratified sub-batches
        lam = self.strat_lambda
        assert strat_key.shape[0] == batch_size
        key_all = self._gather_no_grad(strat_key.detach().float().to(emb_cat.device))

        ## Calculate the chunked variance
        with torch.set_grad_enabled(lam > 0.0):
            var_chunk, var_per_chunk, key_std = self._chunked_terms(z1_all, z2_all, key_all)

        ## Add to the variance loss
        if lam > 0.0:
            variance_loss = (1.0 - lam) * var_global + lam * var_chunk
        else:
            variance_loss = var_global

        loss = (self.sim_coeff * invariance_loss
                + self.std_coeff * variance_loss
                + self.cov_coeff * covariance_loss)

        logs = {
            "invariance": invariance_loss.detach(),
            "variance_global": var_global.detach(),
            "covariance": covariance_loss.detach(),
            "variance": variance_loss.detach(),
            "variance_chunked": var_chunk.detach(),
            "variance_gap": (var_chunk - var_global).detach(),
        }

        ## Track what is happening for each chunk
        width = len(str(self.strat_chunks - 1))
        for i in range(self.strat_chunks):
            logs[f"variance_chunk_{i:0{width}d}"] = var_per_chunk[i].detach()
        
        return loss, logs
