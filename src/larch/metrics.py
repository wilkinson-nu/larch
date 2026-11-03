import torch
from torch import nn
import torch.distributed as dist

@torch.no_grad()
def alignment(z_cat):
    z_cat = nn.functional.normalize(z_cat, dim=1)
    batch_size = z_cat.shape[0] // 2
    z_i, z_j = z_cat[:batch_size], z_cat[batch_size:]    
    cos = (z_i * z_j).sum(dim=1)
    return (2 - 2 * cos).mean()

@torch.no_grad()
def uniformity(z, t=2):
    z = nn.functional.normalize(z.float(), dim=1)
    sim = z @ z.T  # cosine similarity matrix since z is normalised
    sq_pdist = 2 - 2 * sim
    mask = ~torch.eye(z.size(0), device=z.device, dtype=torch.bool)
    vals = torch.exp(-t * sq_pdist)
    masked_sum = (vals * mask).sum()
    masked_cnt = mask.sum()
    return torch.log(masked_sum / masked_cnt)
    
@torch.no_grad()
def argmax_consistency(c_cat):
    batch_size = c_cat.shape[0] // 2
    c_i, c_j = c_cat[:batch_size], c_cat[batch_size:]
    
    argmax_i = torch.argmax(c_i, dim=1)
    argmax_j = torch.argmax(c_j, dim=1)
    
    same = (argmax_i == argmax_j).float()
    return same.mean()


@torch.no_grad()
def geometry_metrics(buffer, device, normalize=True, sim_stats=True):

    '''
    Each element in buffer is the concatenation of the two views in a batch
    Loop over the buffer and calculate values for each batch and then average
    '''

    dim = buffer[0].shape[1]
    ssum = torch.zeros(dim, dtype=torch.float64, device=device)
    smat = torch.zeros(dim, dim, dtype=torch.float64, device=device)

    ## Add separate views
    ssum_m = torch.zeros(dim, dtype=torch.float64, device=device)
    smat_m = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    ssum_d = torch.zeros(dim, dtype=torch.float64, device=device)
    smat_d = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    
    n = 0
    
    ## Keep track of values
    pos_buffer = []
    neg_buffer = []
    negmean_buffer = []

    ## For gathering
    world = dist.get_world_size()

    ## loop over buffer
    for emb_cat_cpu in buffer:

        z_cat = emb_cat_cpu.to(device)
        
        B = z_cat.shape[0]//2

        ## Optionally normalize
        if normalize: z_cat = z_cat / (z_cat.norm(dim=1, keepdim=True) + 1e-8)

        ## Gather from all GPUs
        gi = [torch.zeros_like(z_cat[:B]) for _ in range(world)]
        gj = [torch.zeros_like(z_cat[B:]) for _ in range(world)]
        dist.all_gather(gi, z_cat[:B].contiguous())
        dist.all_gather(gj, z_cat[B:].contiguous())
        z_i_all, z_j_all = torch.cat(gi), torch.cat(gj)
        N = z_i_all.shape[0]

        ## Compare different views
        m = 0.5 * (z_i_all + z_j_all).double()
        d = 0.5 * (z_i_all - z_j_all).double()
        ssum_m += m.sum(0); smat_m += m.T @ m
        ssum_d += d.sum(0); smat_d += d.T @ d

        z_all = torch.cat([z_i_all, z_j_all], dim=0)
        
        #######################
        ### Geometry metrics ##
        #######################

        if sim_stats:
            sim = z_all @ z_all.t()
            idx = torch.arange(2*N, device=device)
            pos_idx = (idx + N) % (2*N)
            pos_buffer .append(sim[idx, pos_idx].clone())
            
            ## Now modify sim for calculating hard negatives
            sim.fill_diagonal_(-float("inf"))
            sim[idx, pos_idx] = -float("inf")
            neg_buffer .append(sim.max(dim=1).values)
            finite = sim[sim > -float("inf")]
            negmean_buffer.append(finite.mean())
            del sim
        
        #######################
        # Effective dimension #
        #######################

        zd = z_all.double()
        ssum += zd.sum(0)
        smat += zd.T @ zd
        n += zd.shape[0]
        
    ## Now calculate the covariance info
    mu = ssum / n
    cov = (smat - n * torch.outer(mu, mu)) / (n - 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    deff = (eigvals.sum() ** 2) / (eigvals.pow(2).sum())
    l1_ratio = eigvals.max() / eigvals.sum()

    ## RankMe
    s = eigvals.sqrt()
    p = s / s.sum() + 1e-7
    rankme = torch.exp(-(p * p.log()).sum())

    def _l1(ssum, smat, n):
        mu = ssum / n
        c = (smat - n * torch.outer(mu, mu)) / (n - 1)
        e = torch.linalg.eigvalsh(c).clamp_min(0)
        return e[-1] / e.sum()

    ## Calculate mean and different l1_ratio between the two views
    mean_l1_ratio = _l1(ssum_m, smat_m, n // 2)
    diff_l1_ratio = _l1(ssum_d, smat_d, n // 2)
    
    
    ## Return without sim_stats
    out = {
        "deff": deff.item(),
        "rankme": rankme.item(),
        "l1_ratio": l1_ratio,
        "l1_ratio_mean": mean_l1_ratio,
        "l1_ratio_diff": diff_l1_ratio,        
        **{
            f"lambda{i}": val.item()
            for i, val in enumerate(eigvals.flip(0)[:10])
        },
    }

    if sim_stats:
        ## Calculate the SimCLR geometry values
        all_pos = torch.cat(pos_buffer, dim=0)
        all_neg = torch.cat(neg_buffer, dim=0)
        all_meanneg = torch.stack(negmean_buffer)
        gap = all_pos - all_neg

        out.update({
            "pos": all_pos.mean().item(),
            "hard_neg": all_neg.mean().item(),
            "mean_neg": all_meanneg.mean().item(),
            "gap": gap.mean().item(),
            "gap_std": gap.std(unbiased=False).item(),
        })
    return out

def simclr_geometry_metrics(buffer, device, normalize=True):
    return geometry_metrics(buffer, device, normalize, sim_stats=True)

def basic_geometry_metrics(buffer, device, normalize=True):
    return geometry_metrics(buffer, device, normalize, sim_stats=False)
