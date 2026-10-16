import torch
import torch.distributed as dist
import MinkowskiEngine as ME

@torch.no_grad()
def extract_features(encoder, loader, device):
    
    was_training = encoder.training
    encoder.eval()
    model = encoder.module if hasattr(encoder, "module") else encoder
    
    feats = []
    labels = None
    
    for bcoords, bfeats, blabels, bs in loader:
        bcoords = bcoords.to(device, non_blocking=True)
        bfeats  = bfeats.to(device,  non_blocking=True)
        batch   = ME.SparseTensor(bfeats, bcoords, device=device)
        feats.append(model(batch, bs).float())

        if labels is None:
            labels = {group: {name: [] for name in names} for group, names in blabels.items()}
        for group, names in blabels.items():
            for name, values in names.items():
                labels[group][name].append(values.to(device, non_blocking=True))
                
    if was_training:
        encoder.train()

    feats = torch.cat(feats)

    def gather_equal_size(tensor):
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor.contiguous())
        return torch.cat(gathered, dim=0)

    feats = gather_equal_size(feats)

    ## Concatenate local batches
    labels = {
        group: {name: gather_equal_size(torch.cat(chunks)) for name, chunks in names.items()}
        for group, names in labels.items()
    }

    return {"feats": feats, "labels": labels}
