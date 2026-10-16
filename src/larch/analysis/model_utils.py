import torch
import argparse
import os
import glob

from larch.models.resnet_encoder import get_encoder
from larch.models.projection_head import get_projhead
from larch.models.clustering_head import get_clusthead

def resolve_path(pattern):
    pattern = os.path.expanduser(os.path.expandvars(pattern))
    if any(c in pattern for c in "*?["):
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no match for {pattern!r}")
        if len(matches) > 1:
            raise ValueError(f"ambiguous pattern {pattern!r}: {matches}")
        return matches[0]
    return pattern

def load_checkpoint(state_file_name):
    checkpoint = torch.load(state_file_name, map_location='cpu')
    
    # Reconstruct args Namespace
    args = argparse.Namespace(**checkpoint['args'])
    return checkpoint, args

HEAD_BUILDERS = {
    "proj":  get_projhead,
    "clust": get_clusthead,
}

def get_models_from_checkpoint(state_file_name):

    ## Expand paths (but fail on any conflicts)
    state_file_name = resolve_path(state_file_name)    
    checkpoint, args = load_checkpoint(state_file_name)

    ## Get the models
    encoder = get_encoder(args)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    nchan = encoder.get_nchan()

    ## Fail loudly on head types this function doesn't know how to build
    saved_heads = {k.removesuffix("_head_state_dict")
                   for k in checkpoint if k.endswith("_head_state_dict")}
    unknown = saved_heads - HEAD_BUILDERS.keys()
    if unknown:
        raise ValueError(f"{state_file_name} contains unknown heads: {sorted(unknown)}")

    ## Load whichever heads were saved
    heads = {}
    for name in sorted(saved_heads):
        heads[name] = HEAD_BUILDERS[name](nchan, args)
        heads[name].load_state_dict(checkpoint[f"{name}_head_state_dict"])

    print(f"Loaded encoder + heads {sorted(heads)} from: {state_file_name}")
    return encoder, heads, args

def get_encoder_from_checkpoint(state_file_name):
    checkpoint, args = load_checkpoint(state_file_name)
    encoder = get_encoder(args)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    return encoder, None, args

def print_model_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
            total_params += param.numel()
    print("Total parameters =", total_params)
