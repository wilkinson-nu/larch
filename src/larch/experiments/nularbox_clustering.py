import argparse
import sys
import MinkowskiEngine as ME
import torch
import time
from collections import defaultdict
from functools import partial
from pathlib import Path
import psutil, os

## The parallelisation libraries
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn

## Includes from my libraries for this project
from larch.losses.clustering import ClusteringLossMergedMultiGPU
from larch.models.resnet_encoder import get_encoder, ENCODER_ARG_KEYS
from larch.models.clustering_head import get_clusthead
from larch.metrics import argmax_consistency, uniformity, alignment
from larch.training.logging import log_scalar, log_grad_norm, log_grad_rms, log_grad_over_wgt, log_weight_norm
from larch.optim.scheduling import get_opt_and_sched, update_weight_decay

## Import datasets
from larch.datasets.base import solo_labelled_collate_fn
from larch.datasets.dataloaders import build_paired_training_data, build_monitoring_data, monitoring_ranges

## For logging
from torch.utils.tensorboard import SummaryWriter

## Import transformations
from larch.datasets.nularbox.augmentations_2d import get_transform

## Cluster-specific metrics
from larch.probes import extract_features
from larch.cluster_metrics import encoder_neighbours, run_cluster_monitoring, log_cluster_metrics

## Utilities for multi-rank training
from larch.distributed import setup_distributed_runtime, print0

## Checkpointing
from larch.training.checkpointing import read_encoder_checkpoint, save_checkpoint

## Config handling
from larch.config import apply_config, load_config, dump_args

## Wrapped training function
def run_training(rank, local_rank, world_size, args, enc_state):

    ## For parallel work
    device = setup_distributed_runtime(
        rank,
        local_rank,
        world_size,
        seed=args.seed,
        num_workers=args.num_workers,
        print_cpu_affinity=True,
    )

    if bool(args.run_profiler) and rank==0:
        torch.cuda.set_sync_debug_mode("warn")

    torch.autograd.set_detect_anomaly(False)
    
    ## For timing
    tstart = time.time()
    
    ## Setup the encoder
    encoder = get_encoder(args)
    encoder = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(encoder)
    encoder_nchan = encoder.get_nchan()
    encoder.load_state_dict(enc_state, strict=True)
    del enc_state

    ## No DDP on the encoder because it must be frozen in this script
    encoder.to(device)
    encoder.eval()
    encoder.requires_grad_(False)

    ## Dictionary of heads
    heads = {}
    
    ## Dictionary of loss functions
    loss_fns = {}

    ## Clustering head
    clust_head = get_clusthead(encoder_nchan, args)
    clust_head = nn.SyncBatchNorm.convert_sync_batchnorm(clust_head)
    clust_head .to(device)
    clust_head = DDP(clust_head, device_ids=[local_rank])
    heads["clust"] = clust_head    
    loss_fns["clust"] = ClusteringLossMergedMultiGPU(args.clust_temp, args.entropy_scale)
        
    ## Set up the training dataset
    train_transform = get_transform(
        args.out_image_size,
        args.aug_type,
        args.aug_prob,
        args.aug_val,
    )

    ## E.g., Independent monitoring
    _, _, train_start = monitoring_ranges(args.monitor_nquery, args.monitor_nbank)
    train_dataset, train_loader = build_paired_training_data(
        data_dir=args.data_dir,
        start=train_start,
        nevents=args.nevents,
        transform=train_transform,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    nbatches   = len(train_loader)

    bank_loader, query_loader = build_monitoring_data(
        data_dir=args.data_dir,
        nbank=args.monitor_nbank,
        nquery=args.monitor_nquery,
        transform=monitor_transform,
        collate_fn=monitor_collate,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    ## Encoder is frozen, so extract features once, and then cache
    NBR_METRICS = ("cosine", "euclidean")

    qry_f, _ = extract_features(encoder, query_loader, device)
    
    norm_encoder = bool(args.norm_encoder)
    monitor_feats = None
    monitor_nbrs = None
    if rank == 0:
        monitor_feats = qry_f.float().to(device)
        
        ## Calculate neighbours
        monitor_nbrs = {metric: encoder_neighbours(monitor_feats, k=args.monitor_k, metric=metric)
                        for metric in NBR_METRICS}
        
        ## Follow the encoder normalization used in training
        if norm_encoder: monitor_feats = torch.nn.functional.normalize(monitor_feats, p=2, dim=1)

        print0(f"Cached {monitor_feats.shape[0]} query events for cluster monitoring")

    ## Loaders are no longer needed
    del qry_f, bank_loader, query_loader
    torch.cuda.empty_cache()

    ## Keep track of assignments from the last iteration
    prev_assign = None

    ## Make the log directory
    log_dir = Path(args.run_dir) / args.log
    log_dir.mkdir(parents=True, exist_ok=True)

    ## Make the state_file
    state_file = Path(args.run_dir) / args.state_file
    
    ## So we don't constantly ask args
    nepoch = args.nepoch
    weight_decay = args.weight_decay
    weight_decay_final = args.weight_decay_final
    
    print0("Training with", nepoch, "epochs")
    writer = None
    if rank==0:
        writer = SummaryWriter(log_dir=log_dir)

    ## Sort out the optimizer (one for each GPU...)
    nstep_total = nbatches*args.nepoch
    optimizer, scheduler = get_opt_and_sched(args, encoder, heads, nstep_total, world_size, print_debug=False)
    
    ## Set up metrics
    metrics = defaultdict(list)

    ## Stuff in a profiler
    if bool(args.run_profiler) and rank==0:
        
        prof = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.__enter__()
        
    ## Loop over the desired epochs
    global_iter = 0
    for epoch in range(nepoch):

        print0(f"Start of epoch {epoch}")
        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(epoch)
        
        tot_loss_tensor = torch.tensor(0.0, device=device)  
        losses_tensor = {name: torch.tensor(0.0, device=device) for name in heads.keys()}       
        entropy_tensor = torch.tensor(0.0, device=device)

        ## For monitoring
        total_acc_tensor = torch.tensor(0.0, device=device)
        total_clust_align_tensor = torch.tensor(0.0, device=device)
        total_clust_unif_tensor = torch.tensor(0.0, device=device)

        # Set train mode for the encoder and any heads
        encoder.eval()
        for h in heads.values(): h.train()
        
        # Iterate over batches of images with the dataloader
        t0 = time.time()
        first_batch_latency = None
        for cat_bcoords, cat_bfeats, this_batch_size in train_loader:

            if first_batch_latency is None:
                first_batch_latency = time.time() - t0

            ## Update weight decay to allow for scheduling
            this_wd = update_weight_decay(optimizer,
			                  weight_decay,
                                          weight_decay_final,
                                          global_iter,
                                          nstep_total)
            
            ## Send to the device, then make the sparse tensors
            cat_bcoords = cat_bcoords.to(device, non_blocking=True)
            cat_bfeats  = cat_bfeats .to(device, non_blocking=True)
            cat_batch   = ME.SparseTensor(cat_bfeats, cat_bcoords, device=device)

            ## No gradients in the encoder here
            with torch.no_grad():
                encoded_batch = encoder(cat_batch, this_batch_size)
                if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)

            clust_batch = heads["clust"](encoded_batch)
            clust_loss, clust_entropy = loss_fns["clust"](clust_batch)
            tot_loss = clust_loss + clust_entropy
            
            losses_tensor["clust"] += clust_loss.detach()
            entropy_tensor += clust_entropy.detach()
            total_acc_tensor += argmax_consistency(clust_batch)
            total_clust_align_tensor += alignment(clust_batch)
            total_clust_unif_tensor += uniformity(clust_batch)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            tot_loss .backward()

            ## Update optimizer and scheduler
            optimizer.step()
            if scheduler: scheduler.step()

            ## Increment global_iter
            global_iter += 1
            
            ## keep track of losses
            tot_loss_tensor += tot_loss.detach()

        # Manage CUDA memory for ME
        torch.cuda.empty_cache()

        ## Although the gradients are handled correctly by GatherLayer, the losses are global
        ## Strictly speaking this step isn't necessary as each mini-batch gives the same loss value
        ## But I kept it in to avoid my own headaches...
        dist.all_reduce(tot_loss_tensor, op=dist.ReduceOp.SUM)
        for name in heads.keys(): dist.all_reduce(losses_tensor[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(entropy_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_acc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_clust_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_clust_unif_tensor, op=dist.ReduceOp.SUM)        
        
        av_tot_loss = tot_loss_tensor.item() / (nbatches * world_size)
        av_losses = {
            name: losses_tensor[name].item() / (nbatches * world_size)
            for name in heads.keys()
        }
        av_entropy = entropy_tensor.item() / (nbatches * world_size)
        av_acc = total_acc_tensor.item() / (nbatches * world_size)
        av_clust_unif = total_clust_unif_tensor.item() / (nbatches * world_size)
        av_clust_align = total_clust_align_tensor.item() / (nbatches * world_size)

        ## Cluster monitoring on cached query features (rank 0 only, milliseconds)
        cluster_results = None
        if rank == 0:
            cluster_results, prev_assign = run_cluster_monitoring(
                heads["clust"].module,
                monitor_feats,
                monitor_nbrs,
                prev_assign,
            )
        
        ## Reporting, but only for rank 0
        if rank==0:
            metrics["epoch"].append(epoch)
            log_scalar(writer, metrics, 'loss/total', av_tot_loss, epoch)              
            log_scalar(writer, metrics, 'loss/clust', av_losses["clust"]+av_entropy, epoch)
            log_scalar(writer, metrics, 'loss/entropy', av_entropy, epoch)
            log_scalar(writer, metrics, 'loss/clust_only', av_losses["clust"], epoch)
            log_scalar(writer, metrics, 'monitor/acc', av_acc, epoch)
            log_scalar(writer, metrics, 'monitor/clust_alignment', av_clust_align, epoch)
            log_scalar(writer, metrics, 'monitor/clust_uniformity', av_clust_unif, epoch)

            ## Cluster metrics
            for name, value in cluster_results.items():
                log_scalar(writer, metrics, f"cluster/{name}", value, epoch)
                
            ## Extensive logging for gradient debugging
            log_grad_norm(heads["clust"].module, "clust", writer, epoch)
            log_grad_rms(heads["clust"].module, "clust", writer, epoch)
            log_grad_over_wgt(heads["clust"].module, "clust", writer, epoch)
            log_weight_norm(heads["clust"].module, "clust", writer, epoch)
            
            if scheduler: 
                log_scalar(writer, metrics, 'train/lr', scheduler.get_last_lr()[0], epoch)
            log_scalar(writer, metrics, 'train/weight_decay', this_wd, epoch)

            ## Build a string to report the outcome
            iter_string = f"Processed {epoch} / {nepoch}; loss = {av_tot_loss:.4f}" + \
                f" ({av_losses['clust']:.4f} + {av_entropy:.4f}); acc = {av_acc:.4f}"
            print0(iter_string)
            print0(f"Time taken: {(time.time()-tstart):.2f}")
            
        ## Add per GPU logging
        allocated_gb = torch.tensor(torch.cuda.memory_allocated() / 1e9, device=device)
        reserved_gb  = torch.tensor(torch.cuda.memory_reserved()  / 1e9, device=device)
        peak_alloc_gb = torch.tensor(torch.cuda.max_memory_allocated() / 1e9, device=device)
        torch.cuda.reset_peak_memory_stats()

        all_allocated  = [torch.zeros(1, device=device) for _ in range(world_size)]
        all_reserved   = [torch.zeros(1, device=device) for _ in range(world_size)]
        all_peak_alloc = [torch.zeros(1, device=device) for _ in range(world_size)]
        
        dist.all_gather(all_allocated,  allocated_gb.unsqueeze(0))
        dist.all_gather(all_reserved,   reserved_gb.unsqueeze(0))
        dist.all_gather(all_peak_alloc, peak_alloc_gb.unsqueeze(0))
            
        ## Enhanced logging
        if rank == 0:
            vm = psutil.virtual_memory()
            proc = psutil.Process(os.getpid())
            io = psutil.disk_io_counters()

            log_scalar(writer, metrics, 'syst_monitor/vm_used_gb', vm.used / 1e9, epoch)
            log_scalar(writer, metrics, 'syst_monitor/vm_avail_gb', vm.available / 1e9, epoch)
            log_scalar(writer, metrics, 'syst_monitor/vm_cached_gb', getattr(vm, "cached", 0) / 1e9, epoch)
            log_scalar(writer, metrics, 'syst_monitor/rss_gb', proc.memory_info().rss / 1e9, epoch)
            log_scalar(writer, metrics, 'syst_monitor/num_fds', proc.num_fds(), epoch)
            log_scalar(writer, metrics, 'syst_monitor/io_read', io.read_bytes, epoch)
            log_scalar(writer, metrics, 'syst_monitor/io_write', io.write_bytes, epoch)
            log_scalar(writer, metrics, 'syst_monitor/mem_pressure', vm.available / vm.total, epoch)
            log_scalar(writer, metrics, 'syst_monitor/first_batch_latency', first_batch_latency, epoch)

            for gpu_rank in range(world_size):
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_allocated_gb',  all_allocated[gpu_rank].item(),  epoch)
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_reserved_gb',   all_reserved[gpu_rank].item(),   epoch)
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_peak_alloc_gb', all_peak_alloc[gpu_rank].item(), epoch)
                
    ## Final version of the model
    if rank==0:
        save_checkpoint(encoder, heads, optimizer, scheduler, state_file, epoch, metrics, args)
        writer.close()

    ## Report profiler if requested
    if bool(args.run_profiler) and rank == 0:
        prof.__exit__(None, None, None)
        
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=100))
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))

    ## Clear things up
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()


def build_parser():

    ## Parse some args
    parser = argparse.ArgumentParser("Contrastive clustering (only) module")

    ## Basic job setup
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--nevents', type=int, required=True)
    parser.add_argument('--nepoch', type=int, required=True)
    parser.add_argument('--pretrained', type=str, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--log', type=str)
    parser.add_argument('--state_file', type=str)
    
    ## Training dynamics
    parser.add_argument('--lr', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--scheduler', type=str)
    parser.add_argument('--lars_trust_coeff', type=float)
    parser.add_argument('--lars_momentum', type=float)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--weight_decay_final', type=float)
    parser.add_argument('--weight_decay_head', type=int, choices=[0,1])
    parser.add_argument('--norm_encoder', type=int, choices=[0,1])
    parser.add_argument('--non_lars_lr_scale', type=float)
    
    ## Image size and augmentations
    parser.add_argument('--aug_type', type=str)
    parser.add_argument('--aug_prob', type=float)
    parser.add_argument('--aug_val', type=float)

    ## Clustering head
    parser.add_argument('--clust_arch', type=str)
    parser.add_argument('--clust_temp', type=float)
    parser.add_argument('--nclusters', type=int)
    parser.add_argument('--entropy_scale', type=float)

    ## Monitoring arguments
    parser.add_argument('--monitor_nbank', type=int)
    parser.add_argument('--monitor_nquery', type=int)
    parser.add_argument('--monitor_k', type=int)
    
    ## Optional profiler
    parser.add_argument('--run_profiler', type=int, choices=[0,1])

    return parser

def main(argv=None):

    ## Parse arguments starting from the config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', required=True)
    known, rest = pre.parse_known_args(argv)

    ## Then look on the command line (for overrides and required CLI args )
    parser = build_parser()
    apply_config(parser, load_config(known.config))
    args = parser.parse_args(argv)

    enc_cfg, enc_state, pretrained_epoch, pretrained_path = read_encoder_checkpoint(
        args.pretrained, ENCODER_ARG_KEYS,
    )

    for key, value in enc_cfg.items():
        setattr(args, key, value)
        args.pretrained = str(pretrained_path)
        args.pretrained_epoch = pretrained_epoch
    
    ## Note global and local ranks to allow multi-node training 
    rank       = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    ## Report arguments 
    if rank == 0:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        ## Note that this yaml file will not be useable as an input to this script
        dump_args(args, run_dir / "args.yaml")
        for k in sorted(vars(args)):
            print(f"{k}: {getattr(args, k)}")

    ## Removed mp.spawn, now requires srun 
    return run_training(rank, local_rank, world_size, args, enc_state)


if __name__ == "__main__":
    main()
