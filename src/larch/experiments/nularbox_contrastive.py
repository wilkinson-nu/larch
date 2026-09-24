import numpy as np
import argparse
import sys
import MinkowskiEngine as ME
import torch
import time
import math
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

## The parallelisation libraries
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity

## Includes from my libraries for this project
from larch.losses.ntxent import NTXentMerged, NTXentMergedMultiGPU
from larch.losses.vicreg import VICRegLossDistributed
from larch.losses.clustering import ClusteringLossMerged, ClusteringLossMergedMultiGPU
from larch.models.resnet_encoder import get_encoder
from larch.models.projection_head import get_projhead
from larch.models.clustering_head import get_clusthead
from larch.metrics import argmax_consistency, uniformity, alignment, simclr_geometry_metrics
from larch.training.logging import log_scalar, log_grad_norm, log_grad_rms, log_grad_over_wgt, log_weight_norm
from larch.optim.scheduling import get_opt_and_sched, cosine_scheduler, update_weight_decay
from larch.optim.lars import log_lars_diagnostics

## Import datasets
from larch.datasets.base import solo_labelled_collate_fn
from larch.datasets.dataloaders import build_paired_training_data, build_monitoring_data

from larch.sysmon import log_memory, log_gpu, log_vmstat
import psutil, os

## For logging
from torch.utils.tensorboard import SummaryWriter

## Import transformations
from larch.datasets.nularbox.augmentations_2d import get_transform

## Supervised for kNN monitoring
from larch.datasets.nularbox.targets import MULTIPLICITY_TARGETS, LABEL_GROUPS, label_clamp
from larch.probes import run_probes, log_probe_results

## Utilities for multi-rank training
from larch.distributed import setup_distributed_runtime, print0

## Checkpointing
from larch.training.checkpointing import load_pretrained, load_checkpoint, save_checkpoint

## Config handling
from larch.config import apply_config, load_config, dump_args

## Wrapped training function
def run_training(rank, local_rank, world_size, args):

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
    encoder .to(device)
    encoder = DDP(encoder, device_ids=[local_rank])  ## Sort out parallel models (e.g., one is sent to each GPU)

    ## Dictionary of heads
    heads = {}
    
    ## Dictionary of loss functions
    loss_fns = {}

    ## Set up head and loss for projection space
    proj_head = get_projhead(encoder_nchan, args)
    proj_head = nn.SyncBatchNorm.convert_sync_batchnorm(proj_head)
    proj_head .to(device)
    proj_head = DDP(proj_head, device_ids=[local_rank])
    heads["proj"] = proj_head

    ## TODO add some protection here in case arguments are missing
    if args.proj_loss == "simclr":
        print0(f"LOSS: SimCLR")
        print0(f"      temp = {args.proj_temp}")
        loss_fns["proj"] = NTXentMergedMultiGPU(args.proj_temp)
    elif args.proj_loss == "vicreg":
        print0(f"LOSS: VICReg")
        print0(f"      sim_coeff = {args.vicreg_sim_coeff}")
        print0(f"      std_coeff = {args.vicreg_std_coeff}")
        print0(f"      cov_coeff = {args.vicreg_cov_coeff}")        
        loss_fns["proj"] = VICRegLossDistributed(args.vicreg_sim_coeff,
                                                 args.vicreg_std_coeff,
                                                 args.vicreg_cov_coeff)
    else:
        raise ValueError(f"Unknown projection head loss: {args.proj_loss}")


    ## Optionally include the head and loss for the clustering space
    if args.clust_arch != "none":
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
    
    train_dataset, train_loader = build_paired_training_data(
        data_dir=args.data_dir,
        nevents=args.nevents,
        transform=train_transform,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    nbatches   = len(train_loader)

    ## Setup the monitoring dataset
    monitor_transform = get_transform(args.out_image_size, "no_aug")

    ## Use the default label groups to run monitoring probes on
    MONITOR_LABEL_GROUPS = LABEL_GROUPS

    ## Apply maxima to the N. particle groups of interest
    MONITOR_CONFIG = MULTIPLICITY_TARGETS
    
    MONITOR_CLAMP = {
        name: label_clamp(MULTIPLICITY_TARGETS)
        for name in MONITOR_LABEL_GROUPS
    }
    
    monitor_collate = partial(
        solo_labelled_collate_fn,
        label_clamp=MONITOR_CLAMP,
    )
    
    bank_loader, query_loader = build_monitoring_data(
        data_dir=args.data_dir,
        train_events=args.nevents,
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

    ## Make the log directory
    log_dir = Path(args.run_dir) / args.log
    log_dir.mkdir(parents=True, exist_ok=True)

    ## Make the state_file
    state_file = Path(args.run_dir) / args.state_file
    
    ## So we don't constantly ask args
    num_iterations = args.nepoch
    clust_loss_scale = args.clust_loss_scale
    norm_encoder = bool(args.norm_encoder)
    weight_decay = args.weight_decay
    weight_decay_final = args.weight_decay_final
    
    print0("Training with", num_iterations, "iterations")
    writer = None
    if rank==0:
        writer = SummaryWriter(log_dir=log_dir)

    ## Sort out the optimizer (one for each GPU...)
    nstep_total = nbatches*args.nepoch
    optimizer, scheduler = get_opt_and_sched(args, encoder, heads, nstep_total, world_size, print_debug=False)
    
    ## Set up metrics
    metrics = defaultdict(list)

    ## Load the checkpoint if one has been given
    start_iteration = 0
    global_iter = 0
    if args.restart:
        start_iteration, metrics = load_checkpoint(encoder, heads, optimizer, scheduler, state_file)
        global_iter = start_iteration*nbatches
        print0("Restarting from iteration", start_iteration)

    ## Load the pretrained model if given
    if args.pretrained:
        if args.restart:
            print0("Restart requested along with a pretraining file, abort!")
            sys.exit()
        load_pretrained(encoder, heads, args.pretrained)

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
        
    ## Loop over the desired iterations
    for iteration in range(start_iteration, args.nepoch):

        print0(f"Start of iteration {iteration}")
        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(iteration)
        
        tot_loss_tensor = torch.tensor(0.0, device=device)  
        losses_tensor = {name: torch.tensor(0.0, device=device) for name in heads.keys()}       
        entropy_tensor = torch.tensor(0.0, device=device)

        ## This is only used by VICReg for now
        proj_part_sums = None
        proj_part_names = None
        
        ## For monitoring
        total_acc_tensor = torch.tensor(0.0, device=device)
        total_enc_align_tensor = torch.tensor(0.0, device=device)
        total_enc_unif_tensor = torch.tensor(0.0, device=device)
        total_proj_align_tensor = torch.tensor(0.0, device=device)
        total_proj_unif_tensor = torch.tensor(0.0, device=device)
        total_clust_align_tensor = torch.tensor(0.0, device=device)
        total_clust_unif_tensor = torch.tensor(0.0, device=device)

        ## Add more monitoring tools
        nbuffer = 5
        buffer_enc = []
        buffer_proj = []
        
        # Set train mode for the encoder and any heads
        encoder.train()
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

            ## Now do the forward passes
            encoded_batch = encoder(cat_batch, this_batch_size)

            if global_iter % args.extra_log_rate == 0 and args.extra_log_rate > 0:
                with torch.no_grad():
                    hn = encoded_batch.detach().float().norm(dim=1)
                    gathered = [torch.zeros_like(hn) for _ in range(world_size)]
                    dist.all_gather(gathered, hn.contiguous())
                    allhn = torch.cat(gathered)
                    if rank == 0:
                        q = torch.quantile(allhn, torch.tensor([0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0], device=allhn.device))
                        for name, v in zip(['min', 'p1', 'p10', 'p50', 'p90', 'p99', 'max'], q.cpu()):
                            log_scalar(writer, metrics, f'hnorm/{name}', v.item(), global_iter)
            
            ## L2 norm the encoder
            if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)

            ## Deal with the projection loss
            proj_batch = heads["proj"](encoded_batch)
            proj_loss, proj_loss_parts = loss_fns["proj"](proj_batch)
                
            tot_loss = proj_loss
            losses_tensor["proj"] += proj_loss.detach()

            ## This is for VICReg for now
            if proj_part_names is None:
                proj_part_names = tuple(sorted(proj_loss_parts))
                proj_part_sums = torch.zeros(
                    len(proj_part_names),
                    device=device,
                    dtype=torch.float64,
                )
            proj_part_sums += torch.stack([
                proj_loss_parts[name].detach().double()
                for name in proj_part_names
            ])
                    
            ## Add to metrics
            total_enc_align_tensor += alignment(encoded_batch)
            total_enc_unif_tensor += uniformity(encoded_batch)
            total_proj_align_tensor += alignment(proj_batch)
            total_proj_unif_tensor += uniformity(proj_batch)
            
            ## Get a few batches for calculating the running deff
            if len(buffer_enc) < nbuffer:
                with torch.no_grad():
                    buffer_enc .append(encoded_batch.detach().to("cpu", non_blocking=False))
                    buffer_proj.append(proj_batch.detach().to("cpu", non_blocking=False))
                    
            ## Optionally deal with clustering loss
            if "clust" in heads:
                clust_batch = heads["clust"](encoded_batch)
                clust_loss, clust_entropy = loss_fns["clust"](clust_batch)
                clust_loss = args.clust_loss_scale * clust_loss
                clust_entropy = args.clust_loss_scale * clust_entropy
                tot_loss += clust_loss + clust_entropy
                losses_tensor["clust"] += clust_loss.detach()
                entropy_tensor += clust_entropy.detach()
                total_acc_tensor += argmax_consistency(clust_batch)
                total_clust_align_tensor += alignment(clust_batch)
                total_clust_unif_tensor += uniformity(clust_batch)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            tot_loss .backward()

            ## Decide whether to collect LARS stats
            collect_lars_stats = (
                args.extra_log_rate > 0
                and global_iter % args.extra_log_rate == 0
                and rank == 0
            )
            optimizer.collect_stats = collect_lars_stats

            ## Update optimizer and scheduler
            optimizer.step()
            if args.extra_log_rate > 0 and global_iter % args.extra_log_rate == 0 and rank == 0:
                log_lars_diagnostics(
                    optimizer=optimizer,
                    writer=writer,
                    metrics=metrics,
                    global_iter=global_iter,
                )
            
            ## ...after all that logging, finally update the scheduler...
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
        dist.all_reduce(total_enc_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_enc_unif_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_proj_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_proj_unif_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_clust_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_clust_unif_tensor, op=dist.ReduceOp.SUM)        
        
        av_tot_loss = tot_loss_tensor.item() / (nbatches * world_size)
        av_losses = {
            name: losses_tensor[name].item() / (nbatches * world_size)
            for name in heads.keys()
        }
        av_entropy = entropy_tensor.item() / (nbatches * world_size)
        av_acc = total_acc_tensor.item() / (nbatches * world_size)
        av_enc_unif = total_enc_unif_tensor.item() / (nbatches * world_size)
        av_enc_align = total_enc_align_tensor.item() / (nbatches * world_size)
        av_proj_unif = total_proj_unif_tensor.item() / (nbatches * world_size)
        av_proj_align = total_proj_align_tensor.item() / (nbatches * world_size)
        av_clust_unif = total_clust_unif_tensor.item() / (nbatches * world_size)
        av_clust_align = total_clust_align_tensor.item() / (nbatches * world_size)

        ## Deal with VICReg components
        if proj_part_sums is not None:
            dist.all_reduce(proj_part_sums, op=dist.ReduceOp.SUM)
        
            av_proj_loss_parts = {
                name: proj_part_sums[i].item() /  (nbatches * world_size)
                for i, name in enumerate(proj_part_names)
            }
        
        ## Other geometry calculations
        enc_geom = simclr_geometry_metrics(buffer_enc, device)
        proj_geom = simclr_geometry_metrics(buffer_proj, device)

        ## kNN and linear probe monitoring
        run_knn = (args.knn_every > 0 and iteration % args.knn_every == 0)
        run_linear = (args.linear_every > 0 and iteration % args.linear_every == 0)

        ## Run probes as desired
        knn_results, linear_results = run_probes(encoder,
                                                 bank_loader,
                                                 query_loader,
                                                 device,
                                                 targets=MONITOR_CONFIG,
                                                 label_groups=MONITOR_LABEL_GROUPS,
                                                 rank=rank,
                                                 run_knn=run_knn,
                                                 run_linear=run_linear,
                                                 knn_k=args.knn_k,
                                                 knn_pca=args.knn_k,
                                                 linear_epochs=args.linear_epochs,
                                                 linear_batch_size=args.linear_batch_size,
                                                 linear_lr=args.linear_lr,
                                                 seed=args.seed)

        ## Reporting, but only for rank 0
        if rank==0:
            metrics["iteration"].append(iteration)
            log_scalar(writer, metrics, 'loss/total', av_tot_loss, iteration)              
            log_scalar(writer, metrics, 'loss/proj', av_losses["proj"], iteration)

            ## Add metrics for debugging/training diagnostics
            log_scalar(writer, metrics, 'monitor/proj_alignment', av_proj_align, iteration)
            log_scalar(writer, metrics, 'monitor/proj_uniformity', av_proj_unif, iteration)
            log_scalar(writer, metrics, 'monitor/enc_alignment', av_enc_align, iteration)
            log_scalar(writer, metrics, 'monitor/enc_uniformity', av_enc_unif, iteration)
                
            ## Extensive logging for gradient debugging
            log_grad_norm(encoder.module, "encoder", writer, iteration)
            log_grad_rms(encoder.module, "encoder", writer, iteration)
            log_grad_over_wgt(encoder.module, "encoder", writer, iteration)
            log_weight_norm(encoder.module, "encoder", writer, iteration)
            
            log_grad_norm(heads["proj"].module, "proj", writer, iteration)
            log_grad_rms(heads["proj"].module, "proj", writer, iteration)
            log_grad_over_wgt(heads["proj"].module, "proj", writer, iteration)
            log_weight_norm(heads["proj"].module, "proj", writer, iteration)
            
            ## More summary quantities about the encoder and projection spaces
            for name, value in enc_geom.items():
                log_scalar(writer, metrics, f"eigen/enc_{name}", value, iteration)
            for name, value in proj_geom.items():
                log_scalar(writer, metrics, f"eigen/proj_{name}", value, iteration)                
                
            if "clust" in heads:
                log_scalar(writer, metrics, 'loss/clust', av_losses["clust"]+av_entropy, iteration)
                log_scalar(writer, metrics, 'loss/entropy', av_entropy, iteration)
                log_scalar(writer, metrics, 'loss/clust_only', av_losses["clust"], iteration)
                log_scalar(writer, metrics, 'monitor/acc', av_acc, iteration)
                log_scalar(writer, metrics, 'monitor/clust_alignment', av_clust_align, iteration)
                log_scalar(writer, metrics, 'monitor/clust_uniformity', av_clust_unif, iteration)

                ## Extensive logging for gradient debugging
                log_grad_norm(heads["clust"].module, "clust", writer, iteration)
                log_grad_rms(heads["clust"].module, "clust", writer, iteration)
                log_grad_over_wgt(heads["clust"].module, "clust", writer, iteration)
                log_weight_norm(heads["clust"].module, "clust", writer, iteration)

            if proj_loss_parts is not None:
                for name, value in av_proj_loss_parts.items():
                    log_scalar(
                        writer,
                        metrics,
                        f"vicreg/{name}",
                        value,
                        iteration,
                    )

            ## Logs all of the kNN and linear probe results
            log_probe_results(writer, metrics, knn_results, linear_results, iteration)
                    
            if scheduler: 
                log_scalar(writer, metrics, 'train/lr', scheduler.get_last_lr()[0], iteration)
            log_scalar(writer, metrics, 'train/weight_decay', this_wd, iteration)

            ## Build a string to report the outcome
            iter_string = f"Processed {iteration} / {start_iteration + num_iterations}; loss = {av_tot_loss:.4f}"
            
            if "clust" in heads:
                iter_string += f" ({av_losses['proj']:.4f} + {av_losses['clust']:.4f} + {av_entropy:.4f}); acc = {av_acc:.4f}"
            print0(iter_string)
            print0(f"Time taken: {(time.time()-tstart):.2f}")
            
        ## For checkpointing
        #if rank==0 and iteration%25 == 0 and iteration != 0:
        #    save_checkpoint(encoder, heads, optimizer, scheduler, state_file+".check"+str(iteration), iteration, metrics, args)

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

            log_scalar(writer, metrics, 'syst_monitor/vm_used_gb', vm.used / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/vm_avail_gb', vm.available / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/vm_cached_gb', getattr(vm, "cached", 0) / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/rss_gb', proc.memory_info().rss / 1e9, iteration)
            log_scalar(writer, metrics, 'syst_monitor/num_fds', proc.num_fds(), iteration)
            log_scalar(writer, metrics, 'syst_monitor/io_read', io.read_bytes, iteration)
            log_scalar(writer, metrics, 'syst_monitor/io_write', io.write_bytes, iteration)
            log_scalar(writer, metrics, 'syst_monitor/mem_pressure', vm.available / vm.total, iteration)
            log_scalar(writer, metrics, 'syst_monitor/first_batch_latency', first_batch_latency, iteration)

            for gpu_rank in range(world_size):
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_allocated_gb',  all_allocated[gpu_rank].item(),  iteration)
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_reserved_gb',   all_reserved[gpu_rank].item(),   iteration)
                log_scalar(writer, metrics, f'syst_monitor/gpu{gpu_rank}_peak_alloc_gb', all_peak_alloc[gpu_rank].item(), iteration)
                
    ## Final version of the model
    if rank==0:
        save_checkpoint(encoder, heads, optimizer, scheduler, state_file, iteration, metrics, args)
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
    parser = argparse.ArgumentParser("Contrastive SSL module")

    ## Basic job setup
    parser.add_argument('--config', required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--nevents', type=int, required=True)
    parser.add_argument('--nepoch', type=int, required=True)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--log', type=str)
    parser.add_argument('--state_file', type=str)
    parser.add_argument('--restart', action='store_true')
    parser.add_argument('--pretrained', type=str, default=None)
    
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
    parser.add_argument('--out_image_size', type=int)
    parser.add_argument('--aug_type', type=str)
    parser.add_argument('--aug_prob', type=float)
    parser.add_argument('--aug_val', type=float)

    ## Encoder architecture choices
    parser.add_argument('--enc_act', type=str)
    parser.add_argument('--enc_arch', type=str)
    parser.add_argument('--enc_arch_pool', type=str)
    parser.add_argument('--enc_res_pool', type=int, choices=[0,1])
    parser.add_argument('--enc_stem_norm', type=int, choices=[0,1])
    parser.add_argument('--enc_init_stem_stride', type=int)
    parser.add_argument('--enc_final_stem_stride', type=int)
    parser.add_argument('--enc_stem_pool', type=str)
    parser.add_argument('--enc_stem_deep', type=int, choices=[0,1])
    parser.add_argument('--enc_layer1_norm', type=int, choices=[0,1])
    parser.add_argument('--enc_final_linear', type=int)
    parser.add_argument('--enc_stem_channels', type=int)

    ## (Optional) clustering head
    parser.add_argument('--clust_arch', type=str)
    parser.add_argument('--clust_temp', type=float)
    parser.add_argument('--nclusters', type=int)
    parser.add_argument('--entropy_scale', type=float)
    parser.add_argument('--instance_scale', type=float)
    parser.add_argument('--clust_loss_scale', type=float)

    ## Projection head architecture
    parser.add_argument('--proj_arch', type=str)
    parser.add_argument('--proj_init_bn', type=int, choices=[0,1])
    parser.add_argument('--proj_final_bn', type=int, choices=[0,1])
    parser.add_argument('--latent', type=int)
    parser.add_argument('--nhidden', type=int)

    ## Projection head loss
    parser.add_argument('--proj_loss', type=str)
    ## TODO: rename to simlar_temp
    parser.add_argument('--proj_temp', type=float)
    parser.add_argument('--vicreg_sim_coeff', type=float)
    parser.add_argument('--vicreg_std_coeff', type=float)
    parser.add_argument('--vicreg_cov_coeff', type=float)
    
    ## kNN and linear probe monitoring options
    parser.add_argument('--monitor_nbank', type=int)
    parser.add_argument('--monitor_nquery', type=int)
    parser.add_argument('--knn_every', type=int)
    parser.add_argument('--knn_k', type=int)
    parser.add_argument('--knn_pca', type=int)
    parser.add_argument("--linear_every", type=int)
    parser.add_argument("--linear_epochs", type=int)
    parser.add_argument("--linear_batch_size", type=int)
    parser.add_argument("--linear_lr", type=float)
    
    ## Optional profiler
    parser.add_argument('--run_profiler', type=int, choices=[0,1])
    parser.add_argument('--extra_log_rate', type=int)

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

    ## Note global and local ranks to allow multi-node training 
    rank       = int(os.environ.get("SLURM_PROCID", 0))
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    ## Report arguments 
    if rank == 0:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_args(args, run_dir / "args.yaml")
        for k in sorted(vars(args)):
            print(f"{k}: {getattr(args, k)}")

    ## Removed mp.spawn, now requires srun 
    return run_training(rank, local_rank, world_size, args)


if __name__ == "__main__":
    main()
