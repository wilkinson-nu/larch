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

## Includes from my libraries for this project
from larch.models.resnet_encoder import get_encoder
from larch.metrics import uniformity, alignment, basic_geometry_metrics
from larch.training.logging import log_scalar, log_grad_norm, log_grad_rms, log_grad_over_wgt, log_weight_norm
from larch.optim.scheduling import get_opt_and_sched, update_weight_decay

## For logging
from torch.utils.tensorboard import SummaryWriter

## Import transformations
from larch.datasets.nularbox.augmentations_2d import get_transform

## Import dataset
from larch.datasets.base import solo_labelled_collate_fn
from larch.datasets.dataloaders import build_labelled_training_data, build_monitoring_data, monitoring_ranges

## Supervised learning specific
from larch.datasets.nularbox.targets import MULTIPLICITY_TARGETS, LABEL_GROUPS, label_clamp
from larch.classification import SupervisedHead, supervised_loss, ClassificationMetrics

## kNN and linear probe monitoring
from larch.probes import run_probes, log_probe_results


## Utilities for multi-rank training
from larch.distributed import setup_distributed_runtime, print0

## Checkpointing
from larch.training.checkpointing import load_pretrained, load_checkpoint, save_checkpoint

## Config
from larch.config import apply_config, load_config, dump_args

def select_labels(blabels, group, device):
    return {
        name: v.to(device, non_blocking=True)
        for name, v in blabels[group].items()
    }

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

    ## Set up supervised head and loss
    ## A subset of labels will be used later for the supervised head
    SUP_GROUP = args.sup_label_group
    SUP_TARGETS = MULTIPLICITY_TARGETS
    sup_head = SupervisedHead(encoder_nchan,
                              classifier_config=SUP_TARGETS)
    sup_head .to(device)
    sup_head = DDP(sup_head, device_ids=[local_rank])
    heads["sup"] = sup_head
    loss_fns["sup"] = supervised_loss
    
    ## Set up the distributed dataset
    train_transform = get_transform(
        args.out_image_size,
        args.aug_type,
        args.aug_prob,
        args.aug_val,
    )

    ## Apply maxima to the N. particle groups of interest
    nested_label_clamp = {
        name: label_clamp(MULTIPLICITY_TARGETS)
        for name in LABEL_GROUPS
    }

    ## Collate all labels for both the training and monitoring dataloaders
    labelled_collate = partial(
        solo_labelled_collate_fn,
        label_clamp=nested_label_clamp,
    )

    _, _, train_start = monitoring_ranges(args.monitor_nquery, args.monitor_nbank)
    train_dataset, train_loader = build_labelled_training_data(
        data_dir=args.data_dir,
        start=train_start,
        nevents=args.nevents,
        transform=train_transform,
        rank=rank,
        world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=labelled_collate,
        seed=args.seed,
    )
    nbatches   = len(train_loader)

    ## Setup the monitoring dataset
    monitor_transform = get_transform(args.out_image_size, "no_aug")
    
    bank_loader, query_loader = build_monitoring_data(
        data_dir=args.data_dir,
        nbank=args.monitor_nbank,
        nquery=args.monitor_nquery,
        transform=monitor_transform,
        collate_fn=labelled_collate,
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
    nepoch = args.nepoch
    clip_gradients = bool(args.clip_gradients)
    norm_encoder = bool(args.norm_encoder)
    weight_decay = args.weight_decay
    weight_decay_final = args.weight_decay_final

    print0("Training with", nepoch, "epochs")
    
    writer = None
    if rank==0:
        writer = SummaryWriter(log_dir=log_dir)

    ## Sort out the optimizer (one for each GPU...)
    nstep_total = nbatches*args.nepoch
    optimizer, scheduler = get_opt_and_sched(args, encoder, heads, nstep_total, world_size)
    
    ## Set up metrics
    metrics = defaultdict(list)
    
    ## Load the checkpoint if one has been given
    start_epoch = 0
    global_iter = 0
    if args.restart:
        start_epoch, metrics = load_checkpoint(encoder, heads, optimizer, scheduler, state_file)
        global_iter = start_epoch*nbatches
        print0("Restarting from epoch", start_epoch)

    ## Load the pretrained model if given
    if args.pretrained:
        if args.restart:
            print0("Restart requested along with a pretraining file, abort!")
            sys.exit()
        load_pretrained(encoder, heads, args.pretrained)

    ## Stuff in a profiler
    if bool(args.run_profiler) and rank==0:
        
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA
                        ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )        
        prof.__enter__()

    ## Set up metrics:
    clf_metrics = ClassificationMetrics(SUP_TARGETS, device=device)
    val_metrics = ClassificationMetrics(SUP_TARGETS, device=device)

    for epoch in range(start_epoch, nepoch):

        # Ensure shuffling with the sampler each epoch
        train_loader.sampler.set_epoch(epoch)
        
        tot_loss_tensor = torch.zeros((), device=device)
        losses_tensor = {name: torch.zeros((), device=device) for name in SUP_TARGETS}
        
        ## For monitoring
        clf_metrics.reset()
        total_enc_align_tensor = torch.zeros((), device=device)
        total_enc_unif_tensor = torch.zeros((), device=device)

        ## Add more monitoring tools
        nbuffer = 5
        buffer_enc = []

        # Set train mode for the encoder and any heads
        encoder.train()
        for h in heads.values(): h.train()
        
        # Iterate over batches of images with the dataloader
        t0 = time.time()
        first_batch_latency = None
        step = 0
        for bcoords, bfeats, blabels, this_batch_size in train_loader:
            
            if first_batch_latency is None:
                first_batch_latency = time.time() - t0
                
            ## Update weight decay to allow for scheduling
            this_wd = update_weight_decay(optimizer,
	        	                  weight_decay,
                                          weight_decay_final,
                                          global_iter,
                                          nstep_total)
            
            ## Send to the device, then make the sparse tensors
            blabels = select_labels(blabels, SUP_GROUP, device)
            bcoords = bcoords.to(device, non_blocking=True)
            bfeats  = bfeats .to(device, non_blocking=True)
            batch   = ME.SparseTensor(bfeats, bcoords, device=device)
            
            ## Now do the forward passes
            encoded_batch = encoder(batch, this_batch_size)
            
            ## L2 norm the encoder
            if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)

            ## Deal with the projection loss
            sup_batch = heads["sup"](encoded_batch)
            sup_loss, sup_loss_dict = loss_fns["sup"](sup_batch, blabels, SUP_TARGETS)
            for name, loss_val in sup_loss_dict.items():
                losses_tensor[name] += loss_val.detach()

            ## Add to metrics
            total_enc_align_tensor += alignment(encoded_batch)
            total_enc_unif_tensor += uniformity(encoded_batch)
            
            ## Get a few batches for cealculating the running deff
            ## If the number of batches is large w.r.t. the total number (e.g., for testing), non_blocking will cause an issue here
            if len(buffer_enc) < nbuffer:
                with torch.no_grad():
                    buffer_enc .append(encoded_batch.detach().to("cpu"))
            
            ## Supervision specific metrics:
            with torch.no_grad(): clf_metrics.update(sup_batch, blabels, outputs_are_logits=True)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            sup_loss .backward()
            
            if clip_gradients:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
                for h in heads.values(): torch.nn.utils.clip_grad_norm_(h.parameters(), max_norm=1.0)

            ## Update optimizer and scheduler
            optimizer.step()
            if scheduler: scheduler.step()
            
            ## Increment global_iter
            global_iter += 1
            step += 1
            
            ## keep track of losses
            tot_loss_tensor += sup_loss.detach()
            
        ## Validation pass
        encoder.eval()
        for h in heads.values(): h.eval()

        val_tot_loss_tensor = torch.zeros((), device=device)
        val_losses_tensor = {name: torch.zeros((), device=device) for name in SUP_TARGETS}
        val_metrics.reset()
        val_nbatches = 0
        step = 0

        with torch.no_grad():
            for bcoords, bfeats, blabels, this_batch_size in query_loader:
        
                blabels = select_labels(blabels, SUP_GROUP, device)
                bcoords = bcoords.to(device, non_blocking=True)
                bfeats  = bfeats .to(device, non_blocking=True)
                batch   = ME.SparseTensor(bfeats, bcoords, device=device)
        
                encoded_batch = encoder(batch, this_batch_size)
                if norm_encoder: encoded_batch = torch.nn.functional.normalize(encoded_batch, p=2, dim=1)
        
                sup_batch = heads["sup"](encoded_batch)
                sup_loss, sup_loss_dict = loss_fns["sup"](sup_batch, blabels, SUP_TARGETS)
        
                for name, loss_val in sup_loss_dict.items():
                    val_losses_tensor[name] += loss_val
                val_tot_loss_tensor += sup_loss
        
                val_metrics.update(sup_batch, blabels, outputs_are_logits=True)
                val_nbatches += 1
                step += 1
                
        torch.cuda.empty_cache()

        # Resume train mode for next epoch
        encoder.train()
        for h in heads.values(): h.train()
        
        ## Although the gradients are handled correctly by GatherLayer, the losses are global
        ## Strictly speaking this step isn't necessary as each mini-batch gives the same loss value
        ## But I kept it in to avoid my own headaches...
        dist.all_reduce(tot_loss_tensor, op=dist.ReduceOp.SUM)
        for name in losses_tensor.keys(): dist.all_reduce(losses_tensor[name], op=dist.ReduceOp.SUM)
        dist.all_reduce(total_enc_align_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_enc_unif_tensor, op=dist.ReduceOp.SUM)

        av_tot_loss = tot_loss_tensor.item() / (nbatches * world_size)
        av_losses = {
            name: losses_tensor[name].item() / (nbatches * world_size)
            for name in losses_tensor.keys()
        }
        av_enc_align = total_enc_align_tensor.item() / (nbatches * world_size)
        av_enc_unif  = total_enc_unif_tensor.item() / (nbatches * world_size)

        ## Other geometry calculations
        with torch.no_grad():
            enc_geom = basic_geometry_metrics(buffer_enc, device, norm_encoder)

        ## kNN and linear probe monitoring
        is_last_epoch = (epoch == nepoch - 1)
        run_knn = args.knn_every > 0 and (epoch % args.knn_every == 0 or is_last_epoch)
        run_linear = args.linear_every > 0 and (epoch % args.linear_every == 0 or is_last_epoch)

        ## Run probes as desired
        knn_results, linear_results = run_probes(encoder,
                                                 bank_loader,
                                                 query_loader,
                                                 device,
                                                 targets=MULTIPLICITY_TARGETS,
                                                 label_groups=LABEL_GROUPS,
                                                 rank=rank,
                                                 run_knn=run_knn,
                                                 run_linear=run_linear,
                                                 knn_k=args.knn_k,
                                                 knn_pca=args.knn_pca,
                                                 linear_epochs=args.linear_epochs,
                                                 linear_batch_size=args.linear_batch_size,
                                                 linear_lr=args.linear_lr,
                                                 seed=args.seed)
            
        ## Sort out the metrics (needs to be on all ranks due to collective ops)
        clf_metrics.reduce()
        metric_results = clf_metrics.compute()

        ## Also deal with validation metrics
        dist.all_reduce(val_tot_loss_tensor, op=dist.ReduceOp.SUM)
        for name in val_losses_tensor.keys():
            dist.all_reduce(val_losses_tensor[name], op=dist.ReduceOp.SUM)
        av_val_tot_loss = val_tot_loss_tensor.item() / (val_nbatches * world_size)
        av_val_losses = {
            name: val_losses_tensor[name].item() / (val_nbatches * world_size)
            for name in val_losses_tensor.keys()
        }
        val_metrics.reduce()
        val_metric_results = val_metrics.compute()
        
        ## Reporting, but only for rank 0
        if rank==0:
            metrics["epoch"].append(epoch)
            metrics['time/total_seconds'].append(time.time()-tstart)
            log_scalar(writer, metrics, 'loss/total', av_tot_loss, epoch)
            for name in losses_tensor.keys():
                log_scalar(writer, metrics, 'loss/'+name, av_losses[name], epoch)

            ## Supervised training metrics
            for part_name, result in metric_results.items():
                for metric_name, val in result.items():
                    log_scalar(writer, metrics, f'acc/{part_name}_{metric_name}', result[metric_name], epoch)
                
            ## Add metrics for debugging/training diagnostics
            log_scalar(writer, metrics, 'monitor/enc_alignment', av_enc_align, epoch)
            log_scalar(writer, metrics, 'monitor/enc_uniformity', av_enc_unif, epoch)
                
            ## Extensive logging for gradient debugging
            log_grad_norm(encoder.module, "encoder", writer, epoch)
            log_grad_rms(encoder.module, "encoder", writer, epoch)
            log_grad_over_wgt(encoder.module, "encoder", writer, epoch)
            log_weight_norm(encoder.module, "encoder", writer, epoch)

            log_grad_norm(heads["sup"].module, "sup", writer, epoch)
            log_grad_rms(heads["sup"].module, "sup", writer, epoch)
            log_grad_over_wgt(heads["sup"].module, "sup", writer, epoch)
            log_weight_norm(heads["sup"].module, "sup", writer, epoch)

            ## More summary quantities about the encoded space
            for name, value in enc_geom.items():
                log_scalar(writer, metrics, f"eigen/enc_{name}", value, epoch)

            ## Logs all of the kNN and linear probe results
            log_probe_results(writer, metrics, knn_results, linear_results, epoch)
                
            if scheduler: 
                log_scalar(writer, metrics, 'train/lr', scheduler.get_last_lr()[0], epoch)
            log_scalar(writer, metrics, 'train/weight_decay', this_wd, epoch)

            ## Build a string to report the outcome
            iter_string = f"Processed {epoch} / {nepoch}; loss = {av_tot_loss:.4f} (val loss = {av_val_tot_loss:.4f})"
            print0(iter_string)
            print0(f"Time taken: {(time.time()-tstart):.2f}")

        ## Log validation now:
        if rank == 0:
            log_scalar(writer, metrics, 'loss/val_total', av_val_tot_loss, epoch)
            for name in val_losses_tensor.keys():
                log_scalar(writer, metrics, 'loss/val_'+name, av_val_losses[name], epoch)
                
            for part_name, result in val_metric_results.items():
                for metric_name, val in result.items():
                    log_scalar(writer, metrics, f'acc/val_{part_name}_{metric_name}', result[metric_name], epoch)
            
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
    parser = argparse.ArgumentParser("Supervised training module")

    # Basic job setup
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
    parser.add_argument('--clip_gradients', type=int, choices=[0,1])
    parser.add_argument('--norm_encoder', type=int, choices=[0,1])
    parser.add_argument('--non_lars_lr_scale', type=float)

    ## Supervised head specific
    parser.add_argument('--sup_label_group', type=str, choices=LABEL_GROUPS)

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

    ## kNN and linear probe monitoring options
    parser.add_argument('--monitor_nbank', type=int)
    parser.add_argument('--monitor_nquery', type=int)
    parser.add_argument('--knn_every', type=int)
    parser.add_argument('--knn_k', type=int)
    parser.add_argument('--knn_pca', type=int)
    parser.add_argument('--linear_every', type=int)
    parser.add_argument('--linear_epochs', type=int)
    parser.add_argument('--linear_batch_size', type=int)
    parser.add_argument('--linear_lr', type=float)

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

