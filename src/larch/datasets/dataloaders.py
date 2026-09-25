import torch
import numpy as np
import random
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader, DistributedSampler, Subset

## Import dataset utils
from larch.datasets.base import paired_2d_dataset_ME, cat_ME_collate_fn, single_2d_dataset_ME

## Basic utils
from larch.distributed import print0

def worker_init_fn(worker_id):
    threadpool_limits(limits=1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def monitoring_ranges(nquery, nbank):
    query = range(0, nquery)
    bank = range(nquery, nquery + nbank)
    train_start = nquery + nbank
    return query, bank, train_start
    
def make_distributed_dataloader(dataset,
                                *,
                                rank,
                                world_size,
                                batch_size,
                                collate_fn,
                                num_workers,
                                shuffle,
                                drop_last,
                                seed=0,
                                pin_memory=True):
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "sampler": sampler,
        "shuffle": False,  # The sampler controls shuffling.
        "num_workers": num_workers,
        "worker_init_fn": worker_init_fn,
        "drop_last": drop_last,
        "pin_memory": pin_memory,
    }

    # These options are invalid or irrelevant with zero workers.
    if num_workers > 0:
        kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 2,
        })

    return DataLoader(**kwargs)

    
def build_paired_training_data(*,
                               data_dir,
                               start,
                               nevents,
                               transform,
                               rank,
                               world_size,
                               batch_size,
                               num_workers,
                               seed):
    
    dataset = paired_2d_dataset_ME(data_dir, aug_transform=transform,
                                   max_events=start+nevents)

    if len(dataset) < start+nevents:
        raise ValueError(f"Requested events [{start}, {start + nevents}), "
                         f"but dataset contains only {len(dataset)}")

    ## This subset approach ensures that any monitoring set is always the same
    dataset = Subset(dataset, range(start, start + nevents))

    loader = make_distributed_dataloader(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        collate_fn=cat_ME_collate_fn,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )

    print0(f"Loaded {len(dataset)} training events")
    return dataset, loader


def build_monitoring_data(*,
                          data_dir,
                          nbank,
                          nquery,
                          transform,
                          collate_fn,
                          rank,
                          world_size,
                          batch_size,
                          num_workers,
                          seed):
    
    # Ensure DistributedSampler does not need to pad either subset
    for name, n in (("nquery", nquery), ("nbank", nbank)):
        if n % world_size:
            raise ValueError(f"{name}={n} is not divisible by world_size={world_size}")

    query_range, bank_range, _ = monitoring_ranges(nquery, nbank)
    required_events = bank_range.stop

    full_dataset = single_2d_dataset_ME(data_dir, transform=transform, max_events=required_events)

    if len(full_dataset) < required_events:
        raise ValueError(f"Monitoring requires {required_events} total events, "
                         f"but dataset contains only {len(full_dataset)}")

    bank_dataset = Subset(full_dataset, bank_range)
    query_dataset = Subset(full_dataset, query_range)

    common = {
        "rank": rank,
        "world_size": world_size,
        "batch_size": batch_size,
        "collate_fn": collate_fn,
        "num_workers": num_workers,
        "shuffle": False,
        "drop_last": False,
        "seed": seed,
    }

    bank_loader = make_distributed_dataloader(bank_dataset, **common)
    query_loader = make_distributed_dataloader(query_dataset, **common)

    print0(f"Loaded {nbank} bank and {nquery} query events for monitoring")
    return bank_loader, query_loader


def build_labelled_training_data(*,
                                 data_dir,
                                 start,
                                 nevents,
                                 transform,
                                 collate_fn,
                                 rank,
                                 world_size,
                                 batch_size,
                                 num_workers,
                                 seed):

    dataset = single_2d_dataset_ME(data_dir, transform=transform, max_events=start+nevents)

    if len(dataset) < start + nevents:
        raise ValueError(f"Requested events [{start}, {start + nevents}), "
                         f"but dataset contains only {len(dataset)}")

    ## This subset approach ensures that any monitoring set is always the same
    dataset = Subset(dataset, range(start, start + nevents))
    
    loader = make_distributed_dataloader(
        dataset,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        seed=seed,
    )

    print0(f"Loaded {len(dataset)} labelled training events")
    return dataset, loader

