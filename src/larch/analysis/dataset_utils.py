import torch
import MinkowskiEngine as ME
import numpy as np
import time
from collections import defaultdict
from collections.abc import Mapping
from torch import nn
from larch.datasets.nularbox.truth_labels import make_cc_category
from larch.datasets.base import single_2d_dataset_ME, solo_ME_collate_fn, solo_ME_collate_fn_with_meta
from larch.datasets.augmentations_2d import FirstRegionCrop

def get_dataset(input_dir, nevents, nom_transform=False, return_metadata=False):

    print("Loading", nevents, "events from", input_dir)

    ## This is a relic as this was initially done for the FSD dataset
    if nom_transform == False:
        nom_transform = transforms.Compose([
            FirstRegionCrop((800, 256), (768, 256)),
        ])
    
    dataset = single_2d_dataset_ME(input_dir, \
                                   transform=nom_transform, \
                                   max_events=nevents,\
                                   return_metadata=return_metadata)
    this_collate = solo_ME_collate_fn
    if return_metadata: this_collate = solo_ME_collate_fn_with_meta
    
    loader = torch.utils.data.DataLoader(dataset,
                                         collate_fn=this_collate,
                                         batch_size=2048,
                                         shuffle=False,
                                         num_workers=8)

    print("Loaded", dataset.__len__(), "events")
    return dataset, loader


def image_loop(encoder,
               heads,
               loader,
               device,
               detailed_info=False,
               return_hidden=False):
    
    start = time.time()

    representations = defaultdict(list)

    # Only one of these paths will be populated.
    old_label_records = []
    new_label_batches = defaultdict(lambda: defaultdict(list))
    label_schema = None

    nhits = []
    maxQ = []
    sumQ = []

    filenames = []
    event_ids = []

    encoder.eval()
    encoder.to(device)

    for head in heads.values():
        head.eval()
        head.to(device)

    for batch in loader:
        batch_coords, batch_feats, batch_labels = batch[:3]

        # The metadata collate function returns five items. A four-item
        # batch may instead be from solo_labelled_collate_fn, whose final
        # item is the batch size.
        if len(batch) >= 5:
            batch_filenames = batch[3]
            batch_eventids = batch[4]
        else:
            batch_filenames = None
            batch_eventids = None

        ## Deal with the multiple file layouts (old can be deprecated... soon...
        if label_schema is None:            
            nested_schema = isinstance(batch_labels, Mapping)
            label_schema = "new" if nested_schema else "old"

        if label_schema == "new":
            # Find the event dimension from any label tensor.
            try:
                first_group = next(iter(batch_labels.values()))
                first_field = next(iter(first_group.values()))
                batch_size = len(first_field)
            except StopIteration:
                raise RuntimeError("Could not infer batch size from empty nested labels")
        else:
            batch_size = len(batch_labels)

        batch_coords = batch_coords.to(device)
        batch_feats = batch_feats.to(device)
        orig_batch = ME.SparseTensor(batch_feats, batch_coords, device=device)            

        dec_coords = orig_batch.decomposed_coordinates
        dec_feats  = orig_batch.decomposed_features
        
        ## Now do the forward passes            
        with torch.no_grad(): 
            encoded_batch = encoder(orig_batch, batch_size)
            if "clust" in heads:
                clust_batch = heads["clust"](encoded_batch, return_hidden=return_hidden)
            proj_batch = heads["proj"](encoded_batch, return_hidden=return_hidden)

        ## Normalize the output (this is a bit fragile, but backwards compatible)
        if not return_hidden:
            proj_batch = {"proj_final": proj_batch}
            if "clust" in heads:
                clust_batch = {"clust_final": clust_batch}

        ## Get the representations all in one place
        for k, v in proj_batch.items():
            representations[k].append(v.detach().cpu())
        if "clust" in heads:
            for k, v in clust_batch.items():
                representations[k].append(v.detach().cpu())
        representations["encoder"].append(encoded_batch.detach().cpu())
        
        ## Collect labels according to their existing interface.
        if label_schema == "new":
            for group_name, fields in batch_labels.items():
                for field_name, values in fields.items():
                    if torch.is_tensor(values):
                        values = values.detach().cpu()
                    else:
                        values = torch.as_tensor(values)

                    new_label_batches[group_name][field_name].append(values)
        else:
            # Preserve the old structured-record behavior.
            old_label_records.extend(batch_labels)

        ## If desired, add a load more info, but this slows things down a lot...
        if detailed_info is True:
            nhits_batch = torch.tensor([f.shape[0] for f in dec_feats], device=device)
            sumQ_batch  = torch.stack([f.sum() for f in dec_feats])
            maxQ_batch  = torch.stack([f.max() for f in dec_feats])
        
            # Move everything to the CPU
            nhits.append(nhits_batch.cpu())
            sumQ.append(sumQ_batch.cpu())
            maxQ.append(maxQ_batch.cpu())

        if batch_filenames is not None:
            filenames.extend(batch_filenames)
        if batch_eventids is not None:
            event_ids.extend(batch_eventids)

        # Manage CUDA memory for ME
        torch.cuda.empty_cache()

    ## Turn into numpy arrays 
    representations = {k: torch.cat(v).numpy() for k, v in representations.items()}

    if label_schema == "old":
        # Preserve the previous flat label output and derived fields.
        label_array = np.asarray(old_label_records)

        label_dict = {name: np.ascontiguousarray(label_array[name])
                      for name in label_array.dtype.names}

        label_dict.update({
            "ncharged": np.ascontiguousarray(
                label_dict["nproton"]
                + label_dict["npipm"]
                + label_dict["nkapm"]
            ),
            "ncluster": np.ascontiguousarray(
                label_dict["ndeuteron"]
                + label_dict["ntritium"]
                + label_dict["nalpha"]
                + label_dict["nhelium3"]
                + label_dict["nnuclfrag"]
            ),
            "cc_category": np.ascontiguousarray(
                make_cc_category(label_array)
            ),
        })

    else:
        # New nested output. Stored derived fields and topologies are
        # retained directly rather than recalculated.
        label_dict = {
            group_name: {field_name: np.ascontiguousarray(torch.cat(chunks, dim=0).numpy())
                for field_name, chunks in fields.items()}
            for group_name, fields in new_label_batches.items()
        }

    out = {"labels": label_dict}
    out .update(representations)

    if "clust" in heads:
        cluster = out["clust_final"]
        out["clust"] = cluster
        out["clust_index"] = np.argmax(cluster, axis=1)
        out["clust_max"] = np.max(cluster, axis=1)

    if detailed_info:
        out["nhits"] = torch.cat(nhits).numpy()
        out["sumQ"] = torch.cat(sumQ).numpy()
        out["maxQ"] = torch.cat(maxQ).numpy()

    if filenames: out["filename"] = filenames
    if event_ids: out["event_id"] = np.asarray(event_ids)

    print(f"Time to process events with image_loop ({label_schema} labels):", time.time() - start)
    return out


## Function to reorder the order of clusters in the processed data
def reorder_clusters(data_processed, sim_processed):

    ## Rage quit if someone tries to call this without cluster labels
    if "clust" not in data_processed: return

    max_cluster = max(
        data_processed['clust_index'].max(),
        sim_processed['clust_index'].max()
    ) + 1

    # histogram of cluster frequency in data
    counts = np.bincount(data_processed['clust_index'], minlength=max_cluster)

    # permutation: clusters sorted by decreasing frequency
    perm = np.argsort(-counts)

    # mapping: old_index -> new_index
    remap = np.argsort(perm)

    data_processed['clust_index'] = remap[data_processed['clust_index']]
    sim_processed['clust_index']  = remap[sim_processed['clust_index']]

    data_processed['clust'] = data_processed['clust'][:, perm]
    sim_processed['clust']  = sim_processed['clust'][:, perm]
