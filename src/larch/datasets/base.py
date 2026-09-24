from torch.utils.data import Dataset
import h5py
import numpy as np
import os
from glob import glob
from bisect import bisect
import MinkowskiEngine as ME
import torch
from collections import OrderedDict
from collections.abc import Mapping

class paired_2d_dataset_ME(Dataset):

    def __init__(self,
                 infile_dir,
                 nom_transform=None,
                 aug_transform=None,
                 max_events=None,
                 projection="xz",
                 max_open=9999
                 ):
        self.infile_dir = os.path.expandvars(infile_dir)
        self.hdf5_files = sorted(glob(os.path.join(self.infile_dir, '*.h5')))
        self.file_indices = []
        self.nom_transform = nom_transform
        self.aug_transform = aug_transform
        self.max_events = max_events

        ## 'xz', 'xy', or 'xyz'
        self.proj = projection

        ## For lazily caching files
        self.max_open = max_open
        self._handles = OrderedDict()

        ## file_index -> offsets array for self.proj
        self._offsets = {}

        ## Sort out the file map
        self.create_file_indices()

        ## Apply some limitation to the size
        if self.max_events and max_events < self.length:
            self.length = self.max_events

    @staticmethod
    def _apply_transform(transform, coords, feats):
        if transform is None:
            return coords, feats
        return transform(coords, feats)
            
    def create_file_indices(self):
        cumulative_size = 0
        
        for file in self.hdf5_files:
            self.file_indices.append(cumulative_size)
            with h5py.File(file, 'r', libver='latest') as f:
                cumulative_size += f.attrs['N']
        self.file_indices.append(cumulative_size)
        self.length = cumulative_size

    def _get_file(self, file_index):
        h = self._handles.get(file_index)
        if h is not None:
            self._handles.move_to_end(file_index)
            return h
        f = h5py.File(self.hdf5_files[file_index], 'r', libver='latest', rdcc_nbytes=0)
        # Cache the dataset objects once, not per __getitem__
        dsets = {}
        for name in (f'{self.proj}_data', f'{self.proj}_row', f'{self.proj}_col',
                     'xyz_data', 'xyz_coords', 'event_id'):
            if name in f:
                dsets[name] = f[name]
        h = {'file': f, 'dsets': dsets}
        self._handles[file_index] = h
        self._offsets[file_index] = f[f'{self.proj}_offsets'][:]
        if len(self._handles) > self.max_open:
            old_idx, old = self._handles.popitem(last=False)
            old['file'].close()
            self._offsets.pop(old_idx, None)
        return h

    def apply_aug_with_retry(self, coords, feats, max_retries=100):
        for _ in range(max_retries):
            out_coords, out_feats = self.aug_transform(coords, feats)
            if out_feats.size > 0:
                return out_coords, out_feats
        ## If no valid augmentation has been found, bail
        raise RuntimeError("Augmentation failed on initial image with feats.size =", feats.size)
    
    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        file_index = bisect(self.file_indices, idx)-1
        this_idx = idx - self.file_indices[file_index]
        f = self._get_file(file_index)
        off = self._offsets[file_index]
        s, e = int(off[this_idx]), int(off[this_idx + 1])
    
        dsets = f['dsets']
        
        ## Note that we can't build the sparse tensor here because ME uses some sort of global indexing
        ## And this function is replicated * num_workers
        if self.proj == 'xyz':
            raw_feats = dsets['xyz_data'][s:e].reshape(-1, 1)
            raw_coords = dsets['xyz_coords'][s:e]
        else:
            raw_feats = dsets[f'{self.proj}_data'][s:e].reshape(-1, 1)
            row = dsets[f'{self.proj}_row'][s:e]
            col = dsets[f'{self.proj}_col'][s:e]
            raw_coords = np.vstack((row, col)).T

        ## Apply a nom_transform if it exists
        nom_coords, nom_feats = self._apply_transform(self.nom_transform, raw_coords, raw_feats)

        ## Apply transforms to augment the data
        if not self.aug_transform:
            aug1_coords,aug1_feats = raw_coords,raw_feats
            aug2_coords,aug2_feats = raw_coords,raw_feats
        else:
            ## Make sure the images aren't empty...            
            aug1_coords, aug1_feats = self.apply_aug_with_retry(raw_coords, raw_feats)
            aug2_coords, aug2_feats = self.apply_aug_with_retry(raw_coords, raw_feats)

        return aug1_coords, aug1_feats, aug2_coords, aug2_feats, nom_coords, nom_feats

    
def triple_ME_collate_fn(batch):
    aug1_coords, aug1_feats, aug2_coords, aug2_feats, raw_coords, raw_feats = zip(*batch)

    # Create batched coordinates for the SparseTensor input
    aug1_bcoords = ME.utils.batched_coordinates(aug1_coords)
    aug2_bcoords = ME.utils.batched_coordinates(aug2_coords)
    raw_bcoords  = ME.utils.batched_coordinates(raw_coords)

    # Concatenate all lists
    aug1_bfeats = torch.from_numpy(np.concatenate(aug1_feats, 0)).float()
    aug2_bfeats = torch.from_numpy(np.concatenate(aug2_feats, 0)).float()
    raw_bfeats  = torch.from_numpy(np.concatenate(raw_feats, 0)).float()

    return aug1_bcoords, aug1_bfeats, aug2_bcoords, aug2_bfeats, raw_bcoords, raw_bfeats


def cat_ME_collate_fn(batch):
    aug1_coords, aug1_feats, aug2_coords, aug2_feats, _, _ = zip(*batch)

    coords_list = list(aug1_coords) + list(aug2_coords)
    feats_list  = list(aug1_feats)  + list(aug2_feats)
    
    # Create batched coordinates for the SparseTensor input
    cat_bcoords = ME.utils.batched_coordinates(coords_list)

    # Concatenate all lists
    cat_bfeats = torch.from_numpy(np.concatenate(feats_list, axis=0)).float()

    return cat_bcoords, cat_bfeats, len(batch)*2


class single_2d_dataset_ME(Dataset):

    def __init__(self,
                 infile_dir,
                 transform,
                 max_events=None,
                 return_metadata=False,
                 projection="xz",
                 max_open=9999):
        
        self.infile_dir = os.path.expandvars(infile_dir)
        self.hdf5_files = sorted(glob(os.path.join(self.infile_dir, '*.h5')))
        self.file_indices = []
        self.file_schemas = []
        self.transform = transform
        self.max_events = max_events
        self.return_metadata = return_metadata

        ## 'xz', 'xy', or 'xyz'
        self.proj = projection
        
        ## For lazily caching files
        self.max_open = max_open
        self._handles = OrderedDict()

        ## file_index -> offsets array for self.proj
        self._offsets = {}
        
        ## Sort out the file map
        self.create_file_indices()
        
        ## Apply some limitation to the size
        if self.max_events and max_events < self.length:
            self.length = self.max_events

    @staticmethod
    def _detect_file_schema(h5file):
        has_old = "labels" in h5file

        new_datasets = {
            "particle_truth",
            "particle_visible",
            "event_labels",
        }
        has_new = new_datasets.issubset(h5file.keys())

        ## Decide what schema was used
        if has_new: return "new"
        if has_old: return "old"
        raise RuntimeError(f"File {h5file.filename!r} is missing labels")

            
    def create_file_indices(self):
        cumulative_size = 0
        
        for file in self.hdf5_files:
            self.file_indices.append(cumulative_size)
            with h5py.File(file, 'r', libver='latest') as f:
                schema = self._detect_file_schema(f)
                self.file_schemas.append(schema)
                cumulative_size += f.attrs['N']
        self.file_indices.append(cumulative_size)
        self.length = cumulative_size

        if cumulative_size == 0:
            raise RuntimeError(f"No files in {self.infile_dir}!")
        
        schemas = set(self.file_schemas)
        if len(schemas) != 1:
            raise RuntimeError("Multiple labeling schemes found")
        self.label_schema = next(iter(schemas))


    def _get_file(self, file_index):
        handle = self._handles.get(file_index)

        if handle is not None:
            self._handles.move_to_end(file_index)
            return handle

        f = h5py.File(self.hdf5_files[file_index],
                      "r",
                      libver="latest",
                      rdcc_nbytes=0)

        if self.proj == "xyz":
            dataset_names = ["xyz_data",
                             "xyz_coords"]
        else:
            dataset_names = [f"{self.proj}_data",
                             f"{self.proj}_row",
                             f"{self.proj}_col"]

        if self.label_schema == "old":
            label_datasets = ["labels"]
        else:
            label_datasets = ["particle_truth",
                              "particle_visible",
                              "event_labels"]
            
        dataset_names += ["event_id",
                          *label_datasets]

        dsets = {}
        for name in dataset_names:
            if name not in f:
                f.close()
                raise RuntimeError(f"{name!r} missing from {self.hdf5_files[file_index]!r}")
            dsets[name] = f[name]

        handle = {"file": f,
                  "dsets": dsets}
        self._handles[file_index] = handle
        offset_name = f"{self.proj}_offsets"
        self._offsets[file_index] = f[offset_name][:]

        if len(self._handles) > self.max_open:
            old_index, old_handle = self._handles.popitem(last=False)
            old_handle["file"].close()
            self._offsets.pop(old_index, None)

        return handle

    def apply_aug_with_retry(self, coords, feats, max_retries=100):

        if self.transform is None:
            return coords, feats
        
        for _ in range(max_retries):
            out_coords, out_feats = self.transform, coords, feats)
            if out_feats.size > 0:
                return out_coords, out_feats
        ## If no valid augmentation has been found, bail
        raise RuntimeError("Augmentation failed on initial image with feats.size =", feats.size)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.length

        if idx < 0 or idx >= self.length:
            raise IndexError(idx)

        file_index = bisect(self.file_indices, idx) - 1
        this_idx = idx - self.file_indices[file_index]

        handle = self._get_file(file_index)
        offsets = self._offsets[file_index]
        dsets = handle["dsets"]

        start = int(offsets[this_idx])
        end = int(offsets[this_idx + 1])

        if self.proj == "xyz":
            feats = dsets["xyz_data"][start:end].reshape(-1, 1)
            coords = dsets["xyz_coords"][start:end]
        else:
            feats = dsets[f"{self.proj}_data"][start:end].reshape(-1, 1)
            row = dsets[f"{self.proj}_row"][start:end]
            col = dsets[f"{self.proj}_col"][start:end]
            coords = np.column_stack((row, col))
            
        coords, feats = self.apply_aug_with_retry(coords, feats)

        if self.label_schema == "old":
            labels = dsets["labels"][this_idx]
        else:
            labels = {"particle_truth": dsets["particle_truth"][this_idx],
                      "particle_visible": dsets["particle_visible"][this_idx],
                      "event": dsets["event_labels"][this_idx],}

        if self.return_metadata:
            filename = os.path.basename(self.hdf5_files[file_index])
            event_id = int(dsets["event_id"][this_idx])
            return coords, feats, labels, filename, event_id
        
        return coords, feats, labels        

    def close(self):
        for handle in self._handles.values():
            handle["file"].close()

        self._handles.clear()
        self._offsets.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
        

def collate_nested_labels(labels):

    ## If these aren't nested labels (backwards compatibility), return
    if not isinstance(labels[0], Mapping):        
        return labels
    
    groups = tuple(labels[0].keys())
    blabels = {}

    for group in groups:
        records = [label[group] for label in labels]
        dtype = records[0].dtype

        if dtype.names is None:
            raise TypeError(f"Expected structured records for group '{group}', "
                            f"found dtype {dtype}")

        array = np.asarray(records, dtype=dtype)

        blabels[group] = {name: torch.from_numpy(np.ascontiguousarray(array[name]))
                          for name in dtype.names}

    return blabels

def solo_ME_collate_fn(batch):
    coords, feats, labels = zip(*batch)
    
    # Create batched coordinates for the SparseTensor input
    bcoords  = ME.utils.batched_coordinates(coords)
    
    # Concatenate all lists
    bfeats  = torch.from_numpy(np.concatenate(feats, 0)).float()

    # Deal with the multiple dictionaries
    blabels = collate_nested_labels(labels)
    
    return bcoords, bfeats, blabels

def solo_ME_collate_fn_with_meta(batch):
    coords, feats, labels, filenames, event_ids = zip(*batch)
    bcoords = ME.utils.batched_coordinates(coords)
    bfeats = torch.from_numpy(np.concatenate(feats, 0)).float()
    blabels = collate_nested_labels(labels)
    return bcoords, bfeats, blabels, filenames, event_ids


## label_clamp allows a configurable maximum to be provided
## Derived labels allows raw labels to be added or otherwise manipulated
def solo_labelled_collate_fn(batch,
                             label_clamp=None,
                             derived_labels=None):
    
    coords, feats, labels = zip(*batch)
    bcoords = ME.utils.batched_coordinates(coords)
    bfeats = torch.from_numpy(np.concatenate(feats, axis=0)).float()

    ## Old schema for backwards compatibility
    if not isinstance(labels[0], Mapping):

        blabels = {name: torch.from_numpy(np.asarray([label[name] for label in labels]))
            for name in labels[0].dtype.names}
        
        if derived_labels:
            for name, function in derived_labels.items():
                blabels[name] = function(blabels)

        if label_clamp:
            for name, clamp_value in label_clamp.items():
                if name in blabels:
                    blabels[name] = torch.clamp(
                        blabels[name],
                        min=0,
                        max=clamp_value,
                    )

    else:
        blabels = collate_nested_labels(labels)

        ## New schema, future default
        if derived_labels:
            blabels["derived"] = {}

            for name, function in derived_labels.items():
                blabels["derived"][name] = function(blabels)

        if label_clamp:
            for group_name, group_clamps in label_clamp.items():
                if group_name not in blabels:
                    continue

                for name, clamp_value in group_clamps.items():
                    if name not in blabels[group_name]:
                        continue

                    blabels[group_name][name] = torch.clamp(
                        blabels[group_name][name],
                        min=0,
                        max=clamp_value,
                    )

    return bcoords, bfeats, blabels, len(batch)
