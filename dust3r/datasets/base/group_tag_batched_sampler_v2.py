# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
import numpy as np
import torch
import sys

# Optional: import wrapper types if you want isinstance checks
try:
    from dust3r.datasets.base.easy_dataset import CatDataset, ResizedDataset, MulDataset
except ImportError:
    CatDataset = ResizedDataset = MulDataset = type('Dummy', (), {})

def round_by(total, multiple, up=False):
    if up:
        total = total + multiple - 1
    return (total // multiple) * multiple

def _is_cat(obj):
    return hasattr(obj, 'datasets') and isinstance(getattr(obj, 'datasets'), list)

def _unwrap_for_tag(ds):
    """
    Return the underlying base dataset to derive a tag:
    - If ds has attribute 'dataset' (ResizedDataset/MulDataset) unwrap one level until no '.dataset'
      OR until we hit something that looks like a BaseStereoViewDataset (has 'scenes').
    - Stop at first object having 'dataset_label' or 'scenes'.
    """
    visited = set()
    cur = ds
    while True:
        if id(cur) in visited:
            break
        visited.add(id(cur))
        if hasattr(cur, 'dataset'):
            inner = getattr(cur, 'dataset')
            # If inner has scenes attribute, consider inner the base; else continue
            if hasattr(inner, 'scenes'):
                cur = inner
                break
            cur = inner
            continue
        break
    return cur

def _collect_leaf_wrappers(root):
    """
    Recursively flatten nested CatDataset structures.
    Returns a list of 'sampling children':
      Each element is a wrapper (possibly ResizedDataset/MulDataset or base dataset) that
      will be sampled independently. Its length is len(child), and its resolutions
      are child._resolutions (inherited through wrappers).
    """
    leaves = []
    if _is_cat(root):
        for c in root.datasets:
            leaves.extend(_collect_leaf_wrappers(c))
    else:
        leaves.append(root)
    return leaves

class GroupByResolutionAndTagBatchSampler:
    """
    Batch sampler enforcing homogeneous (resolution, tag) batches.
    Recursively flattens nested CatDataset wrappers so each base dataset (possibly wrapped
    by ResizedDataset/MulDataset) contributes its own samples/groups.

    Yields tuples: (global_sample_idx, local_ar_idx_in_child)
    where global_sample_idx indexes into a virtual concatenation of flattened children.

    Parameters
    ----------
    dataset : CatDataset-like (possibly nested)
    batch_size : int
    tags_mapping : dict[str, str] or None
        Maps a dataset identifier (dataset_label or class name) to a tag. If None,
        tag = dataset_label if present else class name.
    use_dataset_label : bool
        Prefer dataset_label attribute when present.
    group_weights : list[float] or None
        Optional weights for (resolution, tag) groups.
    drop_last : bool
        If True, drop remainder to keep full batches; otherwise final incomplete batches are padded/repeated.

    Resolution Differences
    ----------------------
    Each child can have its own resolution list. For every resolution the child supports,
    a group (resolution, tag) is formed. Children sharing tag & resolution merge into same group
    (if you want them separate, give them distinct tags in tags_mapping).
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 tags_mapping=None,
                 use_dataset_label=True,
                 world_size=1,
                 rank=0,
                 drop_last=True,
                 group_weights=None):
        self.original_dataset = dataset
        self.batch_size = batch_size
        self.tags_mapping = tags_mapping
        self.use_dataset_label = use_dataset_label
        self.world_size = world_size
        self.rank = rank
        self.drop_last = drop_last
        self.group_weights = group_weights
        self.epoch = None

        # Flatten nested concatenations
        self.children = _collect_leaf_wrappers(dataset)
        assert len(self.children) > 0, "No children found in dataset."

        # Compute lengths & offsets in global index space
        self.child_lens = [len(ch) for ch in self.children]
        self.child_offsets = np.cumsum([0] + self.child_lens[:-1])
        self.n_children = len(self.children)

        # Derive tags per child
        self.child_tags = []

        for ch in self.children:

            base = _unwrap_for_tag(ch)
            identifier = None
            if self.use_dataset_label and hasattr(base, 'dataset_label'):
                identifier = getattr(base, 'dataset_label')
            if identifier is None:
                identifier = type(base).__name__
            if self.tags_mapping:
                # Normalize key lookups (case sensitive unless you choose otherwise)
                tag = self.tags_mapping.get(identifier, identifier)
            else:
                tag = identifier
            self.child_tags.append(tag)

        # Map resolution -> local ar index per child
        self.child_res_to_ar_idx = []
        for ch in self.children:
            res_map = {}
            # Access _resolutions via wrapper (wrappers forward property)
            for j, (w, h) in enumerate(ch._resolutions):
                res_map[(int(w), int(h))] = int(j)
            self.child_res_to_ar_idx.append(res_map)

        # Build groups (resolution, tag)
        group_dict = {}
        for c_idx, (start, length) in enumerate(zip(self.child_offsets, self.child_lens)):
            if length == 0:
                continue
            tag = self.child_tags[c_idx]
            res_map = self.child_res_to_ar_idx[c_idx]
            for res_key, local_ar in res_map.items():
                gkey = (res_key[0], res_key[1], tag)
                if gkey not in group_dict:
                    group_dict[gkey] = {
                        'indices': [],
                        'child_local_ar': -np.ones(self.n_children, dtype=np.int64),
                        'res_key': (res_key[0], res_key[1]),
                        'tag': tag
                    }
                group_dict[gkey]['child_local_ar'][c_idx] = local_ar
                # Add all indices of this child to group's pool
                group_dict[gkey]['indices'].extend(range(start, start + length))

        self.groups = []
        for gkey, entry in group_dict.items():
            if len(entry['indices']) == 0:
                continue
            entry['indices'] = np.array(entry['indices'], dtype=np.int64)
            self.groups.append(entry)

        assert len(self.groups) > 0, "No non-empty (resolution, tag) groups discovered."

        if self.group_weights is not None:
            assert len(self.group_weights) == len(self.groups), \
                f"group_weights length {len(self.group_weights)} != number of groups {len(self.groups)}"
            self.group_weights = np.array(self.group_weights, dtype=np.float64)
            self.group_weights /= self.group_weights.sum()

        

        N = sum(self.child_lens)
        if drop_last:
            self.total_size = round_by(N, batch_size * world_size)
        else:
            self.total_size = N

    def __len__(self):
        return self.total_size // self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _rng(self):
        if self.epoch is None:
            assert self.world_size == 1 and self.rank == 0, "Use set_epoch() in distributed mode."
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
        else:
            seed = self.epoch + 777
        return np.random.default_rng(seed=seed)

    def __iter__(self):
        rng = self._rng()
        if self.drop_last:
            n_batches = self.total_size // self.batch_size
        else:
            n_batches = (self.total_size + self.batch_size - 1) // self.batch_size

        child_offsets = self.child_offsets
        pairs = []

        for _ in range(n_batches):
            # Select a group
            if self.group_weights is None:
                g_idx = int(rng.integers(len(self.groups)))
            else:
                g_idx = int(rng.choice(len(self.groups), p=self.group_weights))

            """
            groups depend upon the resolution passed as well - incase of multi resolution, groups will be duplicated among the tags for each resolution. restrict to a single resolution for uniform sampling; else - add group weights 
            """
            grp = self.groups[g_idx]
            pool = grp['indices']
            if len(pool) == 0:
                continue

            replace = len(pool) < self.batch_size
            sample_idxs = rng.choice(pool, size=self.batch_size, replace=replace)

            child_idx = np.searchsorted(child_offsets, sample_idxs, side='right') - 1
            local_ar = grp['child_local_ar'][child_idx]

            # Safety: filter invalid (should not occur if maps built right)
            if np.any(local_ar < 0):
                valid = local_ar >= 0
                sample_idxs = sample_idxs[valid]
                local_ar = local_ar[valid]
                if len(sample_idxs) < self.batch_size:
                    need = self.batch_size - len(sample_idxs)
                    extra = rng.choice(pool, size=need, replace=(len(pool) < need))
                    extra_child = np.searchsorted(child_offsets, extra, side='right') - 1
                    extra_local = grp['child_local_ar'][extra_child]
                    sample_idxs = np.concatenate([sample_idxs, extra])
                    local_ar = np.concatenate([local_ar, extra_local])

            pairs.extend((int(si), int(ai)) for si, ai in zip(sample_idxs, local_ar))

        # Distributed slicing aligned to batch_size
        size_per_proc = self.batch_size * ((self.total_size + self.world_size * self.batch_size - 1)
                                           // (self.world_size * self.batch_size))
        start = self.rank * size_per_proc
        end = (self.rank + 1) * size_per_proc

        if end > len(pairs):
            need = end - len(pairs)
            if len(pairs) > 0:
                reps = (need + len(pairs) - 1) // len(pairs)
                tiled = pairs * (reps + 1)
                pairs = pairs + tiled[:need]
            else:
                pairs = []
        pairs = pairs[start:end]

        yield from pairs