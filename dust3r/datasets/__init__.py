# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# CAM3R re-exports only the dataset classes it ships. Upstream DUSt3R also
# exported ARKitScenes, BlendedMVS, Co3d, Habitat, ScanNet++, StaticThings3D,
# Waymo and WildRGBD; those loaders are not part of CAM3R, so importing them
# here would make this package fail to load.
from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa

from .megadepth_updated import MegaDepth  # noqa
from .pano2D3DS import Pano2D3DS  # noqa
from .panoFishPin2D3DS import PanoFishPin2D3DS  # noqa
from .fisheyeAria import FisheyeAria  # noqa
from .fisheyePinholeAria import FisheyePinholeAria  # noqa
from .co3dFisheyePinhole import Co3dFisheyePinhole  # noqa
from .pano360Loc import Pano360Loc  # noqa
from .panoFishPin360Loc import PanoFishPin360Loc  # noqa


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        # passing globals() lets eval() see the dataset classes imported above
        dataset = eval(dataset, globals())

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader
