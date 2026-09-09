# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# head factory
# --------------------------------------------------------
from .linear_head import LinearPts3d
from .dpt_head import create_dpt_head
# CAM3R: separate head that regresses the relative pose between the two views.
from .pose_network_relative import RelPoseHeadV2 


def head_factory(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'linear' and output_mode == 'pts3d':
        return LinearPts3d(net, has_conf)
    elif head_type == 'dpt' and output_mode == 'pts3d':
        return create_dpt_head(net, has_conf=has_conf)
    elif head_type == 'relpose_v2' and output_mode == 'relpose_v2':
        return RelPoseHeadV2(in_dim=net.dec_embed_dim, patch_size=net.patch_embed.patch_size[0], mode="multi", num_layers=net.dec_depth, rot_rep='9D')
    else:
        raise NotImplementedError(f"unexpected {head_type=} and {output_mode=}")