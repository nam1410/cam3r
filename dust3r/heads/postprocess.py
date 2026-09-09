# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# post process function for all heads: extract 3D points/confidence from output
# Now supports both:
#   - pts3d output: out.shape[1] in {3,4} (XYZ + optional conf)
#   - distance output: out.shape[1] in {1,2} (radial distance + optional conf)
# --------------------------------------------------------
import torch
import torch.nn.functional as F


def postprocess(out, depth_mode, conf_mode):
    """
    Dispatch to proper postprocess depending on #channels.
      out: B x C x H x W
      returns dict with keys:
        - pts3d (+ conf) for C in {3,4}
        - distance (+ conf) for C in {1,2}
    """
    
    B, C, H, W = out.shape
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C

    if C in (3, 4):
        res = dict(pts3d=reg_dense_pts3d(fmap[..., 0:3], mode=depth_mode))
        if conf_mode is not None and C == 4:
            res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
        return res

    if C in (1, 2):
        dist = reg_dense_distance(fmap[..., 0:1], mode=depth_mode)  # B,H,W,1
        res = dict(distance=dist)
        if conf_mode is not None and C == 2:
            res['conf'] = reg_dense_conf(fmap[..., 1], mode=conf_mode)
        return res

    raise ValueError(f'Unexpected number of channels in head output: C={C}')


def reg_dense_pts3d(xyz, mode):
    """
    Legacy path: extract 3D points from prediction head output (B,H,W,3).
    Mode affects the magnitude scaling by distance (norm).
    """
    mode, vmin, vmax = mode
    no_bounds = (vmin == -float('inf')) and (vmax == float('inf'))
    assert no_bounds

    if mode == 'linear':
        if no_bounds:
            return xyz  # [-inf, +inf]
        return xyz.clip(min=vmin, max=vmax)

    # distance to origin
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)

    if mode == 'square':
        return xyz * d.square()

    if mode == 'exp':
        return xyz * torch.expm1(d)

    raise ValueError(f'bad {mode=}')


def reg_dense_distance(x, mode):
    """
    New path: extract positive radial distance (depth from camera center)
    from single-channel prediction x (B,H,W,1).

    Recommendations:
    - Use mode 'exp' and predict log-distance: distance = exp(x) or expm1(x).
    - Alternatively, use 'softplus' (safer gradients), configured via depth_mode=('softplus', 0, inf).
    """
    mode, vmin, vmax = mode
    # For distance, we enforce non-negativity:
    no_bounds = (vmin == 0.0) and (vmax == float('inf'))
    assert no_bounds, "[reg_dense_distance] Only (0, +inf) bounds supported for distance."

    if mode == 'linear':
        # If model already predicts positive distances (e.g., via last-layer ReLU/Softplus), just clamp
        return x.clip(min=0.0)

    if mode == 'softplus':
        # Smooth, positive; numerically stable
        return F.softplus(x)

    if mode == 'exp':
        # Positive; be careful with very large inputs
        # You can clamp logits to keep stability if needed:
        x_clamped = x.clip(min=-20.0, max=20.0)
        return torch.exp(x_clamped)

    if mode == 'expm1':
        # Smaller values near zero; still positive
        x_clamped = x.clip(min=-20.0, max=20.0)
        return torch.expm1(x_clamped).clip(min=0.0)

    raise ValueError(f'bad {mode=}')


def reg_dense_conf(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')