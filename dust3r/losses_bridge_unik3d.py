# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
"""
Bridge: Use UniK3D's PolarRegression ray loss inside DUSt3R's loss pipeline.

Simple usage from CLI (pair with DUSt3R's Regr3D for pts3d):
  --train_criterion="UniK3DCameraPolar(weight=1.0, polar_weight=3.0, polar_asym=0.7) + Regr3D_ScaleShiftInv(L21, gt_scale=True)"
  --test_criterion="UniK3DCameraPolar(weight=1.0) + Regr3D_ScaleShiftInv(L21, gt_scale=True)"

Design notes:
- We only add ray loss. No depth loss here. Depth is implicitly supervised by Regr3D via pts3d = rays * distance.
- The wrapper adapts DUSt3R's dicts and shapes to UniK3D's expected (B,C,H,W) interface.
- The mask preference is: depth_mask -> valid_mask -> all-ones.
"""

import torch
import dust3r.utils.path_to_unik3d  # noqa: F401  puts <repo>/UniK3D on sys.path
import importlib
import sys
from einops import rearrange

# Reuse DUSt3R's MultiLoss to compose with existing losses (e.g., + Regr3D(...))
from dust3r.losses_unik3d_cam_coords import MultiLoss


# Dynamically import UniK3D loss factories (no hard dependency on its package layout here)
# This module exposes PolarRegression in unik3d.ops.losses.__all__
u_losses = importlib.import_module('unik3d.ops.losses')


# ------------------------------ Helpers ------------------------------ #

def _to_chw(t: torch.Tensor | None) -> torch.Tensor | None:
    """
    Convert tensor to channel-first (B, C, H, W) as UniK3D losses expect.

    Accepts:
      - (B, H, W, C) -> (B, C, H, W)
      - (B, H, W)    -> (B, 1, H, W)
      - (B, C, H, W) -> passthrough
    """
    if t is None:
        return None
    if t.ndim == 4 and t.shape[-1] in (1, 3):
        return t.permute(0, 3, 1, 2).contiguous()
    if t.ndim == 3:
        return t.unsqueeze(1).contiguous()
    if t.ndim == 4 and t.shape[1] in (1, 3):
        return t
    raise ValueError(f'Unexpected tensor shape {t.shape}, expected (B,H,W,C) or (B,C,H,W)')


def _get_rays_gt(view: dict) -> torch.Tensor:
    """
    Fetch GT rays (B,(H*W),3) or (B,H,W,C) from the view dict.

    Priority:
      1) view['gt_rays'] if present
      2) normalize view['pts3d'] to unit-length: pts3d / ||pts3d||
    """
    _, _, H, W = view['img'].shape
    if 'gt_rays' in view:
        gt_rays = view['gt_rays']
        if gt_rays.ndim == 4 and gt_rays.shape[-1] == 3:  # Check for (B, H, W, C)
            gt_rays = rearrange(gt_rays, "b h w c -> b c h w")
        elif gt_rays.ndim == 3:  # Check for (B, (H*W), 3)
            gt_rays = rearrange(gt_rays, "b (h w) c -> b c h w", h=H, w=W)
        return gt_rays
    raise KeyError("Need GT 'rays' in the view dict to build target rays.")


def _get_mask(view: dict, fallback_chw: torch.Tensor) -> torch.Tensor:
    """
    Build a boolean mask (B,1,H,W) to weight the loss.
    Accepts (B,H,W), (B,1,H,W), or (B,H,W,1). Converts to (B,1,H,W).
    """
    m = view.get('valid_mask', None)
    if m is None:
        B, _, H, W = fallback_chw.shape
        return torch.ones((B, 1, H, W), dtype=torch.bool, device=fallback_chw.device)

    # Handle shapes: (B,H,W) -> (B,1,H,W); (B,H,W,1) -> (B,1,H,W); (B,1,H,W) stays
    if m.ndim == 3:
        m = m.unsqueeze(1)
    if m.ndim == 4 and m.shape[-1] == 1:
        m = m.permute(0, 3, 1, 2).contiguous()
    if m.shape[1] != 1:
        m = m[:, :1]
    return m.bool()


# --------------------------- DUSt3R-compatible loss --------------------------- #

class UniK3DCameraPolar(MultiLoss):
    """
    DUSt3R criterion that wraps UniK3D's PolarRegression (angular ray loss).

    Inputs expected from DUSt3R's training loop:
      - gt1, gt2: dict-like views with possible keys:
          'rays' (B,H,W,3), 'pts3d' (B,H,W,3), 'depth_mask' (B,H,W) or 'valid_mask' (B,H,W)
        We build GT rays and a (B,1,H,W) mask from these.
      - pred1, pred2: dict-like preds where we read:
          'rays' (B,H,W,3) from your adapter head.

    Behavior:
      - Converts pred/gt rays to (B,3,H,W)
      - Builds a (B,1,H,W) mask
      - Calls UniK3D PolarRegression forward(...) which returns a per-image vector
      - Averages across images and both views, returning a scalar and a small details dict
    """
    """
                "camera": {
                "name": "PolarRegression",
                "weight": 1.0,
                "gamma": 1.0,
                "alpha": 1.0,
                "fn": "l1",
                "output_fn": "sqrt",
                "input_fn": "linear",
                "dims": [
                    1,
                    2
                ],
                "polar_weight": 3.0,
                "polar_asym": 0.7
            },
    """
    def __init__(self,
                 weight: float = 1.0,           # scalar multiplier applied by your MultiLoss chain (set in CLI)
                 fn: str = 'l1',                # underlying robust penalty ('l1', 'l2', 'charbonnier', etc.)
                 input_fn: str = 'linear',      # UniK3D pre-transform for inputs
                 output_fn: str = 'sqrt',       # UniK3D post-transform for aggregated loss
                 gamma: float = 1.0,            # scale for selected robust penalty
                 alpha: float = 1.0,            # alpha for some robust penalties
                 dims: tuple[int, ...] = (-1, -2),  # spatial dims to reduce inside UniK3D loss
                 polar_weight: float = 3.0,     # extra weight for polar term (UniK3D design)
                 polar_asym: float = 0.7):      # asymmetry factor (biases errors across hemispheres)
        super().__init__()

        # 1) Build the UniK3D loss object using its .build factory.
        #    This returns an nn.Module whose forward(input, target, mask, ...) returns per-image losses.
        self._u_loss = u_losses.PolarRegression.build(dict(
            weight=weight,
            fn=fn,
            gamma=gamma,
            alpha=alpha,
            input_fn=input_fn,
            output_fn=output_fn,
            dims=list(dims),
            polar_weight=polar_weight,
            polar_asym=polar_asym,
        ))

        # 2) Keep a readable name for DUSt3R's logging (details dict keys)
        self._name = 'UniK3DCameraPolar'

    def get_name(self) -> str:
        # Used by DUSt3R's MultiLoss to label metrics in TensorBoard/logs
        return self._name

    def _one_view(self, gt: dict, pred: dict) -> torch.Tensor:
        """
        Compute the PolarRegression loss for a single view.

        Steps:
          a) Read pred rays (B,H,W,3) and convert to (B,3,H,W).
          b) Build GT rays (prefer view['rays'], else normalize view['pts3d']).
          c) Build a (B,1,H,W) mask (prefer depth_mask, else valid_mask, else ones).
          d) Call UniK3D's PolarRegression: returns per-image vector.
          e) Mean-reduce to a scalar (compatible with DUSt3R's MultiLoss contract).
        """
        # a) Pred rays (adapter output) must exist
        if 'rays' not in pred:
            raise KeyError("pred dict must contain key 'rays' with shape (B,H,W,3).")
        

        pred_rays_chw = _to_chw(pred['rays'])     # (B,3,H,W)

        # b) Target rays from view dict (either 'rays' or normalized 'pts3d')
        gt_rays = _get_rays_gt(gt)
        
        gt_rays_chw = _to_chw(gt_rays)            # (B,3,H,W)

        # c) Mask (B,1,H,W), ensures we ignore invalid pixels if provided
        mask = _get_mask(gt, pred_rays_chw)       # (B,1,H,W)

        # d) UniK3D's loss returns per-image vector (shape: (B,))
        per_image = self._u_loss(pred_rays_chw, target=gt_rays_chw, mask=mask)

        # e) Mean over batch to get a scalar; DUSt3R's MultiLoss will average over both views next
        return per_image.mean()

    def compute_loss(self, gt1: dict, gt2: dict, pred1: dict, pred2: dict, **kw):
        """
        DUSt3R's MultiLoss entry point.
        - Compute view-wise losses and average them.
        - Return (scalar_loss, details_dict) so training logs per-view components.
        """
        l1 = self._one_view(gt1, pred1)   # scalar loss for view1
        l2 = self._one_view(gt2, pred2)   # scalar loss for view2
        loss = 0.5 * (l1 + l2)            # symmetric average

        # Optional: expose per-view values in the details dict for monitoring
        details = {
            self.get_name() + '_1': float(l1.detach()),
            self.get_name() + '_2': float(l2.detach()),
        }
        return loss, details