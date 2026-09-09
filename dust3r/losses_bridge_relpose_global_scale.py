# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
import torch
import torch.nn as nn
import torch.nn.functional as F

from dust3r.losses_unik3d_cam_coords import MultiLoss
from dust3r.utils.geometry import inv, geotrf


def _extract_pred_pose(pred: dict):
    """
    Extract predicted pose from view's pred dict.
    Returns:
      dict {'R': (B,3,3), 't': (B,3)}
    """
    if 'R' in pred and 't' in pred:
        return {'R': pred['R'], 't': pred['t']}
    raise KeyError("pred must contain pose as either pred['pose']{'R','t'} or keys 'R','t' directly.")


def _extract_valid_points(pred_dict, gt_dict):
    output_pts3d = pred_dict['pts3d'] # predicted points in cam coord of some view
    gt_pts3d_world  = gt_dict['pts3d'] # ground truth points of some view in world coord
    
    gt_pose = gt_dict['camera_pose'] # view -> cam2world: ground truth camera pose from cam coordinates to world
    # R_cam2world = gt_pose[:, :3, :3] #ground truth rotation of view - cam2world
    t_cam2world = gt_pose[:, :3, 3] #ground truth translation of view - cam2world

    in_camera_pose = inv(gt_pose) # view -> world2cam: ground truth camera pose from world to cam coordinates

    gt_pts3d = geotrf(in_camera_pose, gt_pts3d_world) # ground truth points in cam coordinate of view  
    valid_gt = gt_dict['valid_mask'] # valid mask for view from ground truth 
    valid_pts3d = output_pts3d[valid_gt] # extract valid predicted points (cam coordinates) for view 
    valid_pts3d_gt = gt_pts3d[valid_gt] # extract valid ground truth points (cam coordinates) for view 

    return valid_pts3d, valid_pts3d_gt, gt_pose, t_cam2world

def _compute_global_scale(pairs_list):
    """
    Computes a single robust scale factor using points from MULTIPLE views.
    """
    all_pred_dists = []
    all_gt_dists = []

    # 1. Collect distances from ALL views
    for pts_pred, pts_gt in pairs_list:
        d_pred = torch.norm(pts_pred, dim=-1)
        d_gt = torch.norm(pts_gt, dim=-1)
        
        valid_mask = d_gt > 1e-4
        d_pred_valid = d_pred[valid_mask]
        d_gt_valid = d_gt[valid_mask]

        all_pred_dists.append(d_pred_valid)
        all_gt_dists.append(d_gt_valid)

    # 2. Concatenate into one giant array
    cat_pred = torch.cat(all_pred_dists)
    cat_gt = torch.cat(all_gt_dists)

    if cat_gt.numel() == 0:
        return 1.0

    # 3. Compute Robust Global Scale
    scale_ratios = cat_pred / cat_gt
    global_scale = torch.median(scale_ratios)

    return global_scale

def _slice_dict(d, idx):
    """
    Helper to slice all tensors in a dict by the batch index `idx`.
    Keeps dimension 0 as 1 (e.g., [1, C, H, W]) to satisfy geometry utils expects.
    """
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            # Use slice idx:idx+1 to keep the dimension
            out[k] = v[idx:idx+1]
        else:
            out[k] = v
    return out

class PoseAngularLoss(MultiLoss):
    """
    DUSt3R-compatible pose loss that computes angular errors for rotation and translation
    between predicted relative pose and GT relative pose for each view, then averages.

    Expected inputs from the DUSt3R training loop:
      - gt1, gt2: dicts that contain 'camera_pose' with shape (B,4,4) (cam2world).
      - pred1, pred2: dicts that contain pose predictions with keys 'R' (B,3,3), 't' (B,3).

    Relative GT convention:
      transformation_1to0 = torch.linalg.inv(gt_pose1) @ gt_pose2 # view 2 to view 1 - gt_pose2 is view2's cam2world i.e. converts points in view2 cam coordinate space to world; inv(gt_pose1) is view1's world2cam i.e. converts any world points to cam coordinate space of view 1
    

    Returns:
      (scalar_loss, details_dict) compatible with DUSt3R's MultiLoss.

    CLI:
      --train_criterion="UniK3DCameraPolar(weight=1.0, polar_weight=3.0, polar_asym=0.7) + PoseAngularLoss(weight=1.0) + Regr3D_ScaleShiftInv(L21, gt_scale=True)"
      --test_criterion="UniK3DCameraPolar(weight=1.0) + PoseAngularLoss(weight=1.0) + Regr3D_ScaleShiftInv(L21, gt_scale=True)"
    """
    def __init__(self, weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self._name = 'PoseAngularLoss'
        self.weight = float(weight)
        self.eps = float(eps)

    def get_name(self) -> str:
        return self._name

    def _calculate_pose_loss(self, gt_view0: dict, gt_view1: dict, pred_view0: dict,  pred_view1: dict):
        """
        Relative-pose loss between the two views: an angular term on rotation and
        an MSE term on the (globally rescaled) translation. _extract_valid_points()
        factors out the point/pose extraction shared by both views.
        """
        if 'camera_pose' not in gt_view0 or 'camera_pose' not in gt_view1:
            raise KeyError("gt dicts must contain key 'camera_pose'")
        
        # 1. Extract Prediction
        pose_pred_1to0 = _extract_pred_pose(pred_view1)  # (B, 3, 3) - predicted rotation and (B, 3) - predicted translation from view 1 to view 0
        # predicted R and t from predicted transformation from view 1 to view 0
        pred_R_1to0 = pose_pred_1to0["R"] # (B, 3, 3)
        pred_t_1to0 = pose_pred_1to0["t"] # (B, 3)

        # 2. Extract Points & GT Poses
        valid_pts3d0, valid_pts3d_gt0, gt_pose0, t0_cam2world = _extract_valid_points(pred_view0, gt_view0) # extract valid predicted and ground truth points for view 0; also, extract ground truth rotation and translation for view 0 - cam2world
        valid_pts3d1, valid_pts3d_gt1, gt_pose1, t1_cam2world = _extract_valid_points(pred_view1, gt_view1)  # extract valid predicted and ground truth points for view 1; also, extract ground truth rotation and translation for view 1 - cam2world

        # 3. Compute Global Scale
        # The scale is detached: it is treated as a fixed target so that the
        # gradient moves the predicted translation, not the point-cloud scale.
        scale_factor = _compute_global_scale([
            (valid_pts3d0, valid_pts3d_gt0),
            (valid_pts3d1, valid_pts3d_gt1),
        ]).detach()

        # 4. Adjust GT Translation Magnitudes
        t0_adj = t0_cam2world * scale_factor
        t1_adj = t1_cam2world * scale_factor

        # 5. Reconstruct Scaled GT Relative Pose (View 1 -> View 0)
        # We only need the t_rel component here really, but full matrix is safer
        # Pose 0 Adjusted
        gt_pose0_adj = gt_pose0.clone()
        gt_pose0_adj[:, :3, 3] = t0_adj
        
        # Pose 1 Adjusted
        gt_pose1_adj = gt_pose1.clone()
        gt_pose1_adj[:, :3, 3] = t1_adj

        # deriving R and t from ground truth transformation from view 1 to view 0 after scale adjustment 
        gt_rel_1to0 = torch.linalg.inv(gt_pose0_adj) @ gt_pose1_adj
        gt_adjusted_R_1to0 = gt_rel_1to0[:, :3, :3]
        gt_adjusted_t_1to0 = gt_rel_1to0[:, :3, 3] # This vector now has the CORRECT Magnitude relative to the point cloud
        
        # --- LOSS CALCULATION ---

        # A. Rotation Loss (Keep Angular/Cosine)
        # Rotation is independent of scale, so Angular loss is perfect here.
        R_rel = torch.matmul(pred_R_1to0.transpose(1, 2), gt_adjusted_R_1to0) 
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        cos_theta = ((trace - 1) / 2).clamp(-1 + self.eps, 1 - self.eps)
        rot_loss = torch.acos(cos_theta).mean()

        # B. Translation Loss (Switch to MSE/L1)
        # We use MSE (L2) or L1 to enforce BOTH Direction AND Scale.
        # This will penalize the model if pred_t is too small or too large.
        trans_loss = F.mse_loss(pred_t_1to0, gt_adjusted_t_1to0)

        total_loss = self.weight * (trans_loss + rot_loss)
        return total_loss, trans_loss, rot_loss


    def compute_loss(self, gt1: dict, gt2: dict, pred1: dict, pred2: dict, **kw):
        """
        DUSt3R's MultiLoss entry: compute per-view losses and average them.
        """
        batch_size = gt1['pts3d'].shape[0]
        
        total_loss_accum = 0.0

        for b in range(batch_size):
            # Slice dictionaries to isolate one batch element
            # We use [b:b+1] to keep the dimension (1, ...) for geometry utils compatibility
            g1_s = _slice_dict(gt1, b)
            g2_s = _slice_dict(gt2, b)
            p1_s = _slice_dict(pred1, b)
            p2_s = _slice_dict(pred2, b)

            # Calculate loss for this single pair
            loss_b, trans_b, rot_b = self._calculate_pose_loss(g1_s, g2_s, p1_s, p2_s)
            
            total_loss_accum += loss_b

        # Average over batch
        avg_loss = total_loss_accum / batch_size

        details = {
            f'{self.get_name()}_total': float(avg_loss.detach()),
        }
        return avg_loss, details