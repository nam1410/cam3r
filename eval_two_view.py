# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Derived from the DUSt3R evaluation code.
# --------------------------------------------------------
r"""Relative-pose evaluation of a CAM3R checkpoint on the held-out validation splits.

Default mode: evaluate every pair of each dataset's validation split and report
median RRA/RTA, RRA/RTA at 15 and 30 degrees, and mAA@30 per dataset, followed by
a summary table. These are the numbers reported in the paper.

--two-view-test selects one representative pair for each camera-model combination
(pano-pano, pano-fisheye, fisheye-fisheye, pano-pinhole, fisheye-pinhole,
pinhole-pinhole) from fixed scenes, runs the model on them, and saves pointclouds,
poses, depth, confidence and mutual-nearest-neighbour matches for each.

This mode produces the QUALITATIVE figures only. It searches within each camera-model
combination for a pair whose RRA and RTA are both below RRA_RTA_THRESH (3 deg), so the
pairs it saves are well-reconstructed examples and are NOT a random or representative
sample. The quantitative results reported in the paper come from the default mode of
this script, which evaluates every pair of each dataset's held-out split with no such
filtering. Do not read the per-pair metrics written by --two-view-test as dataset-level
accuracy.

Example:
    python3 eval_two_view.py \
        --ckpt checkpoints/cam3r.pth \
        --pano2d3ds-root /path/to/2D3DS/preprocessed_gt_rays/ \
        --batch-size 4
"""
import argparse
import math
import re
import time
from datetime import timedelta
from typing import Dict, List, Optional, Tuple
import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm  # Progress bar

from dust3r.model_with_unik3d_encoder_dust3r_adapter_relpose import (  # noqa: F401
    AsymmetricCroCo3DStereo,
    inf,
)
from dust3r.datasets.pano2D3DS import Pano2D3DS
from dust3r.datasets.fisheyeAria import FisheyeAria
from dust3r.datasets.megadepth_updated import MegaDepth
from dust3r.datasets.panoFishPin2D3DS import PanoFishPin2D3DS
from dust3r.datasets.fisheyePinholeAria import FisheyePinholeAria
from dust3r.datasets.co3dFisheyePinhole import Co3dFisheyePinhole  # noqa
from dust3r.datasets.pano360Loc import Pano360Loc
from dust3r.datasets.panoFishPin360Loc import PanoFishPin360Loc

from dust3r.datasets.pano2D3DS import (
    Pano2D3DS,
    save_gradient_colored_pcd,
    save_overlaid_gradient_colored_pcd,
)
from dust3r.utils.image import rgb
from dust3r.utils.geometry import geotrf, inv
from dust3r.confidence_geometry import compute_pairwise_mutual_nn_confidence

def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [move_to_device(v, device) for v in obj]
    else:
        return obj


def _format_device(d: torch.device) -> str:
    # Return 'cpu' or 'cuda:N' string
    if d.type == "cuda":
        return f"cuda:{d.index if d.index is not None else torch.cuda.current_device()}"
    return "cpu"


def build_model(device: torch.device, ckpt_path: str) -> torch.nn.Module:
    """
    Instantiate the model with the released CAM3R configuration and load the
    checkpoint at ckpt_path.
    """
    print("Defining DUSt3R model with adapter (relative pose head).")
    model = AsymmetricCroCo3DStereo(
        pos_embed="RoPE100",
        patch_embed_cls="ManyAR_PatchEmbed",
        img_size=(512, 384),
        head_type="dpt",
        output_mode="pts3d",
        depth_mode=("softplus", 0, inf),
        conf_mode=("exp", 1, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
        adapter_config=str(Path(__file__).resolve().parent / "dust3r" / "vitl.json"),
        adapter_pretrained=True,
        adapter_use_layernorm_pool=True,
    )

    model.to(device)

    print(f"Loading pretrained weights from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    pretrained_dict = ckpt["model"]
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(">> Checkpoint loaded (strict=False).")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    # Evaluation runs on a single device: DataParallel scatters the per-view
    # dictionaries across GPUs and ManyAR_PatchEmbed then indexes position
    # tensors with boolean masks that live on a different device. Increase
    # --batch-size to make better use of the selected device instead.
    if torch.cuda.device_count() > 1:
        print(f">> Detected {torch.cuda.device_count()} GPUs, but running evaluation on a single device: {device}.")
    else:
        print(">> Using single GPU or CPU.")

    model.eval()
    return model


def rotation_error_deg(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    """
    Geodesic rotation error in degrees, as in DUSt3R Sec. 4.2:

        theta = arccos( clamp((trace(R_pred^T R_gt) - 1) / 2, -1, 1) ) * 180/pi

    Supports batched tensors of shape (B, 3, 3).
    """
    # R_err = R_pred^T R_gt
    R_err = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    trace = torch.diagonal(R_err, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta) * (180.0 / math.pi)
    return theta


def translation_error_deg(
    t_pred: torch.Tensor,
    t_gt: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Angular translation error in degrees between t_pred and t_gt, as in DUSt3R Sec. 4.2:

        theta = arccos( clamp( (t_pred · t_gt) / (||t_pred|| ||t_gt||), -1, 1) ) * 180/pi

    If either vector has (near-)zero norm (< eps), returns 0.0 to avoid NaNs,
    since translation is undefined in that degenerate case.

    Supports batched tensors of shape (B, 3).
    """
    pred_norm = torch.norm(t_pred, dim=-1)
    gt_norm = torch.norm(t_gt, dim=-1)

    # Handle degenerate translations: where either norm is ~0, define error as 0.
    valid = (pred_norm > eps) & (gt_norm > eps)
    out = torch.zeros_like(pred_norm)
    if not torch.any(valid):
        return out

    dot = (t_pred * t_gt).sum(-1)
    cos_theta = dot / (pred_norm * gt_norm + 1e-12)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta) * (180.0 / math.pi)
    out[valid] = theta[valid]
    return out


def visualize_and_save_pointclouds(
    sample_view_device: List[dict],
    output0: dict,
    output1: dict,
    pair_dir: str,
) -> None:
    """
    Save predicted and GT pointclouds (individual + overlaid) under generic filenames.
    pair_dir: subdirectory for this pair (e.g. out_dir/1_Pano360Loc_pano_pano/)
    """
    os.makedirs(pair_dir, exist_ok=True)
    v0, v1 = sample_view_device[0], sample_view_device[1]
    B, C, H, W = v0["img"].shape

    colors0 = rgb(v0["img"].squeeze(0))
    colors1 = rgb(v1["img"].squeeze(0))

    # --- 1. Predicted 3D points (deidentified names) ---
    pts3d_pred0 = output0["pts3d"].reshape(B, H * W, C).squeeze(0).cpu().numpy()
    pts3d_pred1 = output1["pts3d"]
    R_pred = output1["R"]
    t_pred = output1["t"]
    pts3d_pred1_chw = pts3d_pred1.permute(0, 3, 1, 2).contiguous()
    pts3d_pred1to0_chw = (
        torch.einsum("bij,bjhw->bihw", R_pred, pts3d_pred1_chw)
        + t_pred[:, :, None, None]
    )
    pts3d_pred1to0 = (
        pts3d_pred1to0_chw.permute(0, 2, 3, 1).contiguous()
        .reshape(B, H * W, C)
        .squeeze(0)
        .cpu()
        .numpy()
    )
    pts3d_pred1_np = pts3d_pred1.reshape(B, H * W, C).squeeze(0).cpu().numpy()

    save_gradient_colored_pcd(pts3d_pred0, colors0, os.path.join(pair_dir, "pred_view0.ply"), axis=2, cmap="plasma")
    save_gradient_colored_pcd(pts3d_pred1_np, colors1, os.path.join(pair_dir, "pred_view1.ply"), axis=2, cmap="plasma")
    save_overlaid_gradient_colored_pcd(
        pts3d_pred0, pts3d_pred1to0, colors0, colors1,
        os.path.join(pair_dir, "pred_overlay.ply"), axis=2
    )

    # --- 2. Ground truth 3D points (world -> view0 cam via geotrf) ---
    pts3d_gt0_world = sample_view_device[0]["pts3d"]
    pts3d_gt1_world = sample_view_device[1]["pts3d"]
    T0 = sample_view_device[0]["camera_pose"]
    T1 = sample_view_device[1]["camera_pose"]
    in_cam0 = inv(T0)
    in_cam1 = inv(T1)
    pts3d_gt0_t = geotrf(in_cam0, pts3d_gt0_world)
    pts3d_gt1_t = geotrf(in_cam1, pts3d_gt1_world)
    pts3d_gt0 = pts3d_gt0_t.reshape(B, H * W, C).squeeze(0).cpu().numpy()
    pts3d_gt1 = pts3d_gt1_t.reshape(B, H * W, C).squeeze(0).cpu().numpy()

    T_rel = torch.matmul(inv(T0), T1)
    R_gt = T_rel[:, :3, :3]
    t_gt = T_rel[:, :3, 3]
    pts3d_gt1_chw = pts3d_gt1_t.permute(0, 3, 1, 2).contiguous()
    pts3d_gt1to0_chw = (
        torch.einsum("bij,bjhw->bihw", R_gt, pts3d_gt1_chw) + t_gt[:, :, None, None]
    )
    pts3d_gt1to0 = (
        pts3d_gt1to0_chw.permute(0, 2, 3, 1).reshape(B, H * W, C).squeeze(0).cpu().numpy()
    )

    save_gradient_colored_pcd(pts3d_gt0, colors0, os.path.join(pair_dir, "gt_view0.ply"), axis=2, cmap="plasma")
    save_gradient_colored_pcd(pts3d_gt1, colors1, os.path.join(pair_dir, "gt_view1.ply"), axis=2, cmap="plasma")
    save_overlaid_gradient_colored_pcd(
        pts3d_gt0, pts3d_gt1to0, colors0, colors1,
        os.path.join(pair_dir, "gt_overlay.ply"), axis=2
    )


def _get_inst(v):
    inst = v["instance"]
    return inst[0] if isinstance(inst, (list, tuple)) else inst


def save_pair_outputs(
    pair_dir: str,
    views0: dict,
    views1: dict,
    R_pred: torch.Tensor,
    t_pred: torch.Tensor,
    R_gt: torch.Tensor,
    t_gt: torch.Tensor,
    rot_err: float,
    tra_err: float,
    dataset_name: str,
) -> None:
    """
    Save poses.txt, metrics.txt and image_info.txt in pair_dir.
    Output filenames are generic; source image names and paths are recorded in
    image_info.txt only.
    """
    os.makedirs(pair_dir, exist_ok=True)

    # GT and predicted relative poses
    with open(os.path.join(pair_dir, "poses.txt"), "w") as f:
        f.write("R_pred:\n")
        f.write(np.array2string(R_pred.cpu().numpy(), max_line_width=120) + "\n")
        f.write("t_pred: " + np.array2string(t_pred.cpu().numpy()) + "\n")
        f.write("R_gt:\n")
        f.write(np.array2string(R_gt.cpu().numpy(), max_line_width=120) + "\n")
        f.write("t_gt: " + np.array2string(t_gt.cpu().numpy()) + "\n")

    # Rotation and translation error for this pair
    with open(os.path.join(pair_dir, "metrics.txt"), "w") as f:
        f.write(f"rotation error (deg): {rot_err:.6f}\n")
        f.write(f"translation error (deg): {tra_err:.6f}\n")

    # Source file names, scene, subscene and dataset name
    inst0 = _get_inst(views0)
    inst1 = _get_inst(views1)
    lbl0 = views0.get("label", "")
    lbl1 = views1.get("label", "")
    # Parse label for scene/subscene (format varies: "scene_subscene" or "scene subscene")
    scene0, subscene0 = _parse_label(lbl0)
    scene1, subscene1 = _parse_label(lbl1)
    with open(os.path.join(pair_dir, "image_info.txt"), "w") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"view0_filename: {inst0}\n")
        f.write(f"view0_scene: {scene0}\n")
        f.write(f"view0_subscene: {subscene0}\n")
        f.write(f"view0_label: {lbl0}\n")
        f.write(f"view1_filename: {inst1}\n")
        f.write(f"view1_scene: {scene1}\n")
        f.write(f"view1_subscene: {subscene1}\n")
        f.write(f"view1_label: {lbl1}\n")


def _parse_label(label: str) -> Tuple[str, str]:
    """Extract scene and subscene from label (e.g. 'area_5a hallway_15_5' or 'area_5a_hallway_15_5')."""
    if not label:
        return "", ""
    # Try space-separated first
    parts = str(label).strip().split()
    if len(parts) >= 2:
        return parts[0], parts[1]
    # Try underscore: first part as scene, rest as subscene
    uscore = str(label).split("_")
    if len(uscore) >= 2:
        return uscore[0], "_".join(uscore[1:])
    return label, ""


def save_depth_conf_and_mnn(
    output0: dict,
    output1: dict,
    views0: dict,
    views1: dict,
    pair_dir: str,
    cmap: str = "viridis",
    alpha: float = 0.5,
) -> None:
    """Save depth maps, confidence maps and mutual-NN matches as images."""
    os.makedirs(pair_dir, exist_ok=True)

    if "distance" in output0 and "distance" in output1:
        d0 = output0["distance"].squeeze().cpu().numpy()
        d1 = output1["distance"].squeeze().cpu().numpy()
    else:
        pts0 = output0["pts3d"].squeeze(0).cpu().numpy()
        pts1 = output1["pts3d"].squeeze(0).cpu().numpy()
        d0 = np.linalg.norm(pts0, axis=-1)
        d1 = np.linalg.norm(pts1, axis=-1)

    d0_norm = (d0 - d0.min()) / (d0.max() - d0.min() + 1e-10)
    d1_norm = (d1 - d1.min()) / (d1.max() - d1.min() + 1e-10)
    plt.imsave(os.path.join(pair_dir, "depth_view0.png"), d0_norm, cmap="viridis")
    plt.imsave(os.path.join(pair_dir, "depth_view1.png"), d1_norm, cmap="viridis")

    if "conf" in output0 and "conf" in output1:
        c0 = output0["conf"].squeeze().cpu().numpy()
        c1 = output1["conf"].squeeze().cpu().numpy()
        c0_n = (c0 - c0.min()) / (c0.max() - c0.min() + 1e-10)
        c1_n = (c1 - c1.min()) / (c1.max() - c1.min() + 1e-10)
        plt.imsave(os.path.join(pair_dir, "conf_view0.png"), c0_n, cmap="viridis")
        plt.imsave(os.path.join(pair_dir, "conf_view1.png"), c1_n, cmap="viridis")

    # Mutual nearest-neighbour 3D-3D confidence, saved as images and as mnn.npz
    pred1 = {"pts3d": output0["pts3d"]}
    pred2 = {"pts3d": output1["pts3d"], "R": output1["R"], "t": output1["t"]}
    mnn_d1, mnn_d2, mnn_m1, mnn_m2 = compute_pairwise_mutual_nn_confidence(pred1, pred2)
    m1 = mnn_d1.squeeze(0).cpu().numpy()
    m2 = mnn_d2.squeeze(0).cpu().numpy()
    mask1 = mnn_m1.squeeze(0).cpu().numpy()
    mask2 = mnn_m2.squeeze(0).cpu().numpy()

    vals1 = m1[mask1]
    vals2 = m2[mask2]
    if vals1.size > 0 or vals2.size > 0:
        all_vals = np.concatenate([vals1, vals2]) if vals1.size and vals2.size else (vals1 if vals1.size else vals2)
        d_min, d_max = float(all_vals.min()), float(all_vals.max())
        if np.isfinite(d_min) and np.isfinite(d_max) and d_max > d_min:
            def dist_to_conf_global(d, mask):
                conf = np.full_like(d, np.nan, dtype=np.float64)
                conf[mask] = 1.0 - (d[mask] - d_min) / (d_max - d_min + 1e-10)
                conf[mask] = np.clip(conf[mask], 0.0, 1.0)
                return conf

            c1 = dist_to_conf_global(m1, mask1)
            c2 = dist_to_conf_global(m2, mask2)

            img0 = np.asarray(rgb(views0["img"].squeeze(0)))
            img1 = np.asarray(rgb(views1["img"].squeeze(0)))
            if img0.max() > 1.0:
                img0 = img0.astype(np.float64) / 255.0
            if img1.max() > 1.0:
                img1 = img1.astype(np.float64) / 255.0

            try:
                from scipy.ndimage import zoom
                if img0.shape[:2] != c1.shape:
                    scale = (img0.shape[0] / c1.shape[0], img0.shape[1] / c1.shape[1])
                    c1 = zoom(c1, scale, order=1)
                    mask1 = zoom(mask1.astype(np.float32), scale, order=0) > 0.5
                if img1.shape[:2] != c2.shape:
                    scale = (img1.shape[0] / c2.shape[0], img1.shape[1] / c2.shape[1])
                    c2 = zoom(c2, scale, order=1)
                    mask2 = zoom(mask2.astype(np.float32), scale, order=0) > 0.5
            except ImportError:
                pass  # skip resize if scipy unavailable

            base_cmap = plt.get_cmap(cmap).copy()
            base_cmap.set_bad(color="black")

            overlay0 = base_cmap(c1)[:, :, :3]
            overlay0[np.isnan(c1)] = 0.0
            blend0 = (alpha * overlay0 + (1 - alpha) * img0).clip(0, 1)
            overlay1 = base_cmap(c2)[:, :, :3]
            overlay1[np.isnan(c2)] = 0.0
            blend1 = (alpha * overlay1 + (1 - alpha) * img1).clip(0, 1)

            fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
            axes[0, 0].imshow(img0)
            axes[0, 0].set_title("View 0")
            axes[0, 0].axis("off")
            im0 = axes[0, 1].imshow(c1, cmap=base_cmap, vmin=0.0, vmax=1.0)
            axes[0, 1].set_title("View 0: MNN quality (near=bright)")
            axes[0, 1].axis("off")
            axes[0, 2].imshow(blend0)
            axes[0, 2].set_title("View 0: MNN overlay")
            axes[0, 2].axis("off")
            axes[1, 0].imshow(img1)
            axes[1, 0].set_title("View 1")
            axes[1, 0].axis("off")
            axes[1, 1].imshow(c2, cmap=base_cmap, vmin=0.0, vmax=1.0)
            axes[1, 1].set_title("View 1: MNN quality (near=bright)")
            axes[1, 1].axis("off")
            axes[1, 2].imshow(blend1)
            axes[1, 2].set_title("View 1: MNN overlay")
            axes[1, 2].axis("off")
            fig.colorbar(im0, ax=axes[:, 1], fraction=0.046, pad=0.04, label="Pair-wise MNN confidence")
            plt.suptitle("Pairwise MNN 3D-3D confidence (symmetric scale)")
            plt.savefig(os.path.join(pair_dir, "mnn_combined.png"), dpi=150, bbox_inches="tight")
            plt.close()

            plt.imsave(os.path.join(pair_dir, "mnn_view0.png"), blend0)
            plt.imsave(os.path.join(pair_dir, "mnn_view1.png"), blend1)

    np.savez(
        os.path.join(pair_dir, "mnn.npz"),
        mnn_dist1=mnn_d1.cpu().numpy(),
        mnn_dist2=mnn_d2.cpu().numpy(),
        mnn_mask1=mnn_m1.cpu().numpy(),
        mnn_mask2=mnn_m2.cpu().numpy(),
    )


def _iter_matching_pairs(
    dataset,
    pair_filter: callable,
    max_tries: int = 10000,
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """Yield pair indices that satisfy pair_filter. Uses shuffled order to avoid repeats."""
    n = len(dataset)
    if n == 0:
        return []
    order = np.arange(n)
    if rng is not None:
        rng.shuffle(order)
    else:
        np.random.shuffle(order)
    count = 0
    for idx in order:
        if count >= max_tries:
            break
        count += 1
        try:
            views = dataset[idx]
            if pair_filter(views):
                yield idx
        except Exception:
            continue


@torch.no_grad()
def run_two_view_test(
    model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """
    Save qualitative outputs for one pair per camera-model combination: pointclouds,
    poses, per-pair metrics, depth, confidence and mutual-NN matches.

    Pairs are selected for reconstruction quality (see the module docstring); the
    per-pair metrics written here are not dataset-level accuracy.
    """
    out_dir = getattr(args, "out_dir", "two_view_test_outputs")
    os.makedirs(out_dir, exist_ok=True)

    def _get_digits(s: str) -> List[int]:
        return [int(m) for m in re.findall(r"\d+", s)]

    def _pano360_pair(views):
        v0, v1 = views[0], views[1]
        inst0 = v0["instance"][0] if isinstance(v0["instance"], (list, tuple)) else v0["instance"]
        inst1 = v1["instance"][0] if isinstance(v1["instance"], (list, tuple)) else v1["instance"]
        d0 = _get_digits(str(inst0))
        d1 = _get_digits(str(inst1))
        return all(301 <= x <= 333 for x in d0 + d1) if (d0 and d1) else False

    def _inst(v, i=0):
        inst = v["instance"]
        return (inst[i] if isinstance(inst, (list, tuple)) else inst).lower()

    def _panofish_pair(views):
        v0, v1 = views[0], views[1]
        lbl = (v0.get("label", "") or v1.get("label", "")).lower()
        if "area_5a" not in lbl or "hallway_15_5" not in lbl:
            return False
        n0, n1 = _inst(v0), _inst(v1)
        fish = "fish" in n0 or "fish" in n1
        pano = ("fish" not in n0 and "pin" not in n0 and "persp" not in n0) or (
            "fish" not in n1 and "pin" not in n1 and "persp" not in n1
        )
        return fish and pano

    def _panopin_pair(views):
        v0, v1 = views[0], views[1]
        lbl = (v0.get("label", "") or v1.get("label", "")).lower()
        if "area_5a" not in lbl or "hallway_15_5" not in lbl:
            return False
        n0, n1 = _inst(v0), _inst(v1)
        pin = "pin" in n0 or "persp" in n0 or "pin" in n1 or "persp" in n1
        pano = ("fish" not in n0 and "pin" not in n0 and "persp" not in n0) or (
            "fish" not in n1 and "pin" not in n1 and "persp" not in n1
        )
        return pin and pano

    def _co3d_pair(views):
        v0, v1 = views[0], views[1]
        # Fixed CO3D sequence for the figure: category "apple", sequence 41_1950_5738.
        lbl = v0.get("label", "") or v1.get("label", "")
        if "apple" not in lbl or "41_1950_5738" not in lbl:
            return False
        i0 = v0["instance"] if isinstance(v0["instance"], str) else v0["instance"][0]
        i1 = v1["instance"] if isinstance(v1["instance"], str) else v1["instance"][0]
        m0 = re.search(r"frame(\d+)", str(i0))
        m1 = re.search(r"frame(\d+)", str(i1))
        f0 = int(m0.group(1)) if m0 else 0
        f1 = int(m1.group(1)) if m1 else 0
        return (f0 % 2 == 1 and f1 % 2 == 0) or (f0 % 2 == 0 and f1 % 2 == 1)

    configs = [
        ("1_Pano360Loc_pano_pano", "Pano360Loc", Pano360Loc, {"split": "val", "ROOT": args.pano360loc_root}, _pano360_pair),
        ("2_PanoFishPin2D3DS_pano_fish", "PanoFishPin2D3DS", PanoFishPin2D3DS, {"split": "val", "ROOT": args.panofishpin2d3ds_root}, _panofish_pair),
        ("3_FisheyeAria_fish_fish", "FisheyeAria", FisheyeAria, {"split": "val", "ROOT": args.fisheyearia_root}, lambda _: True),
        ("4_PanoFishPin2D3DS_pano_pin", "PanoFishPin2D3DS", PanoFishPin2D3DS, {"split": "val", "ROOT": args.panofishpin2d3ds_root}, _panopin_pair),
        ("5_Co3dFisheyePinhole_fish_pin", "Co3dFisheyePinhole", Co3dFisheyePinhole, {"split": "test", "ROOT": args.co3d_root}, _co3d_pair),
        ("6_MegaDepth_pin_pin", "MegaDepth", MegaDepth, {"split": "val", "ROOT": args.megadepth_root}, lambda v: "0015" in (v[0].get("label", "") or v[1].get("label", ""))),
    ]

    # Pano360Loc: prefer the 'atrium' scene for the figure pair; fall back to the
    # default val scenes if it is not present in this copy of the data.
    if not args.pano360loc_root:
        raise SystemExit("--two-view-test needs --pano360loc-root (and the other dataset roots).")
    pano360 = Pano360Loc(split="val", ROOT=args.pano360loc_root, resolution=512, aug_crop=16)
    try:
        pano360.select_scene(("atrium",))
    except Exception:
        pass  # scene not available: keep the default val scenes

    # Qualitative-figure selection only: keep searching within each camera-model
    # combination until a pair reconstructs well. This biases WHICH pairs appear in the
    # figures and is deliberately not how the reported numbers are computed (see the
    # module docstring); the default mode of this script evaluates every pair unfiltered.
    RRA_RTA_THRESH = 3.0
    all_rot_err: List[float] = []
    all_tra_err: List[float] = []
    rng = np.random.default_rng(42)

    for pair_name, dataset_name, ds_cls, ds_kw, pair_filter in configs:
        print(f"\n=== Processing {pair_name} ===")
        try:
            if "Pano360Loc" in pair_name:
                dataset = pano360
            else:
                dataset = ds_cls(
                    resolution=512,
                    aug_crop=16,
                    **ds_kw,
                )
        except Exception as e:
            print(f"  Skipping {pair_name}: {e}")
            continue

        if len(dataset) == 0:
            print(f"  Skipping {pair_name}: empty dataset")
            continue

        found_good_pair = False
        for pair_idx in _iter_matching_pairs(dataset, pair_filter, max_tries=5000, rng=rng):
            views = dataset[pair_idx]  # list of 2 view dicts
            batch = default_collate([views])  # [batch_v0, batch_v1]
            batch = move_to_device(batch, device)
            views0, views1 = batch[0], batch[1]

            output0, output1 = model(views0, views1)
            R_pred = output1["R"]
            t_pred = output1["t"]
            T0 = views0["camera_pose"]
            T1 = views1["camera_pose"]
            T_rel = torch.matmul(torch.inverse(T0), T1)
            R_gt = T_rel[:, :3, :3]
            t_gt = T_rel[:, :3, 3]

            rot_err = rotation_error_deg(R_pred, R_gt).item()
            tra_err = translation_error_deg(t_pred, t_gt).item()

            if rot_err >= RRA_RTA_THRESH or tra_err >= RRA_RTA_THRESH:
                print(f"  Pair idx {pair_idx}: RRA={rot_err:.3f}, RTA={tra_err:.3f} (need both < {RRA_RTA_THRESH}), skipping")
                continue

            print(f"  Pair idx {pair_idx}: RRA={rot_err:.3f}, RTA={tra_err:.3f} (both < {RRA_RTA_THRESH}), storing")
            all_rot_err.append(rot_err)
            all_tra_err.append(tra_err)

            pair_dir = os.path.join(out_dir, pair_name)

            # 1. Predicted and 2. GT pointclouds (individual + overlaid)
            visualize_and_save_pointclouds([views0, views1], output0, output1, pair_dir)
            # 3. poses.txt, 4. metrics.txt, 5. image_info.txt
            save_pair_outputs(
                pair_dir, views0, views1,
                R_pred, t_pred, R_gt, t_gt, rot_err, tra_err,
                dataset_name,
            )
            # Depth, conf, mnn (deidentified; mnn saved as images)
            save_depth_conf_and_mnn(output0, output1, views0, views1, pair_dir)
            found_good_pair = True
            break

        if not found_good_pair:
            print(f"  No pair with RRA < {RRA_RTA_THRESH} AND RTA < {RRA_RTA_THRESH} deg found for {pair_name}")

    if all_rot_err:
        print("\n=== Selected qualitative pairs: median error (NOT a dataset metric) ===")
        rra15 = np.median(all_rot_err)
        rta15 = np.median(all_tra_err)
        print(f"Median RRA: {rra15:.3f} deg")
        print(f"Median RTA: {rta15:.3f} deg")
        print(f"Outputs saved to: {out_dir}")


@torch.no_grad()
def evaluate_dataset(
    name: str,
    dataset,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int = 1,
    num_workers: int = 4,
    max_pairs: int = None,
) -> Dict[str, float]:
    """
    Iterate over all (or up to max_pairs) validation pairs of a dataset, compute
    relative pose errors (RRA/RTA) for each pair, and return summary statistics.
    Evaluate dataset with progress bar and timing.
    """
    print(f"\n=== Evaluating dataset: {name} ===")
    print(dataset)

    print(f"Number of pairs: {len(dataset)}")

    start_time = time.time()

    sampler = RandomSampler(dataset)

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    all_rot_err: List[float] = []
    all_tra_err: List[float] = []

    n_processed = 0

    pbar = tqdm(loader, desc=f"{name}", unit="batch")

    for batch in pbar:
        # batch is a list-like of length 2: [view0_batch, view1_batch]
        # IMPORTANT:
        # - If the model is wrapped in DataParallel, let it handle scattering the
        #   CPU batch to the individual GPUs. Pre-moving tensors to a specific
        #   CUDA device can lead to cross-device boolean indexing issues inside
        #   the model (e.g. in ManyAR_PatchEmbed).
        # - For single-GPU / CPU, move everything to `device` manually.
        if isinstance(model, torch.nn.DataParallel):
            views0, views1 = batch[0], batch[1]
        else:
            batch = move_to_device(batch, device)
            views0, views1 = batch[0], batch[1]

        # Forward pass: model predicts pose from view1 -> view0
        output0, output1 = model(views0, views1)

        # When using DataParallel, output tensors are gathered back to the main device.
        # Ensure we access them correctly.
        R_pred = output1["R"]  # (B, 3, 3), view1 -> view0
        t_pred = output1["t"]  # (B, 3)

         # Ground-truth absolute poses (cam->world) for the two views
        T0 = views0["camera_pose"]  # (B, 4, 4)
        T1 = views1["camera_pose"]  # (B, 4, 4)

        # Relative GT: view1 -> view0, T_rel = T0^{-1} @ T1
        T0_inv = torch.inverse(T0)
        T_rel = torch.matmul(T0_inv, T1)
        R_gt = T_rel[:, :3, :3]
        t_gt = T_rel[:, :3, 3]

        # Errors (in degrees)
        rot_err_batch = rotation_error_deg(R_pred, R_gt)
        tra_err_batch = translation_error_deg(t_pred, t_gt)

        all_rot_err.extend(rot_err_batch.detach().cpu().tolist())
        all_tra_err.extend(tra_err_batch.detach().cpu().tolist())

        # Update progress bar stats
        n_processed += R_pred.shape[0]
        if max_pairs is not None and n_processed >= max_pairs:
            break

    if not all_rot_err:
        raise RuntimeError(f"No pose pairs processed for dataset {name}.")

    # Calculate metrics
    rot_arr = np.array(all_rot_err, dtype=np.float64)
    tra_arr = np.array(all_tra_err, dtype=np.float64)

    # Medians (in degrees)
    median_rra = float(np.median(rot_arr))
    median_rta = float(np.median(tra_arr))

    # Percentages of pairs with error < 15 and < 30 degrees
    thr15 = 15.0
    thr30 = 30.0
    rra_at_15 = float((rot_arr < thr15).mean() * 100.0)
    rta_at_15 = float((tra_arr < thr15).mean() * 100.0)
    rra_at_30 = float((rot_arr < thr30).mean() * 100.0)
    rta_at_30 = float((tra_arr < thr30).mean() * 100.0)

    # mAA@30: mean angular accuracy over thresholds 1..30 deg,
    # joint condition on rotation & translation.
    accs = []
    for t in range(1, 31):
        mask = (rot_arr < t) & (tra_arr < t)
        accs.append(mask.mean())
    maa_30 = float(np.mean(accs) * 100.0)

    elapsed_time = time.time() - start_time
    formatted_time = str(timedelta(seconds=int(elapsed_time)))

    print(f"[{name}] Finished in {formatted_time}")
    print(f"[{name}] Median RRA (deg): {median_rra:.3f}")
    print(f"[{name}] Median RTA (deg): {median_rta:.3f}")
    print(f"[{name}] RRA@15: {rra_at_15:.2f}%")
    print(f"[{name}] RTA@15: {rta_at_15:.2f}%")
    print(f"[{name}] RRA@30: {rra_at_30:.2f}%")
    print(f"[{name}] RTA@30: {rta_at_30:.2f}%")
    print(f"[{name}] mAA@30: {maa_30:.2f}%")

    return {
        "median_rra": median_rra,
        "median_rta": median_rta,
        "rra_at_15": rra_at_15,
        "rta_at_15": rta_at_15,
        "rra_at_30": rra_at_30,
        "rta_at_30": rta_at_30,
        "maa_30": maa_30,
        "num_pairs": int(len(all_rot_err)),
        "time_str": formatted_time,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CAM3R relative-pose metrics (RRA/RTA/mAA) on the held-out splits of "
            "Pano2D3DS, Pano360Loc, FisheyeAria, MegaDepth, Co3dFisheyePinhole, "
            "PanoFishPin2D3DS, PanoFishPin360Loc and FisheyePinholeAria."
        )
    )

    # Model / device
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/cam3r.pth",
        help="Path to the CAM3R checkpoint (relative-pose model).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device: 'cuda' or 'cpu'.",
    )

    # Dataset roots. The defaults are placeholders: point each one at your own
    # preprocessed copy of the corresponding dataset.
    parser.add_argument(
        "--pano2d3ds-root",
        type=str,
        default=None,
        help="Root directory for Pano2D3DS preprocessed validation data.",
    )
    parser.add_argument(
        "--pano360loc-root",
        type=str,
        default=None,
        help="Root directory for 360loc preprocessed validation data.",
    )
    parser.add_argument(
        "--fisheyearia-root",
        type=str,
        default=None,
        help="Root directory for FisheyeAria preprocessed validation data.",
    )
    parser.add_argument(
        "--megadepth-root",
        type=str,
        default=None,
        help="Root directory for MegaDepth preprocessed validation data.",
    )
    parser.add_argument(
        "--co3d-root",
        type=str,
        default=None,
        help="Root directory for co3d preprocessed validation data.",
    )
    parser.add_argument(
        "--panofishpin2d3ds-root",
        type=str,
        default=None,
        help="Root directory for PanoFishPin2D3DS preprocessed data.",
    )
    parser.add_argument(
        "--panofishpin360loc-root",
        type=str,
        default=None,
        help="Root directory for PanoFishPin360loc preprocessed data.",
    )
    parser.add_argument(
        "--fisheyepinholearia-root",
        type=str,
        default=None,
        help="Root directory for FisheyePinholeAria preprocessed data.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional limit on the number of pairs evaluated per dataset (for quick tests).",
    )
    parser.add_argument(
        "--two-view-test",
        action="store_true",
        help=(
            "Save qualitative outputs (pointclouds, poses, depth, confidence, MNN) for one "
            "well-reconstructed pair per camera-model combination. Not a representative "
            "sample: use the default mode for quantitative results."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="two_view_test_outputs",
        help="Output directory for 2-view test visualizations (pointclouds, poses, depth, conf, mnn).",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    # Build model once and reuse across datasets
    model = build_model(device, args.ckpt)

    if args.two_view_test:
        run_two_view_test(model, device, args)
        return

    results: List[Tuple[str, Dict[str, float]]] = []

    # Each entry: (label, dataset class, split, the --*-root value).
    # A dataset whose root was not supplied is skipped, so the script can be run
    # against whichever subset of the benchmark you actually have on disk.
    dataset_specs = [
        ("Pano2D3DS-val",          Pano2D3DS,          "val",  args.pano2d3ds_root),
        ("Pano360Loc-val",         Pano360Loc,         "val",  args.pano360loc_root),
        ("FisheyeAria-val",        FisheyeAria,        "val",  args.fisheyearia_root),
        ("MegaDepth-val",          MegaDepth,          "val",  args.megadepth_root),
        ("Co3dFisheyePinhole-val", Co3dFisheyePinhole, "test", args.co3d_root),
        ("PanoFishPin2D3DS-val",   PanoFishPin2D3DS,   "val",  args.panofishpin2d3ds_root),
        ("PanoFishPin360Loc-val",  PanoFishPin360Loc,  "val",  args.panofishpin360loc_root),
        ("FisheyePinholeAria-val", FisheyePinholeAria, "val",  args.fisheyepinholearia_root),
    ]

    for label, cls, split, root in dataset_specs:
        if not root:
            print(f"[skip] {label}: no root supplied")
            continue
        dataset = cls(split=split, ROOT=root, resolution=512, aug_crop=16)
        res = evaluate_dataset(
            label,
            dataset,
            model=model,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_pairs=args.max_pairs,
        )
        results.append((label, res))

    if not results:
        raise SystemExit(
            "No dataset was evaluated. Supply at least one of --pano2d3ds-root, "
            "--pano360loc-root, --fisheyearia-root, --megadepth-root, --co3d-root, "
            "--panofishpin2d3ds-root, --panofishpin360loc-root, --fisheyepinholearia-root."
        )

    total_elapsed = time.time() - total_start
    total_time_str = str(timedelta(seconds=int(total_elapsed)))

    # Final Summary
    print("\n" + "="*70)
    print("\n=== Summary over all datasets ===")
    print(f"Total Run Time: {total_time_str}")
    print("="*70)
    print(
        f"{'Dataset':<25} | {'MedRRA':<8} | {'MedRTA':<8} | "
        f"{'RRA@15':<8} | {'RTA@15':<8} | "
        f"{'RRA@30':<8} | {'RTA@30':<8} | {'mAA@30':<8} | {'Time':<10}"
    )
    print("-" * 110)
    for name, stats in results:
        print(
            f"{name:<25} | "
            f"{stats['median_rra']:.3f} | "
            f"{stats['median_rta']:.3f} | "
            f"{stats['rra_at_15']:.2f}% | "
            f"{stats['rta_at_15']:.2f}% | "
            f"{stats['rra_at_30']:.2f}% | "
            f"{stats['rta_at_30']:.2f}% | "
            f"{stats['maa_30']:.2f}% | "
            f"{stats['time_str']:<10}"
        )
    print("="*70)


if __name__ == "__main__":
    main()
