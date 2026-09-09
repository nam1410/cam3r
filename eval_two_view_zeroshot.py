#!/usr/bin/env python3
# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Derived from the DUSt3R evaluation code.
# --------------------------------------------------------
r"""
Zero-shot two-view reconstruction from a folder of two images.

- Expects exactly 2 images in the input folder
- Resizes to (512, 512) if square, else (512, 384)
- Runs the CAM3R network and aligns view 1 into view 0's coordinate frame
- Writes per-view point clouds, a centred overlaid point cloud, and a
  mutual-nearest-neighbour 3D-3D confidence visualisation
- Uses no ground truth of any kind

Example:
    python3 eval_two_view_zeroshot.py \
        --input /path/to/folder_with_two_images \
        --outdir zeroshot_two_view_out \
        --ckpt checkpoints/cam3r.pth
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import PIL.Image
from PIL.ImageOps import exif_transpose

from dust3r.inference import inference
from dust3r.utils.image import ImgNorm, rgb
from dust3r.datasets.pano2D3DS import save_gradient_colored_pcd, save_overlaid_gradient_colored_pcd
from dust3r.confidence_geometry import compute_pairwise_mutual_nn_confidence
from dust3r.model_with_unik3d_encoder_dust3r_adapter_relpose import AsymmetricCroCo3DStereo, inf

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def build_model(device: torch.device, ckpt_path: str) -> torch.nn.Module:
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
    model.load_state_dict(ckpt["model"], strict=False)
    print(">> Checkpoint loaded (strict=False).")
    model.eval()
    return model


def load_and_resize_image(path: str, target_square, target_rect):
    """
    Load image and resize: (512, 512) if square; else (512, 384).
    Returns PIL Image and final (H, W) for model input format.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Could not load image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = exif_transpose(PIL.Image.fromarray(img)).convert("RGB")
    W1, H1 = img_pil.size

    is_square = (W1 == H1)
    if is_square:
        tw, th = target_square  # (512, 512)
    else:
        tw, th = target_rect   # (512, 384)

    img_resized = cv2.resize(np.array(img_pil), (tw, th), interpolation=cv2.INTER_AREA)
    img_pil = PIL.Image.fromarray(img_resized)
    return img_pil, (th, tw)


def load_images_from_folder(folder: str, target_square=(512, 512), target_rect=(512, 384), max_images=2):
    """Load up to max_images from folder. Resize: (512,512) if square else (512,384)."""
    supported = (".jpg", ".jpeg", ".png")
    content = sorted(os.listdir(folder))
    imgs = []
    for path in content:
        if not path.lower().endswith(supported):
            continue
        if len(imgs) >= max_images:
            break
        full = os.path.join(folder, path)
        try:
            img_pil, (H, W) = load_and_resize_image(full, target_square, target_rect)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue
        img_norm = ImgNorm(img_pil)[None]
        imgs.append(dict(
            img=img_norm,
            true_shape=np.int32([[H, W]]),
            idx=len(imgs),
            instance=Path(path).stem,
        ))
    return imgs


def create_dummy_gt_rays(B, H: int, W: int) -> torch.Tensor:
    return torch.zeros(B, H, W, 3, dtype=torch.float32)


def visualize_geometry_confidence_overlay_pairwise_mnn(
    sample_view, output0, output1, out_path: str, cmap="viridis", alpha=0.5
):
    """
    Pairwise MNN 3D-3D confidence visualization.
    """
    if "pts3d" not in output0 or "pts3d" not in output1:
        print(f"[MNN] Missing pts3d; skipping visualization.")
        return
    R = output1.get("R")
    t = output1.get("t")
    if R is None or t is None:
        T = output1.get("T") or output1.get("rel_pose")
        if T is None:
            print(f"[MNN] Missing R/t or T; skipping visualization.")
            return
        R = T[:, :3, :3]
        t = T[:, :3, 3]

    pred1 = {"pts3d": output0["pts3d"]}
    pred2 = {"pts3d": output1["pts3d"], "R": R, "t": t}

    mnn_dist1, mnn_dist2, mnn_mask1, mnn_mask2 = compute_pairwise_mutual_nn_confidence(pred1, pred2)
    m1 = mnn_dist1.squeeze(0).cpu().numpy()
    m2 = mnn_dist2.squeeze(0).cpu().numpy()
    mask1 = mnn_mask1.squeeze(0).cpu().numpy()
    mask2 = mnn_mask2.squeeze(0).cpu().numpy()

    vals1 = m1[mask1]
    vals2 = m2[mask2]
    if vals1.size == 0 and vals2.size == 0:
        print("[MNN] No mutual matches found.")
        return
    all_vals = np.concatenate([vals1, vals2]) if vals1.size and vals2.size else (vals1 if vals1.size else vals2)
    d_min, d_max = float(all_vals.min()), float(all_vals.max())
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        print("[MNN] Degenerate distance range.")
        return

    def dist_to_conf_global(d, mask):
        conf = np.full_like(d, np.nan, dtype=np.float64)
        conf[mask] = 1.0 - (d[mask] - d_min) / (d_max - d_min + 1e-10)
        conf[mask] = np.clip(conf[mask], 0.0, 1.0)
        return conf

    c1 = dist_to_conf_global(m1, mask1)
    c2 = dist_to_conf_global(m2, mask2)

    img0 = np.asarray(rgb(sample_view[0]["img"].squeeze(0)))
    img1 = np.asarray(rgb(sample_view[1]["img"].squeeze(0)))
    if img0.max() > 1.0:
        img0 = img0.astype(np.float64) / 255.0
    if img1.max() > 1.0:
        img1 = img1.astype(np.float64) / 255.0

    if img0.shape[:2] != c1.shape:
        from scipy.ndimage import zoom
        scale = (img0.shape[0] / c1.shape[0], img0.shape[1] / c1.shape[1])
        c1 = zoom(c1, scale, order=1)
        mask1 = zoom(mask1.astype(np.float32), scale, order=0) > 0.5
    if img1.shape[:2] != c2.shape:
        from scipy.ndimage import zoom
        scale = (img1.shape[0] / c2.shape[0], img1.shape[1] / c2.shape[1])
        c2 = zoom(c2, scale, order=1)
        mask2 = zoom(mask2.astype(np.float32), scale, order=0) > 0.5

    base_cmap = plt.get_cmap(cmap).copy()
    base_cmap.set_bad(color="black")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].imshow(img0)
    axes[0, 0].set_title(f"View 0: {sample_view[0]['instance'][0]}")
    axes[0, 0].axis("off")
    im0 = axes[0, 1].imshow(c1, cmap=base_cmap, vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("View 0: MNN quality (near=bright)")
    axes[0, 1].axis("off")
    overlay0 = base_cmap(c1)[:, :, :3]
    overlay0[np.isnan(c1)] = 0.0
    blend0 = (alpha * overlay0 + (1 - alpha) * img0).clip(0, 1)
    axes[0, 2].imshow(blend0)
    axes[0, 2].set_title("View 0: MNN overlay")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(img1)
    axes[1, 0].set_title(f"View 1: {sample_view[1]['instance'][0]}")
    axes[1, 0].axis("off")
    im1 = axes[1, 1].imshow(c2, cmap=base_cmap, vmin=0.0, vmax=1.0)
    axes[1, 1].set_title("View 1: MNN quality (near=bright)")
    axes[1, 1].axis("off")
    overlay1 = base_cmap(c2)[:, :, :3]
    overlay1[np.isnan(c2)] = 0.0
    blend1 = (alpha * overlay1 + (1 - alpha) * img1).clip(0, 1)
    axes[1, 2].imshow(blend1)
    axes[1, 2].set_title("View 1: MNN overlay")
    axes[1, 2].axis("off")

    fig.colorbar(im0, ax=axes[:, 1], fraction=0.046, pad=0.04)
    plt.suptitle("Pairwise MNN 3D-3D confidence (symmetric scale)")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved MNN visualization: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ZERO-SHOT two-view pipeline from folder (2 images)")
    parser.add_argument("--input", "-i", type=str, required=True, help="Folder with exactly 2 images")
    parser.add_argument("--outdir", "-o", type=str, default="zeroshot_two_view_out",
                        help="Directory for the point clouds and the MNN confidence figure.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to the CAM3R checkpoint.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device: 'cuda' or 'cpu'.")
    args = parser.parse_args()

    device = torch.device(args.device)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    imgs = load_images_from_folder(args.input, max_images=2)
    if len(imgs) != 2:
        print(f"Expected 2 images in {args.input}, found {len(imgs)}.")
        sys.exit(1)

    for v in imgs:
        B, _, H, W = v["img"].shape
        v["gt_rays"] = create_dummy_gt_rays(B,H, W)

    pairs = [(imgs[0], imgs[1])]
    result = inference(pairs, build_model(device, args.ckpt), device=device, batch_size=1, verbose=True, demo=True)

    view1 = result["view1"]
    view2 = result["view2"]
    pred1 = result["pred1"]
    pred2 = result["pred2"]

    sample_view = [view1, view2]
    output0 = pred1
    output1 = pred2

    B = sample_view[0]["img"].shape[0]
    pts3d_0 = output0["pts3d"].reshape(B, -1, 3).squeeze(0).cpu().numpy()
    pts3d_1 = output1["pts3d"]

    if "R" in output1 and "t" in output1:
        R_10 = output1["R"]
        t_10 = output1["t"]
    elif "T" in output1 or "rel_pose" in output1:
        T = output1.get("T") or output1.get("rel_pose")
        R_10 = T[:, :3, :3]
        t_10 = T[:, :3, 3]
    else:
        print("Missing R,t or T in output1; cannot align view1 to view0.")
        sys.exit(1)
    pts3d_1_chw = pts3d_1.permute(0, 3, 1, 2).contiguous()
    pts3d_1to0_chw = torch.einsum("bij,bjhw->bihw", R_10, pts3d_1_chw) + t_10[:, :, None, None]
    pts3d_1to0 = pts3d_1to0_chw.permute(0, 2, 3, 1).contiguous().reshape(B, -1, 3).squeeze(0).cpu().numpy()

    colors0 = rgb(sample_view[0]["img"].squeeze(0))
    colors1 = rgb(sample_view[1]["img"].squeeze(0))
    inst0 = sample_view[0]["instance"][0]
    inst1 = sample_view[1]["instance"][0]

    merged = np.vstack([pts3d_0, pts3d_1to0])
    valid = np.isfinite(merged).all(axis=1) & (np.linalg.norm(merged, axis=1) > 1e-8)
    centroid = merged[valid].mean(axis=0) if valid.any() else np.zeros(3)
    pts3d_0_centered = pts3d_0 - centroid
    pts3d_1to0_centered = pts3d_1to0 - centroid

    save_gradient_colored_pcd(pts3d_0_centered, colors0, str(outdir / f"view0_{inst0}.ply"), axis=2, cmap="plasma")
    save_gradient_colored_pcd(pts3d_1to0_centered, colors1, str(outdir / f"view1_{inst1}.ply"), axis=2, cmap="plasma")
    save_overlaid_gradient_colored_pcd(
        pts3d_0_centered, pts3d_1to0_centered,
        colors0, colors1,
        str(outdir / f"overlay_{inst0}_{inst1}.ply"),
        axis=2,
    )
    np.save(outdir / "overlay_centroid.npy", centroid)

    visualize_geometry_confidence_overlay_pairwise_mnn(
        sample_view, output0, output1,
        str(outdir / f"pairwise_mnn_conf_{inst0}_{inst1}.png"),
    )

    print(f"Done. Outputs in {outdir.resolve()}")


if __name__ == "__main__":
    main()
