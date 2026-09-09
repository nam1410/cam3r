# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed 360Loc dataset
# Adapted from Pano2D3DS loader
# --------------------------------------------------------
import os
import sys
import os.path as osp
import numpy as np
import math
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from datetime import datetime
import open3d as o3d
from matplotlib import cm
import matplotlib.pyplot as plt
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import PIL
from einops import rearrange

def apply_gradient_color(points, axis=2, cmap='viridis'):
    values = points[:, axis]
    norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
    try:
        colormap = plt.colormaps[cmap]  # newer versions (>=3.8)
    except AttributeError:
        colormap = cm.get_cmap(cmap)    # fallback for older versions 
    colors = colormap(norm)[:, :3]
    return colors

def save_gradient_colored_pcd(points, colors=None, filename='output.ply', axis=2, cmap='viridis'):
    # reshape from (H, W, 3) to (N, 3), else will give run time error while converting to open3d format
    if len(points.shape) != 2:
        points = points.reshape(-1, 3)
    # remove NaNs, infs and keep only real invalid data
    valid_mask = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-8)
    points = points[valid_mask]
    if colors is not None:
        if len(colors.shape) != 2:
            colors = colors.reshape(-1,3)
        colors = colors[valid_mask]
    else:
        colors = apply_gradient_color(points, axis, cmap)
    pcd = o3d.geometry.PointCloud()
    # http://open3d.org/docs/latest/python_api/open3d.utility.Vector3dVector.html
    pcd.points = o3d.utility.Vector3dVector(points) #Convert float64 numpy array of shape (n, 3) to Open3D format.
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"saved gradient colored point cloud to {filename}")


def save_overlaid_gradient_colored_pcd(points1=None, points2=None, colors1=None, colors2=None,
                               filename='output.ply', axis=2, cmap1='viridis', cmap2='plasma'):
    def process(points, colors,cmap):
        if len(points.shape) != 2:
            points = points.reshape(-1, 3)
        # remove NaNs, infs and keep only real invalid data
        valid_mask = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) > 1e-8)
        points = points[valid_mask]
        if colors is not None:
            if len(colors.shape) != 2:
                colors = colors.reshape(-1,3)
            colors = colors[valid_mask]

        else:
            colors = apply_gradient_color(points, axis, cmap)
        return points, colors

    if (points1 is not None) and (points2 is not None):
        points1, colors1 = process(points1, colors1, cmap1)
        points2, colors2 = process(points2, colors2, cmap2)
    else:
        print(f"no points; bye")
        sys.exit()
    points_combined = np.vstack((points1, points2))
    colors_combined = np.vstack((colors1, colors2))
    # create and save point cloud
    pcd = o3d.geometry.PointCloud()
    # http://open3d.org/docs/latest/python_api/open3d.utility.Vector3dVector.html
    pcd.points = o3d.utility.Vector3dVector(points_combined) #Convert float64 numpy array of shape (n, 3) to Open3D format.
    pcd.colors = o3d.utility.Vector3dVector(colors_combined)
    o3d.io.write_point_cloud(filename, pcd)
    print(f"saved overlaid point cloud to {filename}")

# --- Main Dataset Class ---

class PanoFishPin360Loc(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data(self.split)
        self.split = split

        # Scene-level train/val/test split; `all_metadata.npz` holds every scene,
        # and select_scene() restricts the pair list to the named scenes.
        # 360Loc split, per the paper: train on Hall and Piatrium,
        # evaluate on the held-out Atrium and Concourse scenes.
        if self.split == 'train':
            self.select_scene(('hall', 'piatrium'))
        elif self.split in ('val', 'test'):
            self.select_scene(('atrium', 'concourse'))
        elif self.split is not None:
            raise ValueError(f'bad {self.split=}')

    def _load_data(self, split):
        # Pair metadata produced by the 360Loc preprocessing step
        npz_path = osp.join(self.ROOT, 'all_metadata.npz')
        if not osp.isfile(npz_path):
            raise FileNotFoundError(f"Metadata not found at {npz_path}")
            
        with np.load(npz_path,  allow_pickle=True) as data:
            self.all_scenes = data['scenes']
            self.all_images = data['images']
            self.pairs = data['pairs']

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.all_scenes]
        
        # If no scenes match (e.g. running on a subset), warn but don't crash immediately unless empty
        if not any(scene_id):
            raise ValueError(
                f"no scene matched {scenes}; available: {list(self.all_scenes[:8])}. "
                f"Refusing to continue: an unfiltered split would mix train and test scenes."
            )

        valid = np.in1d(self.pairs['scene_id'], np.nonzero(scene_id)[0])
        
        if instances:
            image_id = [i.startswith(instances) for i in self.all_images]
            image_id = np.nonzero(image_id)[0]
            assert len(image_id), 'no instance found'
            if len(instances) == 2:
                valid &= np.in1d(self.pairs['im1_id'], image_id) & np.in1d(self.pairs['im2_id'], image_id)
            else:
                valid &= np.in1d(self.pairs['im1_id'], image_id) | np.in1d(self.pairs['im2_id'], image_id)

        if opposite:
            valid = ~valid
        
        assert valid.any(), f"No pairs found for split {scene}"
        self.pairs = self.pairs[valid]

    def _get_views(self, pair_idx, resolution, rng):
        # 1. Retrieve Pair Info
        scene_id_idx, im1_idx, im2_idx, score = self.pairs[pair_idx]
        
        # 2. Parse Scene Name
        # In 360Loc NPZ, scene_id is "scene_name seq_name" (e.g., "atrium daytime_360_0")
        scene_str = self.all_scenes[scene_id_idx]
        scene_parts = scene_str.split()
        scene_name = scene_parts[0]
        seq_name = scene_parts[1]
        
        # 3. Construct Paths
        # The NPZ 'images' array holds paths of the form
        # "scene/mapping/seq/image/name.jpg", while the preprocessed images are
        # laid out as ROOT/scene/seq/name.jpg, so the path is rebuilt below.
        
        views = []
        for im_idx in [im1_idx, im2_idx]:
            rel_path_raw = self.all_images[im_idx] # e.g. atrium/mapping/daytime_360_0/image/0002.jpg
            
            # Parse raw path to get the filename
            parts = rel_path_raw.split('/')
            if len(parts) < 2: parts = rel_path_raw.split('\\') # Windows fallback
            img_filename = parts[-1] # 0002.jpg
            img_basename = osp.splitext(img_filename)[0] # 0002
            
            # Construct path to preprocessed file
            # Preprocessor output: ROOT/scene_name/seq_name/0002.jpg
            # seq_path = osp.join(self.ROOT, scene_name, seq_name)
            seq_path = osp.join(self.ROOT, scene_name, scene_str)
            img_path = osp.join(seq_path, img_basename + '.jpg')
            depth_path = osp.join(seq_path, img_basename + '.exr')
            npz_path = osp.join(seq_path, img_basename + '.npz')

            
            try:
                # Load RGB
                image = imread_cv2(img_path)
                
                # Load Depth (Float32 Meters)
                depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                
                # Load Metadata
                camera_params = np.load(npz_path)
                intrinsics = np.float32(camera_params['intrinsics'])
                camera_pose = np.float32(camera_params['cam2world'])
                gt_rays = np.float32(camera_params['gt_rays'])
                
            except Exception as e:
                print(f"[Error] Failed loading {img_path}")
                raise e

            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)

            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                gt_rays=gt_rays,
                dataset='PanoFishPin360Loc',
                # label=osp.relpath(seq_path, self.ROOT),
                label=f"{scene_name}_{seq_name}",
                instance=img_basename
            ))


        return views

    def compute_pts3d(self, view):
        """
        Compute 3D points from depthmap and rays.
        Adapted for Equirectangular rays (already generated in preprocessing).
        """
        if view['gt_rays'] is not None:
            depthmap = view['depthmap']
            H, W = depthmap.shape
            rays = view['gt_rays'] # (H*W, 3) or (H, W, 3)
            
            # Ensure rays are (3, H, W) for broadcasting
            if rays.ndim == 2:
                rays = rearrange(rays, "(h w) c -> c h w", h=H, w=W)
            elif rays.ndim == 3:
                rays = rearrange(rays, "h w c -> c h w")
            
            # Points in Camera Frame: P = R * D
            pts3d = rays * depthmap # (3, H, W) * (H, W) -> (3, H, W)
            pts3d = pts3d.transpose(1, 2, 0).astype(np.float32) # (H, W, 3)
            
            valid_mask = depthmap > 0
            X_cam = pts3d
        else:
            raise ValueError("GT Rays missing for Panorama")

        # Transform to World Frame
        X_world = X_cam # Default if no pose
        camera_pose = view['camera_pose']
        
        if camera_pose is not None:
            R = camera_pose[:3, :3]
            t = camera_pose[:3, 3]
            # X_world = R * X_cam + t
            X_world = np.einsum("ik, vuk -> vui", R, X_cam) + t[None, None, :]

        return X_world, valid_mask


# --- Testing Block ---
