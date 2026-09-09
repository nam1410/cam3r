# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed MegaDepth
# dataset at https://www.cs.cornell.edu/projects/megadepth/
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


class MegaDepth(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data(self.split)
        self.split = split # this is essential to initialize the splits


        # Evaluation scenes, per the paper: 0015, 0022 and 5000-5018. Training
        # uses their exact complement (0000-4541 excluding 0015 and 0022), so
        # the two splits cannot overlap.
        eval_scenes = ('0015', '0022') + tuple(f'{i:04d}' for i in range(5000, 5019))

        if self.split is None:
            pass
        elif self.split == 'train':
            self.select_scene(eval_scenes, opposite=True)
        elif self.split == 'val':
            self.select_scene(eval_scenes)
        else:
            raise ValueError(f'bad {self.split=}')

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz')) as data:
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
        assert any(scene_id), 'no scene found'

        valid = np.in1d(self.pairs['scene_id'], np.nonzero(scene_id)[0])
        if instances:
            image_id = [i.startswith(instances) for i in self.all_images]
            image_id = np.nonzero(image_id)[0]
            assert len(image_id), 'no instance found'
            # both together?
            if len(instances) == 2:
                valid &= np.in1d(self.pairs['im1_id'], image_id) & np.in1d(self.pairs['im2_id'], image_id)
            else:
                valid &= np.in1d(self.pairs['im1_id'], image_id) | np.in1d(self.pairs['im2_id'], image_id)

        if opposite:
            valid = ~valid
        assert valid.any()
        self.pairs = self.pairs[valid]


    def compute_pts3d(self, view):
       
        if view['gt_rays'] is not None:
            depthmap = view['depthmap']
            H,W = depthmap.shape
            valid_mask = depthmap > 0
            rays = view['gt_rays'] # H*W, 3

            rays = rearrange(rays, " (h w) c ->  c h w", h=H, w=W)
            pts3d = rays * depthmap # 3, H, W


            pts3d = pts3d.transpose(1, 2, 0).astype(np.float32)
            
            X_cam, valid_mask = pts3d, valid_mask # xcam in cam coordinates
        else:
            print("no gt rays; sys exit being called")
            sys.exit()
        X_world = X_cam # default
        # H, W = view['depthmap'].shape
        
        camera_pose = view['camera_pose']

        if camera_pose is not None:
            # R_cam2world = np.float32(camera_params["R_cam2world"])
            # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
            R_cam2world = camera_pose[:3, :3]
            t_cam2world = camera_pose[:3, 3]
            # Express in absolute coordinates (invalid depth values)
            X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        

        return X_world, valid_mask



    def _get_views(self, pair_idx, resolution, rng):
        scene_id, im1_id, im2_id, score = self.pairs[pair_idx]

        scene, subscene = self.all_scenes[scene_id].split()
        seq_path = osp.join(self.ROOT, scene, subscene)

        views = []

        for im_id in [im1_id, im2_id]:
            img = self.all_images[im_id]
            try:
                image = imread_cv2(osp.join(seq_path, img + '.jpg'))
                # depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                depthmap = cv2.imread(osp.join(seq_path, img + ".exr"), cv2.IMREAD_UNCHANGED).astype(np.float32)
                camera_params = np.load(osp.join(seq_path, img + ".npz"))
            except Exception as e:
                raise OSError(f'cannot load {img}, got exception {e}')

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])
            gt_rays = np.float32(camera_params['gt_rays']) 





            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)

            basename = img
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                gt_rays=gt_rays,  # H*W, 3
                dataset='Megadepth',
                label=osp.relpath(seq_path, self.ROOT),
                instance=basename))

        return views
    
