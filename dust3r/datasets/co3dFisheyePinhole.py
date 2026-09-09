# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Co3d_v2, pairing fisheye and pinhole renderings
# dataset at https://github.com/facebookresearch/co3d - Creative Commons Attribution-NonCommercial 4.0 International
# --------------------------------------------------------
import os
import os.path as osp
import json
import itertools
from collections import deque
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import sys
from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset
from dust3r.utils.image import imread_cv2
from einops import rearrange
from datetime import datetime
import time
import open3d as o3d
from matplotlib import cm
import matplotlib.pyplot as plt
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


class Co3dFisheyePinhole(BaseStereoViewDataset):
    def __init__(self, mask_bg=True, *args, ROOT, **kwargs):
        self.ROOT = ROOT
        super().__init__(*args, **kwargs)
        assert mask_bg in (True, False, 'rand')
        self.mask_bg = mask_bg
        self.dataset_label = 'Co3dFisheyePinhole'

        # load all scenes
        with open(osp.join(self.ROOT, f'selected_seqs_{self.split}.json'), 'r') as f:
            self.scenes = json.load(f)
            self.scenes = {k: v for k, v in self.scenes.items() if len(v) > 0}
            self.scenes = {(k, k2): v2 for k, v in self.scenes.items()
                           for k2, v2 in v.items()}
        self.scene_list = list(self.scenes.keys())

        # for each scene, we have 100 images ==> 360 degrees (so 25 frames ~= 90 degrees)
        # we prepare all combinations such that i-j = +/- [5, 10, .., 90] degrees
        self.combinations = [(i, j)
                             for i, j in itertools.combinations(range(100), 2)
                             if 0 < abs(i - j) <= 30 and abs(i - j) % 5 == 0]

        self.invalidate = {scene: {} for scene in self.scene_list}

    def __len__(self):
        return len(self.scene_list) * len(self.combinations)

    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.npz') # ground truth rays will also be loaded in this version

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'images', f'frame{view_idx:06n}.jpg')

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'depths', f'frame{view_idx:06n}.jpg.geometric.png')

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.ROOT, obj, instance, 'masks', f'frame{view_idx:06n}.png')

    def _read_depthmap(self, depthpath, input_metadata):
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = (depthmap.astype(np.float32) / 65535) * np.nan_to_num(input_metadata['maximum_depth'])
        # return depthmap
        return depthmap.astype(np.float32)

    def compute_pts3d(self, view):
        """
        serves the purpose of depthmap_to_absolute_camera_coordinates https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/dust3r/utils/geometry.py#L200
        Args:
            - depthmap (HxW array):
            - camera_intrinsics: a 3x3 matrix
            - camera_pose: a 4x3 or 4x4 cam2world matrix
        Returns:
            pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
        """
        # serves the purpose of depthmap_to_camera_coordinates https://github.com/naver/dust3r/blob/4c24a6ebf04809f2cfe59915e51779c8984aaa40/dust3r/utils/geometry.py#L165
        # X_cam, valid_mask =  self.pano_backproject(view['depthmap']) 
        if view['gt_rays'] is not None:
            depthmap = view['depthmap']
            H,W = depthmap.shape #
            rays = view['gt_rays'] # (H,W,3) - preprocessing code generates the rays in shape H,W,C; 
            
            rays = rearrange(rays, " h w c ->  c h w", h=H, w=W) 

            # Mask out invalid rays (i.e., rays with NaN values)
            valid_rays_mask = ~np.isnan(rays).any(axis=0)  # mask for valid rays (H, W)
            valid_depth_mask = depthmap > 0 #mask for valid depth values (H, W)

            valid_mask = valid_depth_mask & valid_rays_mask # combined valid mask (H, W)


            # 3. Compute 3D points
            pts3d = rays *  depthmap # 3, H, W
            pts3d = pts3d.transpose(1, 2, 0).astype(np.float32) # H, W, 3

    
            X_cam, valid_mask = pts3d, valid_mask # xcam in cam coordinates
        else:
            print("no gt rays; sys exit being called")
            sys.exit()
        X_world = X_cam # default

        
        camera_pose = view['camera_pose'] # [4,4]
        if camera_pose is not None:
            # R_cam2world = np.float32(camera_params["R_cam2world"])
            # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
            R_cam2world = camera_pose[:3, :3]
            t_cam2world = camera_pose[:3, 3]
            X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :] # convert to world coordinates        

        return X_world, valid_mask

    def _get_views(self, idx, resolution, rng):
        # choose a scene
        obj, instance = self.scene_list[idx // len(self.combinations)]
        image_pool = self.scenes[obj, instance]
        im1_idx, im2_idx = self.combinations[idx % len(self.combinations)]

        # add a bit of randomness
        last = len(image_pool) - 1

        if resolution not in self.invalidate[obj, instance]:  # flag invalid images
            self.invalidate[obj, instance][resolution] = [False for _ in range(len(image_pool))]

        # decide now if we mask the bg
        mask_bg = (self.mask_bg == True) or (self.mask_bg == 'rand' and rng.choice(2))

        views = []
        imgs_idxs = [max(0, min(im_idx + rng.integers(-4, 5), last)) for im_idx in [im2_idx, im1_idx]]
        imgs_idxs = deque(imgs_idxs)
        while len(imgs_idxs) > 0:  # some images (few) have zero depth
            im_idx = imgs_idxs.pop()

            if self.invalidate[obj, instance][resolution][im_idx]:
                # search for a valid image
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(image_pool)):
                    tentative_im_idx = (im_idx + (random_direction * offset)) % len(image_pool)
                    if not self.invalidate[obj, instance][resolution][tentative_im_idx]:
                        im_idx = tentative_im_idx
                        break

            view_idx = image_pool[im_idx]

            impath = self._get_impath(obj, instance, view_idx)
            depthpath = self._get_depthpath(obj, instance, view_idx)

            # load camera params
            metadata_path = self._get_metadatapath(obj, instance, view_idx)
            input_metadata = np.load(metadata_path)
            camera_pose = input_metadata['camera_pose'].astype(np.float32)
            intrinsics = input_metadata['camera_intrinsics'].astype(np.float32)
            gt_rays = input_metadata['rays'].astype(np.float32)  # H W 3

            # load image and depth
            rgb_image = imread_cv2(impath)
            depthmap = self._read_depthmap(depthpath, input_metadata)

            if mask_bg:
                # load object mask
                maskpath = self._get_maskpath(obj, instance, view_idx)
                maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                maskmap = (maskmap / 255.0) > 0.1

                # update the depthmap with mask
                depthmap *= maskmap

            # rgb_image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #     rgb_image, depthmap, intrinsics, resolution, rng=rng, info=impath)




            num_valid = (depthmap > 0.0).sum()
            if num_valid == 0:
                # problem, invalidate image and retry
                self.invalidate[obj, instance][resolution][im_idx] = True
                imgs_idxs.append(im_idx)
                continue
            
            if not isinstance(rgb_image, PIL.Image.Image):
                rgb_image = PIL.Image.fromarray(rgb_image) # convert rgb_image to PIL format if not already


            views.append(dict(
                img=rgb_image,
                depthmap=depthmap,
                camera_pose=camera_pose,
                camera_intrinsics=intrinsics,
                gt_rays=gt_rays,  # HxWxC
                dataset=self.dataset_label,
                label=f"{obj}_{instance}", #osp.join(obj, instance),
                instance=osp.split(impath)[1],
            ))
        return views
