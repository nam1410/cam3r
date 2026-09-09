# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Aria Digital Twin (ADT) fisheye sequences
# dataset at https://www.projectaria.com/datasets/adt/
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



class FisheyeAria(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        
        super().__init__(*args, **kwargs)
        self.split = split # this is essential to initialize the splits

        self.loaded_data = self._load_data(self.split)



        if self.split is None:
            pass
        elif self.split == 'train':
            # train = all scenes EXCEPT those containing "skeleton"
            # self.select_scene("skeleton", opposite=True)
            self.select_scene("user", opposite=True)

        elif self.split == 'val':
            # val = scenes that contain "skeleton"
            # self.select_scene("skeleton")
            self.select_scene("user")

        else:
            raise ValueError(f'bad {self.split=}')

    def _load_data(self, split):
        with np.load(osp.join(self.ROOT, 'all_metadata.npz'), allow_pickle=True) as data:
            self.all_scenes = data['scenes']
            self.all_images = data['images']
            self.pairs = data['pairs']
        
        # Map each image to its scene index
        # image 'Apartment_release_clean_seq131/xxxx.jpg'
        # → scene 'Apartment_release_clean_seq131'
        image_scene_name = np.array([img.split('/')[0] for img in self.all_images])

        # build a mapping: scene name → scene index
        scene_to_index = {s: i for i, s in enumerate(self.all_scenes)}

        # For each image, find which scene index it belongs to
        self.image_scene_id = np.array([scene_to_index[name] for name in image_scene_name])

        # For each pair, find scene_id from first image
        # (you can choose either im1 or im2; they should belong to the same scene)
        self.pair_scene_id = self.image_scene_id[self.pairs[:,0]]

    def __len__(self):
        return len(self.pairs)

    def get_stats(self):
        return f'{len(self)} pairs from {len(self.all_scenes)} scenes'

    def select_scene(self, keywords, opposite=False):
        """
        keywords: string or tuple/list of strings
                e.g. "skeleton" or ("skeleton", "meal")

        Matches scenes if ANY keyword appears in the scene name.
        """
        if isinstance(keywords, str):
            keywords = (keywords,)

        # Find scenes whose name contains at least one keyword
        scene_mask = np.array([
            any(k.lower() in s.lower() for k in keywords)
            for s in self.all_scenes
        ])

        assert scene_mask.any(), "No scenes matched the keyword(s)."

        # Convert scene mask → valid scene indices
        valid_scene_ids = np.nonzero(scene_mask)[0]

        # Determine which pairs stay
        valid_pairs = np.in1d(self.pair_scene_id, valid_scene_ids)

        if opposite:
            valid_pairs = ~valid_pairs

        assert valid_pairs.any(), "Filtering removed all pairs."

        self.pairs = self.pairs[valid_pairs]


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
            rays = view['gt_rays'] # now reshaped to H*W from W*H (disregard the below comment for the pts3d calculation)
            # # 1. reshape stored rays back to per-pixel layout
            # """
            # Rays were stored in order (u-major, then v)
            # for u in range(W):
            #     for v in range(H):
            #         append(ray)
            # """
            
            rays = rearrange(rays, " (h w) c ->  c h w", h=H, w=W) 

            # Mask out invalid rays (i.e., rays with NaN values)
            valid_rays_mask = ~np.isnan(rays).any(axis=0)  # mask of valid rays (H, W)
            valid_depth_mask = depthmap > 0

            valid_mask = valid_depth_mask & valid_rays_mask


            # 3. Compute 3D points
            pts3d = rays *  depthmap # 
            pts3d = pts3d.transpose(1, 2, 0).astype(np.float32)

    
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
            # Express in absolute coordinates (invalid depth values)
            X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]
        

        return X_world, valid_mask



    def _get_views(self, pair_idx, resolution, rng):
        im1_id, im2_id = self.pairs[pair_idx]

        views = []

        for im_id in [im1_id, im2_id]:
            img_rel_path = self.all_images[im_id]
            img_full_path = osp.join(self.ROOT, img_rel_path)
            base = osp.splitext(img_full_path)[0]   

            try:
                image = imread_cv2(img_full_path)
                
                # depthmap = imread_cv2(osp.join(seq_path, img + ".exr"))
                depthmap = cv2.imread(base+".exr", cv2.IMREAD_UNCHANGED).astype(np.float32)
                H,W = depthmap.shape
                camera_params = np.load(base+ ".npz")
                gt_rays_npz = np.load(base+ "_rays.npz")
            except Exception as e:
                raise OSError(f'cannot load {img_full_path}, got exception {e}')

            intrinsics = np.float32(camera_params['intrinsics'])
            camera_pose = np.float32(camera_params['cam2world'])
            # gt_rays = np.float32(camera_params['gt_rays'])
            gt_rays = np.float32(gt_rays_npz['gt_rays']) # (W, H) C
            # gt_rays = gt_rays.reshape(W, H, 3).transpose(1, 0, 2)  # → (H, W, 3)
            # gt_rays = gt_rays.reshape(H * W, 3)
            gt_rays = rearrange(gt_rays, " (w h) c ->  (h w) c", h=H, w=W)

            # mask = ~np.isnan(gt_rays).any(axis=1)   # shape (H*W,)

            # rays_clean = np.where(mask[:, None], gt_rays, 0)








            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            
            basename = base.split("/")[-1]
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                camera_intrinsics=intrinsics,
                gt_rays=gt_rays,  # HxW
                dataset='FisheyeAria',
                label=img_rel_path.split("/")[0], # scene name
                instance=basename))

        return views
    
