# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Stanford 2D-3D-S panoramas
# dataset at http://buildingparser.stanford.edu/dataset.html
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


class Pano2D3DS(BaseStereoViewDataset):
    def __init__(self, *args, split, ROOT, **kwargs):
        self.ROOT = ROOT
        
        super().__init__(*args, **kwargs)
        self.loaded_data = self._load_data(self.split)
        self.split = split # this is essential to initialize the splits



        if self.split is None:
            pass
        elif self.split == 'train':
            self.select_scene(('area_5a', 'area_5b', 'area_6'), opposite=True) #based on the recommended splits in the 2D3DS paper
            # self.select_scene(('area_1', 'area_2', 'area_3', 'area_4', 'area_5a', 'area_5b'))
        elif self.split == 'val':
            self.select_scene(('area_5a', 'area_5b', 'area_6'))
            # self.select_scene(('area_6'))
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
            H,W = depthmap.shape
            rays = view['gt_rays']
            rays = rearrange(rays, " (h w) c ->  c h w", h=H, w=W)
            # depth_flat = depthmap.reshape(-1)
            # pts3d = rays * depth_flat[:, np.newaxis]  # shape (N, 3)
            pts3d = rays * depthmap
            pts3d = pts3d.transpose(1, 2, 0).astype(np.float32)
            # pts3d = pts3d.reshape(H, W, 3).astype(np.float32)
            valid_mask = depthmap > 0
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
            # quaternion = np.float32(camera_params['quaternion_cam2w'])
            gt_rays = np.float32(camera_params['gt_rays'])


            # X_cam, _ =  self.pano_backproject(depthmap) 
            # X_world = X_cam # default







            # image, depthmap, intrinsics = self._crop_resize_if_necessary(
            #     image, depthmap, intrinsics, resolution, rng, info=(seq_path, img))
            """ This function:
            - first downsizes the image with LANCZOS inteprolation,
              which is better than bilinear interpolation in
              # downscale with lanczos interpolation so that image.size == resolution
                # cropping centered on the principal point

                this approach makes strong assumptions that:
                The image is rectilinear (i.e. straight lines stay straight).
                Camera intrinsics can be modeled with a 3x3 matrix (focal lengths, principal point, no distortion)
                Cropping and rescaling can be handled via simple transforms on the intrinsics
                
                These assumptions break down for:
                Fisheye cameras (nonlinear projections, radial distortion)
                Spherical / equirectangular / panoramic images (e.g. 360° cameras)
                Principal point might not even make sense
                Cropping leads to major distortions
                Cannot simply apply matrix transforms
            """

            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)




            
            # X_cam, _ =  self.pano_backproject(depthmap) 
            # X_world = X_cam # default



            
            
            basename = img
            views.append(dict(
                img=image,
                depthmap=depthmap,
                camera_pose=camera_pose,  # cam2world
                # quaternion=quaternion, # cam2world
                camera_intrinsics=intrinsics,
                gt_rays=gt_rays,  # HxW
                dataset='Pano2D3DS',
                label=osp.relpath(seq_path, self.ROOT),
                instance=basename))

        return views
    
