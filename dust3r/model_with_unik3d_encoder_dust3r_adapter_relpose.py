"""
CAM3R model, derived from DUSt3R's ``model.py``.

Differences from upstream DUSt3R:
- each downstream head regresses its view's 3D points in that view's own camera
  frame, rather than both views in the first camera's frame;
- a relative-pose head (``self.downstream_head_relpose``) consumes pooled
  features from both decoders and predicts the transform between the two
  cameras, which is what ties the two frames together;
- a UniK3D-derived camera adapter (``self.cam_adapter``) predicts per-pixel rays
  and intrinsics, so the heads regress a radial distance rather than XYZ.
"""
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
import torch
import torch.nn as nn
import os
import sys
from packaging import version
import huggingface_hub

from dust3r.utils.misc import fill_default_args, freeze_all_params, is_symmetrized, interleave, transpose_to_landscape
from dust3r.heads import head_factory
from dust3r.patch_embed import get_patch_embed
from dust3r.unik3d_encoder_dust3r_adapter import Dust3rUnik3dAdapter

import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet  # noqa

inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), ("Outdated huggingface_hub version, "
                                                                     "please reinstall requirements.txt")


def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)


class AsymmetricCroCo3DStereo (
    CroCoNet,
    huggingface_hub.PyTorchModelHubMixin,
    library_name="dust3r",
    repo_url="https://github.com/naver/dust3r",
    tags=["image-to-3d"],
):
    """ Two siamese encoders, followed by two decoders.
    The goal is to output 3d points directly, both images in view1's frame
    (hence the asymmetry).
    Now also computes UniK3D-style camera intrinsics and rays via an adapter head.
    """

    def __init__(self,
                 output_mode='pts3d',
                 head_type='linear',
                 depth_mode=('exp', -inf, inf),
                 conf_mode=('exp', 1, inf),
                 freeze='none',
                 landscape_only=True,
                 patch_embed_cls='PatchEmbedDust3R',  # PatchEmbedDust3R or ManyAR_PatchEmbed
                 # adapter-related
                 adapter_config=None,                 # UniK3D-compatible config dict (expects model.pixel_encoder.output_idx)
                 adapter_pretrained=True,             # directly download model safetensors from hugging face {'repo_id': 'lpiccinelli/unik3d-vitl', 'filename': 'model.safetensors', 'subfolder': None, 'revision': None}
                 adapter_use_layernorm_pool=True,     # apply LN before pooling tokens
                 **croco_kwargs):
        self.patch_embed_cls = patch_embed_cls
        self.croco_args = fill_default_args(croco_kwargs, super().__init__)
        super().__init__(**croco_kwargs)

        # dust3r specific initialization
        self.dec_blocks2 = deepcopy(self.dec_blocks)
        self.set_downstream_head(output_mode, head_type, landscape_only, depth_mode, conf_mode, **croco_kwargs)
        self.set_freeze(freeze)
        # ---------- Camera adapter (UniK3D AngularModule) ----------
        assert adapter_config is not None
        import json
        print(f"reading json config file")
        with open(adapter_config, "r") as f:
            config = json.load(f)
        
        print(f"finished loading json config file")

        self.cam_adapter = Dust3rUnik3dAdapter(
            config
        )

        # Load from HuggingFace
        if adapter_pretrained:
            loaded, not_loaded = self.cam_adapter.load_pretrained_from_hf(
            repo_id="lpiccinelli/unik3d-vitl",
            filename="model.safetensors",
            subfolder=None,
            )
       

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device='cpu')
        else:
            try:
                model = super(AsymmetricCroCo3DStereo, cls).from_pretrained(pretrained_model_name_or_path, **kw)
            except TypeError as e:
                raise Exception(f'tried to load {pretrained_model_name_or_path} from huggingface, but failed')
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_size = patch_size
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        # duplicate all weights for the second decoder if not present
        new_ckpt = dict(ckpt)
        if not any(k.startswith('dec_blocks2') for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith('dec_blocks'):
                    new_ckpt[key.replace('dec_blocks', 'dec_blocks2')] = value
        return super().load_state_dict(new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            'none': [],
            'mask': [self.mask_token],
            'encoder': [self.mask_token, self.patch_embed, self.enc_blocks],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size,
                            **kw):
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self, has_conf=bool(conf_mode))
        # the two downstream heads predict in their own camera frames, so a
        # third head is needed to relate them
        self.downstream_head_relpose = head_factory("relpose_v2", "relpose_v2", self)
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)
    

    def _encode_image(self, image, true_shape, rays_gt=None):

        """
        Runs encoder and returns (intrinsics, rays).
        Returns:
           x: (B, N, D), pos: (B, N, Dpos), cam: tuple(intrinsics, rays) 
        """
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape) # image goes into patch embedding to get patches and associated position coordinates out of the patch embedding 

        # add positional embedding without cls token
        assert self.enc_pos_embed is None
        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks: # iterate every block of the encoder
            x = blk(x, pos) # get the encoder features
        
        x = self.enc_norm(x) # apply normalization to the encoder features
        # till now, we captured features from dust3r encoder


        # feed image into unik3d encoder to obtain tokens that are fed into the angular module to obtain intriniscs and camera rays 

        cam = None
        if self.cam_adapter is not None: # checks if unik3d's adapter (encoder + angular module is defined or not )
            # Adapter expects input images and optional rays_gt
            cam_intrinsics, cam_rays = self.cam_adapter(images=image, rays_gt=rays_gt)
            cam = (cam_intrinsics, cam_rays)
        return x, pos, None, cam

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2, rays1_gt=None, rays2_gt=None):
        """
        Returns:
          out, out2, pos, pos2, intr1, intr2, rays1, rays2
        """
        if img1.shape[-2:] == img2.shape[-2:]:
            if rays1_gt is not None and rays2_gt is not None:
                rays_cat = torch.cat((rays1_gt, rays2_gt), dim=0)
            else:
                rays_cat = None
            out, pos, _, cam = self._encode_image(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0),rays_gt=rays_cat)
            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
            if cam is not None:
                intr_all, rays_all = cam
                intr1, intr2 = intr_all.chunk(2, dim=0)
                rays1, rays2 = rays_all.chunk(2, dim=0)
            else:
                intr1 = intr2 = rays1 = rays2 = None
        else:
            out, pos, _, cam = self._encode_image(img1, true_shape1,rays_gt=rays1_gt)
            out2, pos2, _, cam2 = self._encode_image(img2, true_shape2, rays_gt=rays2_gt)
            if cam is not None:
                intr1, rays1 = cam
            else:
                intr1 = rays1 = None
            if cam2 is not None:
                intr2, rays2 = cam2
            else:
                intr2 = rays2 = None
        return out, out2, pos, pos2, intr1, intr2, rays1, rays2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        rays1_gt = view1['gt_rays']
        rays2_gt = view2['gt_rays']

        B = img1.shape[0]
        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))
        # warning! maybe the images have different portrait/landscape orientations

        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2, intr1, intr2, rays1, rays2 = self._encode_image_pairs(
                img1[::2], img2[::2], shape1[::2], shape2[::2],
                rays1_gt=None if rays1_gt is None else rays1_gt[::2],
                rays2_gt=None if rays2_gt is None else rays2_gt[::2],
            ) # batch sise is halved because of slicing - to avoid redudant pairs and the model processes every alternate one
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
            if intr1 is not None and intr2 is not None:
                intr1, intr2 = interleave(intr1, intr2)
                rays1, rays2 = interleave(rays1, rays2)
        else:
            feat1, feat2, pos1, pos2, intr1, intr2, rays1, rays2 = self._encode_image_pairs(img1, img2, shape1, shape2, rays1_gt=rays1_gt, rays2_gt=rays2_gt)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2), (intr1, intr2), (rays1, rays2)

    def _decoder(self, f1, pos1, f2, pos2):
        final_output = [(f1, f2)]  # before projection
        # takes encoder features 

        # project to decoder dim
        f1 = self.decoder_embed(f1) # projects encoder feature to decoder's desired embedding dim
        f2 = self.decoder_embed(f2) # projects encoder feature to decoder's desired embedding dim

        final_output.append((f1, f2))
        for blk1, blk2 in zip(self.dec_blocks, self.dec_blocks2):
            # img1 side
            f1, _ = blk1(*final_output[-1][::+1], pos1, pos2)
            # img2 side
            f2, _ = blk2(*final_output[-1][::-1], pos2, pos1)
            # store the result
            final_output.append((f1, f2))

        # normalize last output
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.dec_norm, final_output[-1]))
        return zip(*final_output)
    
    def _pts3d_computer(self, result_dict, rays, key_distance='distance', key_pts3d='pts3d'):
        """
        Constructs pts3d from rays and distance if:
          - pts3d not already in result_dict
          - key_distance exists
          - rays is not None
        Expects rays: (B, H, W, 3), distance: (B, H, W, 1).
        Stores pts3d as (B, H, W, 3) matching rays layout.
        """
        if (key_pts3d in result_dict) or (rays is None):
            return
        if key_distance not in result_dict:
            return
        dist = result_dict[key_distance]
        # Validate shapes
        if dist.dim() != 4 or rays.dim() != 4: # number of dimensions (rank) of the tensor
            raise ValueError(f"Unexpected shape for distance {dist.shape} or rays {rays.shape}")
        if dist.shape[-1] != 1 or rays.shape[-1] != 3:
            raise ValueError(f"Bad channel dims: distance last={dist.shape[-1]} rays last={rays.shape[-1]}")
        if dist.shape[0:3] != rays.shape[0:3]:
            raise ValueError(f"Mismatch spatial dims distance={dist.shape} rays={rays.shape}")
        # Broadcast multiply
        pts3d = rays * dist # (B, H, W, 3)
        result_dict[key_pts3d] = pts3d

    def _downstream_head(self, head_num, decout, img_shape):
        B, S, D = decout[-1].shape
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape)

    def forward(self, view1, view2):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2), (intr1, intr2), (rays1, rays2) = self._encode_symmetrized(view1, view2)

        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)
            res_relpose = self.downstream_head_relpose(dec1, dec2, shape1, shape2)  #{"T": T, "t": t, "R": R, "q": q}


        res1.update({'intrinsics': intr1, 'rays': rays1})
        res2.update({'intrinsics': intr2, 'rays': rays2})
        res2.update({'T': res_relpose['T'], 't': res_relpose['t'], 'R': res_relpose['R'], 'q': res_relpose['q']})

        # compute pts3d if not present and distance is available
        self._pts3d_computer(res1, rays1, key_distance='distance', key_pts3d='pts3d')
        self._pts3d_computer(res2, rays2, key_distance='distance', key_pts3d='pts3d')


        return res1, res2
