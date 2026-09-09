"""
Camera/ray adapter, derived from UniK3D's ``unik3d/models/decoder.py``.

Differences from the UniK3D original:
- the Radial module (and everything it depends on) is dropped; only the
  angular branch, which predicts camera intrinsics and per-pixel rays, is kept;
- the class is renamed ``Dust3rUnik3dAdapter`` and exposes a DUSt3R-style
  ``forward(images, rays_gt)``;
- inputs are padded up to a multiple of the UniK3D patch size and the rays are
  cropped back to the original resolution before being returned;
- the pixel encoder and angular module weights are initialised from the public
  UniK3D checkpoint (see ``load_pretrained_from_hf``).
"""

"""
Author: Luigi Piccinelli
Licensed under the CC BY-NC-SA 4.0 license (http://creativecommons.org/licenses/by-nc-sa/4.0/)
"""


from math import tanh

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import trunc_normal_
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
import sys
import importlib
import numpy as np
import matplotlib.pyplot as plt

import dust3r.utils.path_to_unik3d  # noqa: F401  puts <repo>/UniK3D on sys.path
from unik3d.layers import (MLP, AttentionBlock, AttentionLayer, GradChoker,
                           PositionEmbeddingSine, ResUpsampleBil)
from unik3d.utils.coordinate import coords_grid
from unik3d.utils.sht import rsh_cart_3
from unik3d.utils.misc import last_stack



def plot_tensor_img(tensor, save_path=None):
    """
    tensor: [1, 3, H, W] on CUDA or CPU
    save_path: if not None, saves as PNG
    """
    
    # 1. Move to CPU, detach, convert to numpy
    img = tensor.squeeze(0).detach().cpu().numpy()   # [3, H, W]

    # 2. Convert to HWC format for matplotlib
    img = np.transpose(img, (1, 2, 0))               # [H, W, 3]

    # 3. If image is in [0,1] or [-1,1], fix it
    img_min, img_max = img.min(), img.max()
    if img_min < 0 or img_max > 1:
        # normalize to 0–1 for visualization
        img = (img - img_min) / (img_max - img_min + 1e-8)

    # 4. Plot
    plt.figure(figsize=(6,6))
    plt.imshow(img)
    plt.axis("off")

    # 5. Save to PNG if user asked
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"Saved to {save_path}")

    # plt.show()


def orthonormal_init(num_tokens, dims):
    pe = torch.randn(num_tokens, dims)

    # Apply Gram-Schmidt process to make the matrix orthonormal
    # Awful loop..
    for i in range(num_tokens):
        for j in range(i):
            pe[i] -= torch.dot(pe[i], pe[j]) * pe[j]
        pe[i] = F.normalize(pe[i], p=2, dim=0)

    return pe


class ListAdapter(nn.Module):
    def __init__(self, input_dims: list[int], hidden_dim: int):
        super().__init__()
        self.input_adapters = nn.ModuleList([])
        self.num_chunks = len(input_dims)
        self.checkpoint = True
        for input_dim in input_dims:
            self.input_adapters.append(nn.Linear(input_dim, hidden_dim))

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        outs = [self.input_adapters[i](x) for i, x in enumerate(xs)]
        return outs


class AngularModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 8,
        expansion: int = 4,
        dropout: float = 0.0,
        layer_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.pin_params = 3
        self.deg1_params = 3
        self.deg2_params = 5
        self.deg3_params = 7
        self.num_params = (
            self.pin_params + self.deg1_params + self.deg2_params + self.deg3_params
        )

        self.aggregate1 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.aggregate2 = AttentionBlock(
            hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.latents_pos = nn.Parameter(
            torch.randn(1, self.num_params, hidden_dim), requires_grad=True
        )
        self.in_features = nn.Identity()

        self.project_pin = nn.Linear(
            hidden_dim, self.pin_params * hidden_dim, bias=False
        )
        self.project_deg1 = nn.Linear(
            hidden_dim, self.deg1_params * hidden_dim, bias=False
        )
        self.project_deg2 = nn.Linear(
            hidden_dim, self.deg2_params * hidden_dim, bias=False
        )
        self.project_deg3 = nn.Linear(
            hidden_dim, self.deg3_params * hidden_dim, bias=False
        )

        self.out_pinhole = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=1)
        self.out_deg1 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)
        self.out_deg2 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)
        self.out_deg3 = MLP(hidden_dim, expansion=1, dropout=dropout, output_dim=3)

    def fill_intrinsics(self, x):
        hfov, cx, cy = x.unbind(dim=-1)
        hfov = torch.sigmoid(hfov - 1.1)  # 1.1 magic number s.t hfov = pi/2 for x=0
        ratio = self.shapes[0] / self.shapes[1]
        vfov = hfov * ratio
        cx = torch.sigmoid(cx)
        cy = torch.sigmoid(cy)
        correction_tensor = torch.tensor(
            [2 * torch.pi, 2 * torch.pi, self.shapes[1], self.shapes[0]],
            device=x.device,
            dtype=x.dtype,
        )

        intrinsics = torch.stack([hfov, vfov, cx, cy], dim=1)
        intrinsics = correction_tensor.unsqueeze(0) * intrinsics
        return intrinsics

    def forward(self, cls_tokens) -> torch.Tensor:
        latents_pos = self.latents_pos.expand(cls_tokens.shape[0], -1, -1)

        pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens = cls_tokens.chunk(4, dim=1)
        pin_tokens = rearrange(
            self.project_pin(pin_tokens), "b n (h c) -> b (n h) c", h=self.pin_params
        )
        deg1_tokens = rearrange(
            self.project_deg1(deg1_tokens), "b n (h c) -> b (n h) c", h=self.deg1_params
        )
        deg2_tokens = rearrange(
            self.project_deg2(deg2_tokens), "b n (h c) -> b (n h) c", h=self.deg2_params
        )
        deg3_tokens = rearrange(
            self.project_deg3(deg3_tokens), "b n (h c) -> b (n h) c", h=self.deg3_params
        )
        tokens = torch.cat([pin_tokens, deg1_tokens, deg2_tokens, deg3_tokens], dim=1)

        tokens = self.aggregate1(tokens, pos_embed=latents_pos)
        tokens = self.aggregate2(tokens, pos_embed=latents_pos)

        tokens_pinhole, tokens_deg1, tokens_deg2, tokens_deg3 = torch.split(
            tokens,
            [self.pin_params, self.deg1_params, self.deg2_params, self.deg3_params],
            dim=1,
        )
        x = self.out_pinhole(tokens_pinhole).squeeze(-1)
        d1 = self.out_deg1(tokens_deg1)
        d2 = self.out_deg2(tokens_deg2)
        d3 = self.out_deg3(tokens_deg3)

        camera_intrinsics = self.fill_intrinsics(x)
        return camera_intrinsics, torch.cat([d1, d2, d3], dim=1)

    def set_shapes(self, shapes: tuple[int, int]):
        self.shapes = shapes


class Dust3rUnik3dAdapter(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.build(config)
        self.apply(self._init_weights)
    
    def load_pretrained_from_hf(self, repo_id: str, filename: str = "model.safetensors", 
                            subfolder: str = None, revision: str = None):
        """
        Load pretrained weights from HuggingFace safetensors file.
        
        Args:
            repo_id: HuggingFace repository ID (e.g., "username/model-name")
            filename: Name of the safetensors file
            subfolder: Optional subfolder in the repo
            revision: Optional git revision (branch/tag/commit)
        """
        # Download the safetensors file from HuggingFace
        safetensors_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
            cache_dir=None  # Uses default HF cache
        )
        
        # Load the safetensors file
        state_dict = load_file(safetensors_path)
        
        # Map the checkpoint keys onto this adapter: the UniK3D checkpoint
        # prefixes the decoder weights with "pixel_decoder.", which this class
        # does not use, and its radial branch has no counterpart here.
        mapped_state_dict = {}
        for key, value in state_dict.items():
            if (key.startswith("pixel_encoder.")):
                mapped_state_dict[key] = value
            if (key.startswith("pixel_decoder.")) and ("radial" not in key):
                # Remove the "pixel_decoder." prefix
                new_key = key[len("pixel_decoder."):]
                mapped_state_dict[new_key] = value

        # Get current model's state dict
        model_state_dict = self.state_dict()
        
        # Track what gets loaded and what doesn't
        loaded_keys = []
        not_loaded_keys = []
        shape_mismatch_keys = []
        not_in_checkpoint_keys = []
        
        # Check each parameter in the current model
        for key in model_state_dict.keys():
            if key in mapped_state_dict:
                if model_state_dict[key].shape == mapped_state_dict[key].shape:
                    loaded_keys.append(key)
                else:
                    shape_mismatch_keys.append((key, model_state_dict[key].shape, mapped_state_dict[key].shape))
            else:
                not_in_checkpoint_keys.append(key)
        
        # Check for keys in checkpoint but not in model
        extra_keys = set(mapped_state_dict.keys()) - set(model_state_dict.keys())
        
        # Load the matched parameters
        filtered_state_dict = {k: mapped_state_dict[k] for k in loaded_keys}
        self.load_state_dict(filtered_state_dict, strict=False)
        
        # Print detailed report
        print(f"\n{'='*60}")
        print(f"Loading pretrained weights from: {repo_id}/{filename}")
        print(f"{'='*60}")
        
        print(f"\n✓ Successfully loaded {len(loaded_keys)} parameters:")
        for key in sorted(loaded_keys):
            print(f"  - {key}")
        
        if not_in_checkpoint_keys:
            print(f"\n✗ Not found in checkpoint ({len(not_in_checkpoint_keys)} parameters) - keeping initialized:")
            for key in sorted(not_in_checkpoint_keys):
                print(f"  - {key}")
        
        if shape_mismatch_keys:
            print(f"\n✗ Shape mismatches ({len(shape_mismatch_keys)} parameters) - keeping initialized:")
            for key, model_shape, ckpt_shape in shape_mismatch_keys:
                print(f"  - {key}: model {model_shape} vs checkpoint {ckpt_shape}")
        
        if extra_keys:
            print(f"\n⚠ Extra keys in checkpoint (ignored): {len(extra_keys)}")
            for key in sorted(list(extra_keys)[:10]):  # Show first 10
                print(f"  - {key}")
            if len(extra_keys) > 10:
                print(f"  ... and {len(extra_keys) - 10} more")
        
        print(f"\n{'='*60}")
        print(f"Summary: {len(loaded_keys)}/{len(model_state_dict)} parameters loaded")
        print(f"{'='*60}\n")
        
        return loaded_keys, not_in_checkpoint_keys

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


    def _compute_pad_to_multiple(self, H: int, W: int, multiple: int) -> tuple[int, int, int, int]:
        """
        Compute left, right, top, bottom padding needed to make H and W divisible by `multiple`.
        Pads only on the right and bottom by default (top/left = 0), but you can adjust
        policy as needed.

        Returns:
            (pad_left, pad_right, pad_top, pad_bottom)
        """
        need_h = (multiple - (H % multiple)) % multiple
        need_w = (multiple - (W % multiple)) % multiple
        pad_left, pad_top = 0, 0
        pad_right, pad_bottom = need_w, need_h
        return pad_left, pad_right, pad_top, pad_bottom


    def _apply_dynamic_padding(
    self,
    image: torch.Tensor,
    rays: torch.Tensor | None,
    pad_multiple: int = 14,
    ):
        """
        Apply dynamic padding to image (and rays if provided) so that spatial dimensions
        become divisible by `pad_multiple`. This function:
        - Optionally normalizes image
        - Pads image and rays consistently
        - Builds a validity mask indicating original (unpadded) pixels

        Args:
            image: Tensor of shape (B, C, H, W)
            rays: Optional Tensor of shape (B, 3, H, W) or ( 3, H, W)
            pad_multiple: Patch size multiple to align to (default 14)
        Returns:
            padded_image: (B, C, H', W')
            padded_rays: (B, 3, H', W') or None
            paddings: (pad_left, pad_right, pad_top, pad_bottom)
            validity_mask: (B, 1, H', W'), 1 for original pixels, 0 for padded pixels
        """
        assert image.ndim == 4, "image must be (B, C, H, W)"
        B, C, H, W = image.shape
        
        # Compute padding to reach divisible-by-`pad_multiple`
        pad_left, pad_right, pad_top, pad_bottom = self._compute_pad_to_multiple(H, W, pad_multiple)

        # Prepare validity mask for original content region (before padding)
        # Start with all zeros in padded canvas, set original region to 1
        validity_mask = image.new_zeros((B, 1, H + pad_bottom, W + pad_right))
        validity_mask[..., :H, :W] = 1.0

        # Pad image
        padded_image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)

        # Pad rays if provided. 
        padded_rays = None
        if rays is not None:
            if rays.ndim == 4 and rays.shape[1] != 3:  # Check if rays are in (B, H, W, C)
                rays = rearrange(rays, "b h w c -> b c h w")  # Convert to (B, C, H, W)
            elif rays.ndim == 3:  # Handle rays with flattened dimensions
                rays = rearrange(rays, "b (h w) c -> b c h w", h=H, w=W)  # Convert to (B, C, H, W)

            assert rays.ndim == 4, "rays expected shape is B,C,H,W"
            assert rays.shape[0] == B, "rays batch size must match image batch size"
            padded_rays = F.pad(
                rays,
                (max(0, pad_left), max(0, pad_right), max(0, pad_top), max(0, pad_bottom)),
                value=0.0,
            )

        paddings = (pad_left, pad_right, pad_top, pad_bottom)
        return padded_image, padded_rays, paddings, validity_mask

    def run_camera(self, cls_tokens, original_shapes, rays_gt):
        H, W = original_shapes

        # camera layer
        intrinsics, sh_coeffs = self.angular_module(cls_tokens=cls_tokens)
        B, N = intrinsics.shape
        device = intrinsics.device
        dtype = intrinsics.dtype

        id_coords = coords_grid(B, H, W, device=sh_coeffs.device)

        # This is fov based
        longitude = (
            (id_coords[:, 0] - intrinsics[:, 2].view(-1, 1, 1))
            / W
            * intrinsics[:, 0].view(-1, 1, 1)
        )
        latitude = (
            (id_coords[:, 1] - intrinsics[:, 3].view(-1, 1, 1))
            / H
            * intrinsics[:, 1].view(-1, 1, 1)
        )
        x = torch.cos(latitude) * torch.sin(longitude)
        z = torch.cos(latitude) * torch.cos(longitude)
        y = -torch.sin(latitude)
        unit_sphere = torch.stack([x, y, z], dim=-1)
        unit_sphere = unit_sphere / torch.norm(unit_sphere, dim=-1, keepdim=True).clip(
            min=1e-5
        )

        harmonics = rsh_cart_3(unit_sphere)[..., 1:]  # remove constant-value harmonic
        rays_pred = torch.einsum("bhwc,bcd->bhwd", harmonics, sh_coeffs)
        rays_pred = rays_pred / torch.norm(rays_pred, dim=-1, keepdim=True).clip(
            min=1e-5
        )
        rays_pred = rays_pred.permute(0, 3, 1, 2)

        # `camera_gt` and `pure_gt` are set to False in build(), so the released
        # model always uses the predicted rays; the ground-truth branches below
        # are kept for reference only.
        if self.camera_gt:
            if self.pure_gt:
                rays = rays_gt if rays_gt is not None else rays_pred
            elif rays_gt is not None:
                prob = 1 - tanh(self.steps / self.num_steps)
                where_use_gt_rays = torch.rand(B, 1, 1, device=device, dtype=dtype) < prob
                where_use_gt_rays = where_use_gt_rays.int()
                rays = rays_gt * where_use_gt_rays + rays_pred * (1 - where_use_gt_rays)
            else:
                rays = rays_pred
        else:
            rays = rays_pred
        rays = rearrange(rays, "b c h w -> b h w c") 

        return intrinsics, rays # rays are of shape (B, H, W, C)

    def forward(self, images, rays_gt) -> torch.Tensor:
        B, C, H, W = images.shape
        original_H, original_W = H, W
        device = images.device
        rays_gt = rays_gt

        # pad images and rays up to a multiple of the UniK3D patch size
        padded_images, padded_rays, paddings, validity_mask = self._apply_dynamic_padding(
            image=images,
            rays=rays_gt,
            pad_multiple=self.patch_size,
        )
        (pad_left, pad_right, pad_top, pad_bottom) = paddings 
        images = padded_images

        rays_gt = padded_rays
        B, C, H, W = images.shape
        
        _, tokens = self.pixel_encoder(images) # image goes into the encoder of unik3d and returns features and tokens; we are only intersted in the tokens of the encoder for the angular module. 
        tokens = [
            self.stacking_fn(tokens[i:j]).contiguous()
            for i, j in self.slices_encoder_range
        ] # tokens are stacked up based on encoder ranges like (0 to m)(m to n)(n to p)...

    
        # get cls tokens projections
        camera_tokens = tokens
        camera_tokens = [self.choker(x.contiguous()) for x in camera_tokens]
        camera_tokens = self.camera_token_adapter(camera_tokens) 
        self.angular_module.set_shapes((H, W))
        intrinsics, rays = self.run_camera(
            torch.cat(camera_tokens, dim=1),
            original_shapes=(H, W),
            rays_gt=rays_gt,
        )

        padded_H, padded_W = padded_rays.shape[-2], padded_rays.shape[-1]
        H = padded_H - pad_top - pad_bottom
        W = padded_W - pad_left -pad_right

        # crop the rays back to the original (unpadded) resolution
        required_rays = rays[:,pad_top:pad_top+H, pad_left:pad_left+W, :]

        return intrinsics, required_rays

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"latents_pos"}
    


    def build(self, config):
        """
        defining unik3d's encoder which will take image as input and passes the features to the angular module

        """
        mod = importlib.import_module("unik3d.models.encoder")
        pixel_encoder_factory = getattr(mod, config["model"]["pixel_encoder"]["name"])
        pixel_encoder_config = {
            **config["training"],
            **config["model"]["pixel_encoder"],
            **config["data"],
        }
        pixel_encoder = pixel_encoder_factory(pixel_encoder_config)

        pixel_encoder_embed_dims = (
            pixel_encoder.embed_dims
            if hasattr(pixel_encoder, "embed_dims")
            else [getattr(pixel_encoder, "embed_dim") * 2**i for i in range(4)]
        )
        config["model"]["pixel_encoder"]["embed_dim"] = getattr(
            pixel_encoder, "embed_dim"
        )
        config["model"]["pixel_encoder"]["embed_dims"] = pixel_encoder_embed_dims
        config["model"]["pixel_encoder"]["depths"] = pixel_encoder.depths
        config["model"]["pixel_encoder"]["cls_token_embed_dims"] = getattr(
            pixel_encoder, "cls_token_embed_dims", pixel_encoder_embed_dims
        )
        self.patch_size = config["data"]["augmentations"]["shape_mult"]

        input_dims = config["model"]["pixel_encoder"]["embed_dims"]
        hidden_dim = config["model"]["pixel_decoder"]["hidden_dim"]
        expansion = config["model"]["expansion"]
        num_heads = config["model"]["num_heads"]
        dropout = config["model"]["pixel_decoder"]["dropout"]
        layer_scale = config["model"]["layer_scale"]
        depths_encoder = config["model"]["pixel_encoder"]["depths"]
        self.slices_encoder = list(zip([d - 1 for d in depths_encoder], depths_encoder))
        input_dims = [input_dims[d - 1] for d in depths_encoder]
        self.steps = 0
        self.num_steps = config["model"].get("num_steps", 100000)

        camera_dims = input_dims
        self.choker = GradChoker(config["model"]["pixel_decoder"]["detach"])
        self.input_adapter = ListAdapter(input_dims, hidden_dim)
        self.camera_token_adapter = ListAdapter(camera_dims, hidden_dim)
        self.angular_module = AngularModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            layer_scale=layer_scale,
        )
        self.camera_gt = False
        self.pure_gt = False

        self.pixel_encoder = pixel_encoder
        self.slices_encoder_range = list(
            zip([0, *self.pixel_encoder.depths[:-1]], self.pixel_encoder.depths)
        )
        self.stacking_fn = last_stack
