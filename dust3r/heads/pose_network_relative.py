# Copyright (C) 2026-present Namitha Guruprasad and the CAM3R authors.
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
class RelPoseHeadV2(nn.Module):
    """
    Predicts relative pose (translation + rotation) from decoder features.
    Inspired by NianticLabs' PoseHead, but more modular and supports multiple rotation formats.
    """

    def __init__(self, 
                 in_dim, 
                 patch_size,
                 hidden_dim=256, 
                 mode="last", 
                 num_layers=1,
                 rot_rep="9D",         # 'quat', '6D', or '9D'
                 use_resconv=True):
        super().__init__()

        self.mode = mode
        self.num_layers = num_layers
        self.rot_rep = rot_rep
        self.use_resconv = use_resconv
        self.patch_size = patch_size
        self.out_dim = 4 * patch_size ** 2  # following reference logic

        if self.mode == "last":
            eff_in_dim = in_dim 
        elif self.mode == "multi":
            eff_in_dim = in_dim * self.num_layers

        else:
            raise ValueError(f"Unknown mode {self.mode}")

        # Project token features to pseudo-image space for each view
        self.proj1 = nn.Linear(eff_in_dim, self.out_dim)
        self.proj2 = nn.Linear(eff_in_dim, self.out_dim)

        # Optional 1x1 convolution residual blocks, one per view
        if self.use_resconv:
            self.resconv1 = self._make_resconv()
            self.resconv2 = self._make_resconv()

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final MLP layers before regression
        self.mlp = nn.Sequential(
            nn.Linear(self.out_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Pose regression heads
        self.fc_t = nn.Linear(hidden_dim, 3)

        if self.rot_rep == "6D":
            self.fc_rot = nn.Linear(hidden_dim, 6)
        elif self.rot_rep == "9D":
            self.fc_rot = nn.Linear(hidden_dim, 9)
        else:
            raise ValueError("Unsupported rotation representation.")
        
    def _make_resconv(self):
        return nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, dec1, dec2, img_shape1, img_shape2):
        """
        dec1, dec2: decoder outputs (List of tokens per layer), each [B, S, D]
        img_shape1, img_shape2: Tuple[int, int] (H, W) for view1 and view2
        patch_size: int
        """
        B = dec1[-1].shape[0]
        spatial_dim = self.out_dim   # dynamic output dimension

        # Feature fusion
        if self.mode == "last":
            x1 = dec1[-1].mean(dim=1)  # [B, D]
            x2 = dec2[-1].mean(dim=1)
        elif self.mode == "multi":
            xs1, xs2 = [], []
            for i in range(self.num_layers):
                xs1.append(dec1[-(i + 1)])
                xs2.append(dec2[-(i + 1)])

            x1 = torch.cat(xs1, dim =-1)
            x2 = torch.cat(xs2, dim =-1)


        # Pseudo-image dimensions
        H1, W1 = img_shape1[0]
        H2, W2 = img_shape2[0]

        H1p, W1p = H1 // self.patch_size, W1 // self.patch_size
        H2p, W2p = H2 // self.patch_size, W2 // self.patch_size


        # Project token features to pseudo-images
        f1 = self.proj1(x1)  # [B, S1, C]
        f2 = self.proj2(x2)  # [B, S2, C]
        f1 = f1.transpose(1, 2).reshape(B, self.out_dim, H1p, W1p)  # [B, C, H1', W1']
        f2 = f2.transpose(1, 2).reshape(B, self.out_dim, H2p, W2p)  # [B, C, H2', W2']

        # Residual conv blocks
        if self.use_resconv:
            f1 = self.resconv1(f1)
            f2 = self.resconv2(f2)

        # Global average pooling
        f1 = self.pool(f1).view(B, -1)  # [B, C]
        f2 = self.pool(f2).view(B, -1)  # [B, C]

        # Fuse and regress
        fused = torch.cat([f1, f2], dim=-1)  # [B, 2C]
        feat = self.mlp(fused)

        # Predict translation
        t = self.fc_t(feat)

        # Predict rotation
        if self.rot_rep == "6D":
            R = self.rotation_6d_to_matrix(self.fc_rot(feat))  # [B, 3, 3]
            q = None
        elif self.rot_rep == "9D":
            R = self.svd_orthogonalize(self.fc_rot(feat).view(B, 3, 3))  # [B, 3, 3]
            q = None

        # Create homogeneous transformation matrix [B, 4, 4] without in-place ops
        Rt = torch.cat([R, t.unsqueeze(-1)], dim=-1)  # [B, 3, 4]
        last_row = torch.tensor([0, 0, 0, 1], device=R.device, dtype=R.dtype).view(1, 1, 4).expand(B, -1, -1)  # [B, 1, 4]
        T = torch.cat([Rt, last_row], dim=1)  # [B, 4, 4]


        return {"T": T, "t": t, "R": R, "q": q}

    # === Utility methods ===

    @staticmethod
    def quaternion_to_matrix(q):
        """Convert normalized quaternion [x, y, z, w] to rotation matrix"""
        x, y, z, w = q.unbind(-1)
        B = q.size(0)
        tx, ty, tz = 2*x, 2*y, 2*z
        twx, twy, twz = tx*w, ty*w, tz*w
        txx, txy, txz = tx*x, ty*x, tz*x
        tyy, tyz = ty*y, tz*y
        tzz = tz*z

        rot = torch.stack([
            1 - (tyy + tzz), txy - twz, txz + twy,
            txy + twz, 1 - (txx + tzz), tyz - twx,
            txz - twy, tyz + twx, 1 - (txx + tyy)
        ], dim=-1).view(B, 3, 3)
        return rot

    @staticmethod
    def rotation_6d_to_matrix(d6):
        """Convert 6D rotation representation to 3x3 matrix via Gram-Schmidt"""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-2)


    @staticmethod
    def svd_orthogonalize(m):
        """Convert 9D representation to SO(3) using SVD orthogonalization.

        Args:
          m: [BATCH, 3, 3] 3x3 matrices.

        Returns:
          [BATCH, 3, 3] SO(3) rotation matrices.
        """
        if m.dim() < 3:
            m = m.reshape((-1, 3, 3))
        m_transpose = torch.transpose(torch.nn.functional.normalize(m, p=2, dim=-1), dim0=-1, dim1=-2)
        u, s, v = torch.svd(m_transpose)
        det = torch.det(torch.matmul(v, u.transpose(-2, -1)))
        # Check orientation reflection.
        r = torch.matmul(
            torch.cat([v[:, :, :-1], v[:, :, -1:] * det.view(-1, 1, 1)], dim=2),
            u.transpose(-2, -1)
        )
        return r

