"""RGB Decomposition Adapter + Ordinal Depth Witness for DFormerv2-S (n7).

Inserts after the backbone's stage-2 (256-D) output:
  1. RGB Decomposition Adapter: 3x64-ch subspaces (local/mid-range/global)
     -> concat 192-ch -> project to 256 -> add via zero-init gate.
  2. Training-only auxiliary heads (discarded at inference):
     a. RGB-only 40-way segmentation head on 192-ch (loss weight 0.2)
     b. Ordinal depth witness head on 256-D features: 8-bin depth + 2-ch Sobel normals
        (CE + cosine, weight 0.05, stop-grad to depth encoder path)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RGBDecompAdapter(nn.Module):

    def __init__(self, in_dim=256, sub_dim=64, num_classes=40):
        super().__init__()
        self.in_dim = in_dim
        self.sub_dim = sub_dim

        self.local_conv = nn.Sequential(
            nn.Conv2d(in_dim, sub_dim, 3, padding=1),
            nn.BatchNorm2d(sub_dim),
            nn.GELU(),
        )
        self.midrange_d2 = nn.Sequential(
            nn.Conv2d(in_dim, sub_dim, 3, padding=2, dilation=2),
            nn.BatchNorm2d(sub_dim),
            nn.GELU(),
        )
        self.midrange_d4 = nn.Sequential(
            nn.Conv2d(sub_dim, sub_dim, 3, padding=4, dilation=4, groups=sub_dim),
            nn.BatchNorm2d(sub_dim),
            nn.GELU(),
        )
        self.global_pool_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(in_dim, sub_dim),
            nn.GELU(),
        )
        self.global_expand = nn.Conv2d(sub_dim, sub_dim, 1)

        self.merge_proj = nn.Conv2d(sub_dim * 3, in_dim, 1)
        self.gate = nn.Parameter(torch.zeros(1, in_dim, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.local_conv(x)

        mid_feat = self.midrange_d2(x)
        mid_feat = self.midrange_d4(mid_feat)

        global_vec = self.global_pool_proj(x)
        global_feat = self.global_expand(global_vec.unsqueeze(-1).unsqueeze(-1)).expand(-1, -1, H, W)

        concat = torch.cat([local_feat, mid_feat, global_feat], dim=1)
        merged = self.merge_proj(concat)
        out = x + self.gate * merged
        return out, concat


class DepthWitnessHead(nn.Module):

    def __init__(self, in_dim=256, hidden_dim=128, n_bins=8):
        super().__init__()
        self.n_bins = n_bins
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, n_bins + 2, 1),
        )

    def forward(self, x):
        out = self.head(x)
        bin_logits = out[:, :self.n_bins]
        normal_pred = out[:, self.n_bins:]
        return bin_logits, normal_pred


class RGBSegHead(nn.Module):

    def __init__(self, in_dim=192, num_classes=40):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1, groups=in_dim),
            nn.BatchNorm2d(in_dim),
            nn.GELU(),
            nn.Conv2d(in_dim, num_classes, 1),
        )

    def forward(self, x):
        return self.head(x)
