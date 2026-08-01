"""Compute ordinal depth witness targets on-the-fly from the normalized depth tensor.

The depth input (modal_x) is a 3-ch tensor normalized with ImageNet stats. We take channel 0,
reverse the normalization to get relative depth in [0,1], then compute:
  1. 8-bin ordinal labels (quantize into 8 equal-width bins)
  2. 2-ch Sobel-like gradient (dx, dy) as surface normal proxy
"""
import torch
import torch.nn.functional as F


DEPTH_MEAN = 0.485
DEPTH_STD = 0.229


def compute_witness_targets(modal_x, n_bins=8):
    depth_1ch = modal_x[:, 0:1]
    depth_raw = depth_1ch * DEPTH_STD + DEPTH_MEAN
    depth_raw = depth_raw.clamp(0, 1)

    valid_mask = (depth_raw > 1e-4).float()

    bin_edges = torch.linspace(0, 1, n_bins + 1, device=depth_raw.device)
    bin_labels = torch.bucketize(depth_raw, bin_edges[1:-1]).squeeze(1).long()
    bin_labels = bin_labels.clamp(0, n_bins - 1)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=depth_raw.dtype, device=depth_raw.device).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=depth_raw.dtype, device=depth_raw.device).reshape(1, 1, 3, 3)
    dx = F.conv2d(depth_raw, sobel_x, padding=1)
    dy = F.conv2d(depth_raw, sobel_y, padding=1)
    normals = torch.cat([dx, dy], dim=1)

    return bin_labels, normals, valid_mask.squeeze(1)
