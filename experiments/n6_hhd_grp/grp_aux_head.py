"""Geometric Relation Prediction (GRP) auxiliary head.

Attaches to end-of-stage-2 content features and predicts depth geometry:
1. 8-level quantized depth classification (CE loss)
2. Local relative-order ranking on adjacent tokens (margin ranking loss)

Target = original depth. GeoPriorGen depth = augmented (anti-shortcut).
Head discarded at inference — zero deployed cost.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPHead(nn.Module):
    """Auxiliary depth-geometry prediction head."""

    def __init__(self, in_dim: int, n_bins: int = 8):
        super().__init__()
        self.n_bins = n_bins
        self.head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 1, bias=False),
            nn.GroupNorm(8, in_dim // 2),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, n_bins, 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: [B, H, W, C] (RGBD_Block output format) → [B, n_bins, H, W]"""
        x = feat.permute(0, 3, 1, 2).contiguous()
        return self.head(x)


def quantize_depth_target(depth: torch.Tensor, n_bins: int = 8) -> torch.Tensor:
    """Convert raw depth [B, 1, H, W] to bin indices [B, H, W] in [0, n_bins-1].

    Per-sample normalization to handle varying depth ranges.
    Invalid (zero) pixels get target = -1 (ignored in CE loss).
    """
    B = depth.shape[0]
    target = torch.full((B, depth.shape[2], depth.shape[3]),
                        -1, dtype=torch.long, device=depth.device)
    for b in range(B):
        d = depth[b, 0]
        valid = d > 1e-6
        if valid.sum() < 2:
            continue
        vmin, vmax = d[valid].min(), d[valid].max()
        if vmax - vmin < 1e-6:
            target[b][valid] = 0
            continue
        normalized = (d[valid] - vmin) / (vmax - vmin)
        bins = (normalized * (n_bins - 1)).round().long().clamp(0, n_bins - 1)
        target[b][valid] = bins
    return target


def grp_loss(
    logits: torch.Tensor,
    depth_target: torch.Tensor,
    lambda_bin: float = 0.2,
    lambda_rank: float = 0.1,
) -> torch.Tensor:
    """Compute GRP auxiliary loss.

    Args:
        logits: [B, n_bins, H, W] from GRPHead
        depth_target: [B, 1, H, W] original (unaugmented) depth for target computation
    """
    n_bins = logits.shape[1]
    target = quantize_depth_target(depth_target, n_bins)

    target_resized = F.interpolate(
        target.unsqueeze(1).float(), size=logits.shape[2:],
        mode="nearest",
    ).squeeze(1).long()

    loss_bin = F.cross_entropy(logits, target_resized, ignore_index=-1)

    if lambda_rank > 0:
        loss_rank = _ranking_loss(logits, target_resized, n_bins)
    else:
        loss_rank = torch.zeros((), device=logits.device)

    return lambda_bin * loss_bin + lambda_rank * loss_rank


def _ranking_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    n_bins: int,
    margin_per_bin: float = 0.5,
) -> torch.Tensor:
    """Local relative-order ranking loss on adjacent spatial tokens."""
    bin_values = torch.arange(n_bins, dtype=logits.dtype, device=logits.device)
    expected = (F.softmax(logits, dim=1) * bin_values[None, :, None, None]).sum(dim=1)

    valid = target >= 0

    left = expected[:, :, :-1]
    right = expected[:, :, 1:]
    tgt_left = target[:, :, :-1]
    tgt_right = target[:, :, 1:]
    mask = (tgt_left >= 0) & (tgt_right >= 0) & ((tgt_left - tgt_right).abs() >= 1)

    if mask.sum() < 1:
        return torch.zeros((), device=logits.device)

    diff_pred = left[mask] - right[mask]
    diff_tgt = (tgt_left[mask] - tgt_right[mask]).float()
    sign = diff_tgt.sign()

    loss = F.relu(margin_per_bin - sign * diff_pred).mean()
    return loss
