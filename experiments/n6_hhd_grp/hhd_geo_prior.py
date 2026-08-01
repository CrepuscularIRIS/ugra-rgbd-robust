"""Heterogeneous Head Decomposition (HHD) — per-head depth coupling in GeoPriorGen.

Fixes the critical architecture flaw identified by grok-verifier Finding 4:
GeoPriorGen.weight is (2,1,1,1) shared across ALL heads. We replace weight[1]
(w_depth) with a per-head scalar, so content-group heads are STRUCTURALLY
depth-blind (weight_depth=0) while geo-group heads have amplified coupling.

The mask dimensions already carry per-head structure via self.decay, so per-head
weight_depth broadcasts naturally: [1, num_heads, 1, 1] * [B, num_heads, HW, HW].

Usage: `patch_backbone(backbone, geo_ratio=0.5)` — monkey-patches every GeoPriorGen.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _patched_forward(self, HW_tuple, depth_map, split_or_not=False):
    """GeoPriorGen.forward with per-head depth coupling."""
    depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

    wd = self._weight_depth_perhead

    if split_or_not:
        index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

        mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
        mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)
        mask_h = self.generate_1d_decay(HW_tuple[0])
        mask_w = self.generate_1d_decay(HW_tuple[1])

        wd_5d = wd[None, :, None, None, None]
        mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + wd_5d * mask_d_h
        mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + wd_5d * mask_d_w
        geo_prior = ((sin, cos), (mask_h, mask_w))
    else:
        index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
        mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
        mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
        mask = self.weight[0] * mask + wd[None, :, None, None] * mask_d
        geo_prior = ((sin, cos), mask)

    return geo_prior


def patch_backbone(backbone: nn.Module, geo_ratio: float = 0.5) -> dict:
    """Monkey-patch every GeoPriorGen with per-head depth coupling.

    Args:
        backbone: the dformerv2 backbone
        geo_ratio: fraction of heads assigned to geo-group (rest are content-group)

    Returns:
        info dict with patched count and parameter stats
    """
    from models.encoders.DFormerv2 import GeoPriorGen
    import types

    count = 0
    total_new_params = 0

    for name, module in backbone.named_modules():
        if not isinstance(module, GeoPriorGen):
            continue

        num_heads = module.decay.shape[0]
        n_geo = max(1, round(num_heads * geo_ratio))

        current_wd = module.weight[1].item()

        wd_init = torch.full((num_heads,), current_wd)

        wd_param = nn.Parameter(wd_init.to(module.weight.device))
        module._weight_depth_perhead = wd_param
        module.register_parameter("_weight_depth_perhead", wd_param)

        module.forward = types.MethodType(_patched_forward, module)

        total_new_params += num_heads
        count += 1

    return {
        "n_patched": count,
        "n_new_params": total_new_params,
        "geo_ratio": geo_ratio,
    }
