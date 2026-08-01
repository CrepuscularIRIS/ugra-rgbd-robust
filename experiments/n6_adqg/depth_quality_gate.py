"""Adaptive Depth Quality Gate (ADQG) — modifies GeoPriorGen with a per-sample
depth-quality gate that modulates w_depth.

Anchors: Q1a (S w_depth static 1.45), Q1c (L headroom), Q4 (stage-2 accumulation),
Q5 (relational depth augmentation set).

The gate is a lightweight 2→16→1 MLP on depth-map [mean, std] statistics,
producing a per-sample scaling factor in [0, 1] for w_depth.  Under clean depth
the gate outputs ~1 (normal coupling); under degraded/missing depth the std drops
toward zero and the gate attenuates w_depth, limiting error propagation.

Usage: monkey-patch every GeoPriorGen in a DFormerv2 backbone with `patch_backbone()`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DepthQualityGate(nn.Module):
    """Per-sample depth quality gate.

    Input:  depth_map [B, 1, H, W] (after interpolation in GeoPriorGen.forward)
    Output: gate [B, 1, 1, 1] in [0, 1]
    """

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        self._init_to_passthrough()

    def _init_to_passthrough(self):
        nn.init.zeros_(self.fc[0].weight)
        nn.init.zeros_(self.fc[0].bias)
        nn.init.zeros_(self.fc[2].weight)
        self.fc[2].bias.data.fill_(4.0)

    def forward(self, depth_map: torch.Tensor) -> torch.Tensor:
        B = depth_map.shape[0]
        flat = depth_map.reshape(B, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        stats = torch.cat([mean, std], dim=1)
        g = torch.sigmoid(self.fc(stats))
        return g.view(B, 1, 1, 1)


def _patched_forward(self, HW_tuple, depth_map, split_or_not=False):
    """Replacement forward for GeoPriorGen that applies the depth quality gate."""
    import torch.nn.functional as F
    depth_map = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)

    gate = self._dq_gate(depth_map)

    if split_or_not:
        w_depth = (self.weight[1] * gate).unsqueeze(-1)

        index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)

        mask_d_h = self.generate_1d_depth_decay(HW_tuple[0], HW_tuple[1], depth_map.transpose(-2, -1))
        mask_d_w = self.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], depth_map)
        mask_h = self.generate_1d_decay(HW_tuple[0])
        mask_w = self.generate_1d_decay(HW_tuple[1])

        mask_h = self.weight[0] * mask_h.unsqueeze(0).unsqueeze(2) + w_depth * mask_d_h
        mask_w = self.weight[0] * mask_w.unsqueeze(0).unsqueeze(2) + w_depth * mask_d_w
        geo_prior = ((sin, cos), (mask_h, mask_w))
    else:
        w_depth = self.weight[1] * gate

        index = torch.arange(HW_tuple[0] * HW_tuple[1]).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        sin = sin.reshape(HW_tuple[0], HW_tuple[1], -1)
        cos = torch.cos(index[:, None] * self.angle[None, :])
        cos = cos.reshape(HW_tuple[0], HW_tuple[1], -1)
        mask = self.generate_pos_decay(HW_tuple[0], HW_tuple[1])
        mask_d = self.generate_depth_decay(HW_tuple[0], HW_tuple[1], depth_map)
        mask = self.weight[0] * mask + w_depth * mask_d
        geo_prior = ((sin, cos), mask)

    return geo_prior


def patch_backbone(backbone: nn.Module) -> int:
    """Monkey-patch every GeoPriorGen in the backbone with a DepthQualityGate.

    Returns the number of patched modules.
    """
    from models.encoders.DFormerv2 import GeoPriorGen
    count = 0
    for name, module in backbone.named_modules():
        if isinstance(module, GeoPriorGen):
            gate = DepthQualityGate()
            gate = gate.to(module.weight.device)
            module._dq_gate = gate
            module.add_module("_dq_gate", gate)
            import types
            module.forward = types.MethodType(_patched_forward, module)
            count += 1
    return count
