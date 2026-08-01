"""Layer-wise DFormerv2 forward with cache + sequential activation inject.

Unlike decoder-only stage swap, inject at stage s replaces the *stream* so later
stages are recomputed under corrupted depth with clean intermediate activations.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def forward_backbone_cached(
    backbone: nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
) -> Tuple[List[torch.Tensor], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
    """Return (outs BCHW list, cache stage->(x_out BHWC, x_stream BHWC)).

    x_out: stage token map before extra_norm (as returned by BasicLayer)
    x_stream: input to the *next* layer (post downsample when applicable)
    """
    x = backbone.patch_embed(rgb)
    x_e = depth[:, 0, :, :].unsqueeze(1)
    outs: List[torch.Tensor] = []
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    for i in range(backbone.num_layers):
        layer = backbone.layers[i]
        x_out, x = layer(x, x_e)
        cache[i] = (x_out, x)
        if i in backbone.out_indices:
            xo = x_out
            if i != 0:
                xo = backbone.extra_norms[i - 1](xo)
            outs.append(xo.permute(0, 3, 1, 2).contiguous())
    return outs, cache


@torch.no_grad()
def forward_backbone_inject(
    backbone: nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
    inject_stage: int,
    clean_x_out: torch.Tensor,
    clean_x_stream: torch.Tensor,
) -> List[torch.Tensor]:
    """Corrupted-depth run; after layer `inject_stage`, replace activations with clean
    and *continue* remaining layers so later stages recompute.
    """
    x = backbone.patch_embed(rgb)
    x_e = depth[:, 0, :, :].unsqueeze(1)
    outs: List[torch.Tensor] = []

    for i in range(backbone.num_layers):
        layer = backbone.layers[i]
        x_out, x = layer(x, x_e)
        if i == inject_stage:
            # replace both decoder token map and continuing stream
            x_out = clean_x_out
            x = clean_x_stream
        if i in backbone.out_indices:
            xo = x_out
            if i != 0:
                xo = backbone.extra_norms[i - 1](xo)
            outs.append(xo.permute(0, 3, 1, 2).contiguous())
    return outs


@torch.no_grad()
def outs_from_cache(
    backbone: nn.Module,
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
) -> List[torch.Tensor]:
    """Rebuild decoder outs from a full clean/corr cache (no recompute)."""
    outs = []
    for i in backbone.out_indices:
        x_out, _ = cache[i]
        xo = x_out
        if i != 0:
            xo = backbone.extra_norms[i - 1](xo)
        outs.append(xo.permute(0, 3, 1, 2).contiguous())
    return outs
