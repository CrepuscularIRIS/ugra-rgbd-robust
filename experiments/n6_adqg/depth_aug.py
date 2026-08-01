"""Relational depth augmentation for ADQG training.

Applies random depth transformations that preserve relational structure
(Q5: invert, quantize-8 are OK; edge-only and structureless are BAD).

These augmentations differ from the eval corruptions (zeros, holes, gaussian)
so the model learns generalizable depth-quality handling, not eval-condition
memorization.
"""

from __future__ import annotations

import torch


def augment_depth(
    depth: torch.Tensor,
    p_aug: float = 0.3,
    seed: int | None = None,
) -> torch.Tensor:
    """Randomly augment depth during training.

    Args:
        depth: [B, C, H, W] depth tensor (already normalized)
        p_aug: probability of applying any augmentation per sample
        seed: optional seed for reproducibility

    Returns:
        Augmented depth tensor (same shape)
    """
    if seed is not None:
        g = torch.Generator(device=depth.device)
        g.manual_seed(seed)
    else:
        g = None

    B = depth.shape[0]
    apply_mask = torch.rand(B, device=depth.device, generator=g) < p_aug
    if not apply_mask.any():
        return depth

    out = depth.clone()
    aug_type = torch.randint(0, 3, (B,), device=depth.device, generator=g)

    for b in range(B):
        if not apply_mask[b]:
            continue
        t = aug_type[b].item()
        if t == 0:
            out[b] = _invert(out[b])
        elif t == 1:
            out[b] = _quantize(out[b], levels=8)
        else:
            out[b] = _add_mild_noise(out[b], std=0.1, generator=g)

    return out


def _invert(x: torch.Tensor) -> torch.Tensor:
    """Invert depth values: x -> max + min - x (preserves relational structure)."""
    vmin, vmax = x.min(), x.max()
    if vmax - vmin < 1e-6:
        return x
    return vmax + vmin - x


def _quantize(x: torch.Tensor, levels: int = 8) -> torch.Tensor:
    """Quantize to N levels (preserves coarse relational structure)."""
    vmin, vmax = x.min(), x.max()
    if vmax - vmin < 1e-6:
        return x
    normalized = (x - vmin) / (vmax - vmin)
    quantized = torch.round(normalized * (levels - 1)) / (levels - 1)
    return quantized * (vmax - vmin) + vmin


def _add_mild_noise(
    x: torch.Tensor,
    std: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add mild uniform noise (different distribution from gaussian eval corruption)."""
    noise = torch.empty_like(x).uniform_(-std, std, generator=generator)
    return x + noise
