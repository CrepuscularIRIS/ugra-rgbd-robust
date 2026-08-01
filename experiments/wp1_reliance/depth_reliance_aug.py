"""Depth-reliance-reduction augmentation (WP1 / P1) — operates on NORMALIZED depth.

All transforms act on the model-input depth tensor (normalized 3-ch, mean .48/std .28,
as produced by the DFormer loader) — the same tensor the eval-time corruption registry
uses — so train-time and eval-time "missing" are byte-consistent.

Anchors (measured, GOAL_SCORE_PUSH / Phase 1-4):
- Q1b: structureless depth (zeros) == entire_missing == prior-off (~0.51) -> modality
  dropout to zeros is the exact deployment failure condition.
- Q5 clean-equivalent transforms (<=0.6 pt on clean): inverted, quantize-8 (and clean).
  PARTIAL: rank-order, abs-centered, quantize-4. BAD: edge-only, piecewise, structureless.
  So the relational-preserving aug set is restricted to {invert, quantize8, mild-noise}.
- Q1a: S depth coupling is front-loaded (stage0 3.86 / stage2 1.50) -> reducing reliance
  via input-level dropout/aug pressures exactly where the coupling lives.

Affine-normalization note: raw depth d is mapped to x=(d/255-mean)/std (affine). A midpoint
reflection x -> (x.max()+x.min())-x equals the raw inversion (max+min-d) because affine maps
commute with reflection about the interval midpoint; likewise uniform quantization in
normalized space equals uniform quantization of raw levels. So we implement invert/quantize
directly in normalized space (no de/re-normalize round-trip).
"""

from __future__ import annotations

from typing import Optional

import torch


def structureless(depth: torch.Tensor) -> torch.Tensor:
    """Modality dropout target = eval entire_missing = zeros_like(normalized depth)."""
    return torch.zeros_like(depth)


def invert(depth: torch.Tensor) -> torch.Tensor:
    """Midpoint reflection == raw depth inversion (max+min-d). Q5: ~clean-equivalent."""
    lo = depth.amin()
    hi = depth.amax()
    return (hi + lo) - depth


def quantize(depth: torch.Tensor, levels: int = 8) -> torch.Tensor:
    """Uniform quantization into `levels` bins across the tensor range. Q5 q8 ~ clean."""
    lo = depth.amin()
    hi = depth.amax()
    rng = (hi - lo).clamp_min(1e-6)
    norm = (depth - lo) / rng
    q = torch.round(norm * (levels - 1)) / (levels - 1)
    return q * rng + lo


def mild_noise(
    depth: torch.Tensor, sigma: float = 0.08, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """Additive gaussian in normalized space; mild (Q5: sigma 0.15 cost clean only ~0.6)."""
    noise = torch.randn(depth.shape, generator=generator, device=depth.device, dtype=depth.dtype)
    return depth + noise * sigma


class DepthRelianceAug:
    """Per-sample stochastic depth transform for reliance-reduction finetuning.

    For each sample independently:
      - with prob p_drop  -> structureless (zeros)  [modality dropout; == eval missing]
      - else with prob p_aug -> a relational-preserving transform drawn from {invert,
        quantize8, mild_noise} (Q5 OK-set)
      - else -> identity (clean depth kept, protecting clean mIoU)

    Operates on a batched NORMALIZED depth tensor [B, C, H, W]. Transforms that use amin/amax
    are computed per-sample so the relational field is preserved per image.
    """

    def __init__(
        self,
        p_drop: float = 0.3,
        p_aug: float = 0.5,
        noise_sigma: float = 0.08,
        quant_levels: int = 8,
        aug_weights=(1.0, 1.0, 1.0),  # invert, quantize8, mild_noise
        seed: Optional[int] = None,
    ):
        assert 0.0 <= p_drop <= 1.0 and 0.0 <= p_aug <= 1.0
        self.p_drop = p_drop
        self.p_aug = p_aug
        self.noise_sigma = noise_sigma
        self.quant_levels = quant_levels
        self.aug_weights = torch.tensor(aug_weights, dtype=torch.float32)
        self._seed = seed

    def _gen(self, device):
        g = torch.Generator(device=device)
        if self._seed is not None:
            g.manual_seed(self._seed)
        return g

    @torch.no_grad()
    def __call__(self, depth: torch.Tensor) -> torch.Tensor:
        """Return an augmented copy of the batched normalized depth (no in-place)."""
        b = depth.shape[0]
        device = depth.device
        g = self._gen(device)
        out = depth.clone()
        u = torch.rand(b, generator=g, device=device)
        aug_probs = (self.aug_weights / self.aug_weights.sum()).to(device)
        for i in range(b):
            ui = float(u[i])
            if ui < self.p_drop:
                out[i] = structureless(depth[i])
            elif ui < self.p_drop + (1.0 - self.p_drop) * self.p_aug:
                choice = int(torch.multinomial(aug_probs, 1, generator=g).item())
                di = depth[i]
                if choice == 0:
                    out[i] = invert(di)
                elif choice == 1:
                    out[i] = quantize(di, self.quant_levels)
                else:
                    out[i] = mild_noise(di, self.noise_sigma, g)
            # else identity (clean depth retained)
        return out

    def describe(self) -> dict:
        return {
            "p_drop": self.p_drop,
            "p_aug": self.p_aug,
            "noise_sigma": self.noise_sigma,
            "quant_levels": self.quant_levels,
            "aug_weights": self.aug_weights.tolist(),
            "aug_set": ["invert", "quantize8", "mild_noise"],
            "anchor": "Q1b structureless==missing; Q5 invert/quantize8 clean-equivalent",
        }
