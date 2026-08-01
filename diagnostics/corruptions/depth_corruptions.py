"""Depth corruptions for paired clean/corrupted RGB-D runs.

All functions take depth as torch.Tensor [B,C,H,W] or [B,1,H,W] in model input scale
(typically already normalized RGB-like 3ch depth from the DFormer loader).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def entire_missing(depth: torch.Tensor) -> torch.Tensor:
    """Full depth dropout: zero all depth channels."""
    return torch.zeros_like(depth)


def rectangular_holes(
    depth: torch.Tensor,
    num_holes: int = 3,
    hole_frac: float = 0.15,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Zero out random rectangular blocks (shared across batch if seed set)."""
    out = depth.clone()
    b, c, h, w = out.shape
    g = torch.Generator(device=out.device)
    if seed is not None:
        g.manual_seed(seed)
    for _ in range(num_holes):
        fh = max(1, int(h * hole_frac))
        fw = max(1, int(w * hole_frac))
        # random top-left per sample
        for i in range(b):
            y0 = int(torch.randint(0, max(1, h - fh + 1), (1,), generator=g, device=out.device).item())
            x0 = int(torch.randint(0, max(1, w - fw + 1), (1,), generator=g, device=out.device).item())
            out[i, :, y0 : y0 + fh, x0 : x0 + fw] = 0
    return out


def gaussian_noise(
    depth: torch.Tensor,
    sigma: float = 0.1,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """Additive Gaussian noise; clamp to a reasonable range around input."""
    g = torch.Generator(device=depth.device)
    if seed is not None:
        g.manual_seed(seed)
    noise = torch.randn(depth.shape, generator=g, device=depth.device, dtype=depth.dtype) * sigma
    # depth inputs in this codebase are often [0,1]-ish after norm; keep soft clamp
    return (depth + noise).clamp(depth.min().item() - 3 * sigma, depth.max().item() + 3 * sigma)



def half_missing(depth: torch.Tensor, seed=None) -> torch.Tensor:
    """Zero the bottom half of the depth (held-out spatial missing pattern)."""
    out = depth.clone(); h = out.shape[-2]; out[..., h//2:, :] = 0; return out


def salt_pepper(depth: torch.Tensor, frac: float = 0.2, seed=None) -> torch.Tensor:
    """Randomly zero a fraction of pixels (held-out sparse dropout)."""
    g = torch.Generator(device=depth.device)
    if seed is not None: g.manual_seed(seed)
    mask = torch.rand(depth.shape, generator=g, device=depth.device) < frac
    out = depth.clone(); out[mask] = 0; return out

# ---------------------------------------------------------------------------
# STRUCTURE-type corruptions (added 2026-07-25).
#
# Every corruption above is an AMPLITUDE/OCCLUSION corruption: it varies how much
# depth is present. None of them tests whether depth is present but WRONG in a
# structured way — which is what real deployment failures look like (miscalibration,
# RGB-D misregistration, sensor drift, cross-sensor transfer).
#
# These matter because depth enters DFormerv2's attention ONLY as the relational
# bias mask_d = |d_i - d_j| computed on the NORMALIZED depth at token positions.
# That algebra fixes an invariance profile which these corruptions probe directly:
#   offset   -> |(d_i+c)-(d_j+c)| = |d_i-d_j|  => EXACTLY invariant (predicted no effect)
#   gain     -> scales mask_d by g             => equivalent to an authority change
#   gamma    -> distorts differences unevenly  => NOT invariant (a structure corruption)
#   shift    -> compares depth at wrong pixels => NOT invariant (a structure corruption)
# The normalization constants (mean .48, std .28) match utils/dataloader normalization.
_DN_MEAN, _DN_STD = 0.48, 0.28


def depth_offset(depth: torch.Tensor, delta: float = 0.2, **kw) -> torch.Tensor:
    """Additive bias on depth (sensor zero-point drift). mask_d is offset-invariant,
    so this is predicted to have EXACTLY no effect — a strong falsifiable check that
    the single-path relational reading of the architecture is correct."""
    return depth + delta


def depth_gain(depth: torch.Tensor, gain: float = 1.5, **kw) -> torch.Tensor:
    """Multiplicative scale on depth (sensor gain drift / unit change). Scales mask_d
    by the same factor, so this is an AUTHORITY change, not a structure corruption —
    predicted to behave like the lambda sweep, and to be repaired by authority training."""
    d = depth * _DN_STD + _DN_MEAN
    d = d * gain
    return (d - _DN_MEAN) / _DN_STD


def depth_gamma(depth: torch.Tensor, gamma: float = 1.6, **kw) -> torch.Tensor:
    """Monotone nonlinear distortion of the depth response curve (miscalibration,
    cross-sensor transfer). Preserves ORDER but not relative differences, so mask_d is
    corrupted in structure while remaining 'valid-looking' depth."""
    d = (depth * _DN_STD + _DN_MEAN).clamp(min=1e-4)
    d = d.pow(gamma)
    return (d - _DN_MEAN) / _DN_STD


def depth_shift(depth: torch.Tensor, dx: int = 4, dy: int = 0, **kw) -> torch.Tensor:
    """Spatial misregistration between RGB and depth (extrinsic calibration error,
    rolling shutter, temporal offset on a moving camera). Depth values become attached
    to the wrong image locations, so the relational bias compares the wrong pairs."""
    return torch.roll(depth, shifts=(int(dy), int(dx)), dims=(-2, -1))


def depth_constant(depth: torch.Tensor, value: float = None, **kw) -> torch.Tensor:
    """A globally CONSTANT depth field — what a real sensor returns when the scene is a
    featureless plane, beyond max range, or the frame is wholly invalid. Since
    mask_d = |d_i - d_j|, a constant field gives mask_d == 0 EXACTLY, i.e. it lands on the
    same operator configuration as total absence WITHOUT the depth input being 'missing'
    in any obvious way. Predicted to score identically to entire_missing."""
    v = depth.mean() if value is None else torch.tensor(value, device=depth.device, dtype=depth.dtype)
    return torch.full_like(depth, float(v))


def depth_saturate(depth: torch.Tensor, q: float = 0.5, **kw) -> torch.Tensor:
    """Clip everything above the q-quantile to that value: a depth sensor saturating beyond
    its max range, which flattens the far field into a constant plateau while leaving near
    geometry intact. Partial, spatially-structured degeneracy."""
    thr = torch.quantile(depth.float().flatten(), q).to(depth.dtype)
    return torch.minimum(depth, thr)


def depth_invalid(depth: torch.Tensor, frac: float = 0.95, seed=None, **kw) -> torch.Tensor:
    """Zero a very large fraction of pixels — IR washout in sunlight, or specular/transparent
    scenes where most returns are invalid. Extreme sparse invalidity, as opposed to the mild
    15% used by `holes`."""
    g = torch.Generator(device=depth.device)
    if seed is not None:
        g.manual_seed(seed)
    mask = torch.rand(depth.shape, generator=g, device=depth.device) < frac
    out = depth.clone(); out[mask] = 0; return out


def depth_flip(depth: torch.Tensor, axis: str = "h", **kw) -> torch.Tensor:
    """A REAL depth map with PERFECT depth statistics but ZERO correspondence to the RGB.

    This is the decisive test between two rival accounts of what the relational bias
    contributes:
      (i) GEOMETRIC PRIOR  — the bias is useful because |d_i - d_j| encodes the actual
          scene geometry that the RGB pixels depict. Then destroying correspondence while
          keeping depth-like statistics should be as bad as absence, or worse (actively
          misleading routing).
      (ii) NON-SEMANTIC SCAFFOLD — the bias is useful mainly because it supplies familiar
          attention-logit statistics (locality, dynamic range, edge structure, symmetry
          breaking). Then ANY structured, non-degenerate depth-like field should recover
          most of the benefit, and correspondence barely matters.

    Flipping is the cleanest instrument for this: the flipped map is a genuine depth map of
    a real scene, with identical smoothness, gradient and range statistics, but it describes
    the wrong scene layout for this image.
    """
    return torch.flip(depth, dims=(-1,) if axis == "h" else (-2,))


def depth_smooth_random(depth: torch.Tensor, scale: int = 16, seed=None, **kw) -> torch.Tensor:
    """A smooth random field matched to the depth's mean/std — structured and non-degenerate,
    but carrying no scene information at all. The strongest form of the scaffold test."""
    g = torch.Generator(device=depth.device)
    if seed is not None:
        g.manual_seed(seed)
    h, w = depth.shape[-2], depth.shape[-1]
    lo = torch.rand((1, 1, max(h // scale, 2), max(w // scale, 2)), generator=g, device=depth.device)
    field = torch.nn.functional.interpolate(lo, size=(h, w), mode="bicubic", align_corners=False)
    field = (field - field.mean()) / field.std().clamp(min=1e-6)
    return (field * depth.std() + depth.mean()).expand_as(depth).contiguous()


CORRUPTION_REGISTRY = {
    "entire_missing": lambda d, **kw: entire_missing(d),
    # decisive test: structured depth-like field with NO correspondence to the image
    "depth_flip": lambda d, **kw: depth_flip(d, axis=kw.get("axis", "h")),
    "depth_smooth_random": lambda d, **kw: depth_smooth_random(d, scale=kw.get("scale", 16), seed=kw.get("seed")),
    # realistic GLOBAL-DEGENERACY modes (all can drive mask_d toward 0 without the sensor
    # being absent): featureless plane / beyond max range / IR washout / specular scenes
    "depth_constant": lambda d, **kw: depth_constant(d, value=kw.get("value")),
    "depth_saturate": lambda d, **kw: depth_saturate(d, q=kw.get("q", 0.5)),
    "depth_invalid": lambda d, **kw: depth_invalid(d, frac=kw.get("frac", 0.95), seed=kw.get("seed")),
    "holes": lambda d, **kw: rectangular_holes(
        d, num_holes=kw.get("num_holes", 3), hole_frac=kw.get("hole_frac", 0.15), seed=kw.get("seed")
    ),
    "gaussian": lambda d, **kw: gaussian_noise(d, sigma=kw.get("sigma", 0.1), seed=kw.get("seed")),
    "half_missing": lambda d, **kw: half_missing(d, seed=kw.get("seed")),
    "salt_pepper": lambda d, **kw: salt_pepper(d, frac=kw.get("frac",0.2), seed=kw.get("seed")),
    # structure-type (see note above)
    "depth_offset": lambda d, **kw: depth_offset(d, delta=kw.get("delta", 0.2)),
    "depth_gain": lambda d, **kw: depth_gain(d, gain=kw.get("gain", 1.5)),
    "depth_gamma": lambda d, **kw: depth_gamma(d, gamma=kw.get("gamma", 1.6)),
    "depth_shift": lambda d, **kw: depth_shift(d, dx=kw.get("dx", 4), dy=kw.get("dy", 0)),
}


def apply_corruption(depth: torch.Tensor, name: str, **kwargs) -> torch.Tensor:
    if name not in CORRUPTION_REGISTRY:
        raise KeyError(f"Unknown corruption {name}; choose from {list(CORRUPTION_REGISTRY)}")
    return CORRUPTION_REGISTRY[name](depth, **kwargs)


def default_corruption_suite(seed: int = 0) -> Dict[str, dict]:
    """Names and kwargs used in Phase-0 / D1 first wave."""
    return {
        "entire_missing": {},
        "holes": {"num_holes": 3, "hole_frac": 0.15, "seed": seed},
        "gaussian": {"sigma": 0.15, "seed": seed},
    }
