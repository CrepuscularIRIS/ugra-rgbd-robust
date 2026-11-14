"""Availability sampling for depth-availability training (DAT).

Implements Sec. 3.4 and Algorithm 1 of the paper. Every training image receives exactly one of three
depth conditions, sampled with probabilities (complete, missing, partial) = (0.50, 0.25, 0.25):

    complete : M = 1                      depth is kept
    missing  : M = 0                      depth is replaced by the empty-input constant c
    partial  : M ~ Q_p                    depth is removed in one rectangle or in several separated rectangles

and the depth that enters the network is  D~ = M * D + (1 - M) * c   (Eq. 2).
The RGB image and the labels are never touched, no input channel is added, and nothing in the
network changes: DAT only changes which depth the network sees during training.

`c` is the empty-input value of the host architecture (0 after standardization for DFormerv1/v2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch

COMPLETE, MISSING, PARTIAL = "complete", "missing", "partial"
CONDITIONS = (COMPLETE, MISSING, PARTIAL)


@dataclass(frozen=True)
class Mixture:
    """Sampling probabilities of the three availability conditions."""

    complete: float = 0.50
    missing: float = 0.25
    partial: float = 0.25

    def __post_init__(self):
        p = self.as_tuple()
        if min(p) < 0 or abs(sum(p) - 1.0) > 1e-9:
            raise ValueError(f"mixture must be non-negative and sum to 1, got {p}")

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.complete, self.missing, self.partial)


@dataclass(frozen=True)
class PartialMaskSpec:
    """Distribution Q_p of partial masks.

    The paper specifies the two shapes (one rectangle, or several spatially separated rectangles) but not
    the size distribution; the defaults below are this implementation's choice (removed area drawn
    uniformly from `area`, aspect ratio log-uniform in `aspect`, 2-4 regions for the multi-region shape).
    """

    kinds: tuple[str, ...] = ("rect", "multi")  # chosen uniformly
    area: tuple[float, float] = (0.10, 0.50)  # removed fraction of the image, in total
    aspect: tuple[float, float] = (0.5, 2.0)  # width / height of one rectangle
    n_regions: tuple[int, int] = (2, 4)  # inclusive, for "multi"


def _rect(h: int, w: int, area_frac: float, aspect: float, gen: torch.Generator) -> tuple[int, int, int, int]:
    """One rectangle (top, left, height, width) covering about `area_frac` of an h x w image."""
    area = area_frac * h * w
    rh = int(round(min(h, max(1, math.sqrt(area / aspect)))))
    rw = int(round(min(w, max(1, math.sqrt(area * aspect)))))
    top = int(torch.randint(0, h - rh + 1, (1,), generator=gen))
    left = int(torch.randint(0, w - rw + 1, (1,), generator=gen))
    return top, left, rh, rw


def _uniform(lo: float, hi: float, gen: torch.Generator) -> float:
    return lo + (hi - lo) * float(torch.rand(1, generator=gen))


def sample_partial_mask(h: int, w: int, spec: PartialMaskSpec, gen: torch.Generator) -> torch.Tensor:
    """Binary (h, w) float mask; 1 keeps depth, 0 removes it."""
    keep = torch.ones(h, w)
    kind = spec.kinds[int(torch.randint(0, len(spec.kinds), (1,), generator=gen))]
    total = _uniform(*spec.area, gen)
    n = 1 if kind == "rect" else int(torch.randint(spec.n_regions[0], spec.n_regions[1] + 1, (1,), generator=gen))
    for _ in range(n):
        for _try in range(20):  # keep the regions separated when possible
            aspect = math.exp(_uniform(math.log(spec.aspect[0]), math.log(spec.aspect[1]), gen))
            t, l, rh, rw = _rect(h, w, total / n, aspect, gen)
            if keep[t:t + rh, l:l + rw].all() or _try == 19:
                keep[t:t + rh, l:l + rw] = 0.0
                break
    return keep


def sample_conditions(n: int, mixture: Mixture, gen: torch.Generator) -> list[str]:
    """Draw one condition per image (Algorithm 1, lines 2-5)."""
    u = torch.rand(n, generator=gen)
    c0, c1, _ = mixture.as_tuple()
    return [COMPLETE if v < c0 else MISSING if v < c0 + c1 else PARTIAL for v in u.tolist()]


def condition_masks(conditions: Sequence[str], h: int, w: int, spec: PartialMaskSpec,
                    gen: torch.Generator) -> torch.Tensor:
    """(n, 1, h, w) availability masks for the given conditions."""
    masks = []
    for cond in conditions:
        if cond == COMPLETE:
            masks.append(torch.ones(h, w))
        elif cond == MISSING:
            masks.append(torch.zeros(h, w))
        elif cond == PARTIAL:
            masks.append(sample_partial_mask(h, w, spec, gen))
        else:
            raise ValueError(f"unknown availability condition: {cond!r}")
    return torch.stack(masks).unsqueeze(1)


def apply_availability(depth: torch.Tensor, mask: torch.Tensor, c: float) -> torch.Tensor:
    """Eq. (2): D~ = M * D + (1 - M) * c. `mask` broadcasts over the depth channels."""
    mask = mask.to(device=depth.device, dtype=depth.dtype)
    return mask * depth + (1.0 - mask) * c


class AvailabilitySampler:
    """Callable that turns a batch of depth maps into DAT training inputs.

    >>> sampler = AvailabilitySampler(c=0.0, seed=0)
    >>> depth_tilde, conditions = sampler(depth)        # depth: (B, C, H, W), standardized
    """

    def __init__(self, c: float, mixture: Mixture = Mixture(), spec: PartialMaskSpec = PartialMaskSpec(),
                 seed: int | None = None):
        self.c, self.mixture, self.spec = c, mixture, spec
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)

    def __call__(self, depth: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        b, _, h, w = depth.shape
        conditions = sample_conditions(b, self.mixture, self.gen)
        masks = condition_masks(conditions, h, w, self.spec, self.gen)
        return apply_availability(depth, masks, self.c), conditions


def missing_depth(depth: torch.Tensor, c: float) -> torch.Tensor:
    """The missing-input condition at test time: the constant map c * 1 in place of depth."""
    return torch.full_like(depth, c)


def partial_depth(depth: torch.Tensor, c: float, fraction: float, kind: str = "rect",
                  seed: int = 0) -> torch.Tensor:
    """Partially missing depth at test time with a fixed removed `fraction`, reproducible from `seed`."""
    gen = torch.Generator().manual_seed(seed)
    spec = PartialMaskSpec(kinds=(kind,), area=(fraction, fraction))
    b, _, h, w = depth.shape
    masks = condition_masks([PARTIAL] * b, h, w, spec, gen)
    return apply_availability(depth, masks, c)
