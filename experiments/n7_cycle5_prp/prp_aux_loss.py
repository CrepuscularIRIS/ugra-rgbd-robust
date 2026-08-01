#!/usr/bin/env python
"""PRP auxiliary depth-prediction loss (n7).

Scale-and-shift-invariant L1 between the RGB-predicted pseudo-depth field
d-hat and the CLEAN GT inverse-depth, computed on valid pixels only (excludes
zero/invalid GT depth). Invariance is realized by DUAL Z-NORMALIZATION (per
image, both prediction and valid-masked target are normalized to zero-mean /
unit-std before the L1), making the loss free of any global scale+shift. This
is a *training-only* auxiliary; it never touches the frozen eval path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Depth normalization used by the NYU dataloader (TrainPre/ValPre):
#   modal_x = (raw_depth/255 - 0.48) / 0.28   (x_is_single_channel branch)
# so the raw [0,1] depth is recovered by (modal_x * 0.28 + 0.48).clamp(0,1),
# and a raw-0 (invalid/structureless) pixel sits at normalized ~= -1.714, NOT 0.
_DEPTH_MEAN = 0.48
_DEPTH_STD = 0.28


def build_inv_depth_target(modal_x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """From the normalized 3-ch depth batch (b,3,H,W) build a single-channel
    CLEAN inverse-depth target and a validity mask.

    CRITICAL (grok-on-diff fix): modal_x is z-NORMALIZED, so `d > 0` does NOT
    mean "valid pixel" (raw-0 invalid maps to ~= -1.714, and `d>0` would keep
    only above-mean/near pixels). We first DENORMALIZE back to raw [0,1] depth,
    mask on denorm > eps (raw-0 -> invalid), then invert the DENORMED depth.
    Returns target (b,1,H,W) and valid mask (b,1,H,W) bool.
    """
    dn = modal_x[:, 0:1]                                   # (b,1,H,W) normalized
    depth01 = (dn * _DEPTH_STD + _DEPTH_MEAN).clamp(0.0, 1.0)  # raw [0,1] depth
    valid = depth01 > 1e-3                                 # raw-0 == invalid/structureless
    inv = 1.0 / (depth01 + 1e-3)                           # inverse depth; near objects -> large
    return inv, valid


def si_l1_depth_loss(
    d_hat: torch.Tensor, modal_x: torch.Tensor
) -> torch.Tensor:
    """Scale-and-shift-invariant L1 between d_hat (b,1,h,w) and the clean
    inverse-depth target.

    d_hat is at patch-embed token resolution; the target is downsampled to
    match. Invariance is realized by dual z-normalization: per image, BOTH the
    prediction and the (valid-masked) target are normalized to zero-mean /
    unit-std before the L1. This makes the loss free of any global scale+shift
    (a relational/ordinal depth criterion), which is what PRP needs since the
    head carries its own learnable scale. (Not classical median-SI-L1; dual-z
    is the honest description of the code below.)
    """
    inv, valid = build_inv_depth_target(modal_x)
    h, w = d_hat.shape[-2:]
    inv_ds = F.interpolate(inv, size=(h, w), mode="bilinear", align_corners=False)
    # nearest for the mask so we don't fabricate validity on interpolated edges
    valid_ds = (
        F.interpolate(valid.float(), size=(h, w), mode="nearest") > 0.5
    )

    # per-image scale normalization of the target so SI-L1 is well-posed even
    # though d_hat carries its own learnable scale: z-normalize the target per
    # image over valid pixels (mean/std), matching to d_hat's z-stats.
    b = d_hat.shape[0]
    losses = []
    for i in range(b):
        m = valid_ds[i, 0]
        if m.sum() < 16:
            continue
        p = d_hat[i, 0][m]
        t = inv_ds[i, 0][m]
        # normalize both to zero-mean unit-std (SI to global scale+shift)
        p = (p - p.mean()) / (p.std() + 1e-6)
        t = (t - t.mean()) / (t.std() + 1e-6)
        losses.append((p - t).abs().mean())
    if not losses:
        return d_hat.sum() * 0.0
    return torch.stack(losses).mean()
