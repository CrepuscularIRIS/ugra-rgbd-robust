#!/usr/bin/env python3
"""Measure how much each corruption perturbs `mask_d`, the ONLY channel depth uses.

Step 5/6 of the cycle. The correspondence axis `c` was ASSIGNED by rule and failed at low c
(predicted 0.83, observed 1.84). Boundary length cannot replace it: `entire_missing` has no
boundary at all (no valid pixels remain) yet does maximal damage. So the variable must be
MEASURED from the signal, not assigned from geometry.

DFormerv2 uses depth only through the pairwise bias `mask_d(i,j) = |d_i - d_j|` inside local
attention. So the quantity that should govern damage is how far a corruption moves `mask_d`
from its clean value, averaged over the pairs the operator actually forms.

DEFINED BEFORE ANY DAMAGE NUMBER IS CONSULTED:

    P = mean over local pairs (i,j) of  | mask_d_corrupt(i,j) - mask_d_clean(i,j) |
        divided by  mean over local pairs of  mask_d_clean(i,j)

  i.e. the relative perturbation of the depth bias. Pairs are taken within a w x w window,
  which is what "local attention" means here. P = 0 for an untouched map; P = 1 when the
  bias is displaced by as much as its own clean magnitude.

This is not fitted to anything. It is a property of (corruption, dataset), computed on CPU
from real NYU depth maps with the FROZEN corruption functions.

    python3 scripts/maskd_perturbation.py --n 24 --window 7
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
from depth_corruptions import (  # noqa: E402
    rectangular_holes, depth_invalid, depth_flip, gaussian_noise,
)

DFORMER = PROJECT / "repos" / "DFormer"
NYU = DFORMER / "datasets" / "NYUDepthv2"
# normalisation the training/eval path applies to depth
MEAN, STD = 0.48, 0.28


def load_depths(n: int):
    """n normalized depth maps from the held-out split, in the eval's own convention."""
    # the manifest gives RGB and Label; depth is Depth/<stem>.png (config x_root_folder),
    # 4-channel uint8 with x_is_single_channel=True -> the loader uses one channel.
    import cv2
    lines = [l.strip() for l in (NYU / "test.txt").read_text().splitlines() if l.strip()][:n]
    out = []
    for l in lines:
        stem = Path(l.split()[0]).stem
        p = NYU / "Depth" / f"{stem}.png"
        if not p.exists():
            continue
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None:
            continue
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.float32)
        d = d / (d.max() + 1e-8)                      # to [0,1] as the loader does
        d = (d - MEAN) / STD                          # normalized depth
        out.append(torch.from_numpy(d)[None, None])
    return out


def mask_d_stats(d: torch.Tensor, w: int):
    """mean |d_i - d_j| over pairs within a w x w window, via shifted differences."""
    r = w // 2
    tot = torch.zeros(())
    cnt = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            a = d[:, :, max(0, dy):d.shape[2] + min(0, dy), max(0, dx):d.shape[3] + min(0, dx)]
            b = d[:, :, max(0, -dy):d.shape[2] + min(0, -dy), max(0, -dx):d.shape[3] + min(0, -dx)]
            tot = tot + (a - b).abs().sum()
            cnt += a.numel()
    return tot / cnt


def mask_d_delta(clean: torch.Tensor, corr: torch.Tensor, w: int):
    """mean | mask_d_corrupt - mask_d_clean | over the same local pairs."""
    r = w // 2
    tot = torch.zeros(())
    cnt = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            def sh(t, dy, dx):
                a = t[:, :, max(0, dy):t.shape[2] + min(0, dy), max(0, dx):t.shape[3] + min(0, dx)]
                b = t[:, :, max(0, -dy):t.shape[2] + min(0, -dy), max(0, -dx):t.shape[3] + min(0, -dx)]
                return (a - b).abs()
            tot = tot + (sh(corr, dy, dx) - sh(clean, dy, dx)).abs().sum()
            cnt += sh(clean, dy, dx).numel()
    return tot / cnt


CONDITIONS = {
    # label: (function, kwargs)  -- FROZEN corruption implementations, unchanged
    "holes_3x0.15":      (rectangular_holes, {"num_holes": 3, "hole_frac": 0.15}),
    "holes_1blob":       (rectangular_holes, {"num_holes": 1, "hole_frac": 0.316}),
    "holes_2blob":       (rectangular_holes, {"num_holes": 2, "hole_frac": 0.224}),
    "invalid_0.10":      (depth_invalid, {"frac": 0.10}),
    "invalid_0.25":      (depth_invalid, {"frac": 0.25}),
    "invalid_0.55":      (depth_invalid, {"frac": 0.55}),
    "invalid_0.70":      (depth_invalid, {"frac": 0.70}),
    "gaussian_0.15":     (gaussian_noise, {"sigma": 0.15}),
    "depth_flip":        (depth_flip, {"axis": "h"}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--window", type=int, default=7)
    a = ap.parse_args()
    ds = load_depths(a.n)
    if not ds:
        raise SystemExit("no depth maps loaded — cannot proceed")
    print(f"loaded {len(ds)} depth maps, window {a.window}x{a.window}\n")
    base = float(np.mean([mask_d_stats(d, a.window).item() for d in ds]))
    print(f"clean mean mask_d = {base:.4f}\n")
    res = {}
    for lab, (fn, kw) in CONDITIONS.items():
        vals = []
        for i, d in enumerate(ds):
            c = fn(d.clone(), seed=20260721, **kw) if "seed" in fn.__code__.co_varnames else fn(d.clone(), **kw)
            vals.append(mask_d_delta(d, c, a.window).item())
        P = float(np.mean(vals)) / base
        res[lab] = P
        print(f"  {lab:16s} P = {P:.4f}")
    # entire_missing / depth_constant: depth is a constant field -> mask_d identically 0
    for lab in ("entire_missing", "depth_constant"):
        vals = [mask_d_delta(d, torch.zeros_like(d), a.window).item() for d in ds]
        res[lab] = float(np.mean(vals)) / base
        print(f"  {lab:16s} P = {res[lab]:.4f}")
    out = PROJECT / "results" / "gapfill" / "maskd_perturbation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"window": a.window, "n_images": len(ds),
                               "clean_mask_d": base, "P": res}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
