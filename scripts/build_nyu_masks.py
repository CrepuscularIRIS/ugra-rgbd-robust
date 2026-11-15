"""Real Kinect missing-depth masks for NYU Depth V2, in the DFormer file layout.

    python scripts/build_nyu_masks.py --mat data/nyu_depth_v2_labeled.mat --out data/NYU_realmask

Source: the official `nyu_depth_v2_labeled.mat` (2.97 GB). `rawDepths == 0` is NYU's own "no reading" marker.
Mapping: mat index i == DFormer file id i (proven on 2026-09-19, see reference/nyu_rawdepths_20260919/).

Writes  <out>/Mask/<id>.png   uint8, 255 = sensor depth valid, 0 = sensor hole (480x640)
        <out>/holes.json      per-id hole fraction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

N_IMAGES = 1449


def masks_from_raw(raw: np.ndarray) -> np.ndarray:
    """(N, W, H) rawDepths as stored by MATLAB/h5py -> (N, H, W) uint8 valid mask."""
    return (np.transpose(raw, (0, 2, 1)) != 0).astype(np.uint8) * 255


def build(mat: Path, out: Path) -> dict[int, float]:
    (out / "Mask").mkdir(parents=True, exist_ok=True)
    holes: dict[int, float] = {}
    with h5py.File(mat, "r") as f:
        raw = f["rawDepths"]
        if raw.shape[0] != N_IMAGES:
            raise ValueError(f"expected {N_IMAGES} images in rawDepths, got {raw.shape}")
        for i in range(raw.shape[0]):
            m = masks_from_raw(np.asarray(raw[i])[None])[0]
            cv2.imwrite(str(out / "Mask" / f"{i}.png"), m)
            holes[i] = float(1.0 - m.mean() / 255.0)
    (out / "holes.json").write_text(json.dumps(holes, indent=1))
    return holes


def _self_check() -> None:
    raw = np.zeros((1, 640, 480), np.float32)
    raw[0, 10:20, 5:7] = 2.5  # valid block at x in [10,20), y in [5,7)
    m = masks_from_raw(raw)[0]
    assert m.shape == (480, 640) and m[5, 10] == 255 and m[0, 0] == 0 and (m == 255).sum() == 20


if __name__ == "__main__":
    _self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    h = np.array(list(build(a.mat, a.out).values()))
    print(f"{len(h)} masks -> {a.out}/Mask  hole fraction median {np.median(h):.3f} min {h.min():.3f} max {h.max():.3f}")
