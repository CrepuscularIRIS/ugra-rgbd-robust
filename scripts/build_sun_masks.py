"""Real sensor missing-depth masks for SUN RGB-D, in the DFormer file layout.

    python scripts/build_sun_masks.py --sun /data/datasets/rgbd/SUNRGBD --out data/SUN_realmask

The DFormer SUN RGB-D package already ships the official RAW `depth/` maps (uint16, holes = 0), not `depth_bfx`:
checked 2026-11-15 against the official SUNRGBD.zip. Per-sensor shapes and their shares match the official package
(kv2 530x730, xtion 441x591, kv1 427x561, realsense 531x681), the zero-pixel rates match official raw depth, and
official depth_bfx has ~0 holes. No pixel is a hole in >95% of frames for any sensor, so there is no registration
border to strip (unlike NYU, see build_nyu_masks.py).

Writes  <out>/Mask/<name>.png  uint8, 255 = sensor depth valid, 0 = sensor hole (native resolution)
        <out>/holes.json       {name: {"hole": fraction, "sensor": kv1|kv2|realsense|xtion|other}}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

N_IMAGES = 10335
SENSOR_BY_SHAPE = {(427, 561): "kv1", (530, 730): "kv2", (531, 681): "realsense", (441, 591): "xtion"}


def valid_mask(depth: np.ndarray) -> np.ndarray:
    return (depth != 0).astype(np.uint8) * 255


def build(sun: Path, out: Path) -> dict[str, dict]:
    files = sorted((sun / "Depth").glob("*.png"))
    if len(files) != N_IMAGES:
        raise ValueError(f"expected {N_IMAGES} depth maps in {sun / 'Depth'}, got {len(files)}")
    (out / "Mask").mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict] = {}
    for p in files:
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if d is None or d.dtype != np.uint16:
            raise ValueError(f"{p}: expected uint16 raw depth, got {None if d is None else d.dtype}")
        m = valid_mask(d)
        cv2.imwrite(str(out / "Mask" / p.name), m)
        stats[p.stem] = {"hole": float(1.0 - m.mean() / 255.0), "sensor": SENSOR_BY_SHAPE.get(d.shape, "other")}
    (out / "holes.json").write_text(json.dumps(stats, indent=1))
    return stats


def _self_check() -> None:
    d = np.zeros((4, 5), np.uint16)
    d[1:3, 1:4] = 12000
    m = valid_mask(d)
    assert m.dtype == np.uint8 and (m == 255).sum() == 6 and m[0, 0] == 0


if __name__ == "__main__":
    _self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    s = build(a.sun, a.out)
    for sensor in sorted({v["sensor"] for v in s.values()}):
        h = np.array([v["hole"] for v in s.values() if v["sensor"] == sensor])
        print(f"{sensor:9s} n={len(h):5d} hole median {np.median(h):.3f} mean {h.mean():.3f} max {h.max():.3f}")
