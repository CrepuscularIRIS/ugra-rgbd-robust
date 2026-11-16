"""SUN RGB-D corrupted-depth demo on a sampled subset, built from depth_primitives.

    python scripts/build_sun_corrupt.py --split test --frames 0 --out data/SUN_corrupt_test_full

Verified premises (2026-11-15, both reviews):
  - DFormer SUN Depth/ IS the official raw, but stored with a 3-bit circular shift:
    mm = (a>>3)|(a<<13); decoded values > 8 m are treated as no-reading (0).
    DFormer's loader reads it via IMREAD_GRAYSCALE -> a>>8 ~ mm/32, i.e. a fixed
    8-bit metric code with ~32 mm steps -> eps floor = 0.032 m.
  - z_pc = decoded raw itself; output re-encodes with (mm<<3)|(mm>>13) (exact
    involution), unchanged frames copy the original file bytes.
  - Per-sensor pooled eps(z): median over ~100 train frames of per-frame q99 of
    median-filter residuals on eroded (15x15) wall/floor pixels.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import median_filter

from depth_primitives import p_zero, p_offset, p_swap, p_warp, p_noise, select_objects
from nyu_corrupt_core import eps_of, oracle as core_oracle, E_UNDEF

SENSOR_BY_SHAPE = {(427, 561): "kv1", (530, 730): "kv2", (531, 681): "realsense", (441, 591): "xtion"}
WALL, FLOOR = 0, 1
SPECULAR = (8, 18)  # window, mirror (canonical 37-class order)
OBJ_EXCLUDE = {WALL, FLOOR, 21, 255}  # + ceiling + ignore
PER_SENSOR_FRAMES = 3
CAL_FRAMES_PER_SENSOR = 100
EPS_FLOOR = 0.032  # host quantization step: a>>8 ~ mm/32
BAND_DILATE = 3


def decode_sun(a: np.ndarray) -> np.ndarray:
    mm = ((a >> 3) | np.uint16(a << 13))  # 3-bit circular shift, official SUN encoding
    mm = mm.copy() if isinstance(mm, np.ndarray) else mm
    mm[mm > 8000] = 0  # beyond host code range: no reading (review choice)
    return mm.astype(np.float32) / 1000.0


def encode_sun(z_m: np.ndarray) -> np.ndarray:
    mm = np.clip(np.round(z_m * 1000), 0, 8000).astype(np.uint16)
    return ((mm << 3) | (mm >> 13)).astype(np.uint16)


def load(sun: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = cv2.imread(str(sun / "Depth" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    lab = cv2.imread(str(sun / "Label" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
    if a is None or lab is None or a.dtype != np.uint16:
        raise ValueError(f"{name}: bad load")
    return decode_sun(a), lab, a


def plane_residual(z: np.ndarray, lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isin(lab, (WALL, FLOOR)) & (z > 0)
    m = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    if m.sum() < 2000:
        return None, m
    med = median_filter(z, size=15)
    return np.abs(z - med)[m], z[m]


def calibrate_sensor(sun: Path, names: list[str]) -> dict:
    """Pooled eps(z): median across frames of per-frame q99 per 0.5 m bin."""
    per_frame_bins: list[tuple[np.ndarray, np.ndarray]] = []
    for n in names:
        z, lab, _ = load(sun, n)
        r, zb = plane_residual(z, lab)
        if r is None:
            continue
        edges = np.arange(0.5, 8.01, 0.5)
        q = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (zb >= lo) & (zb < hi)
            q.append(np.quantile(r[sel], 0.99) if sel.sum() > 300 else np.nan)
        per_frame_bins.append(np.array(q))
    q = np.nanmedian(np.stack(per_frame_bins), axis=0)
    centers = (np.arange(0.5, 8.0, 0.5) + 0.25).tolist()
    good = ~np.isnan(q)
    return {"bin_centers": [c for c, g in zip(centers, good) if g],
            "q99": [max(float(v), EPS_FLOOR) for v, g in zip(q, good) if g],
            "eps_abs": EPS_FLOOR,  # floor = host quantization step; z-dependence lives in the bins
            "n_frames": len(per_frame_bins)}


def build_bank(sun: Path, names: list[str], rng: np.random.Generator,
               patch: int = 32, want: int = 64) -> np.ndarray:
    """Per-sensor disparity-residual patches; validity from the eroded mask, NOT d!=0
    (quantized zero residuals are common and previously emptied the bank)."""
    fields = []
    for n in names:
        z, lab, _ = load(sun, n)
        m = np.isin(lab, (WALL, FLOOR)) & (z > 0)
        m = cv2.erode(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        if m.sum() < 5000:
            continue
        med = median_filter(z, size=15)
        fields.append((np.where(m, 1.0 / z - 1.0 / np.maximum(med, 0.3), 0.0).astype(np.float32), m))
    out = []
    attempts = 0
    while fields and len(out) < want and attempts < want * 40:  # Sol P2-5: hard cap
        attempts += 1
        d, m = fields[int(rng.integers(len(fields)))]
        y = int(rng.integers(0, d.shape[0] - patch))
        x = int(rng.integers(0, d.shape[1] - patch))
        if m[y:y + patch, x:x + patch].all():
            out.append(d[y:y + patch, x:x + patch])
    if len(out) < want:
        print(f"  bank: only {len(out)}/{want} patches after {attempts} tries")
    return np.stack(out) if out else np.zeros((0, patch, patch), np.float32)


def v_mask(z: np.ndarray, lab: np.ndarray, cal: dict) -> np.ndarray:
    v = (z > 0) & (lab != 255) & ~np.isin(lab, SPECULAR)  # Sol P2-4: ignore stays U
    jump = cv2.dilate(z, np.ones((3, 3), np.uint8)) - cv2.erode(z, np.ones((3, 3), np.uint8))
    thr = eps_of(np.maximum(z, 0.5), cal)
    band = cv2.dilate((jump > thr).astype(np.uint8), np.ones((BAND_DILATE, BAND_DILATE), np.uint8)) > 0
    return v & ~band


def main(sun: Path, out: Path, realmask: Path, split: str = "train",
         n_frames: int = PER_SENSOR_FRAMES, sensor: str | None = None) -> None:
    holes = json.loads((realmask / "holes.json").read_text())
    by_sensor: dict[str, list[str]] = {}
    for name, info in holes.items():
        by_sensor.setdefault(info["sensor"], []).append(name)
    if sensor:
        by_sensor = {sensor: by_sensor[sensor]}
    rng = np.random.default_rng(11)

    def pick(lst: list[str]) -> list[str]:
        pool = [n for n in lst if n.startswith(split)]
        k = len(pool) if n_frames == 0 else min(n_frames, len(pool))
        return list(rng.choice(pool, k, replace=False))

    demo = {s: pick(lst) for s, lst in sorted(by_sensor.items())}
    names = [n for lst in demo.values() for n in lst]
    print(f"subset: {len(names)} frames {names}")

    out.mkdir(parents=True, exist_ok=True)
    calibs, banks = {}, {}
    for s, lst in sorted(by_sensor.items()):
        cal_frames = list(rng.choice([n for n in lst if n.startswith("train")],
                                     min(CAL_FRAMES_PER_SENSOR, len(lst)), replace=False))
        cal = calibrate_sensor(sun, cal_frames)
        calibs[s] = cal
        banks[s] = build_bank(sun, cal_frames, rng)
        print(f"{s}: eps_abs={cal['eps_abs']:.3f} bins={len(cal['bin_centers'])} "
              f"bank={len(banks[s])} (from {cal['n_frames']} cal frames)")

    def save(fam: str, sev: str, name: str, z_t: np.ndarray, z0: np.ndarray,
             orig_path: Path, e: np.ndarray, meta: dict) -> dict:
        d = out / fam / sev
        (d / "E").mkdir(parents=True, exist_ok=True)
        if np.allclose(z_t, z0, atol=1e-9):
            shutil.copyfile(orig_path, d / f"{name}.png")  # identity: original PNG file
        else:
            cv2.imwrite(str(d / f"{name}.png"), encode_sun(z_t))
        cv2.imwrite(str(d / "E" / f"{name}.png"), e)
        return {"frame": name, **meta,
                "e_frac": round(float((e == 1).sum() / max((e != E_UNDEF).sum(), 1)), 4)}

    results: dict[str, list] = {}
    for n in names:
        z, lab, a = load(sun, n)
        sensor = SENSOR_BY_SHAPE[a.shape]
        cal = calibs[sensor]
        v = v_mask(z, lab, cal)
        bank = banks[sensor]
        pool = demo[sensor]
        donor = pool[-1] if pool[-1] != n else pool[0]
        z_d, _, _ = load(sun, donor)
        reg_all = np.ones_like(v)
        cand_ids = [i for i in np.unique(lab) if i not in OBJ_EXCLUDE]

        keep, lcc, fill = select_objects(z, lab, cand_ids, k=3, eps=cal["eps_abs"])
        reg4 = np.isin(lcc, keep)

        def run(tag, z_t, undef, meta):
            e = core_oracle(z_t, {"raw": z}, v, cal)
            if undef is not None:
                e[undef] = E_UNDEF
            fam, sev = tag.split("/")
            row = save(fam, sev, n, z_t, z, sun / "Depth" / f"{n}.png", e, meta)
            results.setdefault(tag, []).append(row)

        run("F0/zero", p_zero(z, reg_all), None, {})
        run("F1/holes", p_zero(z, z_d <= 0), None, {"donor": donor})
        run("F2/a1", p_noise(z, z > 0, 1.0, bank, rng), None, {"alpha": 1.0})
        run("F2/a8", p_noise(z, z > 0, 8.0, bank, rng), None, {"alpha": 8.0})
        z4 = np.where(reg4, fill, z).astype(np.float32)
        if len(bank) and keep:
            z4 = p_noise(z4, reg4, 1.0, bank, rng)
        run("F4/k3", z4, None, {"n_regions": len(keep), "erased_frac": float(reg4.mean())})
        z5 = z.copy()
        for c in keep:
            z5 = p_offset(z5, lcc == c, 0.9, fill)
        run("F5/s90", z5, None, {"n_objects": len(keep)})
        z6, undef6 = p_warp(z, 16)
        run("F6/t16", z6, undef6, {"t": 16})
        run("F8/swap", p_swap(z, reg_all, z_d), None, {"donor": donor})

    summary = {k: {"n": len(r), "mean_e_frac": round(float(np.mean([x["e_frac"] for x in r])), 4)}
               for k, r in sorted(results.items())}
    tag = sensor or "all"
    (out / f"summary_{tag}.json").write_text(json.dumps(
        {"calibs": calibs, "results": results, "summary": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun", type=Path, default=Path("/data/datasets/rgbd/SUNRGBD"))
    ap.add_argument("--out", type=Path, default=Path("data/SUN_corrupt_test_full"))
    ap.add_argument("--realmask", type=Path, default=Path("data/SUN_realmask"))
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--frames", type=int, default=PER_SENSOR_FRAMES,
                    help="frames per sensor (0 = every frame of the split)")
    ap.add_argument("--sensor", choices=["kv1", "kv2", "realsense", "xtion"],
                    help="restrict to one sensor (parallel sharding)")
    a = ap.parse_args()
    main(a.sun, a.out, a.realmask, a.split, a.frames, a.sensor)
