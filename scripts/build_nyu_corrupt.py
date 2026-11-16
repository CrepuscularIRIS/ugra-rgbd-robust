"""Build the NYU corrupted-depth benchmark data (families F1/F4/F6/F8).

    python scripts/build_nyu_corrupt.py \
        --mat data/nyu_depth_v2_labeled.mat --dformer /data/datasets/rgbd/NYUDepthv2 \
        --realmask data/NYU_realmask --out data/NYU_corrupt [--frames 8]

Output layout (drop-in for the DFormer loader, ids unchanged):
    <out>/calib.json                      eps(z) calibration + encoder stats
    <out>/<FAM>/<SEV>/Depth/<id>.png      corrupted depth, 4-ch like the originals
    <out>/<FAM>/<SEV>/E/<id>.png          oracle {0 clean, 1 err, 255 undef}
    <out>/<FAM>/<SEV>/meta.json           params + per-frame stats (donor, v/e frac, m_eff)
Donors are always drawn from the same split as the target (design 4.7).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from zlib import crc32

import h5py
import numpy as np

from nyu_corrupt_core import (
    FrozenEncoder, calibrate_eps, load_names,
    load_png8, load_split, oracle, read_frame, v_mask, write_depth_png, write_e_png,
    E_ERR, E_UNDEF,
)
import nyu_corrupt_families as fam

F1_SEV = [0.05, 0.10, 0.20, 0.40]
F2_SEV = [1.0, 2.0, 4.0, 8.0, 16.0]
F4_SEV = [1, 3, 5]
F5_SEV = [0.5, 0.9]
F6_SEV = [4, 8, 16, 32]
F8_MODES = ["cross", "same"]


def build_noise_bank(f: h5py.File, frame_ids: list[int], names: list[str], out: Path,
                     step: int = 2, patch: int = 64, max_patches: int = 512, seed: int = 1,
                     ) -> np.ndarray:
    """Bank of real Kinect disparity-residual patches from wall/floor/ceiling planes.

    Powers F2 (alpha-mix transfer) and NTR (texture re-synthesis on filled regions).
    """
    import cv2
    from nyu_corrupt_core import WALL_FLOOR_CEIL, label_ids_by_name, fit_plane_ransac, _plane_z_at, FX, FY, CX, CY
    rng = np.random.default_rng(seed)
    wfc_ids = label_ids_by_name(names, WALL_FLOOR_CEIL)
    bank = []
    for i in frame_ids[::step]:
        if len(bank) >= max_patches:
            break
        fr = read_frame(f, i)
        m = np.isin(fr["label"], wfc_ids) & (fr["raw"] > 0)
        m = cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)  # off plane edges
        fit = fit_plane_ransac(fr["raw"], m, rng)
        if fit is None:
            continue
        n, d, _ = fit
        v_idx, u_idx = np.nonzero(m)
        u64, v64 = u_idx.astype(np.float64), v_idx.astype(np.float64)
        denom = n[2] + n[0] * (u64 - CX) / FX + n[1] * (v64 - CY) / FY
        z_raw = fr["raw"][v_idx, u_idx].astype(np.float64)
        z_plane = _plane_z_at(n, d, u64, v64)
        resid = np.abs(1.0 / z_raw - 1.0 / z_plane)
        keep = (denom > 0.2) & (resid < 0.02)  # 1/m units; a few quantization steps
        dmap = np.zeros(fr["raw"].shape, np.float32)
        dmap[v_idx[keep], u_idx[keep]] = (1.0 / z_raw - 1.0 / z_plane)[keep].astype(np.float32)
        valid = dmap != 0
        # carve fully-valid patch blocks
        for _ in range(4):
            y = int(rng.integers(0, dmap.shape[0] - patch))
            x = int(rng.integers(0, dmap.shape[1] - patch))
            blk = valid[y:y + patch, x:x + patch]
            if blk.all():
                bank.append(dmap[y:y + patch, x:x + patch])
                if len(bank) >= max_patches:
                    break
    bank = np.stack(bank)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, bank=bank)
    print(f"noise bank: {len(bank)} patches {bank.shape[1:]} disparity-std "
          f"{bank.std():.5f} 1/m -> {out}")
    return bank


def pick_f1_donor(target: int, a: float, split_ids: list[int], inner_frac: dict[int, float]) -> int:
    cand = [(abs(inner_frac[j] - a), j) for j in split_ids if j != target and inner_frac.get(j, 0) > 0.005]
    return min(cand)[1] if cand else target


def pick_f8_donor(target: int, mode: str, split_ids: list[int], scene_of: dict[int, str],
                  rng: np.random.Generator) -> tuple[int, bool]:
    """Returns (donor, fallback). 'same' falls back to 'cross' for single-frame scenes."""
    if mode == "same":
        same = [j for j in split_ids if j != target and scene_of[j] == scene_of[target]]
        if same:
            return int(rng.choice(same)), False
    diff = [j for j in split_ids if j != target and scene_of[j] != scene_of[target]]
    return int(rng.choice(diff)), mode == "same"


def build(mat: Path, dformer: Path, realmask: Path, out: Path, families: list[str],
          n_frames: int, calib_step: int) -> None:
    families = [f.upper() for f in families]
    inner_frac = {int(k): v for k, v in json.loads((realmask / "inner_holes.json").read_text()).items()}
    with h5py.File(mat, "r") as f:
        names = load_names(f)
        train_ids, test_ids = load_split(dformer)
        all_ids = train_ids + test_ids
        scene_of = {i: read_frame(f, i)["scene"] for i in all_ids}
        targets = all_ids if n_frames == 0 else all_ids[:n_frames]

        calib_path = out / "calib.json"
        if calib_path.exists():
            calib = json.loads(calib_path.read_text())
            print(f"calib: loaded {calib_path} ({calib['n_frames']} frames, {calib['n_pts']} pts)")
        else:
            print("calibrating eps(z) on train wall/floor/ceiling ...")
            calib = calibrate_eps(f, train_ids, names, step=calib_step)
            out.mkdir(parents=True, exist_ok=True)
            calib_path.write_text(json.dumps(calib, indent=1))
            print(f"calib: eps_abs={calib['eps_abs']:.4f} m over {calib['n_frames']} frames -> {calib_path}")

        from nyu_corrupt_core import WALL_FLOOR_CEIL
        structural_names = set(WALL_FLOOR_CEIL) | {"unlabeled", "unknown"}
        obj_ids = [i for i, n in enumerate(names, start=1) if n not in structural_names]
        bank = None
        if any(fam in families for fam in ("F2", "F3", "F4")):
            bank_path = out / "noise_bank.npz"
            if bank_path.exists():
                bank = np.load(bank_path)["bank"]
                print(f"noise bank: loaded {len(bank)} patches from {bank_path}")
            else:
                bank = build_noise_bank(f, train_ids, names, bank_path)
        enc = FrozenEncoder()
        rng = np.random.default_rng(42)
        enc_mad = []
        print("fitting frozen encoders ...")
        for i in all_ids:  # all ids: F1/F4/F6 targets and stats need them
            fr = read_frame(f, i)
            enc_mad.append(enc.fit(i, fr["depth"], load_png8(dformer / "Depth" / f"{i}.png")))
        enc_mad = np.array(enc_mad)
        print(f"encoder reproduction MAD: median {np.median(enc_mad):.2f} p99 {np.quantile(enc_mad, 0.99):.2f} levels")

        jobs: list[tuple[str, str, dict]] = []
        if "F0" in families:
            jobs.append(("F0", "zero", {"c": 0.0}))
        if "F1" in families:
            for a in F1_SEV:
                jobs.append(("F1", f"a{int(a * 100):02d}", {"a": a}))
        if "F2" in families:
            for al in F2_SEV:
                jobs.append(("F2", f"a{al:.0f}", {"alpha": al}))
        if "F3" in families:
            for a in F1_SEV:  # same severities AND same donors as F1 -> matched pair
                jobs.append(("F3", f"a{int(a * 100):02d}", {"a": a}))
        if "F4" in families:
            for k_ in F4_SEV:
                jobs.append(("F4", f"k{k_}", {"k": k_}))
        if "F5" in families:
            for s_ in F5_SEV:
                jobs.append(("F5", f"s{int(s_ * 100):02d}", {"s": s_}))
        if "F6" in families:
            for t in F6_SEV:
                jobs.append(("F6", f"t{t:02d}", {"t": t}))
        if "F8" in families:
            for m in F8_MODES:
                jobs.append(("F8", m, {"mode": m}))

        for fam_name, sev, params in jobs:
            rows, e_fracs, m_effs = [], [], []
            for i in targets:
                # per-task rng (Sol P2-6): outputs depend only on (family, severity,
                # frame), never on which other jobs ran before this one
                rng = np.random.default_rng(crc32(f"{fam_name}/{sev}/{i}".encode()))
                fr = read_frame(f, i)
                v = v_mask(fr, names, calib)
                donor, extra = None, {}
                if fam_name == "F0":
                    z_t, undef, meta = fam.f0_missing(fr["depth"], params["c"])
                elif fam_name == "F1":
                    donor = pick_f1_donor(i, params["a"], train_ids if i in train_ids else test_ids, inner_frac)
                    donor_mask = load_png8(realmask / "InnerMask" / f"{donor}.png")
                    z_t, undef, meta = fam.f1_hole_zero(fr["depth"], donor_mask)
                elif fam_name == "F2":
                    z_t, undef, meta = fam.f2_noise(fr["depth"], bank, params["alpha"], rng)
                elif fam_name == "F3":
                    donor = pick_f1_donor(i, params["a"], train_ids if i in train_ids else test_ids, inner_frac)
                    donor_mask = load_png8(realmask / "InnerMask" / f"{donor}.png")
                    z_t, undef, meta = fam.f3_hole_reinpain(fr["rgb"], fr["raw"], donor_mask, bank, rng)
                elif fam_name == "F4":
                    z_t, undef, meta = fam.f4_erase(fr["depth"], fr["label"], obj_ids, rng,
                                                    k=params["k"], eps_abs=calib["eps_abs"], bank=bank)
                elif fam_name == "F6":
                    z_t, undef, meta = fam.f6_misreg(fr["depth"], params["t"])
                elif fam_name == "F5":
                    z_t, undef, meta = fam.f5_instance_offset(fr["depth"], fr["label"], obj_ids, rng,
                                                              s=params["s"], eps_abs=calib["eps_abs"])
                else:  # F8
                    donor, fallback = pick_f8_donor(i, params["mode"], train_ids if i in train_ids else test_ids,
                                                    scene_of, np.random.default_rng(1000 + i))
                    z_t, undef, meta = fam.f8_swap(read_frame(f, donor)["depth"])
                    extra["fallback_cross"] = fallback

                e = oracle(z_t, fr, v, calib)
                if undef is not None:
                    e[undef] = E_UNDEF
                base = out / fam_name / sev
                if fam_name == "F8":  # donor's own PNG bytes: zero encoding error, maximal plausibility
                    base.joinpath("Depth").mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(dformer / "Depth" / f"{donor}.png", base / "Depth" / f"{i}.png")
                else:
                    write_depth_png(base / "Depth" / f"{i}.png", enc.apply(i, z_t))
                write_e_png(base / "E" / f"{i}.png", e)

                # m_eff over the scored domain only (Sol P2-8): V minus moved-in undef
                stat_mask = v if undef is None else (v & ~undef)
                delta = np.abs(z_t - fr["raw"])
                m_eff = float(np.where(stat_mask, np.minimum(delta / np.maximum(fr["raw"], 0.5), 1.0), 0.0).sum()
                              / max(stat_mask.sum(), 1))
                e_frac = float((e == E_ERR).sum() / max((e != E_UNDEF).sum(), 1))
                rows.append({"id": i, "donor": donor, "v_frac": float(v.mean()),
                             "e_frac": round(e_frac, 4), "m_eff": round(m_eff, 4),
                             "scene": scene_of[i], **meta, **extra})
                e_fracs.append(e_frac)
                m_effs.append(m_eff)
            meta_json = {"family": fam_name, "severity": sev, "params": params,
                         "n": len(rows), "mean_e_frac": round(float(np.mean(e_fracs)), 4),
                         "mean_m_eff": round(float(np.mean(m_effs)), 4), "frames": rows}
            (out / fam_name / sev / "meta.json").write_text(json.dumps(meta_json, indent=1))
            print(f"{fam_name}/{sev}: n={len(rows)} mean_e_frac={np.mean(e_fracs):.3f} "
                  f"mean_m_eff={np.mean(m_effs):.3f} -> {out / fam_name / sev}")

    (out / "build.json").write_text(json.dumps(
        {"encoder_mad_median": float(np.median(enc_mad)),
         "encoder_mad_p99": float(np.quantile(enc_mad, 0.99)),
         "n_frames": len(targets), "families": families}, indent=1))


def _self_check() -> None:
    # donor pickers: determinism, split confinement, same-scene preference
    inner = {1: 0.05, 2: 0.11, 3: 0.40, 4: 0.0}
    assert pick_f1_donor(1, 0.10, [1, 2, 3, 4], inner) == 2
    assert pick_f1_donor(4, 0.05, [1, 2, 3, 4], inner) == 1  # 0.005 floor excludes donor 4 itself
    scene = {1: "a", 2: "a", 3: "b", 4: "b"}
    d, fb = pick_f8_donor(1, "same", [1, 2, 3, 4], scene, np.random.default_rng(0))
    assert d == 2 and not fb
    d, fb = pick_f8_donor(1, "cross", [1, 2, 3, 4], scene, np.random.default_rng(0))
    assert d in (3, 4) and not fb
    scene[9] = "c"  # single-frame scene -> 'same' must fall back to cross
    d, fb = pick_f8_donor(9, "same", [9, 3, 4], scene, np.random.default_rng(0))
    assert d in (3, 4) and fb


if __name__ == "__main__":
    _self_check()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=Path, default=Path("data/nyu_depth_v2_labeled.mat"))
    ap.add_argument("--dformer", type=Path, default=Path("/data/datasets/rgbd/NYUDepthv2"))
    ap.add_argument("--realmask", type=Path, default=Path("data/NYU_realmask"))
    ap.add_argument("--out", type=Path, default=Path("data/NYU_corrupt"))
    ap.add_argument("--families", default="F1,F4,F6,F8")
    ap.add_argument("--frames", type=int, default=0, help="0 = all 1449")
    ap.add_argument("--calib-step", type=int, default=3)
    a = ap.parse_args()
    build(a.mat, a.dformer, a.realmask, a.out, a.families.split(","), a.frames, a.calib_step)
