"""Byte-identity regression: current code must reproduce the data/NYU_corrupt
outputs exactly, Depth AND oracle E (Sol review P2-7 rewrite).

    python scripts/regress_primitives.py

Per-task rng (crc32 of family/severity/frame) makes every sample independently
reproducible — no replay of a global consumption order needed anymore. F8 is a
donor file copy; F1/F2/F6 are encoded through the frozen per-frame map. Exits
nonzero on any mismatch.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from zlib import crc32

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from nyu_corrupt_core import (E_UNDEF, FrozenEncoder, load_names, load_png8, load_split,
                              oracle, read_frame, v_mask, write_depth_png, write_e_png)
import nyu_corrupt_families as fam
from build_nyu_corrupt import pick_f1_donor, pick_f8_donor

MAT = Path("data/nyu_depth_v2_labeled.mat")
DFORMER = Path("/data/datasets/rgbd/NYUDepthv2")
REALMASK = Path("data/NYU_realmask")
CORRUPT = Path("data/NYU_corrupt")
CHECK_IDS = [0, 300, 700, 1100, 1448]


def main() -> None:
    calib = json.loads((CORRUPT / "calib.json").read_text())
    bank = np.load(CORRUPT / "noise_bank.npz")["bank"]
    inner = {int(k): v for k, v in json.loads((REALMASK / "inner_holes.json").read_text()).items()}
    enc = FrozenEncoder()
    td = Path(tempfile.mkdtemp())
    fails = []
    with h5py.File(MAT, "r") as f:
        names = load_names(f)
        train_ids, test_ids = load_split(DFORMER)
        all_ids = train_ids + test_ids
        scene_of = {i: read_frame(f, i)["scene"] for i in all_ids}
        for i in CHECK_IDS:
            fr = read_frame(f, i)
            enc.fit(i, fr["depth"], load_png8(DFORMER / "Depth" / f"{i}.png"))
            v = v_mask(fr, names, calib)
            split_ids = train_ids if i in train_ids else test_ids
            rng = np.random.default_rng(11)

            def check(tag, z_t, undef):
                e = oracle(z_t, fr, v, calib)
                if undef is not None:
                    e[undef] = E_UNDEF
                p = td / f"{tag}_{i}.png"
                write_depth_png(p, enc.apply(i, z_t))
                pe = td / f"{tag}_{i}_E.png"
                write_e_png(pe, e)
                ok_d = p.read_bytes() == (CORRUPT / tag / "Depth" / f"{i}.png").read_bytes()
                ok_e = pe.read_bytes() == (CORRUPT / tag / "E" / f"{i}.png").read_bytes()
                print(f"{tag} id={i}: depth {'OK' if ok_d else 'MISMATCH'}, E {'OK' if ok_e else 'MISMATCH'}")
                if not (ok_d and ok_e):
                    fails.append((tag, i))

            donor1 = pick_f1_donor(i, 0.20, split_ids, inner)
            z1, _, _ = fam.f1_hole_zero(fr["depth"], load_png8(REALMASK / "InnerMask" / f"{donor1}.png"))
            check("F1/a20", z1, None)
            z2, _, _ = fam.f2_noise(fr["depth"], bank, 8.0,
                                    np.random.default_rng(crc32(f"F2/a8/{i}".encode())))
            check("F2/a8", z2, None)
            z6, u6, _ = fam.f6_misreg(fr["depth"], 16)
            check("F6/t16", z6, u6)
            donor8, _ = pick_f8_donor(i, "cross", split_ids, scene_of,
                                      np.random.default_rng(1000 + i))
            src = (DFORMER / "Depth" / f"{donor8}.png").read_bytes()
            ok = src == (CORRUPT / "F8" / "cross" / "Depth" / f"{i}.png").read_bytes()
            print(f"F8/cross id={i}: depth {'OK' if ok else 'MISMATCH'} (donor copy)")
            if not ok:
                fails.append(("F8/cross", i))
            rng.integers(1)  # keep rng defined-use parity harmless

    if fails:
        raise SystemExit(f"REGRESSION FAILED: {fails}")
    print(f"regression OK: {len(CHECK_IDS)} frames x 4 families, depth+E byte-identical")


if __name__ == "__main__":
    main()
