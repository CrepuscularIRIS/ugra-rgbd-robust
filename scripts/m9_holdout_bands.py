"""M9 — the PRE-REGISTERED confirmation of M8's two-band structure, on HELD-OUT conditions.

M8 split 14 conditions into off-support (excess 0.085-0.202, rho +0.829 within group) and
in-support (excess -0.003..+0.005, damage 0.00-5.78, flat). **That split was found by reading the
table**, so it is a hypothesis. This is the test filed before it may be called a structure.

FILED IN plan/STANDING-ASSESSMENT.md BEFORE THIS RAN:
  (a) held-out conditions fall in the SAME TWO BANDS with nothing between;
  (b) within the off-support band, excess orders damage at rho >= 0.7;
  (c) within the in-support band, excess stays FLAT while damage varies.
  "If the bands do not reproduce, this is a description of fourteen numbers rather than a structure."

HELD-OUT SET — measured damage already on disk, NOT used to form the split:
    invalid_0.40   predicted OFF-support (the invalid family was uniformly off-support)
    shift_1/8/16/32  predicted IN-support (shift_4 was in-support at excess +0.0025)

WHY THIS SET IS A REAL TEST AND NOT A REPLAY. The shift family spans damage 0.00 -> 1.63 while
predicted to sit at FLAT support excess. That is the in-support band's defining and most falsifiable
property: damage varying with support pinned. If shift_32 (damage 1.63) shows a large excess, the
band collapses and the two-mechanism account is a post-hoc description.

FILED PREDICTIONS, before any number:
    P-M9a  invalid_0.40 lands in the OFF band (excess > 0.05)                    P_true 0.75
    P-M9b  ALL FOUR shift_* land in the IN band (excess < 0.05)                   P_true 0.65
    P-M9c  the gap survives: no held-out condition falls in 0.006 < excess < 0.08 P_true 0.55
    P-M9d  within the in-support band, shift damage 0.00->1.63 comes with flat
           excess (spread < 0.02)                                                 P_true 0.60

P-M9c is the one that can quietly fail: a single condition landing mid-gap turns a bimodal structure
into a continuum, and a continuum is just `‖Δ‖` again under another name.

    python3 scripts/m9_holdout_bands.py --gpu 1 --n-eval 8 --n-bank 6
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
_CAP = {"store": None, "mode": None}

from m1_mig_full import spearman  # noqa: E402
from m8_local_support import knn_dist, flat  # noqa: E402

# held-out: (gapfill suffix, corruption spec, PREDICTED band)
HELD = {
    "invalid_0.40": ("invalid0.4", ("invalid", 0.40), "OFF"),
    "shift_1":      ("shift1",     ("shift", 1),      "IN"),
    "shift_8":      ("shift8",     ("shift", 8),      "IN"),
    "shift_16":     ("shift16",    ("shift", 16),     "IN"),
    "shift_32":     ("shift32",    ("shift", 32),     "IN"),
}


def held_damage():
    base = {}
    for s in (20260801, 20260802, 20260803):
        f = PROJECT / "results" / "p0" / f"mainB_baseline_s{s}" / "final_eval.json"
        base[s] = json.loads(f.read_text())["results"]["single"]["clean"]["miou"]
    out = {}
    for name, (suf, _, _) in HELD.items():
        vals = []
        for f in sorted(glob.glob(str(GAPFILL / f"mainB_baseline_s*_{suf}.json"))):
            m = re.search(r"_s(\d+)_", f)
            if not m:
                continue
            r = json.loads(open(f).read())["results"]["single"]
            vals.append(base[int(m.group(1))] - r[next(iter(r))]["miou"])
        if vals:
            out[name] = st.mean(vals)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--n-bank", type=int, default=6)
    ap.add_argument("--stage", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    dmg = held_damage()
    print(f"held-out conditions with measured damage: {len(dmg)}/{len(HELD)}")
    if len(dmg) < 4:
        raise SystemExit("REFUSING: too few held-out conditions to test the bands")

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    from depth_corruptions import depth_invalid, depth_shift
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(miss.missing_keys)
    print(f"checkpoint: {got}/{len(sd)} tensors loaded")
    if got < 0.9 * len(sd):
        raise SystemExit("REFUSING: <90% loaded")
    model.eval().cuda()
    _orig = model.decode_head.forward

    def _dec(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def cap_(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    def corr(d, spec):
        if spec[0] == "invalid":
            return depth_invalid(d.clone(), frac=spec[1], seed=SEED)
        if spec[0] == "shift":
            return depth_shift(d.clone(), dx=spec[1], dy=0)
        raise ValueError(spec)

    names = [c for c in HELD if c in dmg]
    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        it = iter(vl)
        bank = []
        for _ in range(a.n_bank):
            b = next(it)
            bank.append(flat(cap_(b["data"].cuda(), b["modal_x"].cuda()), a.stage))
        B = torch.cat(bank)
        if B.shape[0] > 20000:
            B = B[torch.linspace(0, B.shape[0] - 1, 20000).long()]
        print(f"clean bank: {B.shape[0]} per-position features (SAME construction as M8)\n")

        acc = {c: [] for c in names}
        cl = []
        for i, b in enumerate(it):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            cl.append(knn_dist(flat(cap_(rgb, dep), a.stage), B))
            for c in names:
                acc[c].append(knn_dist(flat(cap_(rgb, corr(dep, HELD[c][1])), a.stage), B))

    ref = float(np.mean(cl))
    rows = {c: {"excess": float(np.mean(acc[c])) - ref, "damage": dmg[c],
                "predicted": HELD[c][2]} for c in names}
    print(f"clean reference kNN: {ref:.4f}\n")
    print(f"{'held-out condition':20s} {'damage':>7s} {'excess':>10s} {'pred':>5s} "
          f"{'observed':>9s} {'match':>6s}")
    ok_band = True
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        r = rows[c]
        obs = "OFF" if r["excess"] > 0.05 else ("IN" if r["excess"] < 0.006 else "MID-GAP")
        m = "yes" if obs == r["predicted"] else "NO"
        if m == "NO":
            ok_band = False
        print(f"{c:20s} {r['damage']:7.2f} {r['excess']:10.4f} {r['predicted']:>5s} "
              f"{obs:>9s} {m:>6s}")

    off = {c: r for c, r in rows.items() if r["excess"] > 0.05}
    ins = {c: r for c, r in rows.items() if r["excess"] < 0.006}
    mid = [c for c, r in rows.items() if 0.006 <= r["excess"] <= 0.05]

    p_a = rows.get("invalid_0.40", {}).get("excess", 0) > 0.05
    p_b = all(rows[c]["excess"] < 0.05 for c in rows if c.startswith("shift_"))
    p_c = len(mid) == 0
    spread = (max(r["excess"] for r in ins.values()) - min(r["excess"] for r in ins.values())
              if ins else float("nan"))
    dspan = (max(r["damage"] for r in ins.values()) - min(r["damage"] for r in ins.values())
             if ins else float("nan"))
    p_d = ins and spread < 0.02

    print(f"\nin-support band: excess spread {spread:.4f} while damage spans {dspan:.2f}")
    print(f"mid-gap occupants: {mid if mid else 'NONE'}")
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M9a invalid_0.40 in OFF band -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M9b all shift_* in IN band -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M9c no held-out condition mid-gap -> {'HIT' if p_c else 'MISS'}")
    print(f"P-M9d in-band excess flat (<0.02) while damage varies -> {'HIT' if p_d else 'MISS'}")

    print("\n--- VERDICT ---")
    if p_a and p_b and p_c and p_d:
        print("  BANDS REPRODUCE ON HELD-OUT CONDITIONS. The two-mechanism split is a STRUCTURE,")
        print("  not a post-hoc description of fourteen numbers. Damage varies across the")
        print("  in-support band while support stays pinned — the band's defining property.")
    elif not p_c:
        print(f"  MID-GAP OCCUPANT ({mid}). The structure is a CONTINUUM, not two bands —")
        print("  which makes it ||delta|| again under another name. Report as such.")
    else:
        print("  BANDS DO NOT REPRODUCE. Per the pre-registration this is a description of")
        print("  fourteen numbers, not a structure. Report the failure, do not re-partition.")

    out = GAPFILL / "m9_holdout_bands.json"
    out.write_text(json.dumps({"clean_ref": ref, "rows": rows, "mid_gap": mid,
                               "in_band_spread": spread, "in_band_damage_span": dspan,
                               "P_M9a": bool(p_a), "P_M9b": bool(p_b),
                               "P_M9c": bool(p_c), "P_M9d": bool(p_d)}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
