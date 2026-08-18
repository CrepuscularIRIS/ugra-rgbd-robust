"""M5 — the OWED retrodiction, plus Level-1's seed-reproducibility requirement.

Predictions were filed in plan/MIG-PROGRAM.md BEFORE any spatial statistic was computed:

    P-M5a  leading-mode aligned magnitude predicts damage at Spearman >= 0.90 across the 14
           conditions, i.e. at least as well as ||delta||'s 0.952                    P_true 0.45
    P-M5b  depth_shift's field is MORE DIFFUSE (lower top-1 energy fraction) than
           invalid_0.10's -- the specific claim retracted for being unmeasured       P_true 0.55
    P-M5c  at matched ~10% fraction, invalid_0.10 has HIGHER top-1 aligned magnitude
           than holes_2blob                                                          P_true 0.60

P-M5a is filed BELOW a coin flip on purpose: ||delta|| already reaches 0.952, and M4's rank-1 result
was for ONE corruption. Zero contradictions across 14 conditions would be a WARNING that the leading
mode is a re-description of ||delta||, not a triumph -- the two are expected to be collinear and the
retrodiction's job is to find where they come apart.

ALSO RUNS Level-1's reproducibility requirement: the same statistics on all THREE baseline seeds.
A geometry that does not reproduce across seeds is not a geometry.

    python3 scripts/m5_retrodict.py --gpu 1 --n-eval 16
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

GAPFILL = PROJECT / "results" / "gapfill"
SEEDS = [20260801, 20260802, 20260803]
SEED = 20260721
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, measured_damage, spearman  # noqa: E402


def field_stats(delta):
    """Per-stage leading spatial mode: sigma1 (aligned magnitude) and E1 (energy fraction)."""
    s1_tot, e1_num, e1_den = 0.0, 0.0, 0.0
    for d in delta:
        B, C, H, W = d.shape
        M = d.reshape(B, C, H * W)[0].float()
        S = torch.linalg.svdvals(M)
        s1_tot += float(S[0]) ** 2
        e1_num += float(S[0]) ** 2
        e1_den += float((S ** 2).sum())
    return float(np.sqrt(s1_tot)), (e1_num / e1_den if e1_den > 0 else 0.0)


def run_seed(seed, n_eval, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.chdir(DFORMER)
    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    ck = PROJECT / "results" / "p0" / f"mainB_baseline_s{seed}" / "ckpt" / "latest.pth"
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(miss.missing_keys)
    if got < 0.9 * len(sd):
        raise SystemExit(f"REFUSING seed {seed}: {got}/{len(sd)} loaded")
    model.eval().cuda()
    _orig = model.decode_head.forward

    def _dec(x, *a_, **k_):
        feats = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [f.detach().clone() for f in feats]
        return _orig(x, *a_, **k_)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")
    from m1_mig_full import corrupt

    def cap(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    names = list(COND)
    acc = {c: {"s1": [], "e1": [], "nrm": []} for c in names}
    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            Zc = cap(rgb, dep)
            for c in names:
                Zk = cap(rgb, corrupt(dep, COND[c][1]))
                delta = [k - z for k, z in zip(Zk, Zc)]
                s1, e1 = field_stats(delta)
                acc[c]["s1"].append(s1)
                acc[c]["e1"].append(e1)
                acc[c]["nrm"].append(float(sum(float((d ** 2).sum()) for d in delta) ** 0.5))
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    dmg = measured_damage()
    per_seed = {}
    for s in SEEDS[:a.seeds]:
        print(f"--- seed {s} ---", flush=True)
        per_seed[s] = run_seed(s, a.n_eval, a.gpu)

    names = [c for c in COND if c in dmg]
    d = [dmg[c] for c in names]
    print(f"\n{'condition':22s} {'damage':>7s} " +
          " ".join(f"{'s'+str(s)[-2:]+'_sig1':>10s}" for s in per_seed) + f" {'E1(s01)':>8s}")
    for c in sorted(names, key=lambda x: -dmg[x]):
        row = " ".join(f"{per_seed[s][c]['s1']:10.3f}" for s in per_seed)
        print(f"{c:22s} {dmg[c]:7.2f} {row} {per_seed[SEEDS[0]][c]['e1']:8.3f}")

    print(f"\nSpearman vs damage, per seed ({len(names)} conditions):")
    rows = {}
    for s in per_seed:
        s1 = [per_seed[s][c]["s1"] for c in names]
        nm = [per_seed[s][c]["nrm"] for c in names]
        rows[s] = (spearman(s1, d), spearman(nm, d))
        print(f"  seed {s}: leading-mode sigma1 {rows[s][0]:+.3f} | full-field ||delta|| "
              f"{rows[s][1]:+.3f}")

    # cross-seed reproducibility of the ORDERING — Level 1's requirement
    print("\nLEVEL-1 REPRODUCIBILITY — cross-seed Spearman of sigma1 orderings:")
    ss = list(per_seed)
    for i in range(len(ss)):
        for j in range(i + 1, len(ss)):
            v1 = [per_seed[ss[i]][c]["s1"] for c in names]
            v2 = [per_seed[ss[j]][c]["s1"] for c in names]
            print(f"  seed {ss[i]} vs {ss[j]}: {spearman(v1, v2):+.3f}")

    r_s1 = st.mean(rows[s][0] for s in rows)
    r_nm = st.mean(rows[s][1] for s in rows)
    e1 = {c: per_seed[SEEDS[0]][c]["e1"] for c in names}
    p_a = r_s1 >= 0.90
    p_b = e1.get("depth_shift", 1) < e1.get("invalid_0.10", 0)
    p_c = per_seed[SEEDS[0]].get("invalid_0.10", {}).get("s1", 0) > \
        per_seed[SEEDS[0]].get("holes_2blob", {}).get("s1", 0)
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M5a sigma1 Spearman >= 0.90: {r_s1:+.3f} -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M5b depth_shift E1 < invalid_0.10 E1: "
          f"{e1.get('depth_shift', float('nan')):.3f} vs {e1.get('invalid_0.10', float('nan')):.3f}"
          f" -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M5c invalid_0.10 sigma1 > holes_2blob sigma1 -> {'HIT' if p_c else 'MISS'}")
    print(f"\nleading-mode {r_s1:+.3f} vs full-field ||delta|| {r_nm:+.3f} — "
          f"{'BETTER' if r_s1 > r_nm + 0.01 else 'TIES OR WORSE'}")
    if abs(r_s1 - r_nm) <= 0.01:
        print("  -> the leading mode is a RE-DESCRIPTION of ||delta||, not a better predictor.")
        print("     Level 1's 'predicts better than existing heuristics' is NOT met by it.")

    p = GAPFILL / "m5_retrodict.json"
    p.write_text(json.dumps({"per_seed": {str(k): v for k, v in per_seed.items()},
                             "spearman": {str(k): rows[k] for k in rows},
                             "P_M5a": bool(p_a), "P_M5b": bool(p_b), "P_M5c": bool(p_c)},
                            indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
