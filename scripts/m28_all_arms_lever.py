"""M28 — CAN ANYTHING LOWER R AT FIXED RELIANCE? All four trained arms, both seeds.

M27 showed modality dropout works by GLOBAL PATHWAY ATTENUATION (implied gain 0.525, CV 0.109), not
by moving R. It left one question open and explicitly did not answer it: *dropout does not lower R at
fixed reliance; that does not show nothing can.*

Four arms already exist at 500 epochs, two seeds each, and one of them is a natural experiment:

    baseline      clean 55.18   entire_missing 50.33     gap 4.85
    attenuation   clean 55.23   entire_missing 52.56     gap 2.67   <- +2.23 missing at ZERO clean cost
    constant      clean 55.62   entire_missing 51.61     gap 4.01
    dropout       clean 54.48   entire_missing 53.10     gap 1.38   <- +2.77 missing for -0.70 clean

If the WALK is what costs clean accuracy, then attenuation's zero clean cost predicts preserved
reliance -- and its missing-modality gain would then have to come from somewhere other than
||Delta||. That is the existence proof L3 needs, or its refutation.

INSTRUMENT. Relative ||Delta|| = ||Delta|| / ||Z_clean|| is the depth-reliance observable (C0,
instantiated in M27). The clean-accuracy floor cannot serve: dropout's walk cost -0.70 mIoU against
MDE 1.71.

GATE. Baseline per-condition R must reproduce M14's recorded values for that checkpoint (median
ratio within 10%) -- the same gate M27 passed at 0.9236.

Prereg + outcome table: .grill/prediction-ledger.md (2026-08-10, M28).

    python3 scripts/m28_all_arms_lever.py --gpu 1 --n-eval 30
"""
import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

GAPFILL = PROJECT / "results" / "gapfill"
P0 = PROJECT / "results" / "p0"

ARMS = ["baseline", "attenuation", "constant", "dropout"]
PREFIX = {"baseline": "mainB_baseline", "attenuation": "mainC_attenuation",
          "constant": "mainC_constant", "dropout": "mainC_dropout"}
SEEDS = [20260801, 20260802]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=30)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    from m27_dropout_lever import measure
    from m1_mig_full import COND

    m14 = json.loads((GAPFILL / "m14_holdout_rayleigh.json").read_text())["rows"]
    names = [c for c in COND if c in m14]

    engine = Engine(custom_parser=None)
    vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

    clean_miou, res = {}, {}
    for s in SEEDS:
        for arm in ARMS:
            d = P0 / f"{PREFIX[arm]}_s{s}"
            ck = d / "ckpt" / "latest.pth"
            if not ck.exists():
                print(f"  SKIP {arm} s{s} (no ckpt)")
                continue
            fe = d / "final_eval.json"
            if fe.exists():
                r = json.loads(fe.read_text())["results"]["single"]
                clean_miou[(arm, s)] = {k: v["miou"] for k, v in r.items()}
            print(f"\n=== {arm} s{s} ===")
            res[(arm, s)], _ = measure(ck, cfg, names, lambda: vl, a.n_eval)

    # --- GATE on baseline, both seeds available ---
    gate_conds = [c for c in names if m14[c].get("R") is not None]
    b0 = res.get(("baseline", SEEDS[0]))
    ratios = sorted(b0[c]["R"] / m14[c]["R"] for c in gate_conds)
    med = ratios[len(ratios) // 2]
    gate = abs(med - 1.0) < 0.10
    print(f"\n=== REFERENCE GATE (baseline s{SEEDS[0]} R vs M14) ===")
    print(f"  median ratio {med:.4f}  range [{ratios[0]:.3f}, {ratios[-1]:.3f}]  "
          f"gate: {'PASS' if gate else 'FAIL — internals must not be read'}")

    live = [c for c in names if c != "depth_offset"]        # ||Delta||~0, ratios meaningless
    summary = {}
    print(f"\n{'arm':13s} {'seed':>9s} | {'d|D|% mean':>11s} {'sd':>5s} {'CV':>6s} | "
          f"{'dR% mean':>9s} {'sd':>5s} | {'dclean':>7s} {'dmissing':>9s}")
    for s in SEEDS:
        base = res.get(("baseline", s))
        if base is None:
            continue
        bc = clean_miou.get(("baseline", s), {})
        for arm in ARMS:
            r = res.get((arm, s))
            if r is None:
                continue
            dN = [100 * (r[c]["norm"] / base[c]["norm"] - 1) for c in live]
            dR = [100 * (r[c]["R"] / base[c]["R"] - 1) for c in live]
            g = [1 + v / 100 for v in dN]
            cm = clean_miou.get((arm, s), {})
            dclean = cm.get("clean", float("nan")) - bc.get("clean", float("nan"))
            dmiss = cm.get("entire_missing", float("nan")) - bc.get("entire_missing", float("nan"))
            summary[f"{arm}_s{s}"] = {
                "dN_mean": st.mean(dN), "dN_sd": st.stdev(dN),
                "dR_mean": st.mean(dR), "dR_sd": st.stdev(dR),
                "implied_gain": st.mean(g), "gain_cv": st.stdev(g) / st.mean(g),
                "dclean": dclean, "dmissing": dmiss,
                "relnorm_dconst": r.get("dconst", {}).get("rel_norm"),
            }
            v = summary[f"{arm}_s{s}"]
            print(f"{arm:13s} {s:>9d} | {v['dN_mean']:+11.1f} {v['dN_sd']:5.1f} "
                  f"{v['gain_cv']:6.3f} | {v['dR_mean']:+9.1f} {v['dR_sd']:5.1f} | "
                  f"{dclean:+7.2f} {dmiss:+9.2f}")

    print("\n--- P-M28: does the WALK explain the CLEAN COST across independently trained arms? ---")
    pts = [(v["dN_mean"], v["dclean"]) for k, v in summary.items()
           if not k.startswith("baseline") and v["dclean"] == v["dclean"]]
    if len(pts) >= 3:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
        rho = num / den if den else float("nan")
        print(f"  Pearson(mean d||Delta||%, d clean mIoU) = {rho:+.3f}   n={len(pts)} arm-seeds")
        print("  (positive => the more the pathway is attenuated, the more clean accuracy is lost)")
    else:
        rho = float("nan")
        print("  too few arm-seeds")

    out = {"gate": gate, "gate_median_ratio": med, "n_eval": a.n_eval,
           "summary": summary, "walk_vs_cleancost_pearson": rho,
           "rows": {f"{k[0]}_s{k[1]}": v for k, v in res.items()}}
    (GAPFILL / "m28_all_arms_lever.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GAPFILL / 'm28_all_arms_lever.json'}")


if __name__ == "__main__":
    main()
