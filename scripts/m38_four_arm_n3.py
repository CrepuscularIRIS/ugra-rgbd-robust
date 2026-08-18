"""M38 — the four-arm reliance study at n = 3 seeds. Paper Section 7 main evidence.

M28/M35 established, at n = 2 seeds, that depth-pathway attenuation explains BOTH sides of the
clean/missing trade-off while the conversion coefficient explains neither. The campaign's own
evidence standard is seeds >= 3, and the third seed exists on disk for all four arms. This produces
the final table at n = 3, with the inert stage-0 component excluded throughout (Revision 41).

12 checkpoints, eval-only. Clean and missing mIoU are read from each arm's own `final_eval.json`
(full validation set) rather than transported from another run — the cross-run transport defect this
campaign has already been bitten by twice.

GATE: baseline per-condition R (all-stage convention) reproduces M14's recorded values, median ratio
within 10% — the gate M27/M28/M35 passed at 0.9236.

Prereg: .grill/prediction-ledger.md (2026-08-10, M38).

    python3 scripts/m38_four_arm_n3.py --gpu 1 --n-eval 30
"""
import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
for _p in (DFORMER, DFORMER / "utils", PROJECT / "diagnostics" / "corruptions", PROJECT / "scripts"):
    sys.path.insert(0, str(_p))

GAPFILL = PROJECT / "results" / "gapfill"
P0 = PROJECT / "results" / "p0"
SEED = 20260721
USED = [1, 2, 3]
_CAP = {"store": None, "mode": None}

ARMS = ["baseline", "attenuation", "constant", "dropout"]
PREFIX = {"baseline": "mainB_baseline", "attenuation": "mainC_attenuation",
          "constant": "mainC_constant", "dropout": "mainC_dropout"}
SEEDS = [20260801, 20260802, 20260803]

from m1_mig_full import COND, corrupt  # noqa: E402
from m35_reff_reaudit import measure, pearson, spearman  # noqa: E402


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

    m14 = json.loads((GAPFILL / "m14_holdout_rayleigh.json").read_text())["rows"]
    names = [c for c in COND if c in m14]
    engine = Engine(custom_parser=None)
    vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

    miou = {}
    for s in SEEDS:
        for arm in ARMS:
            f = P0 / f"{PREFIX[arm]}_s{s}" / "final_eval.json"
            r = json.loads(f.read_text())["results"]["single"]
            miou[(arm, s)] = {k: v["miou"] for k, v in r.items()}

    res = {}
    for s in SEEDS:
        for arm in ARMS:
            ck = P0 / f"{PREFIX[arm]}_s{s}" / "ckpt" / "latest.pth"
            print(f"=== {arm} s{s} ===", flush=True)
            res[(arm, s)] = measure(ck, cfg, names, lambda: vl, a.n_eval)

    base0 = res[("baseline", SEEDS[0])]
    gc = [c for c in names if m14[c].get("R") is not None]
    ratios = sorted(base0[c]["R_all"] / m14[c]["R"] for c in gc)
    med = ratios[len(ratios) // 2]
    print(f"\n=== GATE median {med:.4f} -> {'PASS' if abs(med-1) < 0.10 else 'FAIL'}")
    if abs(med - 1) >= 0.10:
        raise SystemExit("REFUSING: gate failed")

    live = [c for c in names if base0[c]["n_all"] > 1e-6 and c != "depth_offset"]

    rows = {}
    for s in SEEDS:
        b = res[("baseline", s)]
        for arm in ARMS:
            if arm == "baseline":
                continue
            r = res[(arm, s)]
            dN = [100 * (r[c]["n_eff"] / b[c]["n_eff"] - 1) for c in live]
            dR = [100 * (r[c]["R_eff"] / b[c]["R_eff"] - 1) for c in live]
            g = [1 + x / 100 for x in dN]
            rows[f"{arm}_s{s}"] = {
                "arm": arm, "seed": s,
                "dN": st.mean(dN), "dN_sd": st.stdev(dN),
                "dR": st.mean(dR), "dR_sd": st.stdev(dR),
                "gain": st.mean(g), "gain_cv": st.stdev(g) / st.mean(g),
                "dclean": miou[(arm, s)]["clean"] - miou[("baseline", s)]["clean"],
                "dmiss": (miou[(arm, s)]["entire_missing"]
                          - miou[("baseline", s)]["entire_missing"]),
            }

    print(f"\n{'arm-seed':22s} {'d|D|%':>8s} {'sd':>6s} {'CV':>6s} {'dR%':>8s} {'sd':>6s} "
          f"{'dclean':>7s} {'dmiss':>7s}")
    for k in sorted(rows, key=lambda x: (rows[x]["arm"], rows[x]["seed"])):
        v = rows[k]
        print(f"{k:22s} {v['dN']:+8.1f} {v['dN_sd']:6.1f} {v['gain_cv']:6.3f} "
              f"{v['dR']:+8.1f} {v['dR_sd']:6.1f} {v['dclean']:+7.2f} {v['dmiss']:+7.2f}")

    # arm-level means (no within-arm pseudo-replication)
    am = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        v = [rows[k] for k in rows if rows[k]["arm"] == arm]
        am[arm] = {kk: st.mean([x[kk] for x in v]) for kk in ("dN", "dR", "gain", "dclean", "dmiss")}
        am[arm]["n"] = len(v)
        am[arm]["dclean_sd"] = st.stdev([x["dclean"] for x in v])
        am[arm]["dmiss_sd"] = st.stdev([x["dmiss"] for x in v])
    print(f"\n{'arm':14s} {'n':>2s} {'d|D|%':>8s} {'gain':>6s} {'dR%':>8s} "
          f"{'dclean':>16s} {'dmiss':>16s}")
    for arm in ("constant", "attenuation", "dropout"):
        v = am[arm]
        print(f"{arm:14s} {v['n']:2d} {v['dN']:+8.1f} {v['gain']:6.3f} {v['dR']:+8.1f} "
              f"{v['dclean']:+8.2f} +- {v['dclean_sd']:4.2f} {v['dmiss']:+8.2f} +- {v['dmiss_sd']:4.2f}")

    ks = sorted(rows)
    dN = [rows[k]["dN"] for k in ks]; dR = [rows[k]["dR"] for k in ks]
    dc = [rows[k]["dclean"] for k in ks]; dm = [rows[k]["dmiss"] for k in ks]
    A = ("constant", "attenuation", "dropout")
    aN = [am[x]["dN"] for x in A]; aR = [am[x]["dR"] for x in A]
    aC = [am[x]["dclean"] for x in A]; aM = [am[x]["dmiss"] for x in A]

    print(f"\n--- POOLED (n={len(ks)} arm-seeds) ---")
    print(f"  Pearson(d|D|, dclean) = {pearson(dN,dc):+.3f}    Pearson(dR, dclean) = {pearson(dR,dc):+.3f}")
    print(f"  Pearson(d|D|, dmiss)  = {pearson(dN,dm):+.3f}    Pearson(dR, dmiss)  = {pearson(dR,dm):+.3f}")
    print(f"\n--- ARM MEANS (n=3 arms, no pseudo-replication) ---")
    print(f"  Pearson(d|D|, dclean) = {pearson(aN,aC):+.3f}    Pearson(dR, dclean) = {pearson(aR,aC):+.3f}")
    print(f"  Pearson(d|D|, dmiss)  = {pearson(aN,aM):+.3f}    Pearson(dR, dmiss)  = {pearson(aR,aM):+.3f}")

    print(f"\n--- WITHIN-SEED ORDERING (the replication test) ---")
    ok = True
    for s in SEEDS:
        o = sorted([k for k in ks if rows[k]["seed"] == s], key=lambda k: rows[k]["dN"])
        seq = [rows[k]["arm"] for k in o]
        ok &= (seq == ["dropout", "attenuation", "constant"])
        print(f"  seed {s}: " + " -> ".join(seq))
    print(f"  identical ordering in all {len(SEEDS)} seeds: {ok}")

    p_a = pearson(aN, aC) >= 0.9
    p_b = abs(pearson(aR, aC)) < 0.5 and abs(pearson(aR, aM)) < 0.5
    p_c = ok
    print(f"\n  P-M38a (arm-mean Pearson(d|D|,dclean) >= 0.9):  {'HIT' if p_a else 'MISS'}")
    print(f"  P-M38b (R explains neither side, |rho| < 0.5):  {'HIT' if p_b else 'MISS'}")
    print(f"  P-M38c (ordering identical in all 3 seeds):     {'HIT' if p_c else 'MISS'}")

    (GAPFILL / "m38_four_arm_n3.json").write_text(json.dumps(
        {"gate_median": med, "n_eval": a.n_eval, "seeds": SEEDS, "rows": rows, "arm_means": am,
         "pooled": {"dN_dclean": pearson(dN, dc), "dR_dclean": pearson(dR, dc),
                    "dN_dmiss": pearson(dN, dm), "dR_dmiss": pearson(dR, dm)},
         "arm_level": {"dN_dclean": pearson(aN, aC), "dR_dclean": pearson(aR, aC),
                       "dN_dmiss": pearson(aN, aM), "dR_dmiss": pearson(aR, aM)},
         "ordering_replicates": bool(ok),
         "P_M38a": bool(p_a), "P_M38b": bool(p_b), "P_M38c": bool(p_c)}, indent=1))
    print(f"\nwrote {GAPFILL / 'm38_four_arm_n3.json'}")


if __name__ == "__main__":
    main()
