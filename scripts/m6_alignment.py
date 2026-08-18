"""M6 — does ALIGNMENT explain what magnitude cannot, and does dropout fall out of it?

M2/M3/M4 showed magnitude does not cause damage, direction is irrelevant, spatial shuffling
destroys damage, and rank-1 retains 90%. All of that says "alignment matters" without ever
DEFINING alignment as a measurable quantity. This script defines it and tests it.

DEFINITION, fixed before any number is computed:

    A = Spearman, over spatial positions, between  ||delta(h,w)||  and  ||Z_clean(h,w)||

i.e. does the displacement land WHERE THE CONTENT IS. This is exactly the property a spatial
permutation destroys (M3: shuffling collapses damage while preserving magnitude and per-channel
statistics), so it is the operational form of the thing those controls were probing.

WHY THIS TEST AND NOT ANOTHER. Two of the five criteria for a principle are unmet:
  * criterion 2 -- existing heuristics fall out as special cases;
  * Level-1 criterion 4 -- predicts behaviour better than existing heuristic descriptions.
Both turn on whether alignment is (a) measurable and (b) explains what the scalars get wrong.
The scalars' single systematic failure is `dconst`: 2nd-largest damage of 14 on the 8th-largest
displacement. If alignment explains dconst, the account earns its keep. If not, it does not.

FILED BEFORE ANY NUMBER (do not edit after seeing output):
    P-M6a  A(dconst) > A(invalid_0.55), despite ||delta||(dconst) < ||delta||(invalid_0.55).
           The scalars' one systematic failure is an ALIGNMENT effect.              P_true 0.55
    P-M6b  damage ~ ||delta|| * A predicts better than ||delta|| alone (Spearman up
           from 0.952)                                                              P_true 0.50
    P-M6c  DROPOUT FALLS OUT: alpha=0 (the dropout arm's training state) has the
           HIGHEST alignment of the alpha sweep {0, 0.25, 0.5, 0.7627, 1.0}         P_true 0.60

P-M6c is the criterion-2 test. Modality dropout trains at alpha=0; if alpha=0 is the
maximally-aligned intervention, then "dropout works" is a corollary of the alignment account
rather than an independent fact -- which is what separates a principle from a heuristic.

P-M6b at 0.50 exactly: a product of two quantities has two chances to be worse, and 0.952 is
already high enough that improvement is hard. Filing it at a coin flip is honest, not modest.

    python3 scripts/m6_alignment.py --gpu 1 --n-eval 16
"""
import argparse
import json
import os
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
SEED = 20260721
ALPHAS = [0.0, 0.25, 0.5, 0.7627, 1.0]
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, measured_damage, spearman, corrupt  # noqa: E402


def align(delta, Zc):
    """A = spatial Spearman between ||delta(h,w)|| and ||Z_clean(h,w)||, pooled over stages."""
    xs, ys = [], []
    for d, z in zip(delta, Zc):
        dm = d[0].pow(2).sum(0).sqrt().reshape(-1)
        zm = z[0].pow(2).sum(0).sqrt().reshape(-1)
        n = min(dm.numel(), 4096)
        idx = torch.linspace(0, dm.numel() - 1, n).long()
        xs += dm[idx].cpu().tolist()
        ys += zm[idx].cpu().tolist()
    return spearman(xs, ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=16)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    dmg = measured_damage()
    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
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

    def _dec(x, *args_, **kw):
        feats = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [f.detach().clone() for f in feats]
        return _orig(x, *args_, **kw)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def cap(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    names = [c for c in COND if c in dmg]
    acc = {c: {"A": [], "nrm": []} for c in names}
    alpha_A = {al: [] for al in ALPHAS}

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            Zc = cap(rgb, dep)
            for c in names:
                Zk = cap(rgb, corrupt(dep, COND[c][1]))
                delta = [k - z for k, z in zip(Zk, Zc)]
                A = align(delta, Zc)
                if A is not None:
                    acc[c]["A"].append(A)
                acc[c]["nrm"].append(float(sum(float((d ** 2).sum()) for d in delta) ** 0.5))
            # the alpha sweep — dropout's training states
            for al in ALPHAS:
                Za = cap(rgb, dep * al)
                da = [k - z for k, z in zip(Za, Zc)]
                A = align(da, Zc)
                if A is not None:
                    alpha_A[al].append(A)

    rows = {c: {"A": float(np.mean(acc[c]["A"])), "nrm": float(np.mean(acc[c]["nrm"])),
                "damage": dmg[c]} for c in names if acc[c]["A"]}
    print(f"\n{'condition':22s} {'damage':>7s} {'|delta|':>10s} {'A(align)':>9s} {'|d|*A':>10s}")
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        r = rows[c]
        print(f"{c:22s} {r['damage']:7.2f} {r['nrm']:10.1f} {r['A']:9.3f} "
              f"{r['nrm']*max(r['A'],0):10.1f}")

    ns = list(rows)
    d = [rows[c]["damage"] for c in ns]
    r_n = spearman([rows[c]["nrm"] for c in ns], d)
    r_a = spearman([rows[c]["A"] for c in ns], d)
    r_p = spearman([rows[c]["nrm"] * max(rows[c]["A"], 0) for c in ns], d)
    print(f"\nSpearman vs damage ({len(ns)} conditions):")
    print(f"  ||delta||        {r_n:+.3f}")
    print(f"  A (alignment)    {r_a:+.3f}")
    print(f"  ||delta|| * A    {r_p:+.3f}")

    print(f"\nALPHA SWEEP — alignment of the dropout/attenuation training states:")
    for al in ALPHAS:
        print(f"  alpha={al:<7.4f} A = {np.mean(alpha_A[al]):+.3f}")

    p_a = ("dconst" in rows and "invalid_0.55" in rows
           and rows["dconst"]["A"] > rows["invalid_0.55"]["A"]
           and rows["dconst"]["nrm"] < rows["invalid_0.55"]["nrm"])
    p_b = r_p > r_n
    best = max(ALPHAS, key=lambda al: np.mean(alpha_A[al]))
    p_c = best == 0.0
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M6a A(dconst) > A(invalid_0.55) with smaller ||delta|| -> {'HIT' if p_a else 'MISS'}")
    if "dconst" in rows and "invalid_0.55" in rows:
        print(f"      dconst A={rows['dconst']['A']:.3f} |d|={rows['dconst']['nrm']:.1f} | "
              f"invalid_0.55 A={rows['invalid_0.55']['A']:.3f} "
              f"|d|={rows['invalid_0.55']['nrm']:.1f}")
    print(f"P-M6b ||delta||*A beats ||delta||: {r_p:+.3f} vs {r_n:+.3f} -> "
          f"{'HIT' if p_b else 'MISS'}")
    print(f"P-M6c alpha=0 is maximally aligned: argmax = {best} -> {'HIT' if p_c else 'MISS'}")

    out = PROJECT / "results" / "gapfill" / "m6_alignment.json"
    out.write_text(json.dumps({"rows": rows, "alpha_A": {str(k): float(np.mean(v))
                                                         for k, v in alpha_A.items()},
                               "spearman": {"norm": r_n, "A": r_a, "prod": r_p},
                               "P_M6a": bool(p_a), "P_M6b": bool(p_b),
                               "P_M6c": bool(p_c)}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
