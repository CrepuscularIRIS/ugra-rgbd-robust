"""M27 (agent item C1) — WHICH LEVER DID MODALITY DROPOUT PULL?

THE QUESTION. The law decomposes damage as KL = R * ||Delta||^2. A model can reduce damage two ways
and they are NOT the same thing:

  * shrink ||Delta||  -> the depth pathway becomes inert. This is the DEGENERATE WALK toward the
                         RGB-only model, the failure mode every penalty-based method collapses into.
  * shrink R at fixed ||Delta||  -> depth still displaces the representation just as much, but the
                         displacement now lands in a FLATTER direction of the decoder's GGN. This is
                         genuine repair.

Nothing else in this campaign distinguishes these two. The clean-accuracy floor cannot: the measured
clean cost of depth-dropout p=0.5 was -0.29 mIoU against MDE 1.71, so the floor returns the same
number whether the depth pathway survived or was mostly abandoned (Revision 32).

THE ARTIFACTS. Both checkpoints already exist, 500 epochs, same seed family:
  mainB_baseline  clean 55.18   entire_missing 50.33
  mainC_dropout   clean 54.48   entire_missing 53.10     (+2.77 missing for -0.70 clean)

THE GATE — and it is sharper than an mIoU band. This probe reads internals, so it must reproduce a
known reference number first. It reproduces the BASELINE's per-condition R against the values M14
recorded for the same checkpoint. mIoU at small n is not usable as a gate here (M17: 35.14 at n=20
vs 52.45 at n=30, identical everything), so the gate is on the exact quantity being read.

Prereg + outcome table: .grill/prediction-ledger.md (2026-08-10, M27).

    python3 scripts/m27_dropout_lever.py --gpu 1 --n-eval 30
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
_CAP = {"store": None, "mode": None}

CKPTS = {
    "baseline": PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth",
    "dropout":  PROJECT / "results" / "p0" / "mainC_dropout_s20260801" / "ckpt" / "latest.pth",
}

from m1_mig_full import COND, corrupt  # noqa: E402


def measure(ckpt_path, cfg, names, loader_fn, n_eval):
    """Return per-condition ||Delta||, KL, R and the clean feature norm, for ONE checkpoint."""
    from models.builder import EncoderDecoder
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(miss.missing_keys)
    print(f"  checkpoint: {got}/{len(sd)} tensors loaded")
    if got < 0.9 * len(sd):
        raise SystemExit("REFUSING: <90% loaded")
    model.eval().cuda()
    _orig = model.decode_head.forward

    def _h(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _h

    def cap_(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    acc = {c: {"kl": [], "nrm": []} for c in names}
    zclean = []
    canary = False
    for i, b in enumerate(loader_fn()):
        if i >= n_eval:
            break
        rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
        size = rgb.shape[-2:]
        Zc = cap_(rgb, dep)
        zl = dec(Zc, size)
        if not canary:
            d = float((zl - dec(Zc, size)).abs().max())
            if d > 0:
                raise SystemExit(f"REFUSING: nondeterministic decoder ({d:.3e})")
            print(f"  CANARY self-comparison max|diff| = {d:.3e}  OK")
            canary = True
        logp = F.log_softmax(zl, 1)
        zclean.append(float(sum(float((t ** 2).sum()) for t in Zc) ** 0.5))
        for c in names:
            Zk = cap_(rgb, corrupt(dep, COND[c][1]))
            acc[c]["nrm"].append(float(sum(float(((k - z) ** 2).sum())
                                           for k, z in zip(Zk, Zc)) ** 0.5))
            acc[c]["kl"].append(float(F.kl_div(F.log_softmax(dec(Zk, size), 1), logp,
                                               log_target=True, reduction="batchmean")))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{n_eval}")
    zc = sum(zclean) / len(zclean)
    out = {}
    for c in names:
        kl = sum(acc[c]["kl"]) / len(acc[c]["kl"])
        nm = sum(acc[c]["nrm"]) / len(acc[c]["nrm"])
        out[c] = {"kl": kl, "norm": nm, "R": kl / max(nm ** 2, 1e-12), "rel_norm": nm / zc}
    return out, zc


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
    print(f"conditions: {len(names)}")

    engine = Engine(custom_parser=None)
    vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

    res = {}
    for tag, path in CKPTS.items():
        print(f"\n=== {tag} ===")
        res[tag], _ = measure(path, cfg, names, lambda: vl, a.n_eval)

    # --- GATE: baseline must reproduce M14's recorded R for the SAME checkpoint ---
    # m14 stores R=None where the denominator vanishes (the guarded-ratio fix from D-1); skip those.
    gate_conds = [c for c in names if m14[c].get("R") is not None]
    ratios = sorted(res["baseline"][c]["R"] / m14[c]["R"] for c in gate_conds)
    med = ratios[len(ratios) // 2]
    print(f"\n=== REFERENCE GATE (baseline R vs M14 recorded) ===")
    print(f"  {len(gate_conds)}/{len(names)} conditions gated (m14 R=None skipped)")
    print(f"  median ratio = {med:.4f}   range [{ratios[0]:.3f}, {ratios[-1]:.3f}]   need |med-1| < 0.10")
    gate = abs(med - 1.0) < 0.10
    print(f"  gate: {'PASS' if gate else 'FAIL — internals must not be read'}")

    print(f"\n{'condition':22s} {'band':>4s} | {'R_base':>8s} {'R_drop':>8s} {'dR%':>7s} "
          f"| {'|D|base':>9s} {'|D|drop':>9s} {'d|D|%':>7s} | {'dKL%':>7s}")
    rows = {}
    for c in sorted(names, key=lambda x: -res["baseline"][x]["kl"]):
        b, d = res["baseline"][c], res["dropout"][c]
        dR = 100 * (d["R"] / b["R"] - 1)
        dN = 100 * (d["norm"] / b["norm"] - 1)
        dK = 100 * (d["kl"] / b["kl"] - 1)
        rows[c] = {"band": m14[c]["band"], "R_base": b["R"], "R_drop": d["R"],
                   "n_base": b["norm"], "n_drop": d["norm"],
                   "kl_base": b["kl"], "kl_drop": d["kl"],
                   "reln_base": b["rel_norm"], "reln_drop": d["rel_norm"],
                   "dR_pct": dR, "dN_pct": dN, "dKL_pct": dK}
        print(f"{c:22s} {m14[c]['band']:>4s} | {b['R']:8.4f} {d['R']:8.4f} {dR:+7.1f} "
              f"| {b['norm']:9.2f} {d['norm']:9.2f} {dN:+7.1f} | {dK:+7.1f}")

    dc = rows.get("dconst") or rows.get("entire_missing")
    key = "dconst" if "dconst" in rows else "entire_missing"
    off = [c for c in rows if rows[c]["band"] == "OFF"]
    off_dR = sum(abs(rows[c]["dR_pct"]) for c in off) / max(len(off), 1)

    p_a = dc["dKL_pct"] <= -30
    p_b = abs(dc["dN_pct"]) > abs(dc["dR_pct"])       # walk dominates
    p_c = off_dR < 20
    print(f"\n--- at the DROPOUT TRAINING POINT ({key}) ---")
    print(f"  dKL = {dc['dKL_pct']:+.1f}%   d||Delta|| = {dc['dN_pct']:+.1f}%   dR = {dc['dR_pct']:+.1f}%")
    print(f"  relative ||Delta|| (norm'd by ||Z_clean||): "
          f"{dc['reln_base']:.4f} -> {dc['reln_drop']:.4f} "
          f"({100*(dc['reln_drop']/dc['reln_base']-1):+.1f}%)")
    print(f"  mean |dR| over OFF-support conditions = {off_dR:.1f}%")
    print(f"\n  P-M27a (dKL at training point <= -30%):        {'HIT' if p_a else 'MISS'}")
    print(f"  P-M27b (||Delta|| reduction DOMINATES R red.): {'HIT' if p_b else 'MISS'}"
          f"   -> {'WALK account' if p_b else 'R account'}")
    print(f"  P-M27c (OFF-support mean |dR| < 20%):          {'HIT' if p_c else 'MISS'}")

    out = {"gate": gate, "gate_median_ratio": med, "n_eval": a.n_eval, "rows": rows,
           "training_point": key, "off_mean_abs_dR": off_dR,
           "P_M27a": bool(p_a), "P_M27b": bool(p_b), "P_M27c": bool(p_c)}
    (GAPFILL / "m27_dropout_lever.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GAPFILL / 'm27_dropout_lever.json'}")


if __name__ == "__main__":
    main()
