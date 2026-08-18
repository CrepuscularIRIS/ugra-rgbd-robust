"""M13 — does damage propagation FACTORISE? The two-stage decomposition, using the KL instrument.

    Delta --(stage A: representation -> output distribution)--> KL --(stage B: -> decision)--> dmIoU

Every correlation on the register collapses A and B into one number. M12 showed KL is far tighter
than mIoU (shuf/real spans 0.106-0.138 in KL vs 0.011-0.444 in mIoU on identical passes), which makes
the split measurable for the first time.

If stage A is a one-parameter function of ||Delta|| and all condition-specific structure sits in
stage B, the in-band exchange-rate puzzle is not about representation geometry at all -- it is about
how close the decisions already were. That also hands stage A to R2 (one dimension is enough).

Prereg, outcome table, severity: .grill/prediction-ledger.md (2026-08-10, M13).

    python3 scripts/m13_two_stage.py --gpu 1 --n-eval 30
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
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
NCLS = 40
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, spearman, corrupt  # noqa: E402

EXTRA_SHIFTS = {"shift_1": 1, "shift_8": 8, "shift_16": 16, "shift_32": 32}


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def miou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    return float((tp / u)[u > 0].mean()) * 100.0


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
    from models.builder import EncoderDecoder
    from depth_corruptions import depth_shift
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

    def _hook(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _hook

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
            out = _orig(list(feats))
        if isinstance(out, dict) and "pred" in out:
            out = out["pred"]
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    names = list(COND) + list(EXTRA_SHIFTS)

    def corrupt_any(d, n):
        return depth_shift(d.clone(), dx=EXTRA_SHIFTS[n], dy=0) if n in EXTRA_SHIFTS \
            else corrupt(d, COND[n][1])

    H = {"clean": torch.zeros(NCLS, NCLS, dtype=torch.long)}
    acc = {}
    for n in names:
        H[n] = torch.zeros(NCLS, NCLS, dtype=torch.long)
        acc[n] = {"kl": [], "nrm": []}

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        canary = False
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            lab = b["label"].cuda().squeeze(0).long()
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)
            if not canary:
                d = float((zl - dec(Zc, size)).abs().max())
                print(f"\nCANARY: max|diff| = {d:.3e}")
                if d > 0:
                    raise SystemExit("REFUSING: nondeterministic decoder")
                print("  bases pinned.\n")
                canary = True
            logp_c = F.log_softmax(zl, 1)
            H["clean"] += hist(zl.argmax(1).squeeze(0).cpu(), lab.cpu())
            for n in names:
                Zk = cap_(rgb, corrupt_any(dep, n))
                nrm = float(sum(float(((k - z) ** 2).sum()) for k, z in zip(Zk, Zc)) ** 0.5)
                zz = dec(Zk, size)
                H[n] += hist(zz.argmax(1).squeeze(0).cpu(), lab.cpu())
                acc[n]["kl"].append(float(F.kl_div(F.log_softmax(zz, 1), logp_c,
                                                   log_target=True, reduction="batchmean")))
                acc[n]["nrm"].append(nrm)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.n_eval}")

    base = miou(H["clean"])
    m8 = json.loads((GAPFILL / "m8_local_support.json").read_text())
    cl8 = m8["clean_ref"]
    band = {n: ("OFF" if (m8["rows"][n]["knn"] - cl8) > 0.05 else "IN")
            if n in m8["rows"] else "IN" for n in names}

    rows = {}
    for n in names:
        rows[n] = {"kl": float(np.mean(acc[n]["kl"])), "norm": float(np.mean(acc[n]["nrm"])),
                   "dmiou": base - miou(H[n]), "band": band[n]}

    use = [n for n in names if rows[n]["norm"] > 1.0]      # exclude depth_offset (norm ~0)
    lx = np.log(np.array([rows[n]["norm"] for n in use]))
    ly = np.log(np.array([max(rows[n]["kl"], 1e-9) for n in use]))
    p = float(np.polyfit(lx, ly, 1)[0])
    for n in names:
        rows[n]["kl_over_normp"] = rows[n]["kl"] / max(rows[n]["norm"] ** p, 1e-12) \
            if rows[n]["norm"] > 1.0 else None
        rows[n]["exchange"] = rows[n]["dmiou"] / rows[n]["kl"] * 1e4 if rows[n]["kl"] > 1e-6 else None

    print(f"\nclean mIoU (subset) {base:.2f}   fitted exponent p in KL ~ ||D||^p : {p:.3f}\n")
    print(f"{'condition':20s} {'band':>4s} {'||D||':>9s} {'KL':>12s} {'dmIoU':>7s} "
          f"{'KL/||D||^p':>11s} {'dmIoU/KL*1e4':>13s}")
    for n in sorted(names, key=lambda x: -rows[x]["kl"]):
        r = rows[n]
        k = f"{r['kl_over_normp']:11.4f}" if r["kl_over_normp"] is not None else "       VOID"
        e = f"{r['exchange']:13.3f}" if r["exchange"] is not None else "         VOID"
        print(f"{n:20s} {r['band']:>4s} {r['norm']:9.3f} {r['kl']:12.1f} {r['dmiou']:7.2f} {k}{e}")

    IN = [n for n in use if rows[n]["band"] == "IN"]
    OFF = [n for n in use if rows[n]["band"] == "OFF"]
    sp_all = spearman([rows[n]["kl"] for n in use], [rows[n]["norm"] for n in use])
    resid = [rows[n]["kl_over_normp"] for n in use]
    spread = max(resid) / max(min(resid), 1e-12)
    ex_in = [rows[n]["exchange"] for n in IN if rows[n]["exchange"] is not None]
    ex_off = [rows[n]["exchange"] for n in OFF if rows[n]["exchange"] is not None]
    band_ratio = (np.mean(ex_off) / np.mean(ex_in)) if ex_in and ex_off else float("nan")
    sp_in = spearman([rows[n]["kl"] for n in IN], [rows[n]["dmiou"] for n in IN])
    sp_off = spearman([rows[n]["kl"] for n in OFF], [rows[n]["dmiou"] for n in OFF])

    p_a = sp_all is not None and sp_all >= 0.90
    p_b = 1.5 <= p <= 2.5
    p_c = spread <= 3.0
    p_d = abs(band_ratio) >= 2.0
    p_e = sp_in is not None and sp_in >= 0.85

    print(f"\n--- STAGE A: representation -> output distribution ---")
    print(f"  Spearman(KL, ||D||) over {len(use)} conditions = {sp_all:+.3f}")
    print(f"  fitted exponent p = {p:.3f}")
    print(f"  residual KL/||D||^p spread = {spread:.2f}x   "
          f"(min {min(resid):.4f}  max {max(resid):.4f})")
    print(f"\n--- STAGE B: output distribution -> decision ---")
    print(f"  exchange rate dmIoU/KL*1e4 : IN mean {np.mean(ex_in):.3f} (n={len(ex_in)})  "
          f"OFF mean {np.mean(ex_off):.3f} (n={len(ex_off)})   ratio {band_ratio:.2f}x")
    print(f"  Spearman(KL, dmIoU)  IN {sp_in:+.3f}   OFF {sp_off:+.3f}")

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M13a Spearman(KL,||D||) >= 0.90        -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M13b p in [1.5,2.5]                    -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M13c residual spread <= 3x             -> {'HIT' if p_c else 'MISS'}")
    print(f"P-M13d band exchange-rate ratio >= 2x    -> {'HIT' if p_d else 'MISS'}")
    print(f"P-M13e in-band Spearman(KL,dmIoU) >= 0.85 -> {'HIT' if p_e else 'MISS'}")

    print(f"\n--- VERDICT ---")
    if p_c:
        print("  STAGE A IS CONDITION-INDEPENDENT. One curve in ||Delta|| carries the")
        print("  representation->output map across every corruption family. All condition-specific")
        print("  structure must live in STAGE B (decisions/margins). R2 wins stage A.")
    else:
        print("  STAGE A IS NOT NORM-GOVERNED. The residual KL/||D||^p is the new ordering object,")
        print("  and it is a representation-level quantity R2 does not supply.")
    if p_d:
        print("  The SUPPORT BANDS are a STAGE-B phenomenon -- they separate on how much decision")
        print("  damage a unit of output-distribution change buys. No current account says this.")

    o = GAPFILL / "m13_two_stage.json"
    o.write_text(json.dumps({"clean_miou": base, "exponent_p": p, "rows": rows,
                             "spearman_kl_norm": sp_all, "residual_spread": spread,
                             "exchange_in": float(np.mean(ex_in)),
                             "exchange_off": float(np.mean(ex_off)),
                             "band_ratio": float(band_ratio),
                             "spearman_kl_dmiou_in": sp_in, "spearman_kl_dmiou_off": sp_off,
                             "P_M13a": p_a, "P_M13b": p_b, "P_M13c": p_c,
                             "P_M13d": bool(p_d), "P_M13e": p_e, "n_eval": a.n_eval},
                            indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
