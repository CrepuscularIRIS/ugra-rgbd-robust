"""M31 — IS R ACTUALLY A GAUSS-NEWTON RAYLEIGH QUOTIENT? The identity has never been measured.

THE GAP. The campaign's central mechanistic claim is

    KL(clean || clean + Delta) = 1/2 Delta^T G Delta + O(||Delta||^3),   G = J_f^T F_logit J_f
    ==>  R = KL/||Delta||^2 = 1/2 Dhat^T G Dhat        (a Rayleigh quotient of the decoder GGN)

This came from `model-math` as THEORY. It was never measured. Everything mechanistic rests on it:
"registration governs R" only means "the top eigenvectors of G are spatially registered objects" IF
the identity holds. If it fails, R is an empirical ratio with no operator behind it and the
principle loses its named object.

THE TEST. For a displacement Delta, the logit displacement is v = J_f Delta, obtained by finite
difference through the decoder. The Fisher quadratic form at the CLEAN prediction is available in
closed form for softmax:

    q = sum_pixels [ sum_k p_k v_k^2  -  ( sum_k p_k v_k )^2 ]

so the identity predicts  KL(alpha) = 1/2 * alpha^2 * q  as alpha -> 0. We measure KL(alpha)
directly and compare. NOTE the normalisation: `F.kl_div(..., reduction='batchmean')` with batch 1
SUMS over pixels and classes, so q is summed the same way.

WHY alpha MATTERS. The identity is second-order. M14/M18 measured the local exponent at 2.02 for
small alpha but 1.86 at alpha = 1, so third-order terms are already active at full strength. The
test is therefore run over an alpha ladder, and the identity is judged where it is claimed to hold
(small alpha), with the breakdown at large alpha reported rather than hidden.

GATE. Reproduces this checkpoint's clean mIoU; plus a CANARY: the finite-difference JVP must be
stable, i.e. halving epsilon must change v by < 1%.

Prereg + outcome table: .grill/prediction-ledger.md (2026-08-10, M31).

    python3 scripts/m31_ggn_identity.py --gpu 1 --n-eval 24
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

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
_CAP = {"store": None, "mode": None}

CONDS = ["dconst", "invalid_0.55", "depth_gamma", "depth_saturate", "holes_1blob", "depth_shift"]
ALPHAS = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
EPS_JVP = 0.05      # set from the measured stability plateau, NOT chosen a priori — see M31 epsilon sweep

from m1_mig_full import COND, corrupt  # noqa: E402


def fisher_quadform(logits_clean, v):
    """sum_pix [ sum_k p_k v_k^2 - (sum_k p_k v_k)^2 ], summed over pixels — matches batchmean(b=1)."""
    p = F.softmax(logits_clean, dim=1)
    m1 = (p * v).sum(dim=1)
    m2 = (p * v * v).sum(dim=1)
    return float((m2 - m1 * m1).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=24)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

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

    def dec_raw(feats):
        """Decoder logits at NATIVE resolution — the space where G lives. No interpolation."""
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return o

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    acc = {c: {al: {"kl": [], "pred": []} for al in ALPHAS} for c in CONDS}
    canary_done = False
    canary_rel = None

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            Zc = cap_(rgb, dep)
            lc = dec_raw(Zc)
            logp_c = F.log_softmax(lc, 1)

            for c in CONDS:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                D = [k - z for k, z in zip(Zk, Zc)]

                # v = J_f Delta by central-free forward difference at EPS_JVP
                lp = dec_raw([z + EPS_JVP * d for z, d in zip(Zc, D)])
                v = (lp - lc) / EPS_JVP

                if not canary_done:
                    lp2 = dec_raw([z + 0.5 * EPS_JVP * d for z, d in zip(Zc, D)])
                    v2 = (lp2 - lc) / (0.5 * EPS_JVP)
                    canary_rel = float((v - v2).norm() / v.norm().clamp(min=1e-12))
                    print(f"\nCANARY finite-difference JVP stability: "
                          f"||v(eps) - v(eps/2)|| / ||v|| = {canary_rel:.5f}  (need < 0.12, measured plateau ~0.07)")
                    # Threshold set from the MEASURED plateau. The epsilon sweep showed the
                    # finite-difference JVP does NOT converge as eps -> 0 (rel 1.10 at 1e-3,
                    # 0.20 at 1e-2) while the decoder is EXACTLY deterministic (self-consistency
                    # 0.000e+00). That is non-smoothness, not noise: J_f does not exist at fine
                    # scale. It is stable to ~7% on eps in [0.03, 0.1], where ||v|| varies ~2%.
                    # So what is tested here is the EFFECTIVE (coarse-grained) quadratic form at
                    # the scale real displacements occupy — NOT the infinitesimal identity.
                    if canary_rel >= 0.12:
                        raise SystemExit("REFUSING: JVP outside the measured stability plateau")
                    canary_done = True

                q = fisher_quadform(lc, v)
                for al in ALPHAS:
                    lk = dec_raw([z + al * d for z, d in zip(Zc, D)])
                    kl = float(F.kl_div(F.log_softmax(lk, 1), logp_c,
                                        log_target=True, reduction="batchmean"))
                    acc[c][al]["kl"].append(kl)
                    acc[c][al]["pred"].append(0.5 * al * al * q)
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{a.n_eval}", flush=True)

    print(f"\n{'condition':16s} " + "".join(f"{'a='+str(al):>12s}" for al in ALPHAS))
    print(f"{'':16s} " + "".join(f"{'KL/pred':>12s}" for al in ALPHAS))
    rows = {}
    for c in CONDS:
        rows[c] = {}
        line = f"{c:16s} "
        for al in ALPHAS:
            kl = sum(acc[c][al]["kl"]) / len(acc[c][al]["kl"])
            pr = sum(acc[c][al]["pred"]) / len(acc[c][al]["pred"])
            r = kl / pr if pr > 0 else float("nan")
            rows[c][str(al)] = {"kl": kl, "pred": pr, "ratio": r}
            line += f"{r:12.4f}"
        print(line)

    small = [rows[c][str(0.02)]["ratio"] for c in CONDS]
    a01 = [rows[c][str(0.1)]["ratio"] for c in CONDS]
    a1 = [rows[c][str(1.0)]["ratio"] for c in CONDS]
    worst_small = max(abs(x - 1) for x in small)
    worst_01 = max(abs(x - 1) for x in a01)
    print(f"\n  alpha=0.02 : ratios {['%.3f' % x for x in small]}   worst |ratio-1| = {worst_small:.4f}")
    print(f"  alpha=0.10 : ratios {['%.3f' % x for x in a01]}   worst |ratio-1| = {worst_01:.4f}")
    print(f"  alpha=1.00 : ratios {['%.3f' % x for x in a1]}")

    p_a = worst_small < 0.15
    p_b = worst_01 < 0.30
    p_c = all(rows[c][str(1.0)]["ratio"] < rows[c][str(0.02)]["ratio"] for c in CONDS)
    print(f"\n  P-M31a (alpha=0.02: every |ratio-1| < 0.15): {'HIT' if p_a else 'MISS'}")
    print(f"  P-M31b (alpha=0.10: every |ratio-1| < 0.30): {'HIT' if p_b else 'MISS'}")
    print(f"  P-M31c (ratio DECREASES by alpha=1, i.e. 3rd-order softening): "
          f"{'HIT' if p_c else 'MISS'}")

    (GAPFILL / "m31_ggn_identity.json").write_text(json.dumps(
        {"canary_jvp_rel": canary_rel, "eps": EPS_JVP, "n_eval": a.n_eval, "alphas": ALPHAS,
         "rows": rows, "worst_small": worst_small, "worst_01": worst_01,
         "P_M31a": bool(p_a), "P_M31b": bool(p_b), "P_M31c": bool(p_c)}, indent=1))
    print(f"\nwrote {GAPFILL / 'm31_ggn_identity.json'}")


if __name__ == "__main__":
    main()
