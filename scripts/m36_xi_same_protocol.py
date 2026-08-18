"""M36 — xi FOR THE REAL FIELD vs xi FOR G's OWN DIRECTION, under the IDENTICAL protocol.

THE TWO OPEN ITEMS THIS CLOSES.

(4) UNIT AMBIGUITY. xi = 26.6 was produced by `block_shuffle(d, b)` applying the SAME block size b in
    every stage's own cells, so b means different physical extents at different stages. M34's
    l_G ~ 6-7 was an autocorrelation half-width converted to "stage-0-equivalent" cells. **Those two
    numbers were never in the same units**, so "l_G is 4x shorter than xi" was not a clean comparison.

(3) AN OPERATOR-COHERENCE MEASURE THAT DOES NOT DEPEND ON THE AUTOCORRELATION CONSTRUCTION. M34's
    l_G came from an autocorrelation half-width on an approximate dominant direction. A different
    estimator is needed before the conclusion rests on it.

BOTH ARE SOLVED BY ONE MOVE: run the ORIGINAL xi protocol — the exact `block_shuffle` from
`m17_cross_arch_alignment.py`, the same block sizes, the same half-height log-interpolation — on
**G's dominant direction** instead of on the real displacement. Then xi_real and xi_G are produced by
identical code in identical units, and the comparison is exact rather than converted.

    xi_real  : block-shuffle the real depth-induced displacement   (must reproduce ~26.6 = CANARY)
    xi_G     : block-shuffle G's dominant direction                (the quantity in question)

R_eff throughout (stage 0 causally inert, Revision 41). The dominant direction is the same 8-step
power iteration as M34, carrying M34's stated caveat that it composes a finite-difference JVP with an
autograd VJP and is therefore approximate.

Prereg: .grill/prediction-ledger.md (2026-08-10, M36).

    python3 scripts/m36_xi_same_protocol.py --gpu 1 --n-eval 16
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
for _p in (DFORMER, DFORMER / "utils", PROJECT / "diagnostics" / "corruptions", PROJECT / "scripts"):
    sys.path.insert(0, str(_p))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
STEP_NORM = 12.0
USED = [1, 2, 3]
N_POWER = 8
BLOCKS = [2, 4, 8, 16, 32, 64]        # identical to m17_cross_arch_alignment.py
XI_REF = 26.6
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402


def block_shuffle(d, b, gen):
    """VERBATIM from m17_cross_arch_alignment.py — the protocol that produced xi = 26.6."""
    B, C, H, W = d.shape
    ph, pw = (b - H % b) % b, (b - W % b) % b
    x = F.pad(d, (0, pw, 0, ph))
    Hp, Wp = x.shape[2], x.shape[3]
    nh, nw = Hp // b, Wp // b
    x = x.reshape(B, C, nh, b, nw, b).permute(0, 1, 2, 4, 3, 5).reshape(B, C, nh * nw, b, b)
    idx = torch.randperm(nh * nw, generator=gen).to(x.device)
    x = x[:, :, idx]
    x = x.reshape(B, C, nh, nw, b, b).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hp, Wp)
    return x[:, :, :H, :W]


def fisher_quadform(lc, v):
    p = F.softmax(lc, dim=1)
    m1 = (p * v).sum(dim=1)
    m2 = (p * v * v).sum(dim=1)
    return float((m2 - m1 * m1).sum())


def fisher_apply(lc, u):
    p = F.softmax(lc, dim=1)
    return p * u - p * (p * u).sum(dim=1, keepdim=True)


def xi_from_curve(curve):
    """VERBATIM logic from m17: half-height between first and last, log-interpolated in b."""
    lo, hi = curve[0], curve[-1]
    half = (lo + hi) / 2
    for i in range(len(curve) - 1):
        if curve[i] < half <= curve[i + 1]:
            f_ = (half - curve[i]) / (curve[i + 1] - curve[i])
            return float(math.exp(math.log(BLOCKS[i]) + f_ * (math.log(BLOCKS[i + 1])
                                                              - math.log(BLOCKS[i]))))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=16)
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
    if len(sd) - len(miss.missing_keys) < 0.9 * len(sd):
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

    OUT = [None]

    def dec_(feats, grad=False):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            o = _orig(list(feats))
            if isinstance(o, dict) and "pred" in o:
                o = o["pred"]
            return F.interpolate(o, size=OUT[0], mode="bilinear", align_corners=False)

    def R_eff(Zc, lc, D):
        n = sum(float((D[j] ** 2).sum()) for j in USED) ** 0.5
        if n < 1e-12:
            return float("nan")
        pert = list(Zc)
        for j in USED:
            pert[j] = Zc[j] + STEP_NORM * D[j] / n
        return 0.5 * fisher_quadform(lc, (dec_(pert) - lc) / STEP_NORM)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    curves = {"real": {b: [] for b in BLOCKS}, "gdir": {b: [] for b in BLOCKS}}
    base = {"real": [], "gdir": []}

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b_ in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b_["data"].cuda(), b_["modal_x"].cuda()
            OUT[0] = tuple(rgb.shape[-2:])
            Zc = cap_(rgb, dep)
            lc = dec_(Zc)
            gen = torch.Generator().manual_seed(SEED + i)

            Zk = cap_(rgb, corrupt(dep, COND["dconst"][1]))
            D = [k - z for k, z in zip(Zk, Zc)]

            # G's dominant direction (same procedure as M34)
            v = [torch.randn(Zc[j].shape, generator=gen, device="cpu").to(Zc[j].device)
                 for j in range(len(Zc))]
            for j in range(len(Zc)):
                if j not in USED:
                    v[j] = torch.zeros_like(v[j])
            for _ in range(N_POWER):
                n = sum(float((v[j] ** 2).sum()) for j in USED) ** 0.5
                pert = list(Zc)
                for j in USED:
                    pert[j] = Zc[j] + STEP_NORM * v[j] / n
                u = (dec_(pert) - lc) / STEP_NORM
                w = fisher_apply(lc, u)
                Zg = [Zc[j].clone().requires_grad_(True) for j in range(len(Zc))]
                out = dec_(Zg, grad=True)
                g = torch.autograd.grad((out * w.detach()).sum(), Zg, allow_unused=True)
                v = [torch.zeros_like(Zc[j]) if (g[j] is None or j not in USED) else g[j].detach()
                     for j in range(len(Zc))]

            for tag, field in (("real", D), ("gdir", v)):
                base[tag].append(R_eff(Zc, lc, field))
                for bb in BLOCKS:
                    sh = [block_shuffle(field[j], bb, gen) if j in USED else field[j]
                          for j in range(len(field))]
                    curves[tag][bb].append(R_eff(Zc, lc, sh))
            if (i + 1) % 4 == 0:
                print(f"  {i+1}/{a.n_eval}", flush=True)

    out = {}
    print(f"\n{'b':>5s} {'real: R_sh/R':>14s} {'G-dir: R_sh/R':>15s}")
    for tag in ("real", "gdir"):
        b0 = sum(base[tag]) / len(base[tag])
        out[tag] = {"R": b0, "curve": [sum(curves[tag][bb]) / len(curves[tag][bb]) / b0
                                       for bb in BLOCKS]}
    for k, bb in enumerate(BLOCKS):
        print(f"{bb:5d} {out['real']['curve'][k]:14.4f} {out['gdir']['curve'][k]:15.4f}")

    xi_r = xi_from_curve(out["real"]["curve"])
    xi_g = xi_from_curve(out["gdir"]["curve"])
    print(f"\n  R (real)  = {out['real']['R']:.4f}     R (G-dir) = {out['gdir']['R']:.4f}")
    print(f"  xi_real = {xi_r if xi_r else float('nan'):.2f}   (recorded reference {XI_REF})")
    print(f"  xi_G    = {xi_g if xi_g else float('nan'):.2f}")

    can = xi_r is not None and (XI_REF / 1.5 <= xi_r <= XI_REF * 1.5)
    print(f"\n  CANARY xi_real in [{XI_REF/1.5:.1f}, {XI_REF*1.5:.1f}]: {'PASS' if can else 'FAIL'}")
    if not can:
        print("  REFUSING to read xi_G: the protocol did not reproduce its own reference.")
    else:
        p_b = xi_g is not None and xi_g < xi_r / 2
        print(f"  P-M36b (xi_G < xi_real / 2, IDENTICAL protocol and units): "
              f"{'HIT' if p_b else 'MISS'}")
        out["P_M36b"] = bool(p_b)

    out.update({"xi_real": xi_r, "xi_G": xi_g, "canary": bool(can),
                "blocks": BLOCKS, "n_eval": a.n_eval, "xi_ref": XI_REF})
    (GAPFILL / "m36_xi_same_protocol.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GAPFILL / 'm36_xi_same_protocol.json'}")


if __name__ == "__main__":
    main()
