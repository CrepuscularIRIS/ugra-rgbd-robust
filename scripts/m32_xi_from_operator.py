"""M32 — IS xi A PROPERTY OF THE OPERATOR G, OR ONLY OF REAL DEPTH DISPLACEMENTS?

THE CLAIM UNDER TEST. `plan/THE-PRINCIPLE.md` states that xi (26.6 DFormerv2 / 24.0 CMNeXt) is "the
spatial coherence length of the decoder's output-sensitive subspace" — i.e. a property of the
operator G. But xi was MEASURED by block-shuffling a REAL depth-induced displacement field. That
procedure can only ever report a property of the real field crossed with the operator. **xi has never
been measured on the operator alone**, so the mechanistic reading is unearned.

THE TEST, and it inverts the original one. Block-shuffle DESTROYS coherence in a real field. Here we
BUILD coherence out of noise and ask at what scale the operator starts to respond:

    generate a random direction v_L with spatial correlation length L, at MATCHED norm
    measure its Rayleigh quotient  R(L) = 1/2 * v_L^T G v_L / ||v_L||^2

Only a JVP is needed, because  v^T G v = || F^(1/2) J_f v ||^2  — no eigendecomposition, no VJP.
The JVP is the finite-difference one validated in M31 (identity accurate to 1.5% at alpha = 0.02),
used at eps inside the measured stability plateau.

    If xi is a property of G, R(L) rises with L and half-saturates near L = xi = 26.6.
    If xi is a property of the real depth field, R(L) will half-saturate somewhere else, and the
    "coherence length of G's sensitive subspace" reading must be withdrawn.

REFERENCE POINTS, both already measured on this checkpoint:
    matched-magnitude RANDOM injection  R = 0.12 - 0.19   (the bulk floor)
    the REAL displacement field         R = 1.84          (the top of the range)

CANARY (blocking): the JVP stability check from M31 must hold at the eps used here, and a
white-noise direction (L -> 0) must land near the measured random floor. If a random direction does
not reproduce the known floor, the probe is not measuring what it claims.

Prereg + outcome table: .grill/prediction-ledger.md (2026-08-10, M32).

    python3 scripts/m32_xi_from_operator.py --gpu 1 --n-eval 24
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
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
# M31's plateau is a property of the PERTURBATION MAGNITUDE, not of eps: there the step was
# eps*||D|| = 0.05 * 245 ~ 12 (about 1.1% of ||Z|| ~ 1110). Directions here are unit-norm, so the
# step must be set in absolute terms or the probe falls back into the unstable eps->0 regime — which
# is exactly what the canary caught on the first run (rel 1.48 at a step of 0.05).
STEP_NORM = 12.0
XI_REF = 26.6                        # block-shuffle value this probe must either confirm or refute
LENGTHS = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 26.6, 40.0, 64.0, 128.0]
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402


def correlated_field(h0, w0, L, gen, device):
    """Unit-norm random field on the stage-0 grid with spatial correlation length ~L.

    L = 0 is white noise (the bulk-floor control). Correlation is imposed by Gaussian smoothing
    white noise with sigma = L/2, which gives an autocorrelation half-width of ~L.
    """
    x = torch.randn((1, 1, h0, w0), generator=gen, device="cpu").to(device)
    if L > 0:
        sigma = L / 2.0
        r = max(1, int(3 * sigma))
        k = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
        g = torch.exp(-(k ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).view(1, 1, -1)
        x = F.conv2d(x, g.view(1, 1, 1, -1), padding=(0, r))
        x = F.conv2d(x, g.view(1, 1, -1, 1), padding=(r, 0))
    return x / x.norm().clamp(min=1e-12)


def fisher_quadform(logits_clean, v):
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

    # The historical floor (0.12-0.19) and real-field value (1.84) were measured on UPSAMPLED
    # logits, where the summed KL is ~16x larger than at native resolution. Reading a native-KL
    # number against an upsampled-KL reference is a unit mismatch, and the canary caught exactly
    # that on the first run (white noise gave 1e-4 against a 0.12-0.19 reference). This probe
    # therefore matches the historical convention so the canary can adjudicate.
    OUT_SIZE = [None]

    def dec_raw(feats):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        if OUT_SIZE[0] is not None:
            o = F.interpolate(o, size=OUT_SIZE[0], mode="bilinear", align_corners=False)
        return o

    def rayleigh(Zc, lc, direction):
        """R(v) = 1/2 v^T G v / ||v||^2, via JVP + closed-form Fisher. Only forwards."""
        nrm = sum(float((d ** 2).sum()) for d in direction) ** 0.5
        if nrm < 1e-12:
            return float("nan")
        u = [d / nrm for d in direction]
        lp = dec_raw([z + STEP_NORM * d for z, d in zip(Zc, u)])
        v = (lp - lc) / STEP_NORM
        return 0.5 * fisher_quadform(lc, v)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    acc = {L: [] for L in LENGTHS}
    real_R = []
    canary_done = False
    canary_rel = None

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            OUT_SIZE[0] = tuple(rgb.shape[-2:])
            Zc = cap_(rgb, dep)
            lc = dec_raw(Zc)
            dev = Zc[0].device
            h0, w0 = Zc[0].shape[-2], Zc[0].shape[-1]
            gen = torch.Generator().manual_seed(SEED + i)

            # reference: the REAL depth-induced displacement, same normalisation
            Zk = cap_(rgb, corrupt(dep, COND["dconst"][1]))
            Dreal = [k - z for k, z in zip(Zk, Zc)]
            rr = rayleigh(Zc, lc, Dreal)
            real_R.append(rr)

            if not canary_done:
                u = [d / (sum(float((x ** 2).sum()) for x in Dreal) ** 0.5) for d in Dreal]
                lp1 = dec_raw([z + STEP_NORM * d for z, d in zip(Zc, u)])
                lp2 = dec_raw([z + 0.5 * STEP_NORM * d for z, d in zip(Zc, u)])
                v1 = (lp1 - lc) / STEP_NORM
                v2 = (lp2 - lc) / (0.5 * STEP_NORM)
                canary_rel = float((v1 - v2).norm() / v1.norm().clamp(min=1e-12))
                print(f"\nCANARY JVP stability at step={STEP_NORM} (1.1% of ||Z||): {canary_rel:.5f} (need < 0.12)")
                if canary_rel >= 0.12:
                    raise SystemExit("REFUSING: outside the measured stability plateau")
                canary_done = True

            # channel loading fixed per image, so ONLY the spatial scale varies across L
            chan = [torch.randn((1, z.shape[1], 1, 1), generator=gen, device="cpu").to(dev)
                    for z in Zc]
            for L in LENGTHS:
                f0 = correlated_field(h0, w0, L, gen, dev)
                direction = []
                for z, cw in zip(Zc, chan):
                    s = F.interpolate(f0, size=(z.shape[-2], z.shape[-1]),
                                      mode="bilinear", align_corners=False)
                    direction.append(s * cw)
                acc[L].append(rayleigh(Zc, lc, direction))
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{a.n_eval}", flush=True)

    R = {L: sum(v) / len(v) for L, v in acc.items()}
    Rreal = sum(real_R) / len(real_R)
    print(f"\n  REAL displacement (dconst) Rayleigh quotient R = {Rreal:.4f}")
    print(f"  reference: matched-magnitude random injection measured 0.12-0.19; real field 1.84\n")
    print(f"{'L (stage-0 cells)':>20s} {'R(L)':>10s} {'frac of real':>13s}")
    for L in LENGTHS:
        print(f"{L:20.1f} {R[L]:10.4f} {R[L]/Rreal:13.3f}")

    R0 = R[0.0]
    Rmax = max(R[L] for L in LENGTHS)
    half = R0 + 0.5 * (Rmax - R0)
    L_half = None
    for j in range(1, len(LENGTHS)):
        a1, b1 = LENGTHS[j - 1], LENGTHS[j]
        r1, r2 = R[a1], R[b1]
        if (r1 - half) * (r2 - half) <= 0 and r2 != r1:
            L_half = a1 + (b1 - a1) * (half - r1) / (r2 - r1)
            break
    order = [R[L] for L in LENGTHS]
    mono = all(order[j] >= order[j - 1] - 1e-9 for j in range(1, len(order) - 1))

    print(f"\n  white-noise floor R(L=0) = {R0:.4f}   (measured random floor 0.12-0.19)")
    print(f"  saturation R_max = {Rmax:.4f}   half-height = {half:.4f}")
    print(f"  HALF-SATURATION LENGTH L_half = {L_half if L_half else float('nan'):.2f}"
          f"   vs block-shuffle xi = {XI_REF}")

    p_can = 0.10 <= R0 <= 0.25
    p_a = mono
    p_b = Rmax >= 0.5 * Rreal
    p_c = (L_half is not None) and (XI_REF / 2 <= L_half <= XI_REF * 2)
    print(f"\n  CANARY (white noise reproduces the 0.12-0.19 floor): {'PASS' if p_can else 'FAIL'}"
          f"  ({R0:.4f})")
    print(f"  P-M32a (R(L) monotone rising):            {'HIT' if p_a else 'MISS'}")
    print(f"  P-M32b (R_max >= 50% of the real field):  {'HIT' if p_b else 'MISS'}"
          f"  ({Rmax/Rreal:.3f})")
    print(f"  P-M32c (L_half within 2x of xi = 26.6):   {'HIT' if p_c else 'MISS'}"
          f"  (L_half = {L_half if L_half else float('nan'):.2f})")

    (GAPFILL / "m32_xi_from_operator.json").write_text(json.dumps(
        {"canary_jvp": canary_rel, "step_norm": STEP_NORM, "n_eval": a.n_eval,
         "R_by_L": {str(L): R[L] for L in LENGTHS}, "R_real": Rreal,
         "R0": R0, "Rmax": Rmax, "L_half": L_half, "xi_ref": XI_REF,
         "canary_floor": bool(p_can), "P_M32a": bool(p_a), "P_M32b": bool(p_b),
         "P_M32c": bool(p_c)}, indent=1))
    print(f"\nwrote {GAPFILL / 'm32_xi_from_operator.json'}")


if __name__ == "__main__":
    main()
