"""M34 — (A) the DOF-matched channel control M33 owed, and (B) xi from G's top eigenvector.

TWO OWED ITEMS, both named in the record.

(A) M33 found that destroying channel alignment collapses R to 3.7% of the real field while
destroying spatial arrangement leaves 13%. But the ORDERING was confounded: `spatial_shuffle` applies
ONE permutation across all channels, whereas `channel_shuffle` applied an INDEPENDENT permutation at
every one of H x W positions — vastly more degrees of freedom scrambled. The DOF-matched control is a
**single GLOBAL channel permutation**, the same at every position: it preserves spatial arrangement,
each channel's spatial pattern, and all magnitudes, and only relabels channel identity. That is the
structural counterpart of `spatial_shuffle` (one permutation applied uniformly across the other axis)
and it is what settles the ordering.

(B) xi (26.6) was measured by block-shuffling a REAL displacement, so it reports the real field
crossed with the operator. If the campaign's wording — a coherence length of the output-sensitive
subspace — is to mean anything, G's own top eigenvector should carry a comparable spatial scale.
Power iteration is now possible because the autograd VJP works (discovered in the M34 feasibility
check, which is also what exposed stage 0's inertness).

STAGE 0 IS EXCLUDED THROUGHOUT. `decode_head.in_index = [1,2,3]`: zeroing the stage-0 displacement
leaves KL bitwise unchanged while stage 0 carries 5.5-20% of ||Delta||^2 (Revision 41). All R values
here are R_eff = KL / ||Delta_{1:3}||^2.

HONEST NOTE ON (B): the power iteration composes a finite-difference JVP with an autograd VJP. Given
the decoder's measured non-smoothness (M31: the JVP diverges as eps -> 0), those two are not the same
linear map, so the resulting vector is an APPROXIMATE dominant direction, not an exact eigenvector.
It is reported as such. A unit caveat also applies: xi was measured per-stage on each stage's own
grid, so coherence lengths are reported per stage in that stage's cells AND in stage-0-equivalent
cells.

CANARY (blocking): `real` must reproduce the historical R, and `spatial_shuffled` must land inside
m14's recorded shuf_ratio range [0.076, 0.139].

Prereg: .grill/prediction-ledger.md (2026-08-10, M34).

    python3 scripts/m34_dof_matched_and_eigen.py --gpu 1 --n-eval 16
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
for _p in (DFORMER, DFORMER / "utils", PROJECT / "diagnostics" / "corruptions", PROJECT / "scripts"):
    sys.path.insert(0, str(_p))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
STEP_NORM = 12.0
USED = [1, 2, 3]                      # stage 0 is causally inert (Revision 41)
N_POWER = 8
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402


def fisher_quadform(lc, v):
    p = F.softmax(lc, dim=1)
    m1 = (p * v).sum(dim=1)
    m2 = (p * v * v).sum(dim=1)
    return float((m2 - m1 * m1).sum())


def fisher_apply(lc, u):
    p = F.softmax(lc, dim=1)
    return p * u - p * (p * u).sum(dim=1, keepdim=True)


def spatial_shuffle(D, gen):
    out = []
    for d in D:
        B, C, H, W = d.shape
        idx = torch.randperm(H * W, generator=gen).to(d.device)
        out.append(d.reshape(B, C, H * W)[:, :, idx].reshape(B, C, H, W))
    return out


def channel_shuffle_perposition(D, gen):
    out = []
    for d in D:
        B, C, H, W = d.shape
        flat = d.reshape(B, C, H * W)
        idx = torch.argsort(torch.rand((C, H * W), generator=gen), dim=0).to(d.device)
        out.append(torch.gather(flat, 1, idx.unsqueeze(0).expand(B, -1, -1)).reshape(B, C, H, W))
    return out


def channel_shuffle_global(D, gen):
    """ONE channel permutation, applied identically at every spatial position. DOF-matched."""
    out = []
    for d in D:
        B, C, H, W = d.shape
        idx = torch.randperm(C, generator=gen).to(d.device)
        out.append(d[:, idx, :, :])
    return out


def coherence_length(energy):
    """Half-width of the radially averaged spatial autocorrelation of a 2-D energy map."""
    e = energy - energy.mean()
    E = torch.fft.rfft2(e)
    ac = torch.fft.irfft2(E * E.conj(), s=e.shape)
    ac = ac / ac.flatten()[0].clamp(min=1e-12)
    H, W = e.shape
    yy = torch.arange(H, device=e.device).view(-1, 1)
    xx = torch.arange(W, device=e.device).view(1, -1)
    ry = torch.minimum(yy, H - yy).float()
    rx = torch.minimum(xx, W - xx).float()
    r = torch.sqrt(ry ** 2 + rx ** 2)
    rmax = int(min(H, W) // 2)
    prof = []
    for k in range(rmax):
        m = (r >= k) & (r < k + 1)
        prof.append(float(ac[m].mean()) if m.any() else float("nan"))
    for k in range(1, len(prof)):
        if prof[k] <= 0.5:
            if prof[k - 1] == prof[k]:
                return float(k)
            return (k - 1) + (prof[k - 1] - 0.5) / (prof[k - 1] - prof[k])
    return float(rmax)


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
    got = len(sd) - len(miss.missing_keys)
    print(f"checkpoint: {got}/{len(sd)} tensors loaded  |  in_index={model.decode_head.in_index}")
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
        """R over the USED stages only — stage 0 is inert (Revision 41)."""
        n = sum(float((D[j] ** 2).sum()) for j in USED) ** 0.5
        if n < 1e-12:
            return float("nan")
        pert = list(Zc)
        for j in USED:
            pert[j] = Zc[j] + STEP_NORM * D[j] / n
        v = (dec_(pert) - lc) / STEP_NORM
        return 0.5 * fisher_quadform(lc, v)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    ARMS = ["real", "spatial_shuffled", "chan_global", "chan_perposition", "both_shuffled"]
    acc = {k: [] for k in ARMS}
    xis = {j: [] for j in USED}

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            OUT[0] = tuple(rgb.shape[-2:])
            Zc = cap_(rgb, dep)
            lc = dec_(Zc)
            gen = torch.Generator().manual_seed(SEED + i)

            Zk = cap_(rgb, corrupt(dep, COND["dconst"][1]))
            D = [k - z for k, z in zip(Zk, Zc)]

            acc["real"].append(R_eff(Zc, lc, D))
            acc["spatial_shuffled"].append(R_eff(Zc, lc, spatial_shuffle(D, gen)))
            acc["chan_global"].append(R_eff(Zc, lc, channel_shuffle_global(D, gen)))
            acc["chan_perposition"].append(R_eff(Zc, lc, channel_shuffle_perposition(D, gen)))
            acc["both_shuffled"].append(
                R_eff(Zc, lc, channel_shuffle_perposition(spatial_shuffle(D, gen), gen)))

            # ---- (B) power iteration for the dominant sensitive direction ----
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
            for j in USED:
                xis[j].append(coherence_length((v[j][0] ** 2).sum(0)))
            if (i + 1) % 4 == 0:
                print(f"  {i+1}/{a.n_eval}", flush=True)

    R = {k: sum(x) / len(x) for k, x in acc.items()}
    print(f"\n=== (A) DOF-MATCHED DECOMPOSITION (R_eff, stage 0 excluded) ===")
    print(f"{'arm':20s} {'R_eff':>10s} {'frac of real':>14s}")
    for k in ARMS:
        print(f"{k:20s} {R[k]:10.4f} {R[k]/R['real']:14.3f}")

    ratio = R["spatial_shuffled"] / R["real"]
    can = 0.076 <= ratio <= 0.139
    print(f"\n  CANARY shuffle ratio vs m14 [0.076,0.139]: {'PASS' if can else 'FAIL'} ({ratio:.4f})")
    if not can:
        print("  REFUSING to read the ordering verdicts.")

    p_a = R["chan_global"] > R["spatial_shuffled"]
    p_b = R["chan_global"] > R["chan_perposition"]
    if can:
        print(f"\n  P-M34a (DOF-matched: global-channel LESS damaging than spatial): "
              f"{'HIT' if p_a else 'MISS'}")
        print(f"  P-M34b (global-channel less damaging than per-position):        "
              f"{'HIT' if p_b else 'MISS'}")

    print(f"\n=== (B) COHERENCE LENGTH OF G's DOMINANT DIRECTION ===")
    print(f"  (approximate: finite-difference JVP composed with autograd VJP, {N_POWER} iterations)")
    scale = {1: 2, 2: 4, 3: 8}
    out_xi = {}
    for j in USED:
        m_ = sum(xis[j]) / len(xis[j])
        out_xi[j] = {"cells": m_, "stage0_equiv": m_ * scale[j]}
        print(f"  stage {j}: coherence length {m_:6.2f} cells  = {m_*scale[j]:6.2f} stage-0-equivalent")
    print(f"  block-shuffle xi (recorded) = 26.6")

    (GAPFILL / "m34_dof_matched_and_eigen.json").write_text(json.dumps(
        {"R_eff": R, "shuf_ratio": ratio, "canary": bool(can), "n_eval": a.n_eval,
         "P_M34a": bool(p_a), "P_M34b": bool(p_b), "xi_by_stage": out_xi,
         "n_power": N_POWER, "used_stages": USED}, indent=1))
    print(f"\nwrote {GAPFILL / 'm34_dof_matched_and_eigen.json'}")


if __name__ == "__main__":
    main()
