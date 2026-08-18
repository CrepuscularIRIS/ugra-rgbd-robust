"""M17 — does ALIGNMENT-sensitivity replicate on a non-NMF decoder? The M16 ladder on CMNeXt-B4.

M16 established on DFormerv2 that a rigid TRANSLATION — which preserves the power spectrum EXACTLY
(measured 0.000000) — still collapses R by 4.1x, killing spatial-coherence accounts and leaving
ALIGNMENT as the causal variable. It also gave a characteristic length xi = 26.6 feature units.

Both results are single-architecture, and DFormerv2's LightHamHead is a rank-r NMF reconstruction.
This ports the decisive rungs to CMNeXt-B4 (SegFormerHead — per-scale MLP, no low-rank structure).

WHY THIS IS THE RIGHT PORT AND NOT A REPLAY. M15 already showed the SHUFFLE collapse reproduces here
(3.0-4.9x). But shuffle is the bundled control — it destroys assignment, coherence and alignment at
once, and Xie et al. ICCV 2017 (arXiv:1703.08603) already published the permutation version. The
TRANSLATION rung is the one that is both unoccupied and decisive, and it has never been run outside
DFormerv2.

Prereg: .grill/prediction-ledger.md (2026-08-10, M17).

    python3 scripts/m17_cross_arch_alignment.py --gpu 1 --n-eval 20
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DELIVER = PROJECT / "repos" / "DELIVER"
sys.path.insert(0, str(DELIVER))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))

CKPT = DELIVER / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
NCLS = 40
GATE_LO, GATE_HI = 52.0, 57.0
TRANS = [1, 2, 4, 8, 16, 32]
BLOCKS = [2, 4, 8, 16, 32, 64]
_CAP = {"store": None, "mode": None}

CONDS = {
    "constant":     lambda d, DC: DC.depth_constant(d.clone()),
    "invalid_0.10": lambda d, DC: DC.depth_invalid(d.clone(), frac=0.10, seed=SEED),
    "invalid_0.55": lambda d, DC: DC.depth_invalid(d.clone(), frac=0.55, seed=SEED),
    "half_missing": lambda d, DC: DC.half_missing(d.clone(), seed=SEED),
}


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def miou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    return float((tp / u)[u > 0].mean()) * 100.0


def translate(d, k):
    return torch.roll(d, shifts=(k, k), dims=(2, 3))


def block_shuffle(d, b, gen):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=20)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DELIVER)

    import depth_corruptions as DC
    from semseg.models.cmnext import CMNeXt
    from semseg.datasets.nyu import NYU
    from semseg.augmentations_mm import get_val_augmentation
    from torch.utils.data import DataLoader

    model = CMNeXt("CMNeXt-B4", NCLS, ["img", "depth"])
    sd = torch.load(str(CKPT), map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    msg = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(msg.missing_keys)
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

    ds = NYU(str(DELIVER / "data" / "NYUDepthv2"), "val",
             get_val_augmentation([480, 640]), ["img", "depth"], None)
    dl = DataLoader(ds, batch_size=1, num_workers=2, pin_memory=False)

    def cap_(mods):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(mods)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            out = _orig(list(feats))
        if isinstance(out, dict) and "pred" in out:
            out = out["pred"]
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)

    arms = ["real"] + [f"trans_{k}" for k in TRANS] + [f"block_{b}" for b in BLOCKS] \
        + ["shuffle", "uniform"]
    acc = {c: {arm: [] for arm in arms} for c in CONDS}
    spec = []
    gen = torch.Generator().manual_seed(SEED)
    Hc = torch.zeros(NCLS, NCLS, dtype=torch.long)
    canary = False

    for i, batch in enumerate(dl):
        if i >= a.n_eval:
            break
        sample, lbl = batch
        mods = [t.cuda() for t in sample]
        lab = lbl.cuda().squeeze(0).long()
        size = mods[0].shape[-2:]
        Zc = cap_(mods)
        zl = dec(Zc, size)
        if not canary:
            dd = float((zl - dec(Zc, size)).abs().max())
            print(f"\nCANARY: max|diff| = {dd:.3e}")
            if dd > 0:
                raise SystemExit("REFUSING: nondeterministic decoder")
            canary = True
        logp_c = F.log_softmax(zl, 1)
        Hc += hist(zl.argmax(1).squeeze(0).cpu(), lab.cpu())

        for c, fn in CONDS.items():
            Zk = cap_([mods[0], fn(mods[1], DC)])
            delta = [k - z for k, z in zip(Zk, Zc)]

            def kl_of(dl_):
                return float(F.kl_div(F.log_softmax(dec([z + d for z, d in zip(Zc, dl_)], size), 1),
                                      logp_c, log_target=True, reduction="batchmean"))
            acc[c]["real"].append(kl_of(delta))
            for k in TRANS:
                acc[c][f"trans_{k}"].append(kl_of([translate(d, k) for d in delta]))
            for bb in BLOCKS:
                acc[c][f"block_{bb}"].append(kl_of([block_shuffle(d, bb, gen) for d in delta]))
            fs = []
            for d in delta:
                B, C, H, W = d.shape
                idx = torch.randperm(H * W, generator=gen).to(d.device)
                fs.append(d.reshape(B, C, H * W)[:, :, idx].reshape(B, C, H, W))
            acc[c]["shuffle"].append(kl_of(fs))
            acc[c]["uniform"].append(kl_of([d.mean(dim=(2, 3), keepdim=True).expand_as(d)
                                            for d in delta]))
            if c == "constant" and i == 0:
                P = lambda t: torch.fft.rfft2(t.float()).abs()
                p0 = P(delta[0])
                spec.append(float((P(translate(delta[0], 32)) - p0).abs().max() / p0.max()))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{a.n_eval}")

    clean = miou(Hc)
    gate = GATE_LO <= clean <= GATE_HI
    print(f"\n=== REFERENCE GATE === clean mIoU {clean:.2f} in [{GATE_LO},{GATE_HI}] -> "
          f"{'PASS' if gate else 'FAIL'}")
    print(f"INSTRUMENT: translate(32) power-spectrum deviation = {np.mean(spec):.6f}")
    if not gate:
        print("*** GATE FAILED — nothing below counts ***")

    rows = {}
    print(f"\n{'arm':12s} " + " ".join(f"{c[:11]:>11s}" for c in CONDS) + f" {'mean':>8s}")
    for arm in arms:
        rr = {c: float(np.mean(acc[c][arm])) / float(np.mean(acc[c]["real"])) for c in CONDS}
        m = float(np.mean(list(rr.values())))
        rows[arm] = dict(rr, _mean=m)
        print(f"{arm:12s} " + " ".join(f"{rr[c]:11.3f}" for c in CONDS) + f" {m:8.3f}")

    t = [rows[f"trans_{k}"]["_mean"] for k in TRANS]
    blocks = [rows[f"block_{b}"]["_mean"] for b in BLOCKS]
    mono = all(blocks[i] <= blocks[i + 1] + 1e-6 for i in range(len(blocks) - 1))
    lo, hi = blocks[0], blocks[-1]
    half = (lo + hi) / 2
    xi = None
    for i in range(len(BLOCKS) - 1):
        if blocks[i] < half <= blocks[i + 1]:
            f_ = (half - blocks[i]) / (blocks[i + 1] - blocks[i])
            xi = float(np.exp(np.log(BLOCKS[i]) + f_ * (np.log(BLOCKS[i + 1]) - np.log(BLOCKS[i]))))
    tmin = TRANS[int(np.argmin(t))]

    p_a = rows["trans_32"]["_mean"] <= 1 / 3
    p_b = mono
    p_c = xi is not None

    print(f"\n--- TRANSLATION (DFormerv2: 0.665/0.402/0.226/0.149/0.207/0.245, min at 8) ---")
    for k, v in zip(TRANS, t):
        print(f"  translate {k:2d} -> {v:.3f}  ({1/v:.1f}x){'   <-- MINIMUM' if k == tmin else ''}")
    print(f"\n--- BLOCK CURVE (DFormerv2 xi = 26.6) ---")
    for b, v in zip(BLOCKS, blocks):
        print(f"  b={b:3d} -> {v:.3f}")
    print(f"  monotone increasing: {mono}    xi = {xi:.1f} feature units" if xi
          else f"  monotone increasing: {mono}    xi: not reached")

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M17gate clean mIoU in band -> {'HIT' if gate else 'MISS'}")
    print(f"P-M17a translate(32) collapses >=3x: {rows['trans_32']['_mean']:.3f} -> "
          f"{'HIT' if p_a else 'MISS'}")
    print(f"P-M17b block curve monotone increasing -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M17c xi is reached (value {xi}) -> {'HIT' if p_c else 'MISS'}")

    print(f"\n--- VERDICT ---")
    if gate and p_a:
        print("  ALIGNMENT-SENSITIVITY REPLICATES ACROSS DECODER FAMILIES. A spectrum-preserving")
        print("  translation collapses the response on an MLP head with no low-rank structure, so")
        print("  the principle is not a property of DFormerv2's NMF decoder.")
    elif gate:
        print("  ALIGNMENT-SENSITIVITY DOES NOT REPLICATE. Translation leaves the response intact")
        print("  on the MLP decoder -> the M16 principle is architecture-specific and must be")
        print("  written as a DFormerv2 result.")

    o = GAPFILL / "m17_cross_arch_alignment.json"
    o.write_text(json.dumps({"clean_miou": clean, "gate": gate, "rows": rows,
                             "trans": TRANS, "trans_curve": t, "trans_min": tmin,
                             "blocks": BLOCKS, "block_curve": blocks, "xi": xi,
                             "spec_dev": float(np.mean(spec)), "monotone": mono,
                             "P_M17a": p_a, "P_M17b": p_b, "P_M17c": p_c,
                             "n_eval": a.n_eval}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
