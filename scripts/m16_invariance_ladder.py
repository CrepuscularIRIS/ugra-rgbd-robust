"""M16 — the INVARIANCE LADDER. Which spatial property of the displacement is causal?

THE GAP. Every arrangement result so far uses ONE transform — full spatial shuffle — which destroys
per-position assignment, internal spatial coherence, and alignment-with-content SIMULTANEOUSLY. So
"arrangement matters" is a bundle, not a mechanism. A principle must name WHICH invariance is
load-bearing.

THE LADDER, each rung chosen for what it PROVABLY preserves:

    translate(k)     preserves power spectrum + autocorrelation EXACTLY; destroys ONLY alignment
    phase_randomize  preserves the power spectrum; destroys phase (hence alignment + local structure)
    block_shuffle(b) preserves statistics within b x b blocks; destroys assignment beyond scale b
    full_shuffle     preserves per-position marginals, norm, per-channel stats; destroys all structure
    uniform          preserves norm and mean; destroys everything spatial

TRANSLATION IS THE DECISIVE RUNG. A rigid shift changes no spatial statistic of the field — the
autocorrelation and power spectrum are mathematically identical — so the ONLY thing it can remove is
registration with the content it perturbs. If R collapses under translation, spatial-coherence
accounts are dead and ALIGNMENT is the causal variable.

Block shuffle then yields xi, a characteristic LENGTH over which assignment matters — a measurable
constant comparable across architectures, which is what a principle owes.

Prereg + outcome table + severity: .grill/prediction-ledger.md (2026-08-10, M16).

    python3 scripts/m16_invariance_ladder.py --gpu 1 --n-eval 20
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
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
BLOCKS = [2, 4, 8, 16, 32, 64]
TRANS = [1, 2, 4, 8, 16, 32]
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402

CONDS = ["dconst", "invalid_0.10", "invalid_0.55", "depth_saturate"]


def translate(d, k):
    """Rigid circular shift. Autocorrelation and power spectrum are EXACTLY preserved."""
    return torch.roll(d, shifts=(k, k), dims=(2, 3))


def block_shuffle(d, b, gen):
    """Permute b x b blocks. Statistics WITHIN a block are untouched; assignment beyond b is destroyed."""
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


def phase_randomize(d, gen):
    """Randomise Fourier phase, keep magnitude. The POWER SPECTRUM is preserved exactly."""
    B, C, H, W = d.shape
    Fx = torch.fft.rfft2(d.float())
    mag = Fx.abs()
    ph = torch.rand(Fx.shape, generator=gen, device="cpu").to(d.device) * 2 * np.pi
    return torch.fft.irfft2(mag * torch.exp(1j * ph), s=(H, W)).to(d.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=20)
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

    arms = ["real"] + [f"trans_{k}" for k in TRANS] + ["phase"] \
        + [f"block_{b}" for b in BLOCKS] + ["shuffle", "uniform"]
    acc = {c: {arm: [] for arm in arms} for c in CONDS}
    spec_check = {"trans_32": [], "phase": [], "shuffle": []}
    gen = torch.Generator().manual_seed(SEED)

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        canary = False
        for i, b_ in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b_["data"].cuda(), b_["modal_x"].cuda()
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)
            if not canary:
                dd = float((zl - dec(Zc, size)).abs().max())
                print(f"\nCANARY: max|diff| = {dd:.3e}")
                if dd > 0:
                    raise SystemExit("REFUSING: nondeterministic decoder")
                print("  bases pinned.\n")
                canary = True
            logp_c = F.log_softmax(zl, 1)

            for c in CONDS:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                delta = [k - z for k, z in zip(Zk, Zc)]

                def kl_of(dl):
                    return float(F.kl_div(F.log_softmax(dec([z + d for z, d in zip(Zc, dl)], size), 1),
                                          logp_c, log_target=True, reduction="batchmean"))

                acc[c]["real"].append(kl_of(delta))
                for k in TRANS:
                    acc[c][f"trans_{k}"].append(kl_of([translate(d, k) for d in delta]))
                acc[c]["phase"].append(kl_of([phase_randomize(d, gen) for d in delta]))
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

                # INSTRUMENT CHECK: does each transform preserve what it claims to?
                if c == "dconst" and i == 0:
                    d0 = delta[0].float()
                    P = lambda t: torch.fft.rfft2(t).abs()
                    p0 = P(d0)
                    spec_check["trans_32"].append(float((P(translate(delta[0], 32).float()) - p0)
                                                        .abs().max() / p0.max()))
                    spec_check["phase"].append(float((P(phase_randomize(delta[0], gen).float()) - p0)
                                                     .abs().max() / p0.max()))
                    sh0 = delta[0].reshape(1, -1, d0.shape[2] * d0.shape[3])
                    idx = torch.randperm(sh0.shape[2], generator=gen).to(sh0.device)
                    spec_check["shuffle"].append(
                        float((P(sh0[:, :, idx].reshape(d0.shape).float()) - p0).abs().max() / p0.max()))
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.n_eval}")

    print("\n=== INSTRUMENT CHECK — relative max power-spectrum deviation from the real field ===")
    for k, v in spec_check.items():
        print(f"  {k:10s} {np.mean(v):.6f}   {'<- preserved as claimed' if np.mean(v) < 1e-3 else ''}")

    rows = {}
    print(f"\n{'arm':12s} " + " ".join(f"{c[:11]:>11s}" for c in CONDS) + f" {'mean':>8s}")
    for arm in arms:
        rr = {}
        for c in CONDS:
            rr[c] = float(np.mean(acc[c][arm])) / float(np.mean(acc[c]["real"]))
        rows[arm] = rr
        m = np.mean(list(rr.values()))
        rows[arm]["_mean"] = float(m)
        print(f"{arm:12s} " + " ".join(f"{rr[c]:11.3f}" for c in CONDS) + f" {m:8.3f}")

    t32 = rows["trans_32"]["_mean"]
    t4 = rows["trans_4"]["_mean"]
    ph = rows["phase"]["_mean"]
    blocks = [rows[f"block_{b}"]["_mean"] for b in BLOCKS]
    mono = all(blocks[i] >= blocks[i + 1] - 1e-6 for i in range(len(blocks) - 1))
    half = 0.5 * blocks[0]
    xi = next((b for b, v in zip(BLOCKS, blocks) if v <= half), None)

    p_a = t32 <= 1 / 3
    p_b = mono
    p_c = xi is not None and xi < 32
    p_d = ph <= 1 / 3
    p_e = t4 > t32

    print(f"\n--- BLOCK-SHUFFLE CURVE (the characteristic scale) ---")
    for b, v in zip(BLOCKS, blocks):
        print(f"  block {b:3d} px  ->  R/R_real = {v:.3f}")
    print(f"  full shuffle  ->  R/R_real = {rows['shuffle']['_mean']:.3f}")
    print(f"  half-collapse scale xi = {xi} px" if xi else "  xi: not reached")

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M16a translate(32) collapses >=3x: {t32:.3f} -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M16b block curve monotone decreasing -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M16c xi < 32 px (xi={xi}) -> {'HIT' if p_c else 'MISS'}")
    print(f"P-M16d phase randomization collapses >=3x: {ph:.3f} -> {'HIT' if p_d else 'MISS'}")
    print(f"P-M16e translate(4) {t4:.3f} > translate(32) {t32:.3f} -> {'HIT' if p_e else 'MISS'}")

    print(f"\n--- VERDICT ---")
    if p_a:
        print("  ALIGNMENT IS THE CAUSAL VARIABLE. A rigid translation preserves the power spectrum")
        print("  and autocorrelation EXACTLY and still destroys the response -- so the effect cannot")
        print("  be about the field's own spatial statistics. Spatial-coherence accounts are KILLED.")
    else:
        print("  ALIGNMENT IS NOT THE VARIABLE. Translation leaves the response intact, so what")
        print("  matters is the field's INTERNAL spatial structure, not its registration with")
        print("  content. The alignment reading is KILLED and coherence survives.")

    o = GAPFILL / "m16_invariance_ladder.json"
    o.write_text(json.dumps({"rows": rows, "blocks": BLOCKS, "block_curve": blocks,
                             "xi": xi, "spec_check": {k: float(np.mean(v))
                                                      for k, v in spec_check.items()},
                             "P_M16a": p_a, "P_M16b": p_b, "P_M16c": p_c,
                             "P_M16d": p_d, "P_M16e": p_e, "n_eval": a.n_eval}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
