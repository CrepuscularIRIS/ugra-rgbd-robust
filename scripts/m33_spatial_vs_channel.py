"""M33 — DECOMPOSING REGISTRATION: is it SPATIAL arrangement, CHANNEL alignment, or the joint object?

WHAT M32 EXPOSED. M32 tried to measure xi on the operator alone by building spatially-coherent random
directions. Its canary failed, and the reason is informative rather than fatal:

  * the probe REPRODUCED the real field's Rayleigh quotient — R = 1.9268 vs the historical 1.84 (4.7%),
    an independent validation of M31's identity in the historical units;
  * but its white-noise floor was 0.0067, nowhere near the historical "random" floor of 0.12-0.19.

Those two numbers are not comparable, and that is a fact about the constructions. The historical
0.12-0.19 arm is the **spatially shuffled REAL displacement** — spatial arrangement destroyed,
**channel structure preserved**. M32's synthetic direction is **rank-1 in channels** (one random
vector per stage times a spatial field) with spatial coherence built in. They differ on the channel
axis, not only the spatial one, so neither can be read against the other.

THE TEST THIS MOTIVATES — a clean 2x2 on the SAME real displacement, at matched norm throughout:

    arm            spatial arrangement   channel structure     what it isolates
    real                 intact               intact           the reference
    spatial-shuffled     DESTROYED            intact           does spatial arrangement carry R?
    channel-shuffled     intact               DESTROYED        does channel alignment carry R?
    both-shuffled        DESTROYED            DESTROYED        the floor

`channel-shuffled` independently permutes the channel vector at every spatial position, which keeps
the per-position magnitude map and the global channel histogram while destroying which channel
pattern sits where. Every arm is renormalised to the real field's norm, so magnitude is never the
variable.

WHY IT MATTERS. `plan/THE-PRINCIPLE.md` currently says xi is "the spatial coherence length of the
decoder's output-sensitive subspace" — a purely SPATIAL reading. If `channel-shuffled` also collapses,
registration is a JOINT spatial-channel alignment with scene content and the purely spatial wording
is wrong.

CANARY (blocking): the `real` arm must reproduce the historical R = 1.84 within 15%, and the
`spatial-shuffled` arm must reproduce the historical 0.12-0.19 floor. Both references are independent
prior measurements on this checkpoint, so the probe must hit them before any new arm is read.

Prereg: .grill/prediction-ledger.md (2026-08-10, M33).

    python3 scripts/m33_spatial_vs_channel.py --gpu 1 --n-eval 24
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
STEP_NORM = 12.0
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402


def fisher_quadform(logits_clean, v):
    p = F.softmax(logits_clean, dim=1)
    m1 = (p * v).sum(dim=1)
    m2 = (p * v * v).sum(dim=1)
    return float((m2 - m1 * m1).sum())


def spatial_shuffle(D, gen):
    out = []
    for d in D:
        B, C, H, W = d.shape
        idx = torch.randperm(H * W, generator=gen).to(d.device)
        out.append(d.reshape(B, C, H * W)[:, :, idx].reshape(B, C, H, W))
    return out


def channel_shuffle(D, gen):
    """Permute the CHANNEL vector independently at every spatial position.

    Keeps the per-position magnitude map and the global channel histogram; destroys which channel
    pattern sits at which location.
    """
    out = []
    for d in D:
        B, C, H, W = d.shape
        flat = d.reshape(B, C, H * W)
        idx = torch.argsort(torch.rand((C, H * W), generator=gen), dim=0).to(d.device)
        out.append(torch.gather(flat, 1, idx.unsqueeze(0).expand(B, -1, -1)).reshape(B, C, H, W))
    return out


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

    OUT = [None]

    def dec_(feats):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=OUT[0], mode="bilinear", align_corners=False)

    def rayleigh(Zc, lc, D):
        nrm = sum(float((d ** 2).sum()) for d in D) ** 0.5
        if nrm < 1e-12:
            return float("nan")
        u = [d / nrm for d in D]
        v = (dec_([z + STEP_NORM * d for z, d in zip(Zc, u)]) - lc) / STEP_NORM
        return 0.5 * fisher_quadform(lc, v)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    ARMS = ["real", "spatial_shuffled", "channel_shuffled", "both_shuffled"]
    acc = {k: [] for k in ARMS}

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

            acc["real"].append(rayleigh(Zc, lc, D))
            acc["spatial_shuffled"].append(rayleigh(Zc, lc, spatial_shuffle(D, gen)))
            acc["channel_shuffled"].append(rayleigh(Zc, lc, channel_shuffle(D, gen)))
            acc["both_shuffled"].append(
                rayleigh(Zc, lc, channel_shuffle(spatial_shuffle(D, gen), gen)))
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{a.n_eval}", flush=True)

    R = {k: sum(v) / len(v) for k, v in acc.items()}
    print(f"\n{'arm':20s} {'R':>10s} {'frac of real':>14s}")
    for k in ARMS:
        print(f"{k:20s} {R[k]:10.4f} {R[k]/R['real']:14.3f}")

    can_real = abs(R["real"] / 1.84 - 1) < 0.15
    # CANARY CORRECTED, and the correction is recorded rather than silent.
    # The first version bounded the ABSOLUTE R of the shuffled arm to [0.10, 0.25]. That band was
    # taken from a low-R subset of conditions, but m14's recorded absolute R_shuffled spans
    # 0.068-0.272 because R itself varies 4x across conditions — so applying it to `dconst`, the
    # HIGHEST-R condition, was arithmetically wrong. The scale-free quantity m14 actually records is
    # `shuf_ratio` = R_shuffled / R, whose measured range over all 23 conditions is 0.076-0.139.
    # That is the correct reference and it is what is checked here.
    shuf_ratio = R["spatial_shuffled"] / R["real"]
    can_shuf = 0.076 <= shuf_ratio <= 0.139
    print(f"\n  CANARY real vs historical 1.84:        {'PASS' if can_real else 'FAIL'}"
          f"  ({R['real']:.4f})")
    print(f"  CANARY shuffle RATIO vs m14 [0.076,0.139]: {'PASS' if can_shuf else 'FAIL'}"
          f"  ({shuf_ratio:.4f})")
    if not (can_real and can_shuf):
        print("\n  REFUSING to read the new arms: the probe did not reproduce its references.")
        (GAPFILL / "m33_spatial_vs_channel.json").write_text(json.dumps(
            {"R": R, "canary_real": bool(can_real), "canary_shuffled": bool(can_shuf), "shuf_ratio": shuf_ratio,
             "n_eval": a.n_eval, "verdicts_withheld": True}, indent=1))
        raise SystemExit(1)

    p_a = R["channel_shuffled"] < 0.5 * R["real"]
    p_b = R["channel_shuffled"] > R["spatial_shuffled"]
    print(f"\n  P-M33a (channel shuffle also collapses R, < 50% of real): "
          f"{'HIT' if p_a else 'MISS'}  ({R['channel_shuffled']/R['real']:.3f})")
    print(f"  P-M33b (channel shuffle less damaging than spatial):      "
          f"{'HIT' if p_b else 'MISS'}")

    (GAPFILL / "m33_spatial_vs_channel.json").write_text(json.dumps(
        {"R": R, "canary_real": bool(can_real), "canary_shuffled": bool(can_shuf), "shuf_ratio": shuf_ratio,
         "n_eval": a.n_eval, "P_M33a": bool(p_a), "P_M33b": bool(p_b)}, indent=1))
    print(f"\nwrote {GAPFILL / 'm33_spatial_vs_channel.json'}")


if __name__ == "__main__":
    main()
