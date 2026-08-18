"""M4 — is the CAUSAL field low-dimensional? The test MIG's definition actually requires.

M3 established that damage is mediated entirely by the decoder-input displacement FIELD, and that
its spatial arrangement is the causal variable (shuffling positions destroys the damage). But
"low-dimensional" was never tested -- and that is the word in MIG's definition.

M0/M1/M2 searched the CHANNEL basis and found nothing: the rank-3 channel coordinate predicts worse
than a scalar, and direction is causally irrelevant. M4 asks the same question in the basis M3 says
is causal: reshape the field to (C, HW) and truncate its SPATIAL modes.

    rank-k field = sum of the top-k singular modes of the displacement field

If a small k reproduces the damage, the causal object IS low-dimensional and MIG exists as
specified -- in a spatial, not spectral, basis. If damage only returns near full rank, the geometry
is real but NOT low-dimensional, and Level 4 is the honest outcome.

FILED BEFORE THE NUMBERS:
    P-M4a  rank-8 retains >= 70% of the full field's damage        P_true 0.55
    P-M4b  rank-1 retains < 40%                                    P_true 0.70

P-M4a at 0.55: M3 showed spatial arrangement matters, which cuts BOTH ways -- a highly structured
field could be low-rank (a few blobs) or full-rank (fine-grained, per-pixel). `invalid_0.10` is
scattered per-pixel invalidity, which argues for high rank. Filed near a coin flip for that reason.

    python3 scripts/m4_rank.py --gpu 1 --n-eval 50
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
SEED = 20260721
RANKS = [1, 2, 4, 8, 16, 32]
_CAP = {"store": None, "inject": None, "mode": None}


def truncate(d, k):
    """Keep the top-k SPATIAL modes of a (B,C,H,W) displacement field."""
    B, C, H, W = d.shape
    M = d.reshape(B, C, H * W)
    out = torch.zeros_like(M)
    for b in range(B):
        U, S, Vh = torch.linalg.svd(M[b].float(), full_matrices=False)
        kk = min(k, S.numel())
        out[b] = (U[:, :kk] * S[:kk]) @ Vh[:kk]
    return out.reshape(B, C, H, W).to(d.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=50)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    from utils.metric import hist_info, compute_score
    from depth_corruptions import depth_invalid

    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(miss.missing_keys)
    print(f"checkpoint: {got}/{len(sd)} tensors loaded")
    if got < 0.9 * len(sd):
        raise SystemExit("REFUSING: <90% loaded")
    model.eval().cuda()
    NC = cfg.num_classes
    _orig = model.decode_head.forward

    def _dec(x, *args, **kw):
        feats = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [f.detach().clone() for f in feats]
        elif _CAP["mode"] == "inject" and _CAP["inject"] is not None:
            feats = [f + d.to(f.device) for f, d in zip(feats, _CAP["inject"])]
        y = feats if isinstance(x, (list, tuple)) else feats[0]
        return _orig(y, *args, **kw)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def run(rgb, dep, mode=None, inject=None):
        _CAP["mode"], _CAP["inject"], _CAP["store"] = mode, inject, None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = model(rgb, dep)
        _CAP["mode"] = None
        return o

    arms = ["clean", "full"] + [f"rank{k}" for k in RANKS]
    hists = {k: np.zeros((NC, NC)) for k in arms}
    corr = {k: [0, 0] for k in arms}
    energy = {k: [] for k in RANKS}

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            gt = b["label"].numpy()[0]
            cdep = depth_invalid(dep.clone(), frac=0.10, seed=SEED)
            run(rgb, dep, mode="store");  Zc = _CAP["store"]
            run(rgb, cdep, mode="store"); Zk = _CAP["store"]
            delta = [k - c for k, c in zip(Zk, Zc)]

            outs = {"clean": run(rgb, dep),
                    "full": run(rgb, dep, mode="inject", inject=delta)}
            for k in RANKS:
                tr = [truncate(d, k) for d in delta]
                energy[k].append(float(sum((t ** 2).sum() for t in tr) /
                                       max(1e-12, sum((d ** 2).sum() for d in delta))))
                outs[f"rank{k}"] = run(rgb, dep, mode="inject", inject=tr)

            for kk, o in outs.items():
                pred = o.argmax(1).cpu().numpy()[0]
                if pred.shape != gt.shape:
                    import torch.nn.functional as F
                    pred = F.interpolate(o.float(), size=gt.shape, mode="bilinear",
                                         align_corners=False).argmax(1).cpu().numpy()[0]
                h, lab, cor = hist_info(NC, pred, gt)
                hists[kk] += h; corr[kk][0] += cor; corr[kk][1] += lab

    res = {k: float(compute_score(hists[k], corr[k][0], corr[k][1])[1]) * 100 for k in arms}
    clean, full = res["clean"], res["full"]
    d_full = clean - full
    print(f"\nfull-field damage (the causal reference): {d_full:.3f}\n")
    print(f"{'rank':>6s} {'energy%':>9s} {'damage':>8s} {'% of full':>10s}")
    frac = {}
    for k in RANKS:
        d = clean - res[f"rank{k}"]
        frac[k] = d / d_full if d_full != 0 else float("nan")
        print(f"{k:6d} {100*np.mean(energy[k]):9.1f} {d:8.3f} {100*frac[k]:10.1f}")

    p_a = frac.get(8, 0) >= 0.70
    p_b = frac.get(1, 1) < 0.40
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M4a rank-8 retains >=70% of full damage: {100*frac.get(8,0):.1f}% "
          f"-> {'HIT' if p_a else 'MISS'}")
    print(f"P-M4b rank-1 retains <40%: {100*frac.get(1,1):.1f}% -> {'HIT' if p_b else 'MISS'}")

    print("\n--- VERDICT ---")
    if p_a:
        print("  THE CAUSAL FIELD IS LOW-DIMENSIONAL. A handful of spatial modes reproduces the")
        print("  damage, so MIG exists AS SPECIFIED -- a model-derived, low-dimensional functional")
        print("  geometry -- in a SPATIAL basis. M0-M2 found nothing only because they searched")
        print("  the channel basis.")
    else:
        print("  THE CAUSAL FIELD IS NOT LOW-DIMENSIONAL. Damage returns only near full rank, so")
        print("  the geometry is real and spatial but NOT low-dimensional. MIG as specified does")
        print("  not exist, and Level 4 is the honest outcome.")

    p = PROJECT / "results" / "gapfill" / "m4_rank.json"
    p.write_text(json.dumps({"n_eval": a.n_eval, "miou": res, "damage_full": d_full,
                             "fraction_of_full": {str(k): frac[k] for k in RANKS},
                             "energy": {str(k): float(np.mean(energy[k])) for k in RANKS},
                             "P_M4a": bool(p_a), "P_M4b": bool(p_b)}, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
