"""M18 — WHERE DOES THE QUADRATIC REGIME END? Dose-response along the natural displacement.

WHY THIS REPLACES M15's FAILED FIT. M15 tried to get the exponent by fitting KL ~ ||Delta||^p ACROSS
corruption conditions on CMNeXt. That fit was degenerate: an 8-point log-log regression over a 3.2x
norm span, where dropping ONE condition moved p from 2.989 to 0.397. Fitting across conditions
conflates "how does KL grow with magnitude" with "do different directions have different gains."

THE CLEAN DESIGN. Take ONE naturally-induced displacement Delta and inject alpha*Delta for a ladder
of alpha. This measures the scaling law ALONG the direction the world hands us, with a 30x span, no
family confound, and a local exponent that can be read off between adjacent points:

    p_local(alpha) = d log KL / d log alpha

WHY IT MATTERS BEYOND THE FIT. arXiv:2603.13911 computes the same class of quantity as an explicit
FINITE-DIFFERENCE PROXY — infinitesimal by construction. Our displacements are at FULL magnitude.
The question "does the quadratic form still hold where the perturbation actually lives, and where
does it stop holding?" is exactly the clause that separates the two, and it has never been measured.
A measured QUADRATIC HORIZON is a validity range, which is what a law owes.

Runs BOTH architectures in one pass so the comparison is not confounded by protocol drift.

    python3 scripts/m18_quadratic_horizon.py --gpu 1 --n-eval 12
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
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
ALPHAS = [0.03, 0.06, 0.12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
_CAP = {"store": None, "mode": None}


def kl_ladder(dec, Zc, delta, logp_c, size):
    out = []
    for al in ALPHAS:
        z = [zz + al * d for zz, d in zip(Zc, delta)]
        out.append(float(F.kl_div(F.log_softmax(dec(z, size), 1), logp_c,
                                  log_target=True, reduction="batchmean")))
    return out


def local_exponents(kl):
    la = np.log(ALPHAS)
    lk = np.log(np.maximum(kl, 1e-12))
    return [(lk[i + 1] - lk[i]) / (la[i + 1] - la[i]) for i in range(len(ALPHAS) - 1)]


def run_dformer(n_eval, gpu):
    DF = PROJECT / "repos" / "DFormer"
    sys.path.insert(0, str(DF)); sys.path.insert(0, str(DF / "utils"))
    sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
    sys.path.insert(0, str(PROJECT / "scripts"))
    os.chdir(DF)
    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    from m1_mig_full import COND, corrupt
    ck = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    if len(sd) - len(miss.missing_keys) < 0.9 * len(sd):
        raise SystemExit("REFUSING: DFormer <90% loaded")
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
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")
    conds = ["dconst", "invalid_0.10", "invalid_0.55"]
    acc = {c: [] for c in conds}
    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)
            if i == 0:
                assert float((zl - dec(Zc, size)).abs().max()) == 0.0, "nondeterministic"
            logp = F.log_softmax(zl, 1)
            for c in conds:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                acc[c].append(kl_ladder(dec, Zc, [k - z for k, z in zip(Zk, Zc)], logp, size))
    return {c: np.mean(np.array(v), axis=0).tolist() for c, v in acc.items()}


def run_cmnext(n_eval, gpu):
    DL = PROJECT / "repos" / "DELIVER"
    sys.path.insert(0, str(DL))
    os.chdir(DL)
    import depth_corruptions as DC
    from semseg.models.cmnext import CMNeXt
    from semseg.datasets.nyu import NYU
    from semseg.augmentations_mm import get_val_augmentation
    from torch.utils.data import DataLoader
    ck = DL / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
    model = CMNeXt("CMNeXt-B4", 40, ["img", "depth"])
    sd = torch.load(str(ck), map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    msg = model.load_state_dict(sd, strict=False)
    if len(sd) - len(msg.missing_keys) < 0.9 * len(sd):
        raise SystemExit("REFUSING: CMNeXt <90% loaded")
    model.eval().cuda()
    _orig = model.decode_head.forward

    def _hook(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _hook

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
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    ds = NYU(str(DL / "data" / "NYUDepthv2"), "val",
             get_val_augmentation([480, 640]), ["img", "depth"], None)
    dl = DataLoader(ds, batch_size=1, num_workers=2)
    conds = {"constant": lambda d: DC.depth_constant(d.clone()),
             "invalid_0.10": lambda d: DC.depth_invalid(d.clone(), frac=0.10, seed=SEED),
             "invalid_0.55": lambda d: DC.depth_invalid(d.clone(), frac=0.55, seed=SEED)}
    acc = {c: [] for c in conds}
    for i, batch in enumerate(dl):
        if i >= n_eval:
            break
        sample, _ = batch
        mods = [t.cuda() for t in sample]
        size = mods[0].shape[-2:]
        Zc = cap_(mods)
        zl = dec(Zc, size)
        if i == 0:
            assert float((zl - dec(Zc, size)).abs().max()) == 0.0, "nondeterministic"
        logp = F.log_softmax(zl, 1)
        for c, fn in conds.items():
            Zk = cap_([mods[0], fn(mods[1])])
            acc[c].append(kl_ladder(dec, Zc, [k - z for k, z in zip(Zk, Zc)], logp, size))
    return {c: np.mean(np.array(v), axis=0).tolist() for c, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=12)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    print("=== DFormerv2-S (Ham/NMF decoder) ===")
    df = run_dformer(a.n_eval, a.gpu)
    print("=== CMNeXt-B4 (SegFormer MLP decoder) ===")
    cm = run_cmnext(a.n_eval, a.gpu)

    res = {"alphas": ALPHAS, "dformer": df, "cmnext": cm}
    mids = [f"{ALPHAS[i]:.2f}-{ALPHAS[i+1]:.2f}" for i in range(len(ALPHAS) - 1)]

    horizons = {}
    for arch, data in (("DFormerv2", df), ("CMNeXt", cm)):
        print(f"\n--- {arch}: local exponent  p = dlogKL/dlogalpha ---")
        print(f"{'condition':14s} " + " ".join(f"{m:>11s}" for m in mids))
        for c, kl in data.items():
            e = local_exponents(kl)
            print(f"{c:14s} " + " ".join(f"{v:11.2f}" for v in e))
            # quadratic horizon: last alpha whose local exponent is still >= 1.5
            h = None
            for i, v in enumerate(e):
                if v >= 1.5:
                    h = ALPHAS[i + 1]
            horizons[f"{arch}|{c}"] = h
        print(f"  quadratic horizon (last alpha with p_local >= 1.5): " +
              ", ".join(f"{c}={horizons[f'{arch}|{c}']}" for c in data))

    small = {}
    for arch, data in (("DFormerv2", df), ("CMNeXt", cm)):
        v = [local_exponents(kl)[0] for kl in data.values()]
        small[arch] = float(np.mean(v))
    print(f"\nsmall-alpha exponent (0.03->0.06): DFormerv2 {small['DFormerv2']:.2f}   "
          f"CMNeXt {small['CMNeXt']:.2f}")
    full = {}
    for arch, data in (("DFormerv2", df), ("CMNeXt", cm)):
        v = [local_exponents(kl)[-3] for kl in data.values()]   # 0.75 -> 1.0
        full[arch] = float(np.mean(v))
    print(f"full-scale exponent (0.75->1.0):   DFormerv2 {full['DFormerv2']:.2f}   "
          f"CMNeXt {full['CMNeXt']:.2f}")

    p_a = all(1.7 <= small[k] <= 2.3 for k in small)
    p_b = all(full[k] < small[k] for k in small)
    hs = [v for v in horizons.values() if v is not None]
    p_c = bool(hs) and (max(hs) / min(hs) <= 2.0)

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M18a small-alpha exponent in [1.7,2.3] both archs -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M18b exponent DECREASES toward full scale (saturation) -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M18c quadratic horizons agree within 2x across archs -> {'HIT' if p_c else 'MISS'}")
    print(f"    horizons: {horizons}")

    res.update({"horizons": horizons, "small_alpha_exp": small, "full_scale_exp": full,
                "P_M18a": p_a, "P_M18b": p_b, "P_M18c": p_c, "n_eval": a.n_eval})
    o = GAPFILL / "m18_quadratic_horizon.json"
    o.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
