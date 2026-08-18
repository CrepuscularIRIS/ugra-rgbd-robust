"""M20 — THE UNIFICATION TEST. Do the alignment law and the quadratic law FACTORIZE?

Three objects are measured and none has been joined to the others:
  (i)   the ALIGNMENT law — R is destroyed by a spectrum-preserving translation; length xi ~ 24-27
  (ii)  the QUADRATIC law — KL ~ ||Delta||^2, p ~ 2.0 over 66x, architecture-general
  (iii) AGGREGATION       — the per-unit Taylor radius does not govern the aggregate (M19)

Listing them side by side is not an account. The account is that they COMPOSE:

    KL(alpha * Delta) = R(structure) * alpha^2 * ||Delta||^2

with ALL magnitude dependence in alpha^2||Delta||^2 and ALL structure dependence in R. That is a
FACTORIZATION claim and it is directly falsifiable: if R depends on alpha, it is false.

THE TEST. Sweep alpha for each of {real, translated, shuffled, uniform}. Under factorization:
  * every arm shares the SAME exponent (the magnitude law does not know about alignment), and
  * every arm's ratio to real is CONSTANT in alpha (the alignment effect does not know about scale).

Prereg: .grill/prediction-ledger.md (2026-08-10, M20).

    python3 scripts/m20_factorization.py --gpu 1 --n-eval 10
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
ALPHAS = [0.06, 0.12, 0.25, 0.5, 1.0]
ARMS = ["real", "trans_8", "shuffle", "uniform"]
_CAP = {"store": None, "mode": None}


def arm_field(delta, arm, gen):
    if arm == "real":
        return delta
    if arm == "trans_8":
        return [torch.roll(d, shifts=(8, 8), dims=(2, 3)) for d in delta]
    if arm == "uniform":
        return [d.mean(dim=(2, 3), keepdim=True).expand_as(d) for d in delta]
    out = []
    for d in delta:
        B, C, H, W = d.shape
        idx = torch.randperm(H * W, generator=gen).to(d.device)
        out.append(d.reshape(B, C, H * W)[:, :, idx].reshape(B, C, H, W))
    return out


def ladder(dec, Zc, delta, logp_c, size, gen):
    res = {}
    for arm in ARMS:
        dl = arm_field(delta, arm, gen)
        res[arm] = [float(F.kl_div(F.log_softmax(dec([z + al * d for z, d in zip(Zc, dl)], size), 1),
                                   logp_c, log_target=True, reduction="batchmean"))
                    for al in ALPHAS]
    return res


def hook(model):
    _orig = model.decode_head.forward

    def _h(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _h
    return _orig


def exponent(kl):
    la, lk = np.log(ALPHAS), np.log(np.maximum(kl, 1e-12))
    return float(np.polyfit(la, lk, 1)[0])


def run_dformer(n_eval):
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
    m = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = m.load_state_dict(sd, strict=False)
    assert len(sd) - len(miss.missing_keys) >= 0.9 * len(sd), "DFormer load failed"
    m = m.eval().cuda()
    _orig = hook(m)

    def cap_(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            m(rgb, dep)
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
    gen = torch.Generator().manual_seed(SEED)
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
                acc[c].append(ladder(dec, Zc, [k - z for k, z in zip(Zk, Zc)], logp, size, gen))
    return {c: {a: np.mean([r[a] for r in v], axis=0).tolist() for a in ARMS}
            for c, v in acc.items()}


def run_cmnext(n_eval):
    DL = PROJECT / "repos" / "DELIVER"
    sys.path.insert(0, str(DL))
    os.chdir(DL)
    import depth_corruptions as DC
    from semseg.models.cmnext import CMNeXt
    from semseg.datasets.nyu import NYU
    from semseg.augmentations_mm import get_val_augmentation
    from torch.utils.data import DataLoader
    ck = DL / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
    m = CMNeXt("CMNeXt-B4", 40, ["img", "depth"])
    sd = torch.load(str(ck), map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    msg = m.load_state_dict(sd, strict=False)
    assert len(sd) - len(msg.missing_keys) >= 0.9 * len(sd), "CMNeXt load failed"
    m = m.eval().cuda()
    _orig = hook(m)

    def cap_(mods):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            m(mods)
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
    gen = torch.Generator().manual_seed(SEED)
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
            acc[c].append(ladder(dec, Zc, [k - z for k, z in zip(Zk, Zc)], logp, size, gen))
    return {c: {a: np.mean([r[a] for r in v], axis=0).tolist() for a in ARMS}
            for c, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=10)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    print("=== DFormerv2-S ===")
    df = run_dformer(a.n_eval)
    print("=== CMNeXt-B4 ===")
    cm = run_cmnext(a.n_eval)

    res = {"alphas": ALPHAS, "arms": ARMS, "dformer": df, "cmnext": cm}
    exps, ratios = {}, {}
    for arch, data in (("DFormerv2", df), ("CMNeXt", cm)):
        print(f"\n=== {arch} ===")
        print(f"{'condition':14s} {'arm':9s} " + " ".join(f"a={al:<6g}" for al in ALPHAS)
              + f" {'exponent':>9s}")
        for c, arms in data.items():
            for arm in ARMS:
                e = exponent(arms[arm])
                exps[f"{arch}|{c}|{arm}"] = e
                print(f"{c:14s} {arm:9s} " + " ".join(f"{v:8.1f}" for v in arms[arm])
                      + f" {e:9.3f}")
            print(f"{'':14s} {'RATIO/real':9s} " + " ".join(
                f"{arms['shuffle'][i]/max(arms['real'][i],1e-12):8.3f}" for i in range(len(ALPHAS)))
                + "   <- shuffle")
            print(f"{'':14s} {'':9s} " + " ".join(
                f"{arms['trans_8'][i]/max(arms['real'][i],1e-12):8.3f}" for i in range(len(ALPHAS)))
                + "   <- translate")
            for arm in ("shuffle", "trans_8"):
                r = [arms[arm][i] / max(arms["real"][i], 1e-12) for i in range(len(ALPHAS))]
                ratios[f"{arch}|{c}|{arm}"] = r

    # P-M20a: exponents agree across arms (exclude uniform: near-degenerate magnitudes)
    spread_a = {}
    for arch, data in (("DFormerv2", df), ("CMNeXt", cm)):
        for c in data:
            e = [exps[f"{arch}|{c}|{arm}"] for arm in ("real", "trans_8", "shuffle")]
            spread_a[f"{arch}|{c}"] = max(e) - min(e)
    p_a = all(v <= 0.15 for v in spread_a.values())

    # P-M20b/c: ratio constancy = max/min over alpha
    drift = {k: max(v) / max(min(v), 1e-12) for k, v in ratios.items()}
    p_b = all(drift[k] <= 1.2 for k in drift if k.endswith("shuffle"))
    p_c = all(drift[k] <= 1.2 for k in drift if k.endswith("trans_8"))
    p_d = p_a and p_b and p_c

    print(f"\n--- EXPONENT SPREAD ACROSS ARMS (real/translate/shuffle) ---")
    for k, v in spread_a.items():
        print(f"  {k:28s} spread {v:.3f} {'OK' if v <= 0.15 else '<-- exceeds 0.15'}")
    print(f"\n--- RATIO DRIFT ACROSS alpha (max/min; 1.00 = perfectly constant) ---")
    for k, v in drift.items():
        print(f"  {k:34s} drift {v:.3f} {'OK' if v <= 1.2 else '<-- exceeds 1.2'}")

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M20a exponent same across arms (<=0.15) -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M20b shuffle ratio constant in alpha (<=1.2x) -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M20c translate ratio constant in alpha (<=1.2x) -> {'HIT' if p_c else 'MISS'}")
    print(f"P-M20d factorization holds on BOTH architectures -> {'HIT' if p_d else 'MISS'}")

    print(f"\n--- OUTCOME-TABLE BRANCH ---")
    print(f"  P-M20a {'HIT' if p_a else 'MISS'} | P-M20b {'HIT' if p_b else 'MISS'} | "
          f"P-M20c {'HIT' if p_c else 'MISS'} | P-M20d {'HIT' if p_d else 'MISS'}")
    print("  (interpretation belongs in STANDING-ASSESSMENT, not here)")

    res.update({"exponents": exps, "ratios": ratios, "exponent_spread": spread_a,
                "ratio_drift": drift, "P_M20a": bool(p_a), "P_M20b": bool(p_b),
                "P_M20c": bool(p_c), "P_M20d": bool(p_d), "n_eval": a.n_eval})
    o = GAPFILL / "m20_factorization.json"
    o.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
