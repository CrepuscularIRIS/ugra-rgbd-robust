"""M19 — does the GHOSTS-OF-SOFTMAX radius predict where our quadratic law breaks?

THE BRIDGE. arXiv:2603.13552 derives a closed-form Taylor-convergence radius for cross-entropy along
a direction, set by the nearest complex singularity of the softmax partition function:

    rho_a = pi / Delta_a,    Delta_a = max_k a_k - min_k a_k,    a_k = grad z_k . v

costing one Jacobian-vector product. It is derived for OPTIMIZATION STEPS in parameter space. We
apply it to a NATURAL DISPLACEMENT in representation space — the same mathematical object in a
setting their derivation does not cover.

UNITS. Our ladder scales the full displacement by alpha, so with a_k measured along the FULL Delta
the predicted breakdown sits at alpha* = pi / Delta_a, PER PIXEL. KL aggregates over pixels, so the
aggregate law should depart from p=2 once an appreciable fraction of pixels pass their own radius.

WHY THIS IS A GENUINE RETRODICTION. M18 measured the saturation ordering BEFORE this bound was read,
and nothing in Delta_a was tuned to it:
    DFormerv2 dconst       p: 2.02 -> 1.80 (alpha=1) -> 1.47 (alpha=2)   saturates HARDEST
    DFormerv2 invalid_*    p: -> 1.91/1.86 -> 1.64/1.55                  saturates less
    CMNeXt    invalid_*    p: -> 1.93/2.01 -> 1.97/2.04                  does NOT saturate

Prereg + outcome table + severity: .grill/prediction-ledger.md (2026-08-10, M19).

    python3 scripts/m19_ghosts_radius.py --gpu 1 --n-eval 10
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
EPS = 1e-3
_CAP = {"store": None, "mode": None}


def radius_stats(dz):
    """alpha* = pi / (max_k dz_k - min_k dz_k), per pixel. dz is (1, K, H, W)."""
    spread = (dz.max(1).values - dz.min(1).values).squeeze(0)      # (H, W)
    a_star = np.pi / spread.clamp(min=1e-9)
    v = a_star.reshape(-1)
    return {"median": float(v.median()),
            "p10": float(v.kthvalue(max(1, int(0.10 * v.numel()))).values),
            "p25": float(v.kthvalue(max(1, int(0.25 * v.numel()))).values),
            "frac_past_1": float((v < 1.0).float().mean()),
            "frac_past_2": float((v < 2.0).float().mean())}


def build_dformer():
    DF = PROJECT / "repos" / "DFormer"
    sys.path.insert(0, str(DF)); sys.path.insert(0, str(DF / "utils"))
    sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
    sys.path.insert(0, str(PROJECT / "scripts"))
    os.chdir(DF)
    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    ck = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
    m = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = m.load_state_dict(sd, strict=False)
    assert len(sd) - len(miss.missing_keys) >= 0.9 * len(sd), "DFormer load failed"
    return m.eval().cuda(), cfg


def hook(model):
    _orig = model.decode_head.forward

    def _h(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _h
    return _orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=10)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    out = {}

    # ---------------- DFormerv2
    model, cfg = build_dformer()
    _orig = hook(model)
    from m1_mig_full import COND, corrupt

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
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)
            if i == 0:
                assert float((zl - dec(Zc, size)).abs().max()) == 0.0
            for c in conds:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                delta = [k - z for k, z in zip(Zk, Zc)]
                dz = (dec([z + EPS * d for z, d in zip(Zc, delta)], size) - zl) / EPS
                acc[c].append(radius_stats(dz))
    out["DFormerv2"] = {c: {k: float(np.mean([r[k] for r in v])) for k in v[0]}
                        for c, v in acc.items()}

    # ---------------- CMNeXt
    DL = PROJECT / "repos" / "DELIVER"
    sys.path.insert(0, str(DL))
    os.chdir(DL)
    import depth_corruptions as DC
    from semseg.models.cmnext import CMNeXt
    from semseg.datasets.nyu import NYU
    from semseg.augmentations_mm import get_val_augmentation
    from torch.utils.data import DataLoader
    ck = DL / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
    cm = CMNeXt("CMNeXt-B4", 40, ["img", "depth"])
    sd = torch.load(str(ck), map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    msg = cm.load_state_dict(sd, strict=False)
    assert len(sd) - len(msg.missing_keys) >= 0.9 * len(sd), "CMNeXt load failed"
    cm = cm.eval().cuda()
    _o2 = hook(cm)

    def cap2(mods):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            cm(mods)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec2(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _o2(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    ds = NYU(str(DL / "data" / "NYUDepthv2"), "val",
             get_val_augmentation([480, 640]), ["img", "depth"], None)
    dl = DataLoader(ds, batch_size=1, num_workers=2)
    c2 = {"constant": lambda d: DC.depth_constant(d.clone()),
          "invalid_0.10": lambda d: DC.depth_invalid(d.clone(), frac=0.10, seed=SEED),
          "invalid_0.55": lambda d: DC.depth_invalid(d.clone(), frac=0.55, seed=SEED)}
    acc2 = {c: [] for c in c2}
    for i, batch in enumerate(dl):
        if i >= a.n_eval:
            break
        sample, _ = batch
        mods = [t.cuda() for t in sample]
        size = mods[0].shape[-2:]
        Zc = cap2(mods)
        zl = dec2(Zc, size)
        if i == 0:
            assert float((zl - dec2(Zc, size)).abs().max()) == 0.0
        for c, fn in c2.items():
            Zk = cap2([mods[0], fn(mods[1])])
            delta = [k - z for k, z in zip(Zk, Zc)]
            dz = (dec2([z + EPS * d for z, d in zip(Zc, delta)], size) - zl) / EPS
            acc2[c].append(radius_stats(dz))
    out["CMNeXt"] = {c: {k: float(np.mean([r[k] for r in v])) for k in v[0]}
                     for c, v in acc2.items()}

    # ---------------- compare to M18's measured saturation
    m18 = json.loads((GAPFILL / "m18_quadratic_horizon.json").read_text())
    ALPH = m18["alphas"]

    def first_below(kl, thr=1.8):
        la, lk = np.log(ALPH), np.log(np.maximum(kl, 1e-12))
        for i in range(len(ALPH) - 1):
            if (lk[i + 1] - lk[i]) / (la[i + 1] - la[i]) < thr:
                return ALPH[i + 1]
        return float("inf")

    meas = {}
    for arch, key in (("DFormerv2", "dformer"), ("CMNeXt", "cmnext")):
        for c, kl in m18[key].items():
            meas[f"{arch}|{c}"] = first_below(kl)

    print(f"\n{'arch|condition':28s} {'med a*':>8s} {'p25 a*':>8s} {'p10 a*':>8s} "
          f"{'%past@1':>8s} {'%past@2':>8s} {'measured break':>15s}")
    rows = {}
    for arch in out:
        for c, s in out[arch].items():
            key = f"{arch}|{c}"
            mk = key if key in meas else f"{arch}|" + ("dconst" if c == "constant" else c)
            mv = meas.get(mk, float("nan"))
            rows[key] = dict(s, measured_break=mv)
            print(f"{key:28s} {s['median']:8.3f} {s['p25']:8.3f} {s['p10']:8.3f} "
                  f"{s['frac_past_1']*100:8.1f} {s['frac_past_2']*100:8.1f} {mv:15.2f}")

    d_dc = out["DFormerv2"]["dconst"]["median"]
    d_i1 = out["DFormerv2"]["invalid_0.10"]["median"]
    p_b = d_dc < d_i1
    cm_inv = [out["CMNeXt"][c]["median"] for c in ("invalid_0.10", "invalid_0.55")]
    p_a = all(v >= 2 for v in cm_inv)
    ks = list(rows)
    fin = [k for k in ks if np.isfinite(rows[k]["measured_break"])]

    def sp(x, y):
        n = len(x)
        if n < 3:
            return float("nan")
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])
    p_c = len(fin) >= 3 and sp([rows[k]["median"] for k in fin],
                               [rows[k]["measured_break"] for k in fin]) >= 0.7
    df_f = np.mean([out["DFormerv2"][c]["frac_past_1"] for c in out["DFormerv2"]])
    cm_f = np.mean([out["CMNeXt"][c]["frac_past_1"] for c in out["CMNeXt"]])
    p_d = df_f > cm_f

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M19a CMNeXt invalid median a* >= 2: {[round(v,2) for v in cm_inv]} -> "
          f"{'HIT' if p_a else 'MISS'}")
    print(f"P-M19b dconst a* ({d_dc:.3f}) < invalid_0.10 a* ({d_i1:.3f}) -> "
          f"{'HIT' if p_b else 'MISS'}")
    if len(fin) >= 3:
        print(f"P-M19c Spearman(median a*, measured break) = "
              f"{sp([rows[k]['median'] for k in fin],[rows[k]['measured_break'] for k in fin]):+.3f}"
              f" -> {'HIT' if p_c else 'MISS'}   (n={len(fin)} finite)")
    else:
        print(f"P-M19c VOID — only {len(fin)} conditions had a finite measured break")
    print(f"P-M19d frac past radius at alpha=1: DFormerv2 {df_f*100:.1f}% vs "
          f"CMNeXt {cm_f*100:.1f}% -> {'HIT' if p_d else 'MISS'}")

    print(f"\n--- VERDICT ---")
    if p_b:
        print("  THE GHOSTS BOUND TRANSFERS ORDINALLY. A closed form derived for optimization steps")
        print("  in PARAMETER space predicts which natural REPRESENTATION-space displacement")
        print("  saturates first. The validity-range clause becomes a confirmed bridge.")
    else:
        print("  THE GHOSTS BOUND DOES NOT TRANSFER. The radius governing optimization steps does")
        print("  not order saturation for natural internal displacements -- a finding about the")
        print("  difference between the two settings, and it must be reported as such.")

    o = GAPFILL / "m19_ghosts_radius.json"
    o.write_text(json.dumps({"rows": rows, "measured_break": meas, "eps": EPS,
                             "P_M19a": bool(p_a), "P_M19b": bool(p_b), "P_M19c": bool(p_c), "P_M19d": bool(p_d),
                             "n_eval": a.n_eval}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
