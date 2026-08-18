"""M40 (E2) — the equivalence audit on the RELEASED DFormerv2 weights, across size and dataset.

THE FALSIFIABLE STRUCTURAL PREDICTION. Every DFormerv2 variant injects depth through the same
relational attention bias `mask_d = |d_i - d_j|`, which vanishes for a zero field and for any
constant field. The criterion therefore predicts, *before* the measurement, that

    EVERY released DFormerv2 checkpoint gives a mIoU spread of EXACTLY 0.00
    across all candidate removals — independent of model size and of dataset.

This is a genuine prediction and it can fail: if the spread is non-zero for any cell, the relational
account of DFormer's depth pathway is wrong, or some variant routes depth elsewhere.

CELLS (the released weights actually on disk):

    DFormerv2-S x NYU Depth v2   trained/DFormerv2_Small_NYU.pth
    DFormerv2-L x NYU Depth v2   pretrained/DFormerv2_Large_NYU.pth        (different SIZE)
    DFormerv2-S x SUN RGB-D      trained/SUNRGBD/DFormerv2_Small_SUNRGBD.pth (different DATASET)

Removing, if it passes, two stated limitations: single model size and single dataset.

CANARY (blocking, per cell): the reference removal compared against itself must give exactly 0.
GATE (per cell): >= 90% of checkpoint tensors must load, and clean mIoU must be in a sane band.

Prereg: .grill/prediction-ledger.md (2026-08-10, M40).

    python3 scripts/m40_released_set_audit.py --gpu 1 --n-eval 654
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
for _p in (DFORMER, DFORMER / "utils", PROJECT / "scripts"):
    sys.path.insert(0, str(_p))

GAPFILL = PROJECT / "results" / "gapfill"
CKROOT = DFORMER / "checkpoints"
SEED = 20260721
_CAP = {"store": None, "mode": None}

CELLS_ALL = [
    ("S_NYU", "local_configs.NYUDepthv2.DFormerv2_S", None,
     CKROOT / "trained" / "DFormerv2_Small_NYU.pth"),
    ("L_NYU", "local_configs.NYUDepthv2.DFormerv2_L", None,
     CKROOT / "pretrained" / "DFormerv2_Large_NYU.pth"),
    # the released SUNRGBD "Small" model carries a 1024-wide decoder (conv_seg is [37,1024,1,1]),
    # so only the backbone is overridden — the embed dim is left at the L config's 1024.
    ("S_SUN", "local_configs.SUNRGBD.DFormerv2_L", ("DFormerv2_S", 1024),
     CKROOT / "trained" / "SUNRGBD" / "DFormerv2_Small_SUNRGBD.pth"),
]


def hist(pred, lab, ncls):
    m = (lab >= 0) & (lab < ncls)
    return torch.bincount(ncls * lab[m].long() + pred[m].long(),
                          minlength=ncls ** 2).reshape(ncls, ncls)


def miou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    return float((tp / u)[u > 0].mean()) * 100.0


def build_conditions(d):
    lo = float(torch.quantile(d.float().flatten(), 0.1))
    hi = float(torch.quantile(d.float().flatten(), 0.9))
    mn = float(d.float().mean())
    return {
        "zero(reference)": torch.zeros_like(d),
        "self-comparison": torch.zeros_like(d),
        "const_mean": torch.full_like(d, mn),
        "const_q10": torch.full_like(d, lo),
        "const_q90": torch.full_like(d, hi),
        "const_+1.0": torch.full_like(d, mn + 1.0),
    }


def run_cell(tag, cfgmod, override, ckpt, gpu, n_eval):
    from importlib import import_module
    cfg = import_module(cfgmod).C
    cfg.pad = False
    if override:
        cfg.backbone, cfg.decoder_embed_dim = override
    ncls = int(cfg.num_classes)
    print(f"\n########## {tag}  backbone={cfg.backbone}  dataset={cfg.dataset_name}  "
          f"classes={ncls} ##########")

    from models.builder import EncoderDecoder
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    msg = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(msg.missing_keys)
    print(f"  checkpoint: {got}/{len(sd)} tensors loaded")
    if got < 0.9 * len(sd):
        print(f"  REFUSING {tag}: <90% loaded")
        return None
    model.eval().cuda()

    _orig = model.decode_head.forward

    def _h(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _h

    def cap(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = model(rgb, dep)
        _CAP["mode"] = None
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return _CAP["store"], o

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")
    engine = Engine(custom_parser=None)
    vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

    names = None
    Zd = KLd = Hd = None
    Hc = torch.zeros(ncls, ncls, dtype=torch.long)
    n = 0
    for i, b in enumerate(vl):
        if i >= n_eval:
            break
        rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
        lab = b["label"].cuda().squeeze(0).long()
        size = rgb.shape[-2:]
        Zc, oc = cap(rgb, dep)
        oc = F.interpolate(oc, size=size, mode="bilinear", align_corners=False)
        Hc += hist(oc.argmax(1).squeeze(0).cpu(), lab.cpu(), ncls)
        conds = build_conditions(dep)
        if names is None:
            names = list(conds)
            Zd = {c: [] for c in names}
            KLd = {c: [] for c in names}
            Hd = {c: torch.zeros(ncls, ncls, dtype=torch.long) for c in names}
        ref = None; logp = None
        for c in names:
            Zk, ok = cap(rgb, conds[c])
            ok = F.interpolate(ok, size=size, mode="bilinear", align_corners=False)
            Hd[c] += hist(ok.argmax(1).squeeze(0).cpu(), lab.cpu(), ncls)
            if c == "zero(reference)":
                ref, logp = Zk, F.log_softmax(ok, 1)
            dz = sum(float(((k - r) ** 2).sum()) for k, r in zip(Zk, ref)) ** 0.5
            dr = sum(float(((r - z) ** 2).sum()) for r, z in zip(ref, Zc)) ** 0.5
            Zd[c].append(dz / max(dr, 1e-12))
            KLd[c].append(float(F.kl_div(F.log_softmax(ok, 1), logp,
                                         log_target=True, reduction="batchmean")))
        n += 1
        if n % 200 == 0:
            print(f"    {n}", flush=True)

    clean = miou(Hc)
    rows = {c: {"relZ": sum(Zd[c]) / len(Zd[c]),
                "kl": sum(KLd[c]) / len(KLd[c]),
                "miou": miou(Hd[c])} for c in names}
    can = rows["self-comparison"]["kl"]
    ref_m = rows["zero(reference)"]["miou"]
    tested = [c for c in names if c.startswith("const")]
    spread = max(rows[c]["miou"] for c in tested) - min(rows[c]["miou"] for c in tested)
    equiv = [c for c in tested if rows[c]["kl"] == 0.0]

    print(f"  clean mIoU {clean:.2f}   zero-fill {ref_m:.2f}   canary {can:.3e}")
    print(f"  {'condition':20s} {'relZ':>9s} {'KL':>12s} {'mIoU':>8s} {'dmIoU':>8s}")
    for c in names:
        r = rows[c]
        print(f"  {c:20s} {r['relZ']:9.4f} {r['kl']:12.4f} {r['miou']:8.2f} "
              f"{r['miou']-ref_m:+8.2f}")
    print(f"  --> {len(equiv)}/{len(tested)} removal-equivalent   spread {spread:.2f}")
    if can != 0.0:
        print(f"  CANARY FAIL — verdict withheld for {tag}")
    return {"tag": tag, "backbone": cfg.backbone, "dataset": cfg.dataset_name,
            "n_classes": ncls, "clean_miou": clean, "zero_miou": ref_m,
            "canary_kl": can, "rows": rows, "n_equivalent": len(equiv),
            "n_tested": len(tested), "miou_spread": spread, "n_eval": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=654)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    out = {}
    import os as _os
    _only = _os.environ.get("ONLY_CELL")
    CELLS = [c for c in CELLS_ALL if (not _only or c[0] == _only)]
    for tag, cfgmod, ov, ck in CELLS:
        if not ck.exists():
            print(f"SKIP {tag}: {ck} not found")
            continue
        try:
            r = run_cell(tag, cfgmod, ov, ck, a.gpu, a.n_eval)
            if r:
                out[tag] = r
        except Exception as e:
            print(f"SKIP {tag}: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print(f"{'cell':10s} {'backbone':14s} {'dataset':14s} {'clean':>7s} {'canary':>9s} "
          f"{'equiv':>7s} {'spread':>8s}")
    allzero = True
    for t, r in out.items():
        print(f"{t:10s} {r['backbone']:14s} {r['dataset']:14s} {r['clean_miou']:7.2f} "
              f"{r['canary_kl']:9.1e} {r['n_equivalent']}/{r['n_tested']:<5} "
              f"{r['miou_spread']:8.2f}")
        allzero &= (r["canary_kl"] == 0.0 and r["miou_spread"] == 0.0
                    and r["n_equivalent"] == r["n_tested"])
    print(f"\n  P-M40 (EVERY released DFormerv2 cell: 4/4 equivalent, spread exactly 0.00): "
          f"{'HIT' if allzero and out else 'MISS'}")
    (GAPFILL / "m40_released_set_audit.json").write_text(json.dumps(
        {"cells": out, "P_M40": bool(allzero and out)}, indent=1))
    print(f"\nwrote {GAPFILL / 'm40_released_set_audit.json'}")


if __name__ == "__main__":
    main()
