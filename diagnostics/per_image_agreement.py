#!/usr/bin/env python
"""Task-space replacement for the saturated CKA instrument.

CKA could not discriminate: baseline-vs-oracle differ by 3.92 mIoU on the absent condition yet their
stage-2 CKA is 0.9966. So move to TASK space, where the resolution demonstrably exists.

CLAIM UNDER TEST (current thesis): the baseline's missing-depth gap is dominated by an UNDER-DEVELOPED
primary pathway, and attenuation develops it. If so, with depth removed the METHOD should behave like
the depth-free ORACLE image-by-image, more than the BASELINE does.

  PREDICTS: corr(method, oracle) > corr(baseline, oracle) over per-image scores.
  REFUTES : if the method's per-image profile is no closer to the oracle's than the baseline's, the
            Pareto result is a bare observation with no pathway mechanism behind it.

VALIDATION GATE (this instrument must prove itself before its internals are read -- the standing rule
after three probes silently ran an untrained model): the mean per-image score must reproduce the
harness ORDERING baseline < method ~ oracle. Per-image mean != dataset mIoU (which pools confusion
matrices), so only the ordering is required, not the absolute values.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT, "repos", "DFormer"))
sys.path.insert(0, os.path.join(_PROJECT, "repos"))


def build(ckpt, cfg):
    from models.builder import EncoderDecoder
    m = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d, syncbn=False)
    w = torch.load(ckpt, map_location="cpu")
    sd = w.get("model", w.get("state_dict", w))
    r = m.load_state_dict(sd, strict=False)
    matched = len(m.state_dict()) - len(r.missing_keys)
    if matched < 0.9 * len(m.state_dict()):
        raise SystemExit(f"FATAL: {ckpt} matched only {matched}/{len(m.state_dict())}")
    print(f"[load] {os.path.basename(os.path.dirname(ckpt))}: {matched}/{len(m.state_dict())}", flush=True)
    return m.cuda().eval()


def per_image_scores(model, loader, n, nclass=40):
    out = []
    seen = 0
    for mb in loader:
        if seen >= n:
            break
        img = mb["data"].cuda().float()
        modal = torch.zeros_like(mb["modal_x"].cuda().float())     # DEPTH REMOVED
        gt = mb["label"].cuda()
        if img.dim() == 3:
            img, modal = img[None], modal[None]
        if gt.dim() == 2:
            gt = gt[None]
        torch.manual_seed(20260721)          # Ham decoder redraws bases every forward
        with torch.no_grad():
            lg = model(img, modal)
        lg = lg[0] if isinstance(lg, (tuple, list)) else lg
        if lg.shape[-2:] != gt.shape[-2:]:
            lg = torch.nn.functional.interpolate(lg, size=gt.shape[-2:], mode="bilinear", align_corners=False)
        pred = lg.argmax(1)
        valid = gt != 255
        ious = []
        for c in torch.unique(gt[valid]).tolist():
            if c >= nclass:
                continue
            p, g = (pred == c) & valid, (gt == c) & valid
            u = (p | g).sum().item()
            if u > 0:
                ious.append((p & g).sum().item() / u)
        out.append(float(np.mean(ious)) if ious else float("nan"))
        seen += 1
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=654)
    a = ap.parse_args()
    import types
    from importlib import import_module
    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset

    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    for k, v in (("pad", False), ("pad_SUNRGBD", False), ("num_workers", 2)):
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    P = _PROJECT
    ck = {
        "baseline": f"{P}/repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth",
        "method":   f"{P}/repos/DFormer/checkpoints/NYUDepthv2_DFormerv2_S_20260725-155828/latest.pth",
        "oracle":   f"{P}/repos/DFormer/checkpoints/NYUDepthv2_DFormerv2_S_20260725-213212/latest.pth",
        # ACCURACY-MATCHED CONTROL: a DIFFERENT recipe (Bernoulli dropout) that lands at essentially
        # the oracle's absent-condition accuracy. If per-image agreement with the oracle is driven
        # merely by ACCURACY LEVEL, this must also correlate ~0.90. If it correlates markedly less
        # at equal accuracy, the accuracy explanation dies.
        "dropout":  f"{P}/results/score_push/runs/l2_nof/screens/alt_bernoulli/ckpt_final_c2.pth",
    }
    S = {}
    for name, path in ck.items():
        loader, _ = get_val_loader(types.SimpleNamespace(distributed=False, world_size=1),
                                   RGBXDataset, cfg, val_batch_size=1)
        model = build(path, cfg)
        S[name] = per_image_scores(model, loader, a.n)
        del model
        torch.cuda.empty_cache()
        print(f"  {name}: mean per-image score {np.nanmean(S[name]):.4f} over {len(S[name])} images", flush=True)
        np.save(f"{P}/results/score_push/runs/l2_nof/perimg_{name}.npy", S[name])

    ok = np.nanmean(S["baseline"]) < np.nanmean(S["method"]) and np.nanmean(S["baseline"]) < np.nanmean(S["oracle"])
    print(f"\n[GATE] ordering baseline < method,oracle : {'PASS' if ok else 'FAIL — do NOT interpret internals'}")
    if not ok:
        return

    m = ~(np.isnan(S["baseline"]) | np.isnan(S["method"]) | np.isnan(S["oracle"]))
    b, me, o = S["baseline"][m], S["method"][m], S["oracle"][m]
    from scipy.stats import pearsonr, spearmanr
    print(f"\nper-image agreement with the DEPTH-FREE ORACLE (depth removed, n={m.sum()})")
    print(f"  corr(baseline, oracle)  pearson={pearsonr(b,o)[0]:.4f}  spearman={spearmanr(b,o)[0]:.4f}")
    print(f"  corr(method,   oracle)  pearson={pearsonr(me,o)[0]:.4f}  spearman={spearmanr(me,o)[0]:.4f}")
    print(f"  corr(method, baseline)  pearson={pearsonr(me,b)[0]:.4f}")
    print(f"\n  PREDICTION corr(method,oracle) > corr(baseline,oracle): "
          f"{'SUPPORTED' if pearsonr(me,o)[0] > pearsonr(b,o)[0] else 'NOT SUPPORTED'}")
    # where the baseline loses most, does the method gain most?
    deficit = o - b          # per-image shortfall of the baseline vs the depth-free model
    gain = me - b            # per-image gain of the method over the baseline
    # NOTE: corr(o-b, m-b) shares the -b term and has an INFLATED null (~+0.5 for independent equal
    # variance), so it is reported ONLY against a permutation null.
    obs = pearsonr(deficit, gain)[0]
    rng = np.random.default_rng(0)
    null = []
    for _ in range(2000):
        perm = rng.permutation(len(me))
        null.append(pearsonr(o - b, me[perm] - b)[0])
    null = np.array(null)
    print(f"  corr(deficit, gain) = {obs:.4f}   permutation null mean={null.mean():.4f} "
          f"p95={np.quantile(null,0.95):.4f}   -> {'ABOVE null' if obs>np.quantile(null,0.95) else 'WITHIN null (not evidence)'}")
    if "dropout" in S:
        d = S["dropout"][m]
        print(f"\n  [accuracy-matched control] dropout mean={np.nanmean(S['dropout']):.4f}")
        print(f"  corr(dropout, oracle) pearson={pearsonr(d,o)[0]:.4f}")
        print(f"  => if ~= corr(method,oracle), agreement tracks ACCURACY not recipe (confound inseparable);"
              f" if much lower at equal accuracy, the pathway account gains support")


if __name__ == "__main__":
    main()
