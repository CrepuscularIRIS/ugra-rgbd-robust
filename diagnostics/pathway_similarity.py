#!/usr/bin/env python
"""Does the METHOD's primary pathway resemble the DEPTH-FREE ORACLE's, or the BASELINE's?

THE CLAIM UNDER TEST (reconstructed thesis): the baseline's missing-depth gap is dominated by an
UNDER-DEVELOPED RGB pathway, and training-time attenuation develops it. If so, then with depth
removed the method's encoder features should resemble the oracle's (which never had a crutch) more
than the baseline's (which did) -- at identical capacity.

  PREDICTS: CKA(method, oracle) - CKA(baseline, oracle) > 0, largest at stage 2 (86.9% of damage).
  REFUTES : if the method's absent-condition features are no closer to the oracle's than the
            baseline's are, the frontier result is a bare Pareto observation with no mechanism.

All three models are run with the DEPTH INPUT ZEROED, so mask_d = |0-0| = 0 for every model and they
all execute the same depth-free operator -- the comparison isolates the learned pathway, not the bias.

Linear CKA on stage feature maps (flattened per-image), a standard representation-similarity measure
that is invariant to rotation and isotropic scaling.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT, "repos", "DFormer"))
sys.path.insert(0, os.path.join(_PROJECT, "repos"))

CAP = {}


def grab(name):
    def hook(_m, _i, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        CAP[name] = o.detach().float().flatten(1).cpu()
    return hook


def linear_cka(X, Y):
    """Linear CKA via the GRAM formulation -- mathematically identical to the feature form but
    O(n^2) instead of O(d^2). Here n=20 images but d~1.2M features per stage, so the feature form
    would build a 1.2M x 1.2M matrix. Always use the Gram form when n << d."""
    X = (X - X.mean(0, keepdim=True)).double()
    Y = (Y - Y.mean(0, keepdim=True)).double()
    K = X @ X.T
    L = Y @ Y.T
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype) - 1.0 / n
    Kc, Lc = H @ K @ H, H @ L @ H
    hsic = (Kc * Lc).sum()
    return float(hsic / (Kc.norm() * Lc.norm() + 1e-12))


def build(ckpt, cfg):
    from models.builder import EncoderDecoder
    m = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d, syncbn=False)
    w = torch.load(ckpt, map_location="cpu")
    sd = w.get("model", w.get("state_dict", w))
    r = m.load_state_dict(sd, strict=False)
    matched = len(m.state_dict()) - len(r.missing_keys)
    if matched < 0.9 * len(m.state_dict()):
        raise SystemExit(f"FATAL: {ckpt} matched only {matched}/{len(m.state_dict())} tensors")
    print(f"[load] {os.path.basename(os.path.dirname(ckpt))}: matched {matched}/{len(m.state_dict())}")
    for nm, mod in m.named_modules():
        if any(nm.endswith(f"backbone.layers.{i}") for i in range(4)):
            mod.register_forward_hook(grab(nm.split("layers.")[1]))
    return m.cuda().eval()


def features(model, loader, n):
    acc = collections.defaultdict(list)
    seen = 0
    for mb in loader:
        if seen >= n:
            break
        img = mb["data"].cuda().float()
        modal = torch.zeros_like(mb["modal_x"].cuda().float())   # DEPTH REMOVED
        if img.dim() == 3:
            img, modal = img[None], modal[None]
        torch.manual_seed(20260721)      # Ham decoder redraws bases every forward
        CAP.clear()
        with torch.no_grad():
            model(img, modal)
        for k, v in CAP.items():
            acc[k].append(v)
        seen += 1
    return {k: torch.cat(v, 0) for k, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
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
    }
    feats = {}
    for name, path in ck.items():
        loader, _ = get_val_loader(types.SimpleNamespace(distributed=False, world_size=1),
                                   RGBXDataset, cfg, val_batch_size=1)
        model = build(path, cfg)
        feats[name] = features(model, loader, a.n)
        del model
        torch.cuda.empty_cache()

    stages = sorted(feats["baseline"].keys())
    print(f"\nLinear CKA to the DEPTH-FREE ORACLE, depth removed, {a.n} images")
    print(f"{'stage':<7}{'baseline':>10}{'method':>10}{'method-baseline':>18}")
    for s in stages:
        cb = linear_cka(feats["baseline"][s], feats["oracle"][s])
        cm = linear_cka(feats["method"][s], feats["oracle"][s])
        star = "  <== predicted largest" if s == "2" else ""
        print(f"{s:<7}{cb:>10.4f}{cm:>10.4f}{cm - cb:>+18.4f}{star}")
    # control: self-similarity must be 1.0, else the metric or the caching is broken
    print(f"\n[control] CKA(oracle, oracle) stage2 = {linear_cka(feats['oracle']['2'], feats['oracle']['2']):.4f} (must be 1.0000)")


if __name__ == "__main__":
    main()
