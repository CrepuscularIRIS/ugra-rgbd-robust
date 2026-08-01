#!/usr/bin/env python
"""Per-stage FUNCTIONAL effect of the depth bias: how much does it move the attention itself?

WHY THIS AND NOT ANOTHER RATIO. Eight explanations for the stage-1 anomaly have been killed, all of
them MAGNITUDE arguments (bias size, authority, bias/logit ratio, share of total bias, saturation,
resolution, attention type, redundancy). The pre-registered reopen condition was an instrument that
measures the bias's FUNCTIONAL effect instead. That is this: within a single forward, compute

    A_with    = softmax(qk + mask)                  # mask = pos + depth
    A_without = softmax(qk + mask - depth_only)     # positional bias only

and report per stage the mean L1 distance between them (attention is a distribution, so L1/2 is the
total-variation distance, in [0,1]).

THE FORK IT RESOLVES:
  - If TV(stage1) ~ 0 while its bias magnitude is normal, the bias is absorbed AT THE SOFTMAX — it
    is present in the logits but does not move the distribution (e.g. swamped by a dominant
    positional term at that stage).
  - If TV(stage1) is comparable to stages 0/2 yet damage is ~0, the attention DOES move and the
    downstream network is simply insensitive to it — the suppression is representational, not
    attentional.
Either answer converts "unexplained" into "localized", which no further magnitude ratio can do.

Forward-only, single pass, no training. n small by default: the W-branch attention tensors are
large at stage 0.
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

TV = collections.defaultdict(list)
_STAGE_OF = {}
_DEPTH_MASK_W = {}


def patch(model):
    from models.encoders.DFormerv2 import Decomposed_GSA, GeoPriorGen, angle_transform

    for name, m in model.named_modules():
        if isinstance(m, (Decomposed_GSA, GeoPriorGen)):
            _STAGE_OF[m] = int(name.split("layers.")[1].split(".")[0])

    def make_geo(mod, orig):
        def fwd(HW_tuple, depth_map, split_or_not=False, d_hat=None):
            out = orig(HW_tuple, depth_map, split_or_not=split_or_not, d_hat=d_hat)
            if split_or_not:
                with torch.no_grad():
                    import torch.nn.functional as F

                    dm = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
                    md_w = mod.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], dm)
                    # same broadcast the real forward applies to the depth term
                    _DEPTH_MASK_W[_STAGE_OF[mod]] = (mod.weight[1] * md_w).detach()
            return out

        return fwd

    def make_gsa(mod, orig):
        def fwd(x, rel_pos, split_or_not=False):
            if split_or_not:
                with torch.no_grad():
                    (sin, cos), (mask_h, mask_w) = rel_pos
                    bsz, h, w, _ = x.size()
                    q = mod.q_proj(x)
                    k = mod.k_proj(x) * mod.scaling
                    q = q.view(bsz, h, w, mod.num_heads, mod.key_dim).permute(0, 3, 1, 2, 4)
                    k = k.view(bsz, h, w, mod.num_heads, mod.key_dim).permute(0, 3, 1, 2, 4)
                    qr = angle_transform(q, sin, cos)
                    kr = angle_transform(k, sin, cos)
                    qk = qr.transpose(1, 2) @ kr.transpose(1, 2).transpose(-1, -2)
                    mw = mask_w.transpose(1, 2)
                    s = _STAGE_OF[mod]
                    dep = _DEPTH_MASK_W.get(s)
                    if dep is not None and dep.shape == mask_w.shape:
                        a1 = torch.softmax(qk + mw, -1)
                        a0 = torch.softmax(qk + (mask_w - dep).transpose(1, 2), -1)
                        TV[s].append(float((0.5 * (a1 - a0).abs().sum(-1)).mean().item()))
                        del a1, a0
            return orig(x, rel_pos, split_or_not)

        return fwd

    from models.encoders.DFormerv2 import GeoPriorGen as G

    for m in [m for m in _STAGE_OF if isinstance(m, G)]:
        m.forward = make_geo(m, m.forward)
    for m in [m for m in _STAGE_OF if isinstance(m, Decomposed_GSA)]:
        m.forward = make_gsa(m, m.forward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    import types
    from importlib import import_module

    from models.builder import EncoderDecoder
    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset

    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    for k, v in (("pad", False), ("pad_SUNRGBD", False), ("num_workers", 2)):
        if not hasattr(cfg, k):
            setattr(cfg, k, v)

    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d, syncbn=False)
    w = torch.load(os.path.join(_PROJECT, "repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth"), map_location="cpu")
    sd = w.get("model", w.get("state_dict", w))
    _r = model.load_state_dict(sd, strict=False)
    _matched = len(model.state_dict()) - len(_r.missing_keys)
    if _matched < 0.9 * len(model.state_dict()):
        raise SystemExit(f"FATAL: checkpoint load matched only {_matched}/{len(model.state_dict())} tensors "
                         f"- refusing to run on an untrained model")
    print(f"[load] matched {_matched}/{len(model.state_dict())} tensors")
    model = model.cuda().eval()
    patch(model)

    val_ds, _ = get_val_loader(types.SimpleNamespace(distributed=False, world_size=1), RGBXDataset, cfg, val_batch_size=1)
    n = 0
    for mb in val_ds:
        if n >= a.n:
            break
        torch.manual_seed(20260721)
        img, modal = mb["data"].cuda().float(), mb["modal_x"].cuda().float()
        if img.dim() == 3:
            img, modal = img[None], modal[None]
        with torch.no_grad():
            model(img, modal)
        n += 1

    print(f"\nper-stage TOTAL-VARIATION distance between attention with vs without the depth bias ({n} images)")
    print(f"{'stage':<7}{'TV distance':>14}   D(s)")
    D = {0: "28.4%", 1: "0.2%", 2: "86.9%"}
    for s in sorted(TV):
        print(f"{s:<7}{np.mean(TV[s]):>14.5f}   {D.get(s,'')}")


if __name__ == "__main__":
    main()
