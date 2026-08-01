#!/usr/bin/env python
"""Per-stage BIAS-TO-LOGIT RATIO — the candidate explanation for the stage-1 anomaly.

WHY. At matched total effective authority, stage 2 delivers 170x and stage 0 80x what stage 1
delivers. So damage is not a function of the bias's own magnitude. But the bias does not act
alone: it is ADDED to the qk logits before the softmax,

    attn = softmax( (qr @ kr^T) + mask )

so what should govern its influence is its size RELATIVE TO THE LOGIT SCALE IT PERTURBS. A bias
that is small next to the qk spread is washed out by the softmax; the same bias next to a narrow
logit distribution dominates it. This measures, per stage, on real data:

    r(s) = std(mask_depth) / std(qk_logits)

and the depth-only part of the mask specifically (weight[1]*mask_d), since weight[0]*mask_pos is
present in both conditions and is not what vanishes.

PREDICTION being tested: r(stage1) << r(stage0), r(stage2). If so, the stage-1 anomaly is
EXPLAINED — its depth bias is negligible against its own logit scale. If r(stage1) is comparable
to the others, the anomaly stands and needs a different account.

Forward-only, no training. Reseeded per forward (the Ham decoder is nondeterministic, and while
this only reads encoder internals, determinism is cheap here).
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

STATS = collections.defaultdict(list)
_STAGE_OF = {}
_PENDING_DEPTH = {}


def patch(model):
    """Record, per stage: the qk logit scale, and the DEPTH-ONLY bias scale.

    The mask returned by GeoPriorGen is weight[0]*mask_pos + weight[1]*mask_d, and the positional
    term dominates it — so measuring the TOTAL mask says nothing about depth. We therefore patch
    GeoPriorGen to compute and stash the depth-only term separately, and read it in the attention
    wrapper where the addition to the logits actually happens.
    """
    from models.encoders.DFormerv2 import Decomposed_GSA, GeoPriorGen, angle_transform

    for name, m in model.named_modules():
        if isinstance(m, (Decomposed_GSA, GeoPriorGen)):
            _STAGE_OF[m] = int(name.split("layers.")[1].split(".")[0])

    # --- GeoPriorGen: stash std of the DEPTH-ONLY contribution weight[1]*mask_d ---
    def make_geo(mod, orig):
        def fwd(HW_tuple, depth_map, split_or_not=False, d_hat=None):
            out = orig(HW_tuple, depth_map, split_or_not=split_or_not, d_hat=d_hat)
            with torch.no_grad():
                import torch.nn.functional as F

                dm = F.interpolate(depth_map, size=HW_tuple, mode="bilinear", align_corners=False)
                if split_or_not:
                    md_w = mod.generate_1d_depth_decay(HW_tuple[1], HW_tuple[0], dm)
                else:
                    md_w = mod.generate_depth_decay(HW_tuple[0], HW_tuple[1], dm)
                mod._depth_only_std = float((mod.weight[1] * md_w).std().item())
            return out

        return fwd

    # --- Decomposed_GSA: qk logit scale, paired with the sibling Geo's depth-only std ---
    def make_gsa(mod, orig):
        def fwd(x, rel_pos, split_or_not=False):
            with torch.no_grad():
                (sin, cos), (mask_h, mask_w) = rel_pos
                bsz, h, w, _ = x.size()
                q = mod.q_proj(x)
                k = mod.k_proj(x) * mod.scaling
                q = q.view(bsz, h, w, mod.num_heads, mod.key_dim).permute(0, 3, 1, 2, 4)
                k = k.view(bsz, h, w, mod.num_heads, mod.key_dim).permute(0, 3, 1, 2, 4)
                qr = angle_transform(q, sin, cos)
                kr = angle_transform(k, sin, cos)
                qk_w = qr.transpose(1, 2) @ kr.transpose(1, 2).transpose(-1, -2)
                STATS[_STAGE_OF[mod]].append(
                    (float(qk_w.std().item()), float(mask_w.std().item()), _PENDING_DEPTH.get(_STAGE_OF[mod], float("nan")))
                )
            return orig(x, rel_pos, split_or_not)

        return fwd

    geos = [m for m in _STAGE_OF if isinstance(m, GeoPriorGen)]
    gsas = [m for m in _STAGE_OF if isinstance(m, Decomposed_GSA)]
    for m in geos:
        o = m.forward

        def wrap(mod=m, orig=o):
            f = make_geo(mod, orig)

            def g(*a, **k):
                r = f(*a, **k)
                _PENDING_DEPTH[_STAGE_OF[mod]] = mod._depth_only_std
                return r

            return g

        m.forward = wrap()
    for m in gsas:
        m.forward = make_gsa(m, m.forward)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()

    from importlib import import_module

    from models.builder import EncoderDecoder
    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset

    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    # attrs the val preprocessor expects but that only the train/eval entrypoints set
    for k, v in (("pad", False), ("pad_SUNRGBD", False), ("num_workers", 2)):
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    ckpt = os.path.join(_PROJECT, "repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth")

    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d, syncbn=False)
    w = torch.load(ckpt, map_location="cpu")
    sd = w.get("model", w.get("state_dict", w))
    _r = model.load_state_dict(sd, strict=False)
    _matched = len(model.state_dict()) - len(_r.missing_keys)
    if _matched < 0.9 * len(model.state_dict()):
        raise SystemExit(f"FATAL: checkpoint load matched only {_matched}/{len(model.state_dict())} tensors "
                         f"- refusing to run on an untrained model")
    print(f"[load] matched {_matched}/{len(model.state_dict())} tensors")
    model = model.cuda().eval()
    patch(model)

    # get_val_loader only touches engine.distributed (and world_size when distributed)
    import types

    val_ds, _ = get_val_loader(types.SimpleNamespace(distributed=False, world_size=1), RGBXDataset, cfg, val_batch_size=1)
    n = 0
    for mb in val_ds:
        if n >= a.n:
            break
        torch.manual_seed(20260721)
        img = mb["data"].cuda().float()
        modal = mb["modal_x"].cuda().float()
        if img.dim() == 3:
            img, modal = img[None], modal[None]
        with torch.no_grad():
            model.encode_decode(img, modal) if hasattr(model, "encode_decode") else model(img, modal)
        n += 1

    print(f"\nper-stage bias-to-logit ratio over {n} images")
    print(f"{'stage':<7}{'std(qk)':>10}{'std(mask_tot)':>14}{'std(depth)':>12}{'depth/qk':>10}   iso-authority")
    iso = {0: "16.3%", 1: "0.2%", 2: "33.8%", 3: "(non-split)"}
    for s in sorted(STATS):
        arr = np.array(STATS[s])
        qk, mk, dp = arr[:, 0].mean(), arr[:, 1].mean(), np.nanmean(arr[:, 2])
        print(f"{s:<7}{qk:>10.4f}{mk:>14.4f}{dp:>12.4f}{dp / max(qk, 1e-9):>10.2f}   {iso.get(s, '')}")


if __name__ == "__main__":
    main()
