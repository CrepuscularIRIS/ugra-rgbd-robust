#!/usr/bin/env python
"""Where does stage 1's depth-driven attention change DIE?

Established: removing the depth bias moves stage 1's attention distribution as much as stage 2's
(TV 0.0142 vs 0.0145) yet stage 1 contributes 0.2% of the gap and stage 2 contributes 86.9%. So the
bias is not absorbed at the softmax — the suppression is downstream. This asks how far downstream.

Two forward passes on identical inputs (DEPTH_LAMBDA_TRAIN toggled 1 -> 0, which zeroes the depth
term inside the forward), capturing:
  - each encoder stage's OUTPUT feature map
  - the final logits
and reporting the relative change ||f_with - f_without|| / ||f_with||.

READING:
  - stage-1 output delta ~ 0  => the attention change dies INSIDE its own block (the residual
    stream dominates; the attention branch contributes little to the block output there).
  - stage-1 output delta comparable to stage 0/2, but logits unchanged => the change survives the
    block and is discarded later (decoder or subsequent stages).
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
        CAP[name] = o.detach().float().clone()

    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
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

    for nm, m in model.named_modules():
        if nm.endswith("backbone.layers.0") or nm.endswith("backbone.layers.1") \
           or nm.endswith("backbone.layers.2") or nm.endswith("backbone.layers.3"):
            m.register_forward_hook(grab(nm.split("layers.")[1]))

    val_ds, _ = get_val_loader(types.SimpleNamespace(distributed=False, world_size=1), RGBXDataset, cfg, val_batch_size=1)

    # Per-stage ISOLATION: gate weight[1] directly (as the eval harness does) so we compare
    # "depth ON at stage s only" against "depth OFF everywhere". A global toggle would give a
    # cumulative delta and could not attribute anything to a single stage.
    from models.encoders.DFormerv2 import GeoPriorGen
    import re as _re

    geos = []
    for nm, m in model.named_modules():
        if isinstance(m, GeoPriorGen):
            geos.append((int(_re.search(r"layers\.(\d+)\.", nm).group(1)), m))
    orig_w1 = {id(m): m.weight[1].detach().clone() for _, m in geos}

    def set_stage(on_stage):
        """on_stage: None = all off, "ALL" = all on, int = that stage only."""
        with torch.no_grad():
            for s, m in geos:
                on = (on_stage == "ALL") or (on_stage is not None and s == on_stage)
                m.weight[1].copy_(orig_w1[id(m)] if on else torch.zeros_like(orig_w1[id(m)]))

    def run(img, modal):
        CAP.clear()
        torch.manual_seed(20260721)  # Ham decoder redraws bases every forward
        with torch.no_grad():
            lg = model(img, modal)
        lg = lg[0] if isinstance(lg, (tuple, list)) else lg
        return {k: v for k, v in CAP.items()}, lg.detach().float().clone()

    acc = collections.defaultdict(list)
    n = 0
    for mb in val_ds:
        if n >= a.n:
            break
        img, modal = mb["data"].cuda().float(), mb["modal_x"].cuda().float()
        gt = mb["label"].cuda()
        if img.dim() == 3:
            img, modal = img[None], modal[None]
        if gt.dim() == 2:
            gt = gt[None]

        set_stage(None)          # depth off everywhere = the reference
        f_off, l_off = run(img, modal)
        # SELF-CONSISTENCY GATE: depth ON everywhere vs OFF everywhere must show a change
        # consistent with the harness's 6.63-point gap. If a FULL removal barely moves this
        # probe's predictions, the probe does not reproduce the process that generates D(s) and
        # NONE of its per-stage internals may be interpreted.
        set_stage("ALL")
        f_all, l_all = run(img, modal)
        g_full = torch.nn.functional.interpolate(gt[:, None].float(), size=l_all.shape[-2:],
                                                 mode="nearest")[:, 0].long()
        vmask = g_full != 255
        acc["GATE_flip"].append(float((l_all.argmax(1) != l_off.argmax(1)).float().mean()))
        acc["GATE_acc_on"].append(float(((l_all.argmax(1) == g_full) & vmask).sum().item() / max(vmask.sum().item(), 1)))
        acc["GATE_acc_off"].append(float(((l_off.argmax(1) == g_full) & vmask).sum().item() / max(vmask.sum().item(), 1)))

        for s in (0, 1, 2, 3):   # depth restored at exactly one stage
            set_stage(s)
            f_on, l_on = run(img, modal)
            for k in sorted(f_on):
                if int(k) >= s:  # only sites at or downstream of the gated stage can be affected
                    d = (f_on[k] - f_off[k]).norm() / max(f_off[k].norm().item(), 1e-9)
                    acc[f"s{s}->feat{k}"].append(float(d))
            acc[f"s{s}->logits"].append(float((l_on - l_off).norm() / max(l_off.norm().item(), 1e-9)))
            # DIRECT test of the "dies at the argmax" reading: segmentation reads argmax, so only
            # the component of the perturbation that crosses a class boundary changes a prediction.
            p_on, p_off = l_on.argmax(1), l_off.argmax(1)
            ch = p_on != p_off
            acc[f"s{s}->flip"].append(float(ch.float().mean()))
            # THE decisive quantity: of the pixels whose prediction changed, how many moved
            # TOWARD the ground truth vs away? A label-uninformative perturbation flips as many
            # pixels but gains nothing; that is the only account left for stages 1 and 3.
            if ch.any():
                g = torch.nn.functional.interpolate(gt[:, None].float(), size=p_on.shape[-2:],
                                                    mode="nearest")[:, 0].long()
                valid = ch & (g != 255)
                if valid.any():
                    gain = ((p_on == g) & valid).sum().item()
                    loss = ((p_off == g) & valid).sum().item()
                    acc[f"s{s}->net"].append((gain - loss) / max(valid.sum().item(), 1))
        n += 1
    set_stage(None)
    with torch.no_grad():        # restore
        for _, m in geos:
            m.weight[1].copy_(orig_w1[id(m)])

    gf, ga_on, ga_off = np.mean(acc["GATE_flip"]) * 100, np.mean(acc["GATE_acc_on"]) * 100, np.mean(acc["GATE_acc_off"]) * 100
    print(f"\n=== SELF-CONSISTENCY GATE (depth ALL-ON vs ALL-OFF) ===")
    print(f"  pixel flip {gf:.3f}%   pixel-acc on {ga_on:.2f}%  off {ga_off:.2f}%   delta {ga_on-ga_off:+.2f} pts")
    print(f"  harness reference: full depth removal costs 6.63 mIoU")
    OK = (ga_on - ga_off) > 1.0
    print(f"  GATE: {'PASS - probe reproduces a substantial depth effect' if OK else 'FAIL - probe does NOT reproduce the harness; do NOT interpret internals'}")
    if not OK:
        print("  (per-stage rows below are UNINTERPRETABLE until this gate passes)")
    print(f"\nISOLATED downstream footprint of each stage's depth bias ({n} images)")
    print("rel delta = ||f(depth on at stage s only) - f(depth off)|| / ||f(depth off)||")
    ds = {0: "28.4%", 1: "0.2%", 2: "86.9%", 3: "0.0%"}
    tv = {0: "0.00899", 1: "0.01421", 2: "0.01452", 3: "n/a"}
    for s in (0, 1, 2, 3):
        row = [f"feat{k}={np.mean(v):.5f}" for k, v in sorted(acc.items()) if k.startswith(f"s{s}->feat")]
        lg = np.mean(acc[f"s{s}->logits"])
        fl = np.mean(acc[f"s{s}->flip"]) * 100.0
        net = np.mean(acc[f"s{s}->net"]) * 100.0 if acc.get(f"s{s}->net") else float("nan")
        print(f"  stage {s}  D(s)={ds[s]:<7} flip={fl:6.3f}%  NET-CORRECT-among-flips={net:+7.2f}%   logits={lg:.5f}")


if __name__ == "__main__":
    main()
