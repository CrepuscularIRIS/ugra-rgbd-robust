"""Endpoint evaluation: S_c (complete depth), S_m (constant depth), optional partial depth, and T/B/G.

    python -m dat.evaluate --host DFormerv2-S --ckpt C.pth --rgb-only-ckpt RGB.pth --out result.json
    python -m dat.evaluate --host DFormerv2-S --ckpt dat.pth --s-rgb 53.1 --partial 0.25

Conditions: "complete", "missing", "partial:<fraction>[:rect|multi]" (e.g. "partial:0.25:rect"); the removed
region is reproducible from --seed. Single-scale, full-image inference, as in the main table.
Note: the upstream HAM decoder draws a random initialisation on every forward pass, so repeated evaluation of
one checkpoint differs by about +-0.1-0.2 mIoU; compare differences larger than that.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import metrics
from .availability import missing_depth, partial_depth


def condition_input(depth: torch.Tensor, cond: str, c: float, seed: int) -> torch.Tensor:
    if cond == "complete":
        return depth
    if cond == "missing":
        return missing_depth(depth, c)
    if cond.startswith("partial"):
        parts = cond.split(":")
        fraction = float(parts[1]) if len(parts) > 1 else 0.25
        kind = parts[2] if len(parts) > 2 else "rect"
        return partial_depth(depth, c, fraction, kind, seed)
    raise ValueError(f"unknown condition {cond!r}")


OFFICIAL_SCALES = (0.5, 0.75, 1.0, 1.25, 1.5)  # upstream infer.py: multi-scale + flip is the published protocol


def predict(host, model, rgb: torch.Tensor, depth: torch.Tensor, scales: Sequence[float] = (1.0,),
            flip: bool = False) -> torch.Tensor:
    """Class probabilities (B, K, H, W) summed over scales and the horizontal flip.

    Mirrors upstream `utils/val_mm.evaluate_msf`: images are resized to a multiple of 32 with
    `align_corners=True`, logits are resized back, softmax outputs are added, and the flipped pass is un-flipped
    before it is added. `scales=(1.0,)` with `flip=False` is the single-scale protocol (no resizing).
    """
    h, w = rgb.shape[-2:]
    if tuple(scales) == (1.0,) and not flip:
        return host.forward(model, rgb, depth).softmax(1)
    total = 0
    for s in scales:
        nh, nw = (int(math.ceil(int(s * h) / 32)) * 32, int(math.ceil(int(s * w) / 32)) * 32)
        r, d = (F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=True) for x in (rgb, depth))
        views = [(r, d, False)] + ([(torch.flip(r, (3,)), torch.flip(d, (3,)), True)] if flip else [])
        for rr, dd, flipped in views:
            logits = host.forward(model, rr, dd)
            if flipped:
                logits = torch.flip(logits, (3,))
            total = total + F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=True).softmax(1)
    return total


@torch.no_grad()
def evaluate(host, model, loader, conditions: Sequence[str] = ("complete", "missing"), seed: int = 0,
             limit: int | None = None, device: str = "cuda", ignore_index: int = 255,
             scales: Sequence[float] = (1.0,), flip: bool = False) -> dict:
    """Confusion matrices and mIoU per condition over `loader` (batch size 1, dict batches).

    `scales`/`flip` select the inference protocol; the published DFormer numbers use `OFFICIAL_SCALES` + flip.
    """
    k = host.num_classes
    hists = {c: torch.zeros(k, k, dtype=torch.long, device=device) for c in conditions}
    model.eval()
    n = 0
    for i, b in enumerate(loader):
        if limit is not None and i >= limit:
            break
        rgb, depth, label = b["data"].to(device), b["modal_x"].to(device), b["label"].to(device)
        for cond in conditions:
            pred = predict(host, model, rgb, condition_input(depth, cond, host.empty_value, seed + i), scales,
                           flip).argmax(1)
            keep = label != ignore_index
            hists[cond] += torch.bincount(label[keep] * k + pred[keep], minlength=k * k).view(k, k)
        n += 1
    out = {}
    for cond, h in hists.items():
        h = h.cpu().numpy()
        out[cond] = {"mIoU": round(metrics.miou_from_hist(h), 2), "IoU": metrics.iou_from_hist(h).round(2).tolist(),
                     "hist": h.tolist(), "n": n}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="e.g. DFormerv2-S, DFormerv1-T")
    ap.add_argument("--dataset", default="NYUDepthv2", choices=("NYUDepthv2", "SUNRGBD"))
    ap.add_argument("--ckpt", required=True, help="checkpoint of the model to evaluate")
    ap.add_argument("--rgb-only-ckpt", help="RGB-only baseline checkpoint; evaluated with constant depth")
    ap.add_argument("--s-rgb", type=float, help="RGB-only mIoU when already known (e.g. from Table 1)")
    ap.add_argument("--partial", type=float, action="append", default=[], help="also evaluate partial depth")
    ap.add_argument("--protocol", default="single", choices=("single", "official"),
                    help="single: one scale, no flip; official: scales 0.5-1.5 + flip, as upstream infer.py")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args(argv)

    from .data import eval_loader
    from .hosts.dformer import make_host

    host = make_host(a.host, a.dataset)
    loader = eval_loader(host)
    conds = ["complete", "missing"] + [f"partial:{f}:rect" for f in a.partial]

    model = host.build(device=a.device)
    host.load_weights(model, a.ckpt)
    kw = dict(scales=OFFICIAL_SCALES, flip=True) if a.protocol == "official" else {}
    res = evaluate(host, model, loader, conds, a.seed, a.limit, a.device, **kw)
    s_rgb = a.s_rgb
    if a.rgb_only_ckpt:
        host.load_weights(model, a.rgb_only_ckpt)
        s_rgb = evaluate(host, model, loader, ["missing"], a.seed, a.limit, a.device, **kw)["missing"]["mIoU"]
    e = metrics.Endpoints(res["complete"]["mIoU"], res["missing"]["mIoU"], s_rgb)
    e.check_identity()
    print(metrics.format_row(a.host, e))
    for c in conds[2:]:
        print(f"  {c:<22} mIoU {res[c]['mIoU']:.2f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump({"host": a.host, "dataset": a.dataset, "ckpt": a.ckpt, "protocol": a.protocol, "endpoints": e.as_dict(),
                   "conditions": {k: {kk: v for kk, v in d.items() if kk != "hist"} for k, d in res.items()},
                   "hist": {k: d["hist"] for k, d in res.items()}}, open(a.out, "w"))


if __name__ == "__main__":
    main()
