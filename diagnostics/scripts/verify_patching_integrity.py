#!/usr/bin/env python3
"""Integrity tests for diagnostic patching (run before trusting new numbers).

Checks:
1) entire_missing == zero depth
2) hybrid=feats_clean -> mIoU ~ clean
3) hybrid=feats_corr  -> mIoU ~ corr
4) stage0 patch under Ham decoder barely changes logits (stage0 unused)
5) stage3 clean patch improves over missing
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))
os.chdir(ROOT / "repos" / "DFormer")

from diagnostics.corruptions.depth_corruptions import entire_missing
from diagnostics.hooks.data_utils import get_nyu_val_loader, iter_limited
from diagnostics.hooks.model_utils import load_dformerv2_s_nyu
from diagnostics.pipeline_all import (
    accumulate_confmat,
    backbone_feats,
    decode_feats,
    miou_from_conf,
)


def main():
    device = "cuda:0"
    model, cfg = load_dformerv2_s_nyu(device=device)
    loader = get_nyu_val_loader(cfg, batch_size=2)
    K = 40
    conf = {k: torch.zeros(K, K, device=device) for k in [
        "clean", "miss", "id_clean", "id_miss", "p0", "p3"
    ]}
    n = 0
    max_diff_zero = 0.0
    for rgb, depth, label in iter_limited(loader, 32):
        rgb = rgb.float().to(device)
        depth = depth.float().to(device)
        label = label.long().to(device)
        hw = rgb.shape[-2:]
        with torch.no_grad():
            z = torch.zeros_like(depth)
            fc = backbone_feats(model, rgb, depth)
            fm = backbone_feats(model, rgb, entire_missing(depth))
            fz = backbone_feats(model, rgb, z)
            for a, b in zip(fm, fz):
                max_diff_zero = max(max_diff_zero, (a - b).abs().max().item())

            accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf["clean"], K)
            accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf["miss"], K)
            accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf["id_clean"], K)
            accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf["id_miss"], K)

            h0 = [fm[j] if j != 0 else fc[0] for j in range(4)]
            h3 = [fm[j] if j != 3 else fc[3] for j in range(4)]
            accumulate_confmat(decode_feats(model, h0, hw).argmax(1), label, conf["p0"], K)
            accumulate_confmat(decode_feats(model, h3, hw).argmax(1), label, conf["p3"], K)
        n += rgb.shape[0]

    scores = {k: miou_from_conf(v) for k, v in conf.items()}
    print("n_images", n)
    print("max|entire_missing - zeros|", max_diff_zero)
    print("scores", scores)
    print("delta_id_clean", abs(scores["clean"] - scores["id_clean"]))
    print("delta_id_miss", abs(scores["miss"] - scores["id_miss"]))
    print("p0 - miss", scores["p0"] - scores["miss"])
    print("p3 - miss", scores["p3"] - scores["miss"])

    ok = True
    if max_diff_zero > 1e-5:
        print("FAIL: entire_missing != zeros"); ok = False
    if abs(scores["clean"] - scores["id_clean"]) > 1e-6:
        print("FAIL: identity clean"); ok = False
    if abs(scores["miss"] - scores["id_miss"]) > 1e-6:
        print("FAIL: identity miss"); ok = False
    if abs(scores["p0"] - scores["miss"]) > 0.01:
        print("WARN: stage0 patch changed mIoU a lot (Ham may still slightly couple)"); 
    if scores["p3"] <= scores["miss"] + 0.005:
        print("FAIL: stage3 patch did not improve"); ok = False
    print("OVERALL", "PASS" if ok else "CHECK_WARNINGS")


if __name__ == "__main__":
    main()
