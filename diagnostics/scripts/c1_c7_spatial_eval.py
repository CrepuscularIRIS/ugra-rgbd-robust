#!/usr/bin/env python3
"""C1 spatial rank-k inject + C7 trust-direction steering on stage features.

Uses conclusions from pipeline_results.json (best_stage, go_c1).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))
os.chdir(ROOT / "repos" / "DFormer")

from diagnostics.corruptions.depth_corruptions import apply_corruption
from diagnostics.hooks.data_utils import get_nyu_val_loader, iter_limited
from diagnostics.hooks.model_utils import load_dformerv2_s_nyu
from diagnostics.pipeline_all import (
    accumulate_confmat,
    backbone_feats,
    decode_feats,
    miou_from_conf,
    project_delta_topk,
)


def main():
    device = "cuda:0"
    num = 500
    batch_size = 2
    stage = 3
    ranks = [1, 2, 4, 8, 16, 32]
    alphas = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]

    model, config = load_dformerv2_s_nyu(device=device)
    loader = get_nyu_val_loader(config, batch_size=batch_size)
    K = config.num_classes

    conf_clean = torch.zeros(K, K, device=device)
    conf_miss = torch.zeros(K, K, device=device)
    conf_c1 = {r: torch.zeros(K, K, device=device) for r in ranks}
    conf_c7 = {a: torch.zeros(K, K, device=device) for a in alphas}

    # load trust direction if exists
    td_path = ROOT / "results" / "diagnostics" / f"trust_dir_stage{stage}.pt"
    trust_dir = torch.load(td_path, map_location=device) if td_path.exists() else None

    n = 0
    t0 = time.time()
    for rgb, depth, label in iter_limited(loader, num):
        rgb = rgb.float().to(device)
        depth = depth.float().to(device)
        label = label.long().to(device)
        b = rgb.shape[0]
        hw = rgb.shape[-2:]
        with torch.no_grad():
            fc = backbone_feats(model, rgb, depth)
            fm = backbone_feats(model, rgb, apply_corruption(depth, "entire_missing"))
            accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf_clean, K)
            accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf_miss, K)

            delta = fc[stage] - fm[stage]
            for r in ranks:
                dk = project_delta_topk(delta, r)
                hybrid = list(fm)
                hybrid[stage] = fm[stage] + dk
                accumulate_confmat(decode_feats(model, hybrid, hw).argmax(1), label, conf_c1[r], K)

            if trust_dir is not None:
                # broadcast direction over spatial dims of stage features
                d = trust_dir.to(device).view(1, -1, 1, 1)
                for a in alphas:
                    hybrid = list(fm)
                    hybrid[stage] = fm[stage] + a * d
                    accumulate_confmat(
                        decode_feats(model, hybrid, hw).argmax(1), label, conf_c7[a], K
                    )
        n += b
        if n % 100 == 0:
            print(f"progress {n}/{num} {time.time()-t0:.1f}s")

    m_clean = miou_from_conf(conf_clean)
    m_miss = miou_from_conf(conf_miss)

    def rec(m):
        den = m_clean - m_miss
        return float((m - m_miss) / den) if abs(den) > 1e-6 else 0.0

    out = {
        "clean_miou": m_clean,
        "entire_missing_miou": m_miss,
        "C1_spatial_rank_inject_stage": stage,
        "C1": {str(r): {"miou": miou_from_conf(conf_c1[r]), "recovery": rec(miou_from_conf(conf_c1[r]))} for r in ranks},
        "C7_trust_steer_stage": stage,
        "C7": {str(a): {"miou": miou_from_conf(conf_c7[a]), "recovery": rec(miou_from_conf(conf_c7[a]))} for a in alphas},
        "seconds": time.time() - t0,
        "num": n,
    }
    path = ROOT / "results" / "diagnostics" / "c1_c7_spatial.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
