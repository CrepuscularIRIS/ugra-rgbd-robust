#!/usr/bin/env python3
"""Phase-0 smoke: load model, apply 3 corruptions, cache stage feats for N images.

Default: N=8 for smoke; use --num 500 for full Phase-0.
GPU: set CUDA_VISIBLE_DEVICES externally (recommend 1).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# project root = ugra-rgbd-robust
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))

from diagnostics.corruptions.depth_corruptions import apply_corruption, default_corruption_suite
from diagnostics.hooks.data_utils import get_nyu_val_loader, iter_limited
from diagnostics.hooks.model_utils import forward_with_stage_feats, load_dformerv2_s_nyu, logits_to_miou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=8, help="Number of images to process")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, default=str(ROOT / "cache" / "activations" / "smoke"))
    ap.add_argument("--save-feats", action="store_true", help="Save stage features (large)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    model, config = load_dformerv2_s_nyu(device=device)
    # run from DFormer root so relative dataset paths work
    import os

    os.chdir(ROOT / "repos" / "DFormer")
    loader = get_nyu_val_loader(config, batch_size=args.batch_size)

    suite = default_corruption_suite(seed=0)
    conditions = ["clean"] + list(suite.keys())
    metrics = {c: [] for c in conditions}

    n_done = 0
    for rgb, depth, label in iter_limited(loader, args.num):
        rgb = rgb.float().to(device)
        depth = depth.float().to(device)
        label = label.to(device)
        b = rgb.shape[0]

        # fix collate: dataloader may return numpy
        if not torch.is_tensor(rgb):
            rgb = torch.from_numpy(rgb).float().to(device)
        if not torch.is_tensor(depth):
            depth = torch.from_numpy(depth).float().to(device)
        if not torch.is_tensor(label):
            label = torch.from_numpy(label).long().to(device)

        for name in conditions:
            if name == "clean":
                d_in = depth
            else:
                d_in = apply_corruption(depth, name, **suite[name])
            logits, feats = forward_with_stage_feats(model, rgb, d_in)
            miou = logits_to_miou(logits, label, num_classes=config.num_classes)
            metrics[name].append(miou)

            if args.save_feats:
                # save mean-pooled stage features only (compact) for smoke+500
                for si, f in enumerate(feats):
                    # [B,C,H,W] -> [B,C]
                    pooled = f.float().mean(dim=(2, 3)).cpu()
                    torch.save(
                        pooled,
                        out_dir / f"img{n_done:04d}_b{b}_{name}_stage{si}_gapool.pt",
                    )

        n_done += b
        print(f"processed {min(n_done, args.num)}/{args.num}")

    summary = {
        c: {
            "mean_miou": float(sum(v) / max(len(v), 1)),
            "n_batches": len(v),
        }
        for c, v in metrics.items()
    }
    (out_dir / "smoke_metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out_dir / "smoke_metrics.json")


if __name__ == "__main__":
    main()
