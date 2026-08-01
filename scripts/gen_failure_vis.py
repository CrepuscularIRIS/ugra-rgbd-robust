#!/usr/bin/env python3
"""Generate side-by-side failure visualizations: RGB | GT | clean_pred | missing_pred | diff.

Usage (from repos/DFormer/):
    CUDA_VISIBLE_DEVICES=1 python ../../scripts/gen_failure_vis.py \
        --ckpt checkpoints/trained/DFormerv2_Small_NYU.pth \
        --out ../../spine/runs/n7/bottleneck-run/failure_vis \
        --n_images 12 --seed 20260721
"""
import argparse
import sys
import os
import pathlib
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "repos", "DFormer"))
sys.path.insert(0, PROJECT_ROOT)

from importlib import import_module
from models.builder import EncoderDecoder as segmodel

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NYU40_PALETTE = np.array([
    [0,0,0],[174,199,232],[152,223,138],[31,119,180],[255,187,120],
    [188,189,34],[140,86,75],[255,152,150],[214,39,40],[197,176,213],
    [148,103,189],[196,156,148],[23,190,207],[247,182,210],[219,219,141],
    [255,127,14],[158,218,229],[44,160,44],[112,128,144],[227,119,194],
    [82,84,163],[128,128,0],[190,153,153],[0,0,128],[30,30,30],
    [220,220,220],[255,193,0],[160,0,0],[0,80,100],[0,0,230],
    [128,64,128],[102,102,156],[244,35,232],[107,142,35],[0,0,70],
    [70,130,180],[220,20,60],[255,0,0],[0,0,142],[119,11,32],
], dtype=np.uint8)


def colorize(seg, palette):
    h, w = seg.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(min(len(palette), seg.max() + 1)):
        out[seg == c] = palette[c]
    return out


def load_and_preprocess(rgb_path, depth_path, label_path, config):
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    h, w = config.eval_crop_size
    rgb_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
    label_resized = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

    rgb_vis = rgb_resized.copy()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    rgb_norm = rgb_resized.astype(np.float32) / 255.0
    rgb_norm = (rgb_norm - mean) / std
    rgb_t = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()

    depth_3ch = np.stack([depth_resized] * 3, axis=-1).astype(np.float32) / 255.0
    depth_mean = np.array([0.485, 0.456, 0.406])
    depth_std = np.array([0.229, 0.224, 0.225])
    depth_norm = (depth_3ch - depth_mean) / depth_std
    depth_t = torch.from_numpy(depth_norm.transpose(2, 0, 1)).float()

    return rgb_t, depth_t, label_resized, rgb_vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    config = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = segmodel(cfg=config)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "state_dict" in state:
        state = state["state_dict"]
    elif "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    test_list = config.eval_source
    with open(test_list, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    rng = np.random.RandomState(args.seed)
    indices = rng.choice(len(lines), size=min(args.n_images, len(lines)), replace=False)
    indices.sort()

    print(f"Generating failure visualizations for {len(indices)} images...")

    for idx in indices:
        parts = lines[idx].split()
        rgb_rel = parts[0]
        label_rel = parts[1] if len(parts) > 1 else rgb_rel.replace("RGB", "Label")
        img_id = pathlib.Path(rgb_rel).stem
        depth_rel = f"Depth/{img_id}.png"

        base = config.dataset_path
        rgb_path = os.path.join(base, rgb_rel.lstrip("/"))
        depth_path = os.path.join(base, depth_rel.lstrip("/"))
        label_path = os.path.join(base, label_rel.lstrip("/"))

        name = pathlib.Path(rgb_path).stem

        rgb_t, depth_t, label, rgb_vis = load_and_preprocess(
            rgb_path, depth_path, label_path, config
        )

        rgb_batch = rgb_t.unsqueeze(0).to(device)
        depth_batch = depth_t.unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            pred_clean = model(rgb_batch, depth_batch).softmax(dim=1).argmax(dim=1).cpu().squeeze().numpy()

            depth_missing = torch.zeros_like(depth_batch)
            pred_missing = model(rgb_batch, depth_missing).softmax(dim=1).argmax(dim=1).cpu().squeeze().numpy()

        gt_color = colorize(label.astype(np.uint8), NYU40_PALETTE)
        clean_color = colorize(pred_clean.astype(np.uint8), NYU40_PALETTE)
        missing_color = colorize(pred_missing.astype(np.uint8), NYU40_PALETTE)

        diff_mask = (pred_clean != pred_missing).astype(np.uint8)
        diff_color = np.zeros_like(clean_color)
        diff_color[diff_mask == 1] = [255, 0, 0]
        diff_color[diff_mask == 0] = (clean_color[diff_mask == 0] * 0.3).astype(np.uint8)

        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        titles = ["RGB Input", "Ground Truth", "Pred (clean depth)", "Pred (missing depth)", "Diff (red=changed)"]
        images = [rgb_vis, gt_color, clean_color, missing_color, diff_color]

        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.set_title(title, fontsize=10)
            ax.axis("off")

        valid = label != 255
        clean_acc = (pred_clean[valid] == label[valid]).mean() * 100 if valid.any() else 0
        miss_acc = (pred_missing[valid] == label[valid]).mean() * 100 if valid.any() else 0
        changed_pct = diff_mask.sum() / diff_mask.size * 100

        fig.suptitle(
            f"{name}  |  clean pixAcc: {clean_acc:.1f}%  |  missing pixAcc: {miss_acc:.1f}%  |  changed: {changed_pct:.1f}%",
            fontsize=11, y=0.02,
        )
        plt.tight_layout()
        save_path = outdir / f"{name}_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved: {save_path}  (clean={clean_acc:.1f}% miss={miss_acc:.1f}% diff={changed_pct:.1f}%)")

    print(f"\nDone. {len(indices)} visualizations in {outdir}")


if __name__ == "__main__":
    main()
