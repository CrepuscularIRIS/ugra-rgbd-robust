#!/usr/bin/env python
"""Qualitative render: baseline vs n7 predictions, side-by-side, per condition.

DIAGNOSTIC (failure post-mortem) — NOT a promotable artifact.

For a fixed set of deterministic val indices, produce a 4-tile row panel
[ RGB input | GT seg | BASELINE pred | n7 pred ] under `clean` and `gaussian`
(sigma 0.15, seed 20260721 — the condition n7 lost 0.90 mIoU on).

The model-construction and corruption path MIRROR scripts/eval_robust_suite.py
exactly:
  - baseline  = segmodel(...)                        (N7_ADAPTER=0 load path)
  - n7        = DFormerv2WithAdapter(segmodel(...))  (N7_ADAPTER=1 load path)
  - single-scale inference = evaluate_msf(scales=[1.0]) internals:
        slide_inference + horizontal-flip TTA, softmax-summed, argmax
  - corruption injected on the normalized 3-ch depth tensor via apply_corruption,
    identical to the frozen harness pre-hook.

Palette: repos/DFormer/utils/nyucmap.npy (the repo's NYUv2 40-class map).

Env (all optional; defaults match the diagnostic brief):
  BASE_CKPT   baseline checkpoint  [checkpoints/trained/DFormerv2_Small_NYU.pth]
  N7_CKPT     n7 checkpoint (best) [checkpoints/NYUDepthv2_DFormerv2_S_20260723-151506/best_n7.pth]
  VAL_INDICES comma list          [0,100,200,300,400,500]
  SEED        corruption seed      [20260721]
  OUT_DIR     png output dir       [<project>/results/score_push/runs/n7_best_ep324/failure_renders]
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- path setup: mirror scripts/eval_robust_suite.py ---------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))  # ugra-rgbd-robust
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
_EXP = os.path.join(_PROJECT, "experiments", "n7_decomp_witness")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT, _EXP):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module  # noqa: E402

from models.builder import EncoderDecoder as segmodel  # noqa: E402
from utils.dataloader.dataloader import get_val_loader  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.engine.engine import Engine  # noqa: E402
from val_mm import slide_inference  # noqa: E402
from diagnostics.corruptions.depth_corruptions import (  # noqa: E402
    apply_corruption,
    default_corruption_suite,
)
from model_wrapper import DFormerv2WithAdapter  # noqa: E402

CONFIG_MODULE = "local_configs.NYUDepthv2.DFormerv2_S"


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _build_model(config, criterion, device, n7: bool, ckpt: str):
    """Construct baseline or n7-wrapped model and load the checkpoint.

    Mirrors eval_robust_suite.py's N7_ADAPTER branch byte-for-byte.
    """
    model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
    if n7:
        model = DFormerv2WithAdapter(
            model,
            num_classes=int(config.num_classes),
            adapter_stage_idx=int(_env("N7_ADAPTER_STAGE_IDX", "2")),
            witness_weight=float(_env("N7_WITNESS_WEIGHT", "0.05")),
            seg_aux_weight=float(_env("N7_SEG_AUX_WEIGHT", "0.2")),
            n_bins=int(_env("N7_N_BINS", "8")),
        )
    weight = torch.load(ckpt, map_location="cpu")
    if "model" in weight:
        weight = weight["model"]
    elif "state_dict" in weight:
        weight = weight["state_dict"]
    rep = model.load_state_dict(weight, strict=False)
    tag = "n7" if n7 else "baseline"
    print(f"[load:{tag}] missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}")
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _predict(model, rgb, modal_x, config, device, cond, cond_kwargs):
    """Single-scale inference == evaluate_msf(scales=[1.0]) internals.

    rgb, modal_x: (1,3,H,W) normalized tensors on device.
    Returns argmax label map (H,W) uint8.
    """
    _, _, H, W = rgb.shape
    # apply corruption to normalized depth exactly like the harness pre-hook
    if cond and cond != "clean":
        modal_x = apply_corruption(modal_x, cond, **cond_kwargs)

    n_classes = config.num_classes
    scaled_logits = torch.zeros(1, n_classes, H, W, device=device)
    # scale = 1.0, round up to /32
    new_H = int(np.ceil(H / 32)) * 32
    new_W = int(np.ceil(W / 32)) * 32
    img = F.interpolate(rgb, size=(new_H, new_W), mode="bilinear", align_corners=True)
    mdl = F.interpolate(modal_x, size=(new_H, new_W), mode="bilinear", align_corners=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = slide_inference(model, img, mdl, config)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
    scaled_logits += logits.softmax(dim=1)
    # horizontal-flip TTA
    img_f = torch.flip(img, dims=(3,))
    mdl_f = torch.flip(mdl, dims=(3,))
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = slide_inference(model, img_f, mdl_f, config)
    logits = torch.flip(logits, dims=(3,))
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=True)
    scaled_logits += logits.softmax(dim=1)

    pred = scaled_logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred


def _colorize(label_map, palette, ignore_index):
    """Map a (H,W) label map to (H,W,3) RGB using the NYU palette.

    ignore/background (config.background, typically 255) -> black.
    """
    lm = label_map.copy()
    valid = lm < len(palette)
    out = np.zeros((*lm.shape, 3), dtype=np.uint8)
    out[valid] = palette[lm[valid]]
    # gt uses 255 for ignore -> black already (palette[255] may be nonzero, so force)
    out[label_map == ignore_index] = 0
    return out


def _label_tile(img_rgb, text):
    """Draw a small text label bar on the top-left of an RGB tile (in place copy)."""
    im = np.ascontiguousarray(img_rgb.copy())
    cv2.rectangle(im, (0, 0), (max(90, 10 * len(text)), 22), (0, 0, 0), -1)
    cv2.putText(im, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return im


def main():
    seed = int(_env("SEED", "20260721"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True

    base_ckpt = _env("BASE_CKPT", "checkpoints/trained/DFormerv2_Small_NYU.pth")
    n7_ckpt = _env(
        "N7_CKPT",
        "checkpoints/NYUDepthv2_DFormerv2_S_20260723-151506/best_n7.pth",
    )
    indices = [int(x) for x in _env("VAL_INDICES", "0,100,200,300,400,500").split(",")]
    out_dir = _env(
        "OUT_DIR",
        os.path.join(_PROJECT, "results/score_push/runs/n7_best_ep324/failure_renders"),
    )
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    palette = np.load(os.path.join(_ROOT, "utils", "nyucmap.npy"))  # (256,3) uint8, RGB

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        config.pad = False
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)

        base_model = _build_model(config, criterion, device, n7=False, ckpt=base_ckpt)
        n7_model = _build_model(config, criterion, device, n7=True, ckpt=n7_ckpt)

        val_loader, _ = get_val_loader(engine, RGBXDataset, config, val_batch_size=1)
        base_ds = val_loader.dataset
        n_val = len(base_ds)
        print(f"[data] val set size = {n_val}; rendering indices {indices}")

        suite = default_corruption_suite(seed=seed)
        conds = {
            "clean": {},
            "gaussian": suite["gaussian"],
        }

        written = []
        for idx in indices:
            if idx >= n_val:
                print(f"[skip] index {idx} >= n_val {n_val}")
                continue
            sample = base_ds[idx]
            rgb_t = sample["data"].unsqueeze(0).to(device)      # (1,3,H,W) normalized
            modal_t = sample["modal_x"].unsqueeze(0).to(device)  # (1,3,H,W) normalized
            gt = sample["label"].numpy().astype(np.int32)        # (H,W)
            fn = sample["fn"]

            # displayable RGB: re-read raw file (BGR->RGB), resize to gt size
            raw = cv2.imread(fn, cv2.IMREAD_COLOR)  # BGR
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            H, W = gt.shape
            if raw.shape[:2] != (H, W):
                raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)

            gt_rgb = _colorize(gt.astype(np.uint8), palette, config.background)

            for cond, ckw in conds.items():
                base_pred = _predict(base_model, rgb_t, modal_t, config, device, cond, ckw)
                n7_pred = _predict(n7_model, rgb_t, modal_t, config, device, cond, ckw)
                base_rgb = _colorize(base_pred, palette, config.background)
                n7_rgb = _colorize(n7_pred, palette, config.background)

                tiles = [
                    _label_tile(raw, "RGB"),
                    _label_tile(gt_rgb, "GT"),
                    _label_tile(base_rgb, "baseline"),
                    _label_tile(n7_rgb, f"n7 [{cond}]"),
                ]
                sep = np.full((H, 6, 3), 255, dtype=np.uint8)
                row = tiles[0]
                for t in tiles[1:]:
                    row = np.concatenate([row, sep, t], axis=1)
                # upscale x1.5 for legibility, cap width
                scale = 1.5
                row = cv2.resize(
                    row, (int(row.shape[1] * scale), int(row.shape[0] * scale)),
                    interpolation=cv2.INTER_NEAREST,
                )
                out_path = os.path.join(out_dir, f"{cond}_{idx}.png")
                cv2.imwrite(out_path, cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
                written.append(out_path)
                print(f"[write] {out_path}  ({row.shape[1]}x{row.shape[0]})")

        print("\n[done] rendered PNGs:")
        for p in written:
            print("  " + p)


if __name__ == "__main__":
    main()
