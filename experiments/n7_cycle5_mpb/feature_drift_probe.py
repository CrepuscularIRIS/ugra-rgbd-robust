#!/usr/bin/env python
"""MP-B feature-drift probe (EVAL-ONLY, frozen checkpoints, NO training).

Kill test 1 for MP-B ("S content-feature spatial-disambiguation deficit under
depth absence"): on the FROZEN official NYU checkpoints for DFormerv2 S / B / L,
run PAIRED forward passes on n val images -- (1) clean RGB-D, (2) zeros depth
(entire_missing == zeros_like(normalized_depth), byte-identical to the frozen
harness corruption) -- and measure the per-image cosine similarity of the
stage-2 EXIT feature (backbone(rgb, modal_x)[2]; C=256 for S, 448 for L, 320
for B). Report mean cosine per model.

MP-B predicts: S LOW cosine (high drift) while B/L HIGH cosine (near
depth-invariance). If S cosine is NOT lower than B/L, the "S-specific" claim is
weakened.

This reuses the frozen eval harness model-load + normalization path exactly:
the val loader (get_val_loader) yields the SAME normalized (data, modal_x)
tensors the frozen eval feeds the model; we call model.backbone(rgb, modal_x)
directly to read the stage-2 exit feature. NO parameter is changed, NO training
occurs, no gradient is taken (torch.no_grad).

All params via env vars:
  MODEL         one of {S, B, L}                                 [required]
  CKPT          checkpoint path (absolute or rel to repos/DFormer) [required]
  N_IMAGES      number of val images to probe                    [default: 200]
  SEED          torch/np seed                                    [default: 20260721]
  OUT_JSON      output JSON path                                 [required]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))  # ugra-rgbd-robust
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module

from models.builder import EncoderDecoder as segmodel  # noqa: E402
from utils.dataloader.dataloader import get_val_loader  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset    # noqa: E402
from utils.engine.engine import Engine                  # noqa: E402

_CONFIG = {
    "S": "local_configs.NYUDepthv2.DFormerv2_S",
    "B": "local_configs.NYUDepthv2.DFormerv2_B",
    "L": "local_configs.NYUDepthv2.DFormerv2_L",
}
_STAGE2_CH = {"S": 256, "B": 320, "L": 448}


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def main():
    model_key = _env("MODEL")
    ckpt = _env("CKPT")
    out_json = _env("OUT_JSON")
    if not model_key or model_key not in _CONFIG:
        raise SystemExit("MODEL env var required, one of S|B|L")
    if not ckpt:
        raise SystemExit("CKPT env var required")
    if not out_json:
        raise SystemExit("OUT_JSON env var required")
    n_images = int(_env("N_IMAGES", "200"))
    seed = int(_env("SEED", "20260721"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(_CONFIG[model_key]), "C")
        config.pad = False
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
        model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)

        weight = torch.load(ckpt, map_location=torch.device("cpu"))
        if "model" in weight:
            weight = weight["model"]
        elif "state_dict" in weight:
            weight = weight["state_dict"]
        rep = model.load_state_dict(weight, strict=False)
        missing = list(rep.missing_keys)
        unexpected = list(rep.unexpected_keys)
        # Count only backbone-relevant load health (decode_head unused here but loaded).
        bb_missing = [k for k in missing if k.startswith("backbone.")]
        bb_unexpected = [k for k in unexpected if k.startswith("backbone.")]
        print(f"[load] {model_key}: missing={len(missing)} unexpected={len(unexpected)} "
              f"backbone_missing={len(bb_missing)} backbone_unexpected={len(bb_unexpected)}")
        if bb_missing:
            print(f"[load] backbone first missing: {bb_missing[:5]}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        val_loader, _ = get_val_loader(engine, RGBXDataset, config, val_batch_size=1)
        n_total = len(val_loader.dataset)
        n_probe = min(n_images, n_total)

        cos_vals = []
        exp_ch = _STAGE2_CH[model_key]
        t0 = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for idx, minibatch in enumerate(val_loader):
                if idx >= n_probe:
                    break
                rgb = minibatch["data"]
                modal_x = minibatch["modal_x"]
                if rgb.dim() == 3:
                    rgb = rgb.unsqueeze(0)
                if modal_x.dim() == 3:
                    modal_x = modal_x.unsqueeze(0)
                rgb = rgb.to(device)
                modal_x = modal_x.to(device)
                modal_zero = torch.zeros_like(modal_x)  # entire_missing

                feats_clean = model.backbone(rgb, modal_x)
                feats_missing = model.backbone(rgb, modal_zero)
                f_c = feats_clean[2]  # (1, C, H, W)  stage-2 exit
                f_m = feats_missing[2]
                if f_c.shape[1] != exp_ch:
                    raise SystemExit(
                        f"stage-2 channel mismatch for {model_key}: got {f_c.shape[1]}, expected {exp_ch}"
                    )
                # per-image cosine over the full flattened stage-2 feature (C*H*W)
                a = f_c.reshape(-1).float()
                b = f_m.reshape(-1).float()
                cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                cos_vals.append(cos)
        dt = round(time.time() - t0, 1)

        cos_arr = np.array(cos_vals, dtype=np.float64)
        payload = {
            "model": model_key,
            "checkpoint": ckpt,
            "config": _CONFIG[model_key],
            "stage2_channels": exp_ch,
            "n_images_probed": int(len(cos_vals)),
            "n_val_total": int(n_total),
            "seed": seed,
            "corruption": "entire_missing == zeros_like(normalized_depth)",
            "feature": "backbone(rgb, modal_x)[2]  (stage-2 exit, extra_norm applied)",
            "metric": "per-image cosine(clean_stage2, missing_stage2) over flattened C*H*W",
            "mean_cosine": float(cos_arr.mean()),
            "std_cosine": float(cos_arr.std()),
            "median_cosine": float(np.median(cos_arr)),
            "min_cosine": float(cos_arr.min()),
            "max_cosine": float(cos_arr.max()),
            "seconds": dt,
            "torch_version": torch.__version__,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[done] {model_key}: mean_cos={payload['mean_cosine']:.4f} "
              f"std={payload['std_cosine']:.4f} n={payload['n_images_probed']} ({dt}s)")
        print(f"[done] wrote {out_json}")


if __name__ == "__main__":
    main()
