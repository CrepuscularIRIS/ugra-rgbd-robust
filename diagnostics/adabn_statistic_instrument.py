#!/usr/bin/env python
"""INSTRUMENT: how much of the missing-depth gap is pure NORMALIZATION-STATISTIC miscalibration?

WHY. A checkpoint diff localized C2's change to normalization shifts and running
statistics (largest relative moves: extra_norms.*.bias ~0.21,
downsample.norm.running_mean ~0.17, decoder BN shifts ~0.13) rather than to the depth
pathway (which cannot move -- it is frozen by the group_weight bug) or to flattening the
authority-response curve (that sub-hypothesis was disconfirmed).

That localization mixes two things a single measurement can separate:
  (a) RUNNING STATISTICS  -- adaptable with forward passes alone, no learning;
  (b) LEARNED SHIFTS      -- the norm biases, which require gradients.
This instrument re-estimates only (a), under the depth-absent condition, with NO
gradients and NO optimizer, then evaluates. The recovered mIoU is the part of the gap
attributable to statistic miscalibration alone.

PRE-REGISTERED PREDICTION (.grill/prediction-ledger.md): +1.0 .. +3.0 if statistics are
load-bearing; < +0.2 means the LEARNED shifts are what matter and a statistics-only fix is
insufficient -- a different and more interesting claim, since it would mean gradient-learned
biases are required.

GOVERNANCE. Re-estimating statistics ON the depth-absent condition IS adapting to the
evaluation failure condition, so this is n2-class and **DISQUALIFIED AS A METHOD**. It is
admissible only as an INSTRUMENT that bounds how much of the gap is statistic
miscalibration, on the same footing as the spatial-oracle and the MP-B screen. It is
NON-PROMOTABLE BY CONSTRUCTION and must never be reported as a robustness result.

Writes an adapted checkpoint so the FROZEN eval harness does the scoring -- the eval path
is not reimplemented here.

USAGE
  CUDA_VISIBLE_DEVICES=1 N_BATCHES=60 python diagnostics/adabn_statistic_instrument.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module  # noqa: E402

from models.builder import EncoderDecoder as segmodel  # noqa: E402
from utils.dataloader.dataloader import get_train_loader  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.engine.engine import Engine  # noqa: E402

CONFIG_MODULE = os.environ.get("CONFIG_MODULE", "local_configs.NYUDepthv2.DFormerv2_S")


def main() -> None:
    ckpt = os.environ.get(
        "CKPT", os.path.join(_ROOT, "checkpoints/trained/DFormerv2_Small_NYU.pth")
    )
    n_batches = int(os.environ.get("N_BATCHES", "60"))
    out_ckpt = os.environ.get(
        "OUT_CKPT",
        os.path.join(_PROJECT, "results", "score_push", "runs", "l2_nof", "adabn_absent.pth"),
    )
    momentum = float(os.environ.get("BN_MOMENTUM", "0.05"))

    torch.manual_seed(0)
    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        config.pad = "SUNRGBD" in str(getattr(config, "dataset_name", "")).upper()
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
        os.environ["DEPTH_LAMBDA_TRAIN"] = "1.0"
        model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
        w = torch.load(ckpt, map_location="cpu")
        w = w.get("model", w.get("state_dict", w)) if isinstance(w, dict) else w
        rep = model.load_state_dict(w, strict=False)
        print(f"[load] {ckpt} missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Everything in eval mode EXCEPT the modules that own running statistics.
        model.eval()
        bn_mods = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        for m in bn_mods:
            m.train()                 # re-enable running-stat updates
            m.momentum = momentum     # gentle EMA over many batches
        print(f"[adabn] {len(bn_mods)} BatchNorm modules put in train mode, momentum={momentum}")
        print("[adabn] NOTE LayerNorm carries no running statistics, so this adapts BN only —"
              " which is exactly the 'statistics' half of the split.")

        # Snapshot the statistics so we can report how far they moved.
        before = [m.running_mean.detach().clone() for m in bn_mods if m.running_mean is not None]

        train_loader, _ = get_train_loader(engine, RGBXDataset, config)
        print(f"[data] adapting over {n_batches} train batches with depth ZEROED (the absent condition)")

        seen = 0
        with torch.no_grad():
            for mb in train_loader:
                rgb = mb["data"].to(device).float()
                dep = mb["modal_x"].to(device).float()
                dep = torch.zeros_like(dep)      # the depth-absent condition
                model(rgb, dep)                  # forward only: updates running stats
                seen += 1
                if seen >= n_batches:
                    break
        print(f"[adabn] adapted over {seen} batches")

        after = [m.running_mean.detach().clone() for m in bn_mods if m.running_mean is not None]
        rel = [
            (b - a).norm().item() / max(a.norm().item(), 1e-12)
            for a, b in zip(before, after)
        ]
        rel_sorted = sorted(rel, reverse=True)
        print(f"[adabn] running_mean relative shift: max={rel_sorted[0]:.4f} "
              f"median={rel_sorted[len(rel_sorted)//2]:.4f} min={rel_sorted[-1]:.4f}")
        if rel_sorted[0] < 1e-4:
            print("[adabn] WARNING: statistics barely moved — the adaptation did not take effect. "
                  "Any null result below would be VACUOUS, not evidence.")

        model.eval()
        torch.save({"model": model.state_dict()}, out_ckpt)
        print(f"[out] adapted checkpoint -> {out_ckpt}")
        print("[next] score it with the FROZEN harness:")
        print(f"  CKPT={out_ckpt} EVAL_CONDITIONS=clean,entire_missing "
              f"OUT_JSON=.../adabn_absent.json bash scripts/eval_robust_suite.sh")


if __name__ == "__main__":
    main()
