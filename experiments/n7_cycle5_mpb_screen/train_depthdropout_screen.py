#!/usr/bin/env python
"""MP-B removability SCREEN — depth-dropout finetune (DIAGNOSTIC ONLY, NOT promotable).

Tests the "self-inflicted vs capacity-necessity" question for MP-B: does forcing
DFormerv2-S toward depth-invariance (via per-sample depth dropout to zeros ==
the eval entire_missing condition) HOLD/RAISE clean mIoU while shrinking the
entire_missing gap (self-inflicted / removable), or DROP clean (capacity-necessity)?

This is a SCREENING finetune-proxy. It is EXPLICITLY never promoted (it trains on
the eval failure condition == n2 RPD-Drop reward-hacking pattern). It exists only
to answer the removability question cheaply. Do NOT ccnative_ingest / merge_branch
its output.

Design (owned by the experimenter per the brief):
  - init from the FROZEN converged S ckpt.
  - PURE depth-dropout: for each sample independently, with prob p replace the
    normalized depth branch with zeros (== structureless == eval entire_missing);
    else keep clean depth. No Q5 relational aug (p_aug=0) -- this is the cleanest
    removability signal, not a robustness cocktail.
  - USE_AMP=0 HARD: fp32 forward/backward, NO autocast, NO GradScaler. AMP silently
    overflows grads on this arch (campaign rule) -> a screen under AMP would give a
    misleading answer.
  - saves only the FINAL ckpt (screen).

Env vars:
  CKPT_INIT   frozen ckpt to finetune from (abs or rel to repos/DFormer)  [required]
  SAVE_DIR    output dir for the final ckpt (abs or rel to project)        [required]
  P_DROP      per-sample depth-dropout prob (constant, no warmup)          [required]
  EPOCHS      finetune epochs                                              [default 15]
  TRAIN_BATCH batch size                                                   [default 16]
  LR          base LR                                                      [default 6e-5]
  SEED        seed                                                         [default 20260721]
  MAX_STEPS   cap steps/epoch (smoke)                                      [default 0=off]
  GPU         cuda index to pin (informational; caller sets CUDA_VISIBLE_DEVICES)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))          # ugra-rgbd-robust
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module

from models.builder import EncoderDecoder as segmodel          # noqa: E402
from utils.dataloader.dataloader import get_train_loader       # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset           # noqa: E402
from utils.engine.engine import Engine                         # noqa: E402
from utils.init_func import group_weight                       # noqa: E402
from utils.lr_policy import WarmUpPolyLR                        # noqa: E402

CONFIG_MODULE = "local_configs.NYUDepthv2.DFormerv2_S"


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _depth_dropout(modal: torch.Tensor, p: float, gen: torch.Generator) -> torch.Tensor:
    """Per-sample: with prob p replace depth with zeros (== eval entire_missing);
    else keep clean depth. Returns a NEW tensor (no in-place mutation of input)."""
    if p <= 0.0:
        return modal
    b = modal.shape[0]
    out = modal.clone()
    u = torch.rand(b, generator=gen, device=modal.device)
    drop_mask = u < p  # per-sample
    if drop_mask.any():
        out[drop_mask] = 0.0  # zeros_like == structureless == entire_missing
    return out


def main():
    ckpt_init = _env("CKPT_INIT")
    save_dir = _env("SAVE_DIR")
    p_drop = _env("P_DROP")
    if not ckpt_init or not save_dir or p_drop is None:
        raise SystemExit("CKPT_INIT, SAVE_DIR, P_DROP are required")
    p_drop = float(p_drop)
    if not (0.0 <= p_drop <= 1.0):
        raise SystemExit(f"P_DROP must be in [0,1], got {p_drop}")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_PROJECT, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(_env("EPOCHS", "15"))
    train_batch = int(_env("TRAIN_BATCH", "16"))
    base_lr = float(_env("LR", "6e-5"))
    seed = int(_env("SEED", "20260721"))
    max_steps = int(_env("MAX_STEPS", "0"))

    # --- reproducibility (campaign rule) ---
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True  # perf; screen only
    torch.set_float32_matmul_precision("high")

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        config.pad = False
        config.batch_size = train_batch
        config.num_workers = int(_env("NUM_WORKERS", "4"))

        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
        model = segmodel(
            cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False
        )
        weight = torch.load(ckpt_init, map_location="cpu")
        if "model" in weight:
            weight = weight["model"]
        elif "state_dict" in weight:
            weight = weight["state_dict"]
        rep = model.load_state_dict(weight, strict=False)
        print(f"[load] init from {ckpt_init} missing={len(rep.missing_keys)} "
              f"unexpected={len(rep.unexpected_keys)}", flush=True)
        if len(rep.missing_keys) > 0:
            print(f"[load] first missing: {list(rep.missing_keys)[:5]}", flush=True)

        device = torch.device("cuda")
        model.to(device)

        params = group_weight([], model, nn.BatchNorm2d, base_lr)
        optimizer = torch.optim.AdamW(
            params, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay
        )

        train_loader, _ = get_train_loader(engine, RGBXDataset, config)
        niters = len(train_loader)
        if max_steps > 0:
            niters = min(niters, max_steps)
        total_iters = epochs * niters
        warmup_iters = min(niters * 1, total_iters)  # 1-ep warmup for a finetune
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iters, warmup_iters)

        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        print(f"[cfg] AMP=OFF(fp32) epochs={epochs} niters/ep={niters} batch={train_batch} "
              f"lr={base_lr} p_drop={p_drop} seed={seed} save_dir={save_dir}", flush=True)

        # one-time consistency assert: dropout target == eval entire_missing (zeros)
        _probe = torch.randn(4, 3, 8, 8, device=device)
        _g2 = torch.Generator(device=device); _g2.manual_seed(0)
        _dropped = _depth_dropout(_probe, 1.0, _g2)
        assert torch.equal(_dropped, torch.zeros_like(_probe)), \
            "depth_dropout(p=1) != zeros; would desync from eval entire_missing"

        global_step = 0
        t0 = time.time()
        skipped = 0
        for epoch in range(1, epochs + 1):
            model.train()
            it = iter(train_loader)
            run_loss = 0.0
            n_seen = 0
            lr = base_lr
            for idx in range(niters):
                try:
                    mb = next(it)
                except StopIteration:
                    break
                imgs = mb["data"].to(device, non_blocking=True)
                gts = mb["label"].to(device, non_blocking=True)
                modal = mb["modal_x"].to(device, non_blocking=True)
                if not torch.isfinite(imgs).all() or not torch.isfinite(modal).all():
                    skipped += 1
                    continue
                modal = _depth_dropout(modal, p_drop, gen)  # PURE depth dropout
                setattr(model, "global_step", global_step)
                # USE_AMP=0: NO autocast, NO GradScaler -- pure fp32
                loss = model(imgs, modal, gts)
                lv = float(loss.detach().item())
                if not (lv == lv and abs(lv) < float("inf")):
                    optimizer.zero_grad(set_to_none=True)
                    skipped += 1
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr = lr_policy.get_lr(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                global_step += 1
                run_loss += lv
                n_seen += 1
            avg = run_loss / max(1, n_seen)
            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] loss={avg:.4f} lr={lr:.2e} "
                  f"n_seen={n_seen} skipped={skipped} ({dt:.0f}s)", flush=True)

        final_path = os.path.join(save_dir, f"ckpt_final_p{p_drop}.pth")
        torch.save({"model": model.state_dict(), "epochs": epochs, "p_drop": p_drop}, final_path)
        print(f"[save] {final_path}", flush=True)

        meta = {
            "screen": "mpb_removability_depth_dropout",
            "promotable": False,
            "ckpt_init": ckpt_init, "epochs": epochs, "train_batch": train_batch,
            "lr": base_lr, "p_drop": p_drop, "seed": seed,
            "amp": "OFF(fp32)", "niters_per_epoch": niters,
            "global_steps": global_step, "skipped_batches": skipped,
            "wall_seconds": round(time.time() - t0, 1),
            "final_ckpt": final_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(os.path.join(save_dir, f"train_meta_p{p_drop}.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} ep in {time.time()-t0:.0f}s -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
