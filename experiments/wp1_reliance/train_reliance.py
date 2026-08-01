#!/usr/bin/env python
"""RPD-Drop finetune driver (WP1 / P1) — single-GPU, env-driven.

Finetunes the trained DFormerv2-S with per-batch depth reliance-reduction augmentation
(modality dropout to structureless + Q5-legal relational-preserving aug) applied to the
NORMALIZED depth right before the model forward. Reuses the repo's model builder, optimizer
grouping, poly-LR, AMP, and train loader; adds nothing to the inference graph.

Env vars:
  CKPT_INIT   trained ckpt to finetune from (rel to repos/DFormer)  [required]
  SAVE_DIR    output dir for checkpoints (abs or rel to project)     [required]
  EPOCHS      finetune epochs                                        [default 40]
  TRAIN_BATCH batch size                                             [default 8]
  LR          base LR                                                [default 6e-5]
  P_DROP_MAX  max modality-dropout prob                              [default 0.30]
  P_DROP_WARMUP_EP  epochs to ramp p_drop 0->max                     [default 10]
  P_AUG       relational-aug prob on non-dropped samples            [default 0.5]
  NOISE_SIGMA mild-noise sigma (normalized space)                    [default 0.08]
  SAVE_EVERY  save ckpt every N epochs                               [default 10]
  SEED        seed                                                   [default 20260721]
  ABLATE      "1" -> p_drop=p_aug=0 (plain-finetune control)         [default 0]
  MAX_STEPS   cap steps/epoch (smoke)                                [default 0=off]
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

sys.path.insert(0, _HERE)
from depth_reliance_aug import DepthRelianceAug                 # noqa: E402

CONFIG_MODULE = "local_configs.NYUDepthv2.DFormerv2_S"


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def main():
    ckpt_init = _env("CKPT_INIT")
    save_dir = _env("SAVE_DIR")
    if not ckpt_init or not save_dir:
        raise SystemExit("CKPT_INIT and SAVE_DIR are required")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_PROJECT, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(_env("EPOCHS", "40"))
    train_batch = int(_env("TRAIN_BATCH", "8"))
    base_lr = float(_env("LR", "6e-5"))
    p_drop_max = float(_env("P_DROP_MAX", "0.30"))
    p_drop_warmup = int(_env("P_DROP_WARMUP_EP", "10"))
    p_aug = float(_env("P_AUG", "0.5"))
    noise_sigma = float(_env("NOISE_SIGMA", "0.08"))
    save_every = int(_env("SAVE_EVERY", "10"))
    seed = int(_env("SEED", "20260721"))
    ablate = _env("ABLATE", "0") == "1"
    max_steps = int(_env("MAX_STEPS", "0"))

    if ablate:
        p_drop_max = 0.0
        p_aug = 0.0

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        config.pad = False
        config.batch_size = train_batch
        config.num_workers = int(_env("NUM_WORKERS", "4"))

        # reduction="none": builder.py indexes per-pixel loss by a mask then .mean()
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
              f"unexpected={len(rep.unexpected_keys)}")

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
        warmup_iters = min(niters * 1, total_iters)  # 1-epoch warmup for a finetune
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iters, warmup_iters)

        aug = DepthRelianceAug(
            p_drop=p_drop_max, p_aug=p_aug, noise_sigma=noise_sigma, seed=seed
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        # one-time consistency assert: dropout target == eval entire_missing (zeros)
        _probe = torch.randn(2, 3, 8, 8, device=device)
        from depth_reliance_aug import structureless
        assert torch.equal(structureless(_probe), torch.zeros_like(_probe)), \
            "structureless != zeros; would desync from eval entire_missing"

        print(f"[cfg] epochs={epochs} niters/ep={niters} batch={train_batch} lr={base_lr} "
              f"p_drop_max={p_drop_max} warmup_ep={p_drop_warmup} p_aug={p_aug} "
              f"ablate={ablate} save_dir={save_dir}")
        print(f"[aug] {json.dumps(aug.describe())}")

        global_step = 0
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            # per-epoch p_drop warmup
            if p_drop_warmup > 0:
                cur_pdrop = min(p_drop_max, p_drop_max * epoch / p_drop_warmup)
            else:
                cur_pdrop = p_drop_max
            aug.p_drop = cur_pdrop
            it = iter(train_loader)
            run_loss = 0.0
            n_seen = 0
            for idx in range(niters):
                try:
                    mb = next(it)
                except StopIteration:
                    break
                imgs = mb["data"].to(device, non_blocking=True)
                gts = mb["label"].to(device, non_blocking=True)
                modal = mb["modal_x"].to(device, non_blocking=True)
                if not torch.isfinite(imgs).all() or not torch.isfinite(modal).all():
                    continue
                if not ablate:
                    modal = aug(modal)  # reliance-reduction on normalized depth
                setattr(model, "global_step", global_step)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(imgs, modal, gts)
                lv = float(loss.detach().item())
                if not (lv == lv and abs(lv) < float("inf")):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr = lr_policy.get_lr(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                global_step += 1
                run_loss += lv
                n_seen += 1
            avg = run_loss / max(1, n_seen)
            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] loss={avg:.4f} p_drop={cur_pdrop:.3f} "
                  f"lr={lr:.2e} ({dt:.0f}s)", flush=True)
            if epoch % save_every == 0 or epoch == epochs:
                path = os.path.join(save_dir, f"ckpt_ep{epoch}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch}, path)
                print(f"[save] {path}", flush=True)

        meta = {
            "ckpt_init": ckpt_init, "epochs": epochs, "train_batch": train_batch,
            "lr": base_lr, "p_drop_max": p_drop_max, "p_drop_warmup_ep": p_drop_warmup,
            "p_aug": p_aug, "noise_sigma": noise_sigma, "seed": seed, "ablate": ablate,
            "niters_per_epoch": niters, "aug": aug.describe(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} epochs in {time.time()-t0:.0f}s -> {save_dir}")


if __name__ == "__main__":
    main()
