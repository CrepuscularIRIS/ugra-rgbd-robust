#!/usr/bin/env python
"""ADQG Training — DFormerv2-S with Adaptive Depth Quality Gate.

Standard 500-ep recipe from DFormerv2_Small_pretrained.pth, with:
1. Depth quality gate monkey-patched into every GeoPriorGen
2. Relational depth augmentation (invert, quantize-8, mild noise) during training

No teacher model, no KD — same compute as vanilla baseline (~4.5 GPU-h).

Env: SAVE_DIR (req); EPOCHS(500), LR(6e-5), P_AUG(0.3),
TRAIN_BATCH(16), SAVE_EVERY(25), SEED(20260721), NUM_WORKERS(4),
MAX_STEPS(0), USE_AMP(1).
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
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.dataloader import get_train_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR

from depth_quality_gate import patch_backbone
from depth_aug import augment_depth

S_CONFIG = "local_configs.NYUDepthv2.DFormerv2_S"


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def main():
    save_dir = _env("SAVE_DIR")
    if not save_dir:
        raise SystemExit("SAVE_DIR required")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_PROJECT, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(_env("EPOCHS", "500"))
    lr = float(_env("LR", "6e-5"))
    p_aug = float(_env("P_AUG", "0.3"))
    batch = int(_env("TRAIN_BATCH", "16"))
    save_every = int(_env("SAVE_EVERY", "25"))
    seed = int(_env("SEED", "20260721"))
    max_steps = int(_env("MAX_STEPS", "0"))
    use_amp = _env("USE_AMP", "1") == "1"

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    with Engine(custom_parser=None) as engine:
        cfg = getattr(import_module(S_CONFIG), "C")
        cfg.pad = False
        cfg.batch_size = batch
        cfg.num_workers = int(_env("NUM_WORKERS", "4"))
        bg = cfg.background

        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=bg)

        model = segmodel(cfg=cfg, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
        device = torch.device("cuda")
        model.to(device)

        n_patched = patch_backbone(model.backbone)
        print(f"[adqg] Patched {n_patched} GeoPriorGen modules with DepthQualityGate")

        gate_param_ids = set()
        gate_params = []
        for m in model.backbone.modules():
            if hasattr(m, "_dq_gate"):
                for p in m._dq_gate.parameters():
                    gate_params.append(p)
                    gate_param_ids.add(id(p))
        n_gate_params = sum(p.numel() for p in gate_params)
        print(f"[adqg] Gate parameters: {n_gate_params} ({n_gate_params/1e3:.1f}K)")

        base_params = group_weight([], model, nn.BatchNorm2d, lr)
        for pg in base_params:
            pg["params"] = [p for p in pg["params"] if id(p) not in gate_param_ids]
        base_params = [pg for pg in base_params if len(pg["params"]) > 0]
        base_params.append({"params": gate_params, "lr": lr * 10, "weight_decay": 0.0})
        optimizer = torch.optim.AdamW(base_params, lr=lr, betas=(0.9, 0.999), weight_decay=cfg.weight_decay)

        from utils.dataloader.dataloader import TrainPre
        import torch.utils.data as data
        train_pre = TrainPre(cfg.norm_mean, cfg.norm_std, cfg.x_is_single_channel, cfg)
        train_setting = {
            "rgb_root": cfg.rgb_root_folder, "rgb_format": cfg.rgb_format,
            "gt_root": cfg.gt_root_folder, "gt_format": cfg.gt_format,
            "transform_gt": cfg.gt_transform, "x_root": cfg.x_root_folder,
            "x_format": cfg.x_format, "x_single_channel": cfg.x_is_single_channel,
            "class_names": cfg.class_names, "train_source": cfg.train_source,
            "eval_source": cfg.eval_source, "dataset_name": cfg.dataset_name,
            "backbone": cfg.backbone,
        }
        n_train = 795
        niters_full = n_train // batch + 1
        train_ds = RGBXDataset(train_setting, "train", train_pre, batch * niters_full)
        train_loader = data.DataLoader(
            train_ds, batch_size=batch, num_workers=cfg.num_workers,
            drop_last=True, shuffle=True, pin_memory=True,
        )
        niters = len(train_loader)
        print(f"[loader] batch={batch} niters={niters} ds_len={len(train_ds)}")
        if max_steps > 0:
            niters = min(niters, max_steps)
        total_iters = epochs * niters
        warmup_iters = niters * min(10, epochs)
        lr_policy = WarmUpPolyLR(lr, cfg.lr_power, total_iters, warmup_iters)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        print(f"[cfg] epochs={epochs} niters/ep={niters} batch={batch} lr={lr} "
              f"p_aug={p_aug} amp={use_amp}")

        def seg_loss(logits, gts):
            return criterion(logits, gts.long())[gts.long() != bg].mean()

        global_step = 0
        t0 = time.time()
        best_loss = float("inf")

        for epoch in range(1, epochs + 1):
            model.train()
            it = iter(train_loader)
            run_ce, n_seen = 0.0, 0

            for idx in range(niters):
                try:
                    mb = next(it)
                except StopIteration:
                    break
                imgs = mb["data"].to(device, non_blocking=True)
                gts = mb["label"].to(device, non_blocking=True)
                depth = mb["modal_x"].to(device, non_blocking=True)
                if not torch.isfinite(imgs).all() or not torch.isfinite(depth).all():
                    continue

                depth_input = augment_depth(depth, p_aug=p_aug)

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits, _, _ = model.encode_decode(imgs, depth_input)
                    loss = seg_loss(logits, gts)

                lv = float(loss.detach().item())
                if not (lv == lv and abs(lv) < float("inf")):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                if torch.isfinite(gnorm):
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                cur_lr = lr_policy.get_lr(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr
                global_step += 1
                run_ce += lv
                n_seen += 1

            avg_ce = run_ce / max(1, n_seen)
            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] ce={avg_ce:.4f} lr={cur_lr:.2e} ({dt:.0f}s)", flush=True)

            if epoch % save_every == 0 or epoch == epochs:
                path = os.path.join(save_dir, f"ckpt_ep{epoch}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch}, path)
                print(f"[save] {path}", flush=True)

        meta = {"method": "ADQG", "epochs": epochs, "lr": lr,
                "p_aug": p_aug, "seed": seed, "niters_per_epoch": niters,
                "n_gate_params": n_gate_params, "n_patched": n_patched,
                "use_amp": use_amp,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} epochs in {time.time()-t0:.0f}s -> {save_dir}")


if __name__ == "__main__":
    main()
