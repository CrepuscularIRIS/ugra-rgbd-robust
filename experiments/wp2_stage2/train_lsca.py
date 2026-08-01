#!/usr/bin/env python
"""LSCA finetune driver (WP2 / P2) — single-GPU, env-driven.

Layer-wise Stage-2 Content Anchoring: finetune trained DFormerv2-S with the RPD depth
reliance-reduction augmentation on the STUDENT depth, PLUS a per-block stage-2 cosine
consistency loss to a FROZEN clean-depth teacher (a t=0 snapshot of the trained weights).
Setting CONS_WEIGHT=0 reduces this exactly to RPD-Drop -> clean on/off ablation for the
stage-2 term.

Anchors: sequential stage-2 inject recovery 0.995 (trained stage-2 state is a near-optimal
target); Q4 stage-2 damage monotone in #blocks; T3 gradual drift 0.990->0.940 across the 18
stage-2 blocks; Q6 loss is spatial/directional -> cosine (not L2). Tap: backbone.layers[2].
blocks[0..17] (DFormerv2-S depths=[3,4,18,4]).

Env (superset of train_reliance.py):
  CKPT_INIT, SAVE_DIR (required); EPOCHS(40), TRAIN_BATCH(8), LR(6e-5),
  P_DROP_MAX(0.30), P_DROP_WARMUP_EP(10), P_AUG(0.5), NOISE_SIGMA(0.08),
  CONS_WEIGHT(1.0)  -> stage-2 consistency weight (0 == RPD-Drop control),
  SAVE_EVERY(10), SEED(20260721), MAX_STEPS(0), NUM_WORKERS(4).
"""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
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

sys.path.insert(0, os.path.join(_PROJECT, "experiments", "wp1_reliance"))
from depth_reliance_aug import DepthRelianceAug, structureless  # noqa: E402

CONFIG_MODULE = "local_configs.NYUDepthv2.DFormerv2_S"
STAGE2_LAYER = 2


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _make_hook(store, key):
    def hook(_module, _inp, out):
        # BasicLayer block returns the updated token tensor (B,H,W,C)
        store[key] = out
    return hook


def _register_stage2_hooks(model, store):
    blocks = model.backbone.layers[STAGE2_LAYER].blocks
    handles = []
    for j, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(_make_hook(store, j)))
    return handles, len(blocks)


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
    cons_weight = float(_env("CONS_WEIGHT", "1.0"))
    save_every = int(_env("SAVE_EVERY", "10"))
    seed = int(_env("SEED", "20260721"))
    max_steps = int(_env("MAX_STEPS", "0"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        config.pad = False
        config.batch_size = train_batch
        config.num_workers = int(_env("NUM_WORKERS", "4"))

        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
        model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
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

        # frozen clean-depth teacher = t=0 snapshot
        teacher = copy.deepcopy(model).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        s_feats, t_feats = {}, {}
        _register_stage2_hooks(model, s_feats)
        _, n_blocks = _register_stage2_hooks(teacher, t_feats)
        print(f"[lsca] tapping {n_blocks} stage-2 blocks; cons_weight={cons_weight}")

        params = group_weight([], model, nn.BatchNorm2d, base_lr)
        optimizer = torch.optim.AdamW(
            params, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay
        )
        train_loader, _ = get_train_loader(engine, RGBXDataset, config)
        niters = len(train_loader)
        if max_steps > 0:
            niters = min(niters, max_steps)
        total_iters = epochs * niters
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iters, min(niters, total_iters))
        aug = DepthRelianceAug(p_drop=p_drop_max, p_aug=p_aug, noise_sigma=noise_sigma)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        print(f"[cfg] epochs={epochs} niters/ep={niters} batch={train_batch} lr={base_lr} "
              f"p_drop_max={p_drop_max} p_aug={p_aug} cons_weight={cons_weight} save_dir={save_dir}")

        def consistency_loss():
            # per-block directional (cosine) match, teacher detached; auto-~0 on clean rows
            tot = 0.0
            for j in range(n_blocks):
                sf = s_feats[j].float()
                tf = t_feats[j].float().detach()
                sf = sf.reshape(-1, sf.shape[-1])
                tf = tf.reshape(-1, tf.shape[-1])
                tot = tot + (1.0 - F.cosine_similarity(sf, tf, dim=-1).mean())
            return tot / n_blocks

        global_step = 0
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            cur_pdrop = min(p_drop_max, p_drop_max * epoch / p_drop_warmup) if p_drop_warmup > 0 else p_drop_max
            aug.p_drop = cur_pdrop
            it = iter(train_loader)
            run_seg, run_cons, n_seen = 0.0, 0.0, 0
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
                modal_student = aug(modal)  # reliance-reduction on normalized depth
                setattr(model, "global_step", global_step)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    if cons_weight > 0:
                        with torch.no_grad():
                            _ = teacher.backbone(imgs, modal)  # clean-depth teacher -> t_feats
                    loss_seg = model(imgs, modal_student, gts)   # student -> s_feats + seg loss
                    if cons_weight > 0:
                        loss_cons = consistency_loss()
                        loss = loss_seg + cons_weight * loss_cons
                    else:
                        loss_cons = torch.zeros((), device=device)
                        loss = loss_seg
                lv = float(loss.detach().item())
                if not (lv == lv and abs(lv) < float("inf")):
                    optimizer.zero_grad(set_to_none=True)
                    s_feats.clear(); t_feats.clear()
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # defensive: never step on a non-finite grad norm (prevents weight NaN)
                if torch.isfinite(gnorm):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                s_feats.clear(); t_feats.clear()
                lr = lr_policy.get_lr(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                global_step += 1
                run_seg += float(loss_seg.detach().item())
                run_cons += float(loss_cons.detach().item())
                n_seen += 1
            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] seg={run_seg/max(1,n_seen):.4f} "
                  f"cons={run_cons/max(1,n_seen):.4f} p_drop={cur_pdrop:.3f} lr={lr:.2e} ({dt:.0f}s)",
                  flush=True)
            if epoch % save_every == 0 or epoch == epochs:
                path = os.path.join(save_dir, f"ckpt_ep{epoch}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch}, path)
                print(f"[save] {path}", flush=True)

        meta = {
            "method": "LSCA", "ckpt_init": ckpt_init, "epochs": epochs,
            "train_batch": train_batch, "lr": base_lr, "p_drop_max": p_drop_max,
            "p_aug": p_aug, "cons_weight": cons_weight, "n_stage2_blocks": n_blocks,
            "seed": seed, "niters_per_epoch": niters, "aug": aug.describe(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} epochs in {time.time()-t0:.0f}s -> {save_dir}")


if __name__ == "__main__":
    main()
