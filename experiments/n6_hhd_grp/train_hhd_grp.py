#!/usr/bin/env python
"""HHD+GRP Training — DFormerv2-S with Heterogeneous Head Decomposition +
Geometric Relation Prediction auxiliary objective.

Standard 500-ep recipe from DFormerv2_Small_pretrained.pth, with:
1. Per-head depth coupling (geo-heads amplified, content-heads zeroed)
2. Aux depth-geometry prediction on end-of-stage-2 content features
3. Anti-shortcut: GeoPriorGen depth augmented (invert/quantize-8), aux target = original

No teacher model — same compute as vanilla baseline (~4.5 GPU-h).

Env: SAVE_DIR (req); EPOCHS(500), LR(6e-5), LAMBDA_BIN(0.2), LAMBDA_RANK(0.1),
GEO_RATIO(0.5), AUG_P(0.5), AUX_WARMUP(10), TRAIN_BATCH(16), SAVE_EVERY(25),
SEED(20260721), NUM_WORKERS(4), MAX_STEPS(0), USE_AMP(1).
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
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module
from models.builder import EncoderDecoder as segmodel
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR

from hhd_geo_prior import patch_backbone
from grp_aux_head import GRPHead, grp_loss

S_CONFIG = "local_configs.NYUDepthv2.DFormerv2_S"
S_STAGE2_DIM = 256


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _augment_depth(depth: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Anti-shortcut depth augmentation for GeoPriorGen input.

    Applies invert or quantize-8 (Q5-admissible) with probability p per sample.
    The aux target always uses the ORIGINAL depth.
    """
    if p <= 0:
        return depth
    B = depth.shape[0]
    out = depth.clone()
    for b in range(B):
        if torch.rand(1).item() >= p:
            continue
        aug_type = torch.randint(0, 2, (1,)).item()
        d = out[b]
        vmin, vmax = d.min(), d.max()
        if vmax - vmin < 1e-6:
            continue
        if aug_type == 0:
            out[b] = vmax + vmin - d
        else:
            normalized = (d - vmin) / (vmax - vmin)
            out[b] = torch.round(normalized * 7) / 7 * (vmax - vmin) + vmin
    return out


def main():
    save_dir = _env("SAVE_DIR")
    if not save_dir:
        raise SystemExit("SAVE_DIR required")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_PROJECT, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(_env("EPOCHS", "500"))
    full_epochs = int(_env("FULL_EPOCHS", "0")) or epochs
    lr = float(_env("LR", "6e-5"))
    lam_bin = float(_env("LAMBDA_BIN", "0.2"))
    lam_rank = float(_env("LAMBDA_RANK", "0.1"))
    geo_ratio = float(_env("GEO_RATIO", "0.5"))
    aug_p = float(_env("AUG_P", "0.5"))
    aux_warmup = int(_env("AUX_WARMUP", "10"))
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

        hhd_info = patch_backbone(model.backbone, geo_ratio=geo_ratio)
        print(f"[hhd] Patched {hhd_info['n_patched']} GeoPriorGen modules, "
              f"{hhd_info['n_new_params']} per-head depth params, geo_ratio={geo_ratio}")

        grp_head = GRPHead(in_dim=S_STAGE2_DIM, n_bins=8).to(device)
        n_grp_params = sum(p.numel() for p in grp_head.parameters())
        print(f"[grp] Aux head: {n_grp_params} params ({n_grp_params/1e3:.1f}K)")

        stage2_feat = {}
        def _stage2_hook(_m, _inp, out):
            stage2_feat["x"] = out[0] if isinstance(out, tuple) else out
        model.backbone.layers[2].register_forward_hook(_stage2_hook)

        hhd_param_ids = set()
        hhd_params = []
        for m in model.backbone.modules():
            if hasattr(m, "_weight_depth_perhead"):
                p = m._weight_depth_perhead
                hhd_params.append(p)
                hhd_param_ids.add(id(p))

        base_params = group_weight([], model, nn.BatchNorm2d, lr)
        for pg in base_params:
            pg["params"] = [p for p in pg["params"] if id(p) not in hhd_param_ids]
        base_params = [pg for pg in base_params if len(pg["params"]) > 0]
        base_params.append({"params": hhd_params, "lr": lr * 5, "weight_decay": 0.0})
        base_params.append({"params": grp_head.parameters(), "lr": lr * 10, "weight_decay": 0.01})
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
        total_iters = full_epochs * niters
        warmup_iters = niters * min(10, full_epochs)
        lr_policy = WarmUpPolyLR(lr, cfg.lr_power, total_iters, warmup_iters)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        print(f"[cfg] epochs={epochs} niters/ep={niters} batch={batch} lr={lr} "
              f"lam_bin={lam_bin} lam_rank={lam_rank} aug_p={aug_p} "
              f"aux_warmup={aux_warmup} amp={use_amp}")

        def seg_loss(logits, gts):
            return criterion(logits, gts.long())[gts.long() != bg].mean()

        global_step = 0
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            grp_head.train()
            it = iter(train_loader)
            run_ce, run_grp, n_seen = 0.0, 0.0, 0
            use_aux = epoch > aux_warmup

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

                depth_for_geo = _augment_depth(depth, p=aug_p) if use_aux else depth

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    logits, _, _ = model.encode_decode(imgs, depth_for_geo)
                    loss_ce = seg_loss(logits, gts)

                    loss_aux = torch.zeros((), device=device)
                    if use_aux and "x" in stage2_feat:
                        aux_logits = grp_head(stage2_feat["x"])
                        loss_aux = grp_loss(aux_logits, depth,
                                            lambda_bin=lam_bin, lambda_rank=lam_rank)

                    loss = loss_ce + loss_aux

                stage2_feat.clear()
                lv = float(loss.detach().item())
                if not (lv == lv and abs(lv) < float("inf")):
                    optimizer.zero_grad(set_to_none=True)
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                all_params = list(model.parameters()) + list(grp_head.parameters())
                gnorm = torch.nn.utils.clip_grad_norm_(all_params, 100.0)
                if torch.isfinite(gnorm):
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                cur_lr = lr_policy.get_lr(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr
                global_step += 1
                run_ce += float(loss_ce.detach().item())
                run_grp += float(loss_aux.detach().item())
                n_seen += 1

            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] ce={run_ce/max(1,n_seen):.4f} "
                  f"grp={run_grp/max(1,n_seen):.4f} lr={cur_lr:.2e} ({dt:.0f}s)", flush=True)
            if epoch % save_every == 0 or epoch == epochs:
                path = os.path.join(save_dir, f"ckpt_ep{epoch}.pth")
                torch.save({"model": model.state_dict(), "epoch": epoch}, path)
                print(f"[save] {path}", flush=True)

        meta = {"method": "HHD+GRP", "epochs": epochs, "lr": lr,
                "lam_bin": lam_bin, "lam_rank": lam_rank, "geo_ratio": geo_ratio,
                "aug_p": aug_p, "aux_warmup": aux_warmup,
                "seed": seed, "niters_per_epoch": niters,
                "n_hhd_params": hhd_info["n_new_params"],
                "n_grp_params": n_grp_params,
                "use_amp": use_amp,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} epochs in {time.time()-t0:.0f}s -> {save_dir}")


if __name__ == "__main__":
    main()
