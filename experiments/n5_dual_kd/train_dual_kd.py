#!/usr/bin/env python
"""Dual-Branch L→S Cross-Capacity Distillation (n5) — single-GPU, env-driven.

Branch 1 (all images): CE + KL(S_logits/T, L_logits/T) * T² * λ_clean
  → imports L's superior representation into S via clean-depth response distillation
Branch 2 (missing_frac images): CE + L2 feature distillation on 6 stage-2 block checkpoints
  via 256→448 projection heads → imports L's depth-invariant features under missing depth

L teacher = frozen DFormerv2_Large_NYU.pth (58.4 mIoU, depth-invariant cos=0.9995).
S student = trained from DFormerv2_Small_pretrained.pth (full 500-ep recipe).
Projection heads discarded at inference.

Env: CKPT_S_PRETRAINED, L_CKPT (req); SAVE_DIR (req); EPOCHS(500), LR(6e-5),
LAMBDA_CLEAN(1.0), LAMBDA_MISS(0.5), KD_T(4.0), MISS_FRAC(0.5),
TRAIN_BATCH(16), SAVE_EVERY(25), SEED(20260721), NUM_WORKERS(4), MAX_STEPS(0).
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
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT):
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

S_CONFIG = "local_configs.NYUDepthv2.DFormerv2_S"
L_CONFIG = "local_configs.NYUDepthv2.DFormerv2_L"
S_STAGE2_DIM = 256
L_STAGE2_DIM = 448
S_STAGE2_BLOCKS = 18
L_STAGE2_BLOCKS = 25
S_HOOK_INDICES = [3, 6, 9, 12, 15, 17]
L_HOOK_INDICES = [4, 8, 13, 17, 21, 24]


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


class Stage2Projector(nn.Module):
    def __init__(self, s_dim=S_STAGE2_DIM, l_dim=L_STAGE2_DIM):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(s_dim, l_dim, bias=False), nn.LayerNorm(l_dim))
    def forward(self, x):
        return self.proj(x)


def _make_hook(store, key):
    def hook(_m, _inp, out):
        store[key] = out
    return hook


def main():
    s_pretrained = _env("CKPT_S_PRETRAINED")
    l_ckpt = _env("L_CKPT")
    save_dir = _env("SAVE_DIR")
    if not all([s_pretrained, l_ckpt, save_dir]):
        raise SystemExit("CKPT_S_PRETRAINED, L_CKPT, SAVE_DIR required")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_PROJECT, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(_env("EPOCHS", "500"))
    lr = float(_env("LR", "6e-5"))
    lam_clean = float(_env("LAMBDA_CLEAN", "1.0"))
    lam_miss = float(_env("LAMBDA_MISS", "0.5"))
    kd_T = float(_env("KD_T", "4.0"))
    miss_frac = float(_env("MISS_FRAC", "0.5"))
    batch = int(_env("TRAIN_BATCH", "16"))
    save_every = int(_env("SAVE_EVERY", "25"))
    seed = int(_env("SEED", "20260721"))
    max_steps = int(_env("MAX_STEPS", "0"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    with Engine(custom_parser=None) as engine:
        # CRITICAL: load L config FIRST because DFormer configs mutate a shared global C.
        # Loading S second restores the correct S dimensions.
        cfg_L = getattr(import_module(L_CONFIG), "C")
        cfg_L_backbone = cfg_L.backbone  # save before S overwrites
        cfg_L_decoder_embed = cfg_L.decoder_embed_dim
        import copy
        cfg_L = copy.deepcopy(cfg_L)
        cfg_L.pad = False

        # Reload S config (this overwrites the shared C back to S dims)
        from importlib import reload
        s_mod = import_module(S_CONFIG)
        reload(s_mod)
        cfg_S = getattr(s_mod, "C")
        cfg_S.pad = False
        cfg_S.batch_size = batch
        cfg_S.num_workers = int(_env("NUM_WORKERS", "4"))
        bg = cfg_S.background
        print(f"[config] S backbone={cfg_S.backbone} decoder_embed={cfg_S.decoder_embed_dim} "
              f"L backbone={cfg_L_backbone} L decoder_embed={cfg_L_decoder_embed}")

        criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=bg)

        # Student S
        student = segmodel(cfg=cfg_S, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
        # The builder auto-loads pretrained backbone; we start from pretrained (full train)
        device = torch.device("cuda")
        student.to(device)

        # Teacher L (frozen) — use the deepcopied L config
        teacher = segmodel(cfg=cfg_L, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
        w_L = torch.load(l_ckpt, map_location="cpu")
        if "model" in w_L: w_L = w_L["model"]
        teacher.load_state_dict(w_L, strict=False)
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # Projection heads (S stage-2 → L stage-2)
        projectors = nn.ModuleList([Stage2Projector() for _ in S_HOOK_INDICES]).to(device)

        # Hooks on stage-2 blocks (only when Branch 2 is active)
        s_feats, l_feats = {}, {}
        if lam_miss > 0:
            for i, idx in enumerate(S_HOOK_INDICES):
                student.backbone.layers[2].blocks[idx].register_forward_hook(_make_hook(s_feats, i))
            for i, idx in enumerate(L_HOOK_INDICES):
                teacher.backbone.layers[2].blocks[idx].register_forward_hook(_make_hook(l_feats, i))

        # Optimizer (student + projectors)
        params = group_weight([], student, nn.BatchNorm2d, lr)
        params.append({"params": projectors.parameters(), "lr": lr, "weight_decay": 0.01})
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=cfg_S.weight_decay)

        # Build train loader manually to guarantee batch size (get_train_loader
        # uses config.batch_size which can be corrupted by multi-config imports)
        from utils.dataloader.dataloader import TrainPre
        import torch.utils.data as data
        train_pre = TrainPre(cfg_S.norm_mean, cfg_S.norm_std, cfg_S.x_is_single_channel, cfg_S)
        train_setting = {
            "rgb_root": cfg_S.rgb_root_folder, "rgb_format": cfg_S.rgb_format,
            "gt_root": cfg_S.gt_root_folder, "gt_format": cfg_S.gt_format,
            "transform_gt": cfg_S.gt_transform, "x_root": cfg_S.x_root_folder,
            "x_format": cfg_S.x_format, "x_single_channel": cfg_S.x_is_single_channel,
            "class_names": cfg_S.class_names, "train_source": cfg_S.train_source,
            "eval_source": cfg_S.eval_source, "dataset_name": cfg_S.dataset_name,
            "backbone": cfg_S.backbone,
        }
        n_train = 795  # NYUDepthv2 train size
        niters_full = n_train // batch + 1
        train_ds = RGBXDataset(train_setting, "train", train_pre, batch * niters_full)
        train_loader = data.DataLoader(
            train_ds, batch_size=batch, num_workers=cfg_S.num_workers,
            drop_last=True, shuffle=True, pin_memory=True,
        )
        niters = len(train_loader)
        print(f"[loader] batch={batch} niters={niters} ds_len={len(train_ds)}")
        if max_steps > 0:
            niters = min(niters, max_steps)
        total_iters = epochs * niters
        warmup_iters = niters * min(10, epochs)
        lr_policy = WarmUpPolyLR(lr, cfg_S.lr_power, total_iters, warmup_iters)
        use_amp = _env("USE_AMP", "1") == "1"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        print(f"[cfg] epochs={epochs} niters/ep={niters} batch={batch} lr={lr} "
              f"lam_clean={lam_clean} lam_miss={lam_miss} T={kd_T} miss_frac={miss_frac}")

        def seg_loss(logits, gts):
            return criterion(logits, gts.long())[gts.long() != bg].mean()

        def kd_logit_loss(s_logits, t_logits, gts, T):
            mask = (gts.long() != bg)
            logp = F.log_softmax(s_logits / T, dim=1)
            q = F.softmax(t_logits / T, dim=1)
            kl = F.kl_div(logp, q, reduction="none").sum(1)
            return (kl * mask).sum() / mask.sum().clamp_min(1) * (T * T)

        def feat_distill_loss(miss_mask):
            tot = 0.0
            for i in range(len(S_HOOK_INDICES)):
                sf = projectors[i](s_feats[i][miss_mask].float())
                tf = l_feats[i].float().detach()
                tot += F.mse_loss(sf, tf)
            s_feats.clear(); l_feats.clear()
            return tot / len(S_HOOK_INDICES)

        global_step = 0
        t0 = time.time()
        for epoch in range(1, epochs + 1):
            student.train(); projectors.train()
            it = iter(train_loader)
            run_ce, run_kd_c, run_kd_m, n_seen = 0.0, 0.0, 0.0, 0
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

                B = imgs.shape[0]
                n_miss = max(1, int(B * miss_frac))
                miss_mask = torch.zeros(B, dtype=torch.bool, device=device)
                miss_mask[:n_miss] = True

                depth_input = depth.clone()
                if lam_miss > 0:
                    depth_input[miss_mask] = 0.0  # zeros = entire_missing for KD condition

                setattr(student, "global_step", global_step)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    # Teacher forward FIRST (no grad, then free its memory)
                    loss_kd_clean = torch.zeros((), device=device)
                    t_logits = None
                    if lam_clean > 0:
                        with torch.no_grad():
                            t_logits, _, _ = teacher.encode_decode(imgs, depth)
                            t_logits = t_logits.detach()
                        torch.cuda.empty_cache()

                    # Student forward (clean + missing mixed)
                    s_logits, _, _ = student.encode_decode(imgs, depth_input)
                    loss_ce = seg_loss(s_logits, gts)

                    # Branch 1: clean KD
                    if lam_clean > 0 and t_logits is not None:
                        loss_kd_clean = kd_logit_loss(s_logits, t_logits, gts, kd_T)

                    # Branch 2: missing feature KD (missing images only)
                    loss_kd_miss = torch.zeros((), device=device)
                    if lam_miss > 0 and n_miss > 0:
                        with torch.no_grad():
                            _ = teacher.backbone(imgs[miss_mask], depth_input[miss_mask])
                        # s_feats from full-batch student forward; slice to missing subset
                        loss_kd_miss = feat_distill_loss(miss_mask)

                    loss = loss_ce + lam_clean * loss_kd_clean + lam_miss * loss_kd_miss

                lv = float(loss.detach().item())
                if not (lv == lv and abs(lv) < float("inf")):
                    optimizer.zero_grad(set_to_none=True)
                    s_feats.clear(); l_feats.clear()
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gnorm = torch.nn.utils.clip_grad_norm_(
                    list(student.parameters()) + list(projectors.parameters()), 100.0
                )
                if torch.isfinite(gnorm):
                    scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                s_feats.clear(); l_feats.clear()

                cur_lr = lr_policy.get_lr(global_step)
                for g in optimizer.param_groups:
                    g["lr"] = cur_lr
                global_step += 1
                run_ce += float(loss_ce.detach().item())
                run_kd_c += float(loss_kd_clean.detach().item())
                run_kd_m += float(loss_kd_miss.detach().item())
                n_seen += 1

            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] ce={run_ce/max(1,n_seen):.4f} "
                  f"kd_c={run_kd_c/max(1,n_seen):.4f} kd_m={run_kd_m/max(1,n_seen):.4f} "
                  f"lr={cur_lr:.2e} ({dt:.0f}s)", flush=True)
            if epoch % save_every == 0 or epoch == epochs:
                path = os.path.join(save_dir, f"ckpt_ep{epoch}.pth")
                torch.save({"model": student.state_dict(), "epoch": epoch}, path)
                print(f"[save] {path}", flush=True)

        meta = {"method": "dual_branch_L2S_KD", "epochs": epochs, "lr": lr,
                "lam_clean": lam_clean, "lam_miss": lam_miss, "kd_T": kd_T,
                "miss_frac": miss_frac, "seed": seed, "niters_per_epoch": niters,
                "L_ckpt": l_ckpt, "S_pretrained": s_pretrained,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}
        with open(os.path.join(save_dir, "train_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} epochs in {time.time()-t0:.0f}s -> {save_dir}")


if __name__ == "__main__":
    main()
