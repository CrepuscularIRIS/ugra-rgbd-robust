#!/usr/bin/env python
"""PRP (Pseudo-Depth Relational Prior) 100-ep finetune-proxy SCREEN — n7.

DIAGNOSTIC ONLY (NOT promotable). Cheap kill test for the prereg tripwire:
KILL (no full-train) unless BOTH after 100 ep:
  - clean mIoU >= baseline-0.2 (55.72 -> 55.52), AND
  - stage-2 clean-vs-zeros cosine RISING toward >=0.97 (drift below 0.0713).

Design (mirrors the frozen recipe, finetune-from-converged for the screen):
  - PRP_ON=1: the pseudo-depth head + zero-init mask_p bias are active. At
    epoch 0 w_p==0 => forward == baseline exactly, so init from the converged
    S ckpt is a clean start (PRP params are new: w_p=0, decay_p=decay,
    pseudo_depth_head random -- but the head only affects mask_p, gated by w_p=0).
  - CLEAN depth ONLY -- NO depth dropout, NO corruption in the train path
    (reward_hack_guard P1: assert train depth min>0).
  - Aux loss L = L_seg + lambda * SI-L1(d_hat, clean inverse-depth), valid-mask
    excludes zero/invalid GT pixels. lambda default 0.1.
  - USE_AMP=0 HARD (fp32; AMP silently overflows grads on this arch).
  - saves only the FINAL ckpt (screen).

Env:
  CKPT_INIT   converged S ckpt to finetune from            [required]
  SAVE_DIR    output dir                                    [required]
  EPOCHS      finetune epochs                               [default 100]
  TRAIN_BATCH batch size                                    [default 16]
  LR          base LR                                       [default 6e-5]
  LAMBDA_AUX  aux depth loss weight                         [default 0.1]
  SEED        seed                                          [default 20260721]
  MAX_STEPS   cap steps/epoch (smoke)                       [default 0=off]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

os.environ["PRP_ON"] = "1"  # enable the PRP pathway for this trainer

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))          # ugra-rgbd-robust
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT, _HERE):
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
from prp_aux_loss import si_l1_depth_loss                      # noqa: E402

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

    epochs = int(_env("EPOCHS", "100"))
    train_batch = int(_env("TRAIN_BATCH", "16"))
    base_lr = float(_env("LR", "6e-5"))
    lambda_aux = float(_env("LAMBDA_AUX", "0.1"))
    seed = int(_env("SEED", "20260721"))
    max_steps = int(_env("MAX_STEPS", "0"))

    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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
        # Expected missing = PRP params only (pseudo_depth_head.*, decay_p, w_p).
        prp_missing = [k for k in rep.missing_keys if
                       ("pseudo_depth_head" in k or "decay_p" in k or "w_p" in k)]
        non_prp_missing = [k for k in rep.missing_keys if k not in prp_missing]
        print(f"[load] init from {ckpt_init} missing={len(rep.missing_keys)} "
              f"(PRP={len(prp_missing)} non-PRP={len(non_prp_missing)}) "
              f"unexpected={len(rep.unexpected_keys)}", flush=True)
        if non_prp_missing:
            print(f"[load] WARNING non-PRP missing (unexpected!): {non_prp_missing[:8]}", flush=True)
        if rep.unexpected_keys:
            print(f"[load] unexpected: {list(rep.unexpected_keys)[:8]}", flush=True)

        # sanity: w_p must be zero at init so epoch-0 == baseline. w_p params
        # exist in every GeoPriorGen (29 blocks) but only the 18 stage-2 blocks
        # (layers.2) ever receive d_hat, so only theirs are actually used.
        wps = [p for n, p in model.named_parameters() if n.endswith("w_p")]
        wps_stage2 = [p for n, p in model.named_parameters()
                      if n.endswith("w_p") and ".layers.2." in n]
        assert len(wps_stage2) == 18, f"expected 18 stage-2 w_p, got {len(wps_stage2)}"
        assert all(float(p.detach().abs().sum()) == 0.0 for p in wps), "w_p not zero-init"
        print(f"[prp] {len(wps)} total w_p ({len(wps_stage2)} used @ stage-2), "
              f"all zero-init OK", flush=True)

        device = torch.device("cuda")
        model.to(device)

        params = group_weight([], model, nn.BatchNorm2d, base_lr)
        optimizer = torch.optim.AdamW(params, lr=base_lr, betas=(0.9, 0.999),
                                      weight_decay=config.weight_decay)

        train_loader, _ = get_train_loader(engine, RGBXDataset, config)
        niters = len(train_loader)
        if max_steps > 0:
            niters = min(niters, max_steps)
        total_iters = epochs * niters
        warmup_iters = min(niters * 1, total_iters)
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iters, warmup_iters)

        print(f"[cfg] AMP=OFF(fp32) PRP_ON=1 epochs={epochs} niters/ep={niters} "
              f"batch={train_batch} lr={base_lr} lambda_aux={lambda_aux} seed={seed} "
              f"save_dir={save_dir}", flush=True)

        global_step = 0
        t0 = time.time()
        skipped = 0
        depth_min_seen = float("inf")
        for epoch in range(1, epochs + 1):
            model.train()
            it = iter(train_loader)
            run_seg = 0.0
            run_aux = 0.0
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
                # P1 reward_hack_guard: NO corruption (zeros/holes/gaussian) in
                # the train path. The eval corruptions are: entire_missing ==
                # a per-sample ALL-ZERO (structureless) depth tensor; holes ==
                # patches zeroed. A CLEAN normalized depth sample is never the
                # all-zeros tensor and always has spatial variation. So the
                # authoritative reward-hack check is: no sample in the batch is
                # structureless (per-sample std==0 or exactly all-zero). We track
                # the MIN per-sample depth std seen; a leaked entire_missing
                # sample -> std 0.  (modal is z-normalized so raw min<0 is
                # NORMAL, not a corruption signal.)
                per_sample_std = modal[:, 0].reshape(modal.shape[0], -1).std(dim=1)
                min_std = float(per_sample_std.min().item())
                depth_min_seen = min(depth_min_seen, min_std)
                # HARD early guard: a structureless (all-zero / entire_missing-like)
                # depth sample must fail the run IMMEDIATELY, not after 100 epochs.
                if min_std <= 0.0:
                    raise AssertionError(
                        f"reward_hack_guard P1 FAILED @ step {global_step}: a "
                        f"structureless depth sample (std==0) leaked into the "
                        f"train path (== entire_missing/holes corruption).")

                setattr(model, "global_step", global_step)
                # USE_AMP=0: pure fp32, NO autocast, NO GradScaler.
                seg_loss = model(imgs, modal, gts)
                d_hat = model.backbone.last_dhat  # stashed during forward
                if d_hat is None:
                    raise RuntimeError("PRP d_hat is None after forward -- head not wired")
                aux_loss = si_l1_depth_loss(d_hat, modal)
                loss = seg_loss + lambda_aux * aux_loss

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
                run_seg += float(seg_loss.detach().item())
                run_aux += float(aux_loss.detach().item())
                n_seen += 1
            avg_seg = run_seg / max(1, n_seen)
            avg_aux = run_aux / max(1, n_seen)
            # log w_p growth (a live signal the mask_p term is being used)
            wp_abs = float(torch.stack([p.detach().abs().sum() for p in wps_stage2]).sum().item())
            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] seg={avg_seg:.4f} aux={avg_aux:.4f} "
                  f"lr={lr:.2e} |w_p|sum={wp_abs:.4f} n={n_seen} skip={skipped} "
                  f"min_depth_std={depth_min_seen:.4f} ({dt:.0f}s)", flush=True)

        # P1 assert AFTER the run: no structureless (all-zero) depth sample ever
        # entered the train path (== no entire_missing/holes corruption leak).
        assert depth_min_seen > 0.0, (
            f"reward_hack_guard P1 FAILED: min per-sample depth std={depth_min_seen} "
            f"== 0 (a structureless/zeroed depth sample leaked into the train path)")

        final_path = os.path.join(save_dir, "ckpt_final_prp.pth")
        torch.save({"model": model.state_dict(), "epochs": epochs,
                    "lambda_aux": lambda_aux, "prp": True}, final_path)
        print(f"[save] {final_path}", flush=True)

        meta = {
            "screen": "n7_prp_finetune_proxy",
            "promotable": False,
            "prp_on": True,
            "ckpt_init": ckpt_init, "epochs": epochs, "train_batch": train_batch,
            "lr": base_lr, "lambda_aux": lambda_aux, "seed": seed,
            "amp": "OFF(fp32)", "niters_per_epoch": niters,
            "global_steps": global_step, "skipped_batches": skipped,
            "train_min_per_sample_depth_std": depth_min_seen,
            "reward_hack_guard_P1_pass": bool(depth_min_seen > 0.0),
            "final_wp_abs_sum": float(torch.stack([p.detach().abs().sum() for p in wps_stage2]).sum().item()),
            "wall_seconds": round(time.time() - t0, 1),
            "final_ckpt": final_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(os.path.join(save_dir, "train_meta_prp.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} ep in {time.time()-t0:.0f}s -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
