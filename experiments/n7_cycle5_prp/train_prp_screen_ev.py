#!/usr/bin/env python
"""PRP ESCAPE-VELOCITY re-screen (n7 cycle 5, RE-SCREEN).

The prior PRP screen KILLED on a METHODOLOGY ARTIFACT: zero-init w_p never
moved from 0 under AdamW weight-decay + a decaying finetune LR, so mask_p was
inert and the mechanism was NEVER exercised. This re-screen gives w_p escape
velocity so mask_p ACTIVELY engages, then applies a hard kill rule that can
refute the mechanism itself.

ESCAPE-VELOCITY FIX (both applied):
  1. AUX-ONLY WARMUP then RAMP the mask_p gate g:
       - ep 1..WARMUP_EP (default 10): g=0. mask_p OFF, only the aux SI-L1 loss
         trains the pseudo-depth head h_p. Forward == baseline exactly (ckpt
         untouched, no noise injected while d-hat is random).
       - ep WARMUP_EP+1 .. WARMUP_EP+RAMP_EP (default +20): g ramps 0 -> 1
         linearly. mask_p engages with a trained-enough d-hat.
       - after: g=1. mask_p fully on.
     w_p is WARM-INIT to WP_WARM (default 0.2, a fraction of the sensor
     w_d~1.449) at the moment g first becomes >0, so mask_p is nonzero &
     produces real seg-loss gradients into w_p from the instant it engages.
  2. DEDICATED PARAM GROUP for w_p + decay_p: NO weight decay + LR_MULT (default
     8) x base LR, so they can move and are not pulled back toward 0 by WD.

Everything else is identical to the prior PRP screen (h_p on patch-embed RGB
-- P2 passed; aux SI-L1 with the denorm fix grok approved; USE_AMP=0; clean
depth only; single GPU).

Env:
  CKPT_INIT   converged S ckpt to finetune from            [required]
  SAVE_DIR    output dir                                    [required]
  EPOCHS      finetune epochs                               [default 100]
  WARMUP_EP   aux-only (g=0) warmup epochs                  [default 10]
  RAMP_EP     epochs to ramp g 0->1 after warmup            [default 20]
  WP_WARM     warm-init value for w_p when g first >0       [default 0.2]
  LR_MULT     LR multiplier for the w_p+decay_p group       [default 8.0]
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
from models.encoders.DFormerv2 import set_prp_gate             # noqa: E402
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


def compute_gate(epoch: int, warmup_ep: int, ramp_ep: int) -> float:
    """g in [0,1] for the current epoch (1-indexed)."""
    if epoch <= warmup_ep:
        return 0.0
    if epoch <= warmup_ep + ramp_ep:
        return (epoch - warmup_ep) / float(ramp_ep)
    return 1.0


def main():
    ckpt_init = _env("CKPT_INIT")
    save_dir = _env("SAVE_DIR")
    if not ckpt_init or not save_dir:
        raise SystemExit("CKPT_INIT and SAVE_DIR are required")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(_PROJECT, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    epochs = int(_env("EPOCHS", "100"))
    warmup_ep = int(_env("WARMUP_EP", "10"))
    ramp_ep = int(_env("RAMP_EP", "20"))
    wp_warm = float(_env("WP_WARM", "0.2"))
    lr_mult = float(_env("LR_MULT", "8.0"))
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
        prp_missing = [k for k in rep.missing_keys if
                       ("pseudo_depth_head" in k or "decay_p" in k or "w_p" in k)]
        non_prp_missing = [k for k in rep.missing_keys if k not in prp_missing]
        print(f"[load] init from {ckpt_init} missing={len(rep.missing_keys)} "
              f"(PRP={len(prp_missing)} non-PRP={len(non_prp_missing)}) "
              f"unexpected={len(rep.unexpected_keys)}", flush=True)
        if non_prp_missing:
            print(f"[load] WARNING non-PRP missing (unexpected!): {non_prp_missing[:8]}", flush=True)

        # Identify the stage-2 w_p (used) and ALL w_p / decay_p (the EV group).
        wps_stage2 = [p for n, p in model.named_parameters()
                      if n.endswith("w_p") and ".layers.2." in n]
        assert len(wps_stage2) == 18, f"expected 18 stage-2 w_p, got {len(wps_stage2)}"

        # At load, w_p is zero-init (module default) => forward == baseline while
        # g=0 during warmup (double safety: g=0 AND w_p=0). We WARM-INIT w_p to
        # wp_warm exactly at the moment the gate first opens (start of ep
        # warmup_ep+1), so mask_p is nonzero the instant it engages.
        print(f"[prp] {len(wps_stage2)} stage-2 w_p, zero-init at load "
              f"(warm-init to {wp_warm} at gate-open ep {warmup_ep+1})", flush=True)

        device = torch.device("cuda")
        model.to(device)

        # --- Dedicated no-WD, higher-LR param group for w_p + decay_p (EV #2) ---
        ev_names = set()
        ev_params = []
        for n, p in model.named_parameters():
            if n.endswith("w_p") or n.endswith("decay_p"):
                ev_params.append(p)
                ev_names.add(n)
        ev_id = {id(p) for p in ev_params}
        base_params = group_weight([], model, nn.BatchNorm2d, base_lr)
        # strip EV params out of the base groups so they aren't double-optimized
        for grp in base_params:
            grp["params"] = [p for p in grp["params"] if id(p) not in ev_id]
        ev_group = {"params": ev_params, "lr": base_lr * lr_mult,
                    "weight_decay": 0.0, "is_ev": True}
        optimizer = torch.optim.AdamW(base_params + [ev_group], lr=base_lr,
                                      betas=(0.9, 0.999), weight_decay=config.weight_decay)
        n_ev = len(ev_params)
        print(f"[ev] {n_ev} params in the no-WD LRx{lr_mult} escape-velocity group "
              f"(w_p+decay_p across all blocks); base groups stripped of them", flush=True)

        train_loader, _ = get_train_loader(engine, RGBXDataset, config)
        niters = len(train_loader)
        if max_steps > 0:
            niters = min(niters, max_steps)
        total_iters = epochs * niters
        warmup_iters = min(niters * 1, total_iters)
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iters, warmup_iters)

        print(f"[cfg] AMP=OFF(fp32) PRP_ON=1 EV epochs={epochs} warmup_ep={warmup_ep} "
              f"ramp_ep={ramp_ep} wp_warm={wp_warm} lr_mult={lr_mult} niters/ep={niters} "
              f"batch={train_batch} lr={base_lr} lambda_aux={lambda_aux} seed={seed}", flush=True)

        global_step = 0
        t0 = time.time()
        skipped = 0
        depth_min_seen = float("inf")
        warm_init_done = False
        wp_traj = []
        for epoch in range(1, epochs + 1):
            g = compute_gate(epoch, warmup_ep, ramp_ep)
            set_prp_gate(g)
            # warm-init w_p exactly at gate-open (first epoch with g>0)
            if g > 0.0 and not warm_init_done:
                with torch.no_grad():
                    for p in wps_stage2:
                        p.fill_(wp_warm)
                warm_init_done = True
                print(f"[ev] gate OPEN @ ep {epoch}: warm-init all stage-2 w_p "
                      f"= {wp_warm} (mask_p now active)", flush=True)

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
                per_sample_std = modal[:, 0].reshape(modal.shape[0], -1).std(dim=1)
                min_std = float(per_sample_std.min().item())
                depth_min_seen = min(depth_min_seen, min_std)
                if min_std <= 0.0:
                    raise AssertionError(
                        f"reward_hack_guard P1 FAILED @ step {global_step}: a "
                        f"structureless depth sample (std==0) leaked into the train path.")

                setattr(model, "global_step", global_step)
                seg_loss = model(imgs, modal, gts)
                d_hat = model.backbone.last_dhat
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
                # scale base LR normally; keep the EV group at lr_mult x current LR
                for grp in optimizer.param_groups:
                    grp["lr"] = lr * lr_mult if grp.get("is_ev") else lr
                global_step += 1
                run_seg += float(seg_loss.detach().item())
                run_aux += float(aux_loss.detach().item())
                n_seen += 1
            avg_seg = run_seg / max(1, n_seen)
            avg_aux = run_aux / max(1, n_seen)
            wp_abs = float(torch.stack([p.detach().abs().sum() for p in wps_stage2]).sum().item())
            wp_mean = wp_abs / max(1, len(wps_stage2))
            wp_traj.append({"ep": epoch, "g": round(g, 4), "wp_abs_sum": round(wp_abs, 6),
                            "wp_abs_mean": round(wp_mean, 6), "seg": round(avg_seg, 4),
                            "aux": round(avg_aux, 4)})
            dt = time.time() - t0
            print(f"[ep {epoch:3d}/{epochs}] g={g:.3f} seg={avg_seg:.4f} aux={avg_aux:.4f} "
                  f"lr={lr:.2e} |w_p|sum={wp_abs:.4f} (mean={wp_mean:.4f}) n={n_seen} "
                  f"skip={skipped} min_depth_std={depth_min_seen:.4f} ({dt:.0f}s)", flush=True)

        assert depth_min_seen > 0.0, (
            f"reward_hack_guard P1 FAILED: min per-sample depth std={depth_min_seen} == 0")

        # ensure gate is fully open in the saved model (eval uses g=1 default)
        set_prp_gate(1.0)
        final_path = os.path.join(save_dir, "ckpt_final_prp_ev.pth")
        torch.save({"model": model.state_dict(), "epochs": epochs,
                    "lambda_aux": lambda_aux, "prp": True, "ev": True}, final_path)
        print(f"[save] {final_path}", flush=True)

        final_wp = float(torch.stack([p.detach().abs().sum() for p in wps_stage2]).sum().item())
        meta = {
            "screen": "n7_prp_escape_velocity_rescreen",
            "promotable": False,
            "prp_on": True, "escape_velocity": True,
            "ckpt_init": ckpt_init, "epochs": epochs, "warmup_ep": warmup_ep,
            "ramp_ep": ramp_ep, "wp_warm": wp_warm, "lr_mult": lr_mult,
            "train_batch": train_batch, "lr": base_lr, "lambda_aux": lambda_aux,
            "seed": seed, "amp": "OFF(fp32)", "niters_per_epoch": niters,
            "global_steps": global_step, "skipped_batches": skipped,
            "train_min_per_sample_depth_std": depth_min_seen,
            "reward_hack_guard_P1_pass": bool(depth_min_seen > 0.0),
            "final_wp_abs_sum": final_wp,
            "final_wp_abs_mean": final_wp / max(1, len(wps_stage2)),
            "wp_moved_from_warm_init": bool(abs(final_wp / 18.0 - wp_warm) > 1e-4),
            "wp_trajectory": wp_traj,
            "wall_seconds": round(time.time() - t0, 1),
            "final_ckpt": final_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(os.path.join(save_dir, "train_meta_prp_ev.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} ep in {time.time()-t0:.0f}s |w_p|sum={final_wp:.4f} "
              f"-> {final_path}", flush=True)


if __name__ == "__main__":
    main()
