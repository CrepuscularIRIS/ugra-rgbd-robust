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
import os as _os
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

# Overridable so the SAME trainer runs the SUN-RGBD falsification arm (E2) under an
# identical protocol. NYU default is unchanged, so every existing number is untouched.
CONFIG_MODULE = _os.environ.get("CONFIG_MODULE", "local_configs.NYUDepthv2.DFormerv2_S")


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def _depth_attenuate(modal: torch.Tensor, gen: torch.Generator, p_clean: float, alpha_min: float) -> torch.Tensor:
    """C2 legal augmentation (NOT n2). Per-sample: with prob p_clean keep alpha=1
    (clean depth, protects clean mIoU); else alpha ~ U(alpha_min, 1). Multiply the
    normalized depth by alpha, which scales the GeoPriorGen depth bias
    mask_d = |d_i - d_j| * decay by EXACTLY alpha (offset-invariant). alpha is ALWAYS
    >= alpha_min > 0, so mask_d is never exactly 0 -> the model NEVER visits the
    eval-absent (spatial-only) operator during training. Trains the full depth-
    authority magnitude range so the model degrades gracefully; the absent limit
    (alpha=0) is reached only by continuity/extrapolation at eval. NEW tensor."""
    b = modal.shape[0]
    out = modal.clone()
    u = torch.rand(b, generator=gen, device=modal.device)
    alpha = alpha_min + (1.0 - alpha_min) * torch.rand(b, generator=gen, device=modal.device)
    keep = u < p_clean
    alpha = torch.where(keep, torch.ones_like(alpha), alpha)
    return out * alpha.view(b, 1, 1, 1)


def main():
    ckpt_init = _env("CKPT_INIT")
    save_dir = _env("SAVE_DIR")
    if not ckpt_init or not save_dir:
        raise SystemExit("CKPT_INIT, SAVE_DIR are required")
    p_clean = float(_env("P_CLEAN", "0.5"))   # prob of alpha=1 (clean-protect)
    alpha_min = float(_env("ALPHA_MIN", "0.05"))
    if not (0.0 <= p_clean <= 1.0) or not (0.0 < alpha_min < 1.0):
        raise SystemExit(f"bad P_CLEAN={p_clean} or ALPHA_MIN={alpha_min}")
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
        # SUN-RGBD uses the padded eval/train protocol; NYU does not. Derive from the
        # dataset so the SUN arm gets its own correct protocol.
        config.pad = "SUNRGBD" in str(getattr(config, "dataset_name", "")).upper()
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

        # --- FREEZE_MODE: constrained-update experiment. Restricts which parameters may move,
        # to test WHERE a repair CAN live rather than where an unconstrained finetune happens to
        # put it. The transplant only shows where an independently-trained delta transfers; it
        # cannot rule out that q/k could have done the job if it were the only option.
        #   qk_only   : only q_proj / k_proj trainable  (the SCORE path)
        #   vout_only : only v_proj / out_proj trainable (the REPRESENTATION path)
        #   geo_only  : only GeoPriorGen.weight trainable (needs GROUP_WEIGHT_FIX=1 to be reachable)
        #   none      : unconstrained (default; matches the existing arms)
        _fm = _env("FREEZE_MODE", "none")
        if _fm != "none":
            sel = {
                "qk_only": (".q_proj", ".k_proj"),
                "vout_only": (".v_proj", ".out_proj"),
                "geo_only": ("Geo.weight",),
            }[_fm]
            n_train = 0
            for pname, prm in model.named_parameters():
                keep = any(s in pname for s in sel)
                prm.requires_grad_(keep)
                n_train += int(keep)
            print(f"[freeze] FREEZE_MODE={_fm}: {n_train} tensors trainable", flush=True)
            if n_train == 0:
                raise SystemExit(f"FATAL: FREEZE_MODE={_fm} selected 0 tensors")

        # --- GROUP_WEIGHT_FIX: the campaign-documented defect is that group_weight() never adds
        # bare nn.Parameters to the optimizer, so all 29 GeoPriorGen depth-bias-authority weights
        # NEVER TRAIN. With the fix on, they are appended explicitly. This is the confound control:
        # if the phenomenon and the repair survive with the bias authority trainable, neither is an
        # artifact of the bug.
        params = group_weight([], model, nn.BatchNorm2d, base_lr)
        if _env("GROUP_WEIGHT_FIX", "0") == "1":
            seen = {id(p) for g in params for p in g["params"]}
            extra = [p for p in model.parameters() if p.requires_grad and id(p) not in seen]
            if extra:
                params.append({"params": extra, "lr": base_lr, "weight_decay": 0.0})
            print(f"[fix] GROUP_WEIGHT_FIX=1: added {len(extra)} previously-unoptimized tensors", flush=True)

        # Drop frozen tensors so the optimizer state matches the constraint.
        for g in params:
            g["params"] = [p for p in g["params"] if p.requires_grad]
        params = [g for g in params if g["params"]]
        optimizer = torch.optim.AdamW(
            params, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay
        )
        print(f"[opt] {sum(p.numel() for g in params for p in g['params']):,} trainable params "
              f"in {len(params)} groups", flush=True)

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
              f"lr={base_lr} p_clean={p_clean} alpha_min={alpha_min} seed={seed} save_dir={save_dir}", flush=True)
        # PROVENANCE. These dials live in the MODEL/DATALOADER forward, not in this script, so
        # nothing here would otherwise record them -- and an arm whose defining flag is absent
        # from its own log is unverifiable after the fact. The 2026-07-26 rgb_only_finetune arm
        # was defined entirely by DEPTH_LAMBDA_TRAIN=0 and neither its log nor its meta says so.
        fwd_env = {k: os.environ.get(k, "<unset>") for k in
                   ("DEPTH_LAMBDA_TRAIN", "C2_ATTEN", "C2_PCLEAN", "C2_ALPHAMIN", "C2_DIST", "PRP_ON")}
        print("[fwd-env] " + " ".join(f"{k}={v}" for k, v in fwd_env.items()), flush=True)

        # one-time legality assert: attenuation NEVER yields zeros (the eval-absent
        # operator). p_clean=0 forces the attenuated branch for all samples.
        _probe = torch.randn(4, 3, 8, 8, device=device)
        _g2 = torch.Generator(device=device); _g2.manual_seed(0)
        _att = _depth_attenuate(_probe, _g2, p_clean=0.0, alpha_min=alpha_min)
        assert not torch.equal(_att, torch.zeros_like(_probe)) and _att.abs().sum() > 0, \
            "attenuate produced zeros == eval-absent operator; would be n2 (illegal)"

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
                # --- ATTEN_DIST: the ALTERNATIVE-BASELINE comparators a reviewer will demand.
                # All share this trainer/harness so the comparison is exact. E[alpha] is matched
                # to our method's U(0.05,1) mean of 0.525 wherever possible.
                #   u005_1    : alpha ~ U(0.05,1)   -- OURS (never 0), E=0.525
                #   bernoulli : alpha in {0,1}, P(0)=0.475 -> E=0.525  -- ordinary modality dropout
                #   uniform01 : alpha ~ U(0,1), E=0.5      -- includes the absent state
                #   constant  : alpha = 0.525              -- separates STOCHASTICITY from mean shrinkage
                # bernoulli/uniform01 DO instantiate the absent operator: they are BASELINES to
                # compare against, never promotable as our method (the n2 rule governs what we
                # promote, not what we compare to).
                _dist = _env("ATTEN_DIST", "u005_1")
                if _dist != "u005_1" and _env("ATTEN_TARGET", "depth") != "pos":
                    b = modal.shape[0]
                    if _dist == "bernoulli":
                        a = (torch.rand(b, generator=gen, device=modal.device) >= 0.475).float()
                    elif _dist == "uniform01":
                        a = torch.rand(b, generator=gen, device=modal.device)
                    elif _dist == "constant":
                        a = torch.full((b,), 0.525, device=modal.device)
                    else:
                        raise SystemExit(f"unknown ATTEN_DIST={_dist}")
                    # p_clean still applies, matching the structure of the main arm
                    keep = torch.rand(b, generator=gen, device=modal.device) < p_clean
                    a = torch.where(keep, torch.ones_like(a), a)
                    modal = modal * a.view(b, 1, 1, 1)
                elif _env("ATTEN_TARGET", "depth") == "pos":
                    # P2-CTRL: the MATCHED NEGATIVE CONTROL. Attenuate the POSITIONAL bias
                    # instead of the depth bias — same module (GeoPriorGen), same functional
                    # form (a scalar on one of the two bias terms), same U(alpha_min,1) family,
                    # same per-sample draw shared across all 29 modules, same schedule/LR/steps.
                    # The only thing that changes is WHICH term is attenuated. Depth is left
                    # untouched, so if the vo:qk repair split is depth-SPECIFIC this arm must
                    # NOT reproduce it; if the split is a property of the transplant SITES, it
                    # will. Eval never removes the positional term, so this is not n2-class.
                    from models.encoders.DFormerv2 import set_pos_alpha

                    _b = modal.shape[0]
                    _u = torch.rand(_b, generator=gen, device=modal.device)
                    _a = alpha_min + (1.0 - alpha_min) * torch.rand(_b, generator=gen, device=modal.device)
                    set_pos_alpha(torch.where(_u < p_clean, torch.ones_like(_a), _a))
                else:
                    modal = _depth_attenuate(modal, gen, p_clean, alpha_min)  # C2 legal alpha-attenuation
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

        final_path = os.path.join(save_dir, "ckpt_final_c2.pth")
        torch.save({"model": model.state_dict(), "epochs": epochs,
                    "p_clean": p_clean, "alpha_min": alpha_min}, final_path)
        print(f"[save] {final_path}", flush=True)

        meta = {
            "screen": "l2_c2_depth_attenuate",
            "promotable": "screen (legal; promote to 500ep if direction confirmed)",
            "ckpt_init": ckpt_init, "epochs": epochs, "train_batch": train_batch,
            "lr": base_lr, "p_clean": p_clean, "alpha_min": alpha_min, "seed": seed,
            "amp": "OFF(fp32)", "niters_per_epoch": niters, "forward_env": fwd_env,
            "global_steps": global_step, "skipped_batches": skipped,
            "wall_seconds": round(time.time() - t0, 1),
            "final_ckpt": final_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        }
        with open(os.path.join(save_dir, "train_meta_c2.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[done] {epochs} ep in {time.time()-t0:.0f}s -> {final_path}", flush=True)


if __name__ == "__main__":
    main()
