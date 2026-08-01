"""n7 cycle-4 (M3) training: DFormerv2-S + Detached Semantic-Boundary Gate + Zero-Init Parallel Adapter.

Minimal modification of the standard DFormer training script (mirrors train_n7.py's fixes):
  1. Wraps the base segmodel with DFormerv2WithBoundaryGate
  2. Adds boundary_head + adapter params to the optimizer
  3. Scheduled modality (depth) dropout: p ramps 0 -> 0.40 over epochs [0,150], held at 0.40 after
  4. USE_AMP=0 (fp16 breaks training on this arch — confirmed across cycle-3 attempts)
  5. Epoch-0 identity check: M3 forward == base forward at init (zero-init adapter)
"""
import argparse
import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
from importlib import import_module
from torch.nn.parallel import DistributedDataParallel

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DFORMER_ROOT = os.path.join(EXP_DIR, "..", "..", "repos", "DFormer")
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, DFORMER_ROOT)
sys.path.insert(0, os.path.join(DFORMER_ROOT, "utils"))

from models.builder import EncoderDecoder as segmodel
from model_wrapper_m3 import DFormerv2WithBoundaryGate
from val_mm import evaluate
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", default=2, type=int)
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--local-rank", default=0)
parser.add_argument("--clip_grad_norm", type=float, default=1.0)
parser.add_argument("--boundary_weight", type=float, default=0.10)
parser.add_argument("--boundary_pos_weight", type=float, default=30.0)
parser.add_argument("--dropout_max", type=float, default=0.40)
parser.add_argument("--dropout_ramp_epochs", type=int, default=150)
parser.add_argument("--seed", type=int, default=20260721)
# ablation switches (2x2 + dropout): default = canonical M3
parser.add_argument("--supervision", choices=["label", "depth"], default="label")
parser.add_argument("--stop_grad", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_dropout", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--tag", type=str, default="m3_canonical")
parser.add_argument("--init_from", type=str, default="",
                    help="finetune-proxy SCREEN: load a converged baseline into model.base (strict=False; "
                         "new M3 params keep zero-init). Empty = full-train from pretrained backbone.")
parser.add_argument("--eval_iter", type=int, default=-1,
                    help="override config.eval_iter for THIS run only (SCREEN needs early+late val); "
                         "-1 keeps the config default. Does NOT touch the frozen eval harness.")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def dropout_prob(epoch, pmax, ramp):
    if ramp <= 0:
        return pmax
    return min(pmax, pmax * epoch / ramp)


with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")
    if not hasattr(config, "pad"):
        config.pad = False
    if args.eval_iter > 0:
        config.eval_iter = args.eval_iter  # SCREEN-only override; frozen eval harness untouched
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    logger.info(f"[n7c4] M3 Detached Semantic-Boundary Gate + Zero-Init Parallel Adapter")
    logger.info(f"[n7c4] tag={args.tag} supervision={args.supervision} stop_grad={args.stop_grad} "
                f"use_dropout={args.use_dropout} boundary_weight={args.boundary_weight} AMP={args.amp}")
    # fp16 breaks this arch (cycle-3); there is intentionally no autocast path, but refuse --amp
    if args.amp:
        raise SystemExit("[n7c4] FATAL: --amp is not supported (USE_AMP=0 required). Relaunch without --amp.")

    if args.use_seed:
        random.seed(args.seed); np.random.seed(args.seed)
        torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    BatchNorm2d = nn.SyncBatchNorm if args.syncbn else nn.BatchNorm2d

    base_model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, syncbn=args.syncbn)
    model = DFormerv2WithBoundaryGate(
        base_model, num_classes=config.num_classes,
        boundary_weight=args.boundary_weight, boundary_pos_weight=args.boundary_pos_weight,
        supervision=args.supervision, stop_grad=args.stop_grad,   # canonical: label + stop-grad
    )
    new_params = list(model.boundary_head.parameters()) + list(model.adapter.parameters())
    n_new = sum(p.numel() for p in new_params)
    logger.info(f"[n7c4] added {n_new:,} new params ({n_new/1e6:.3f}M): boundary_head + zero-init adapter")

    # finetune-proxy SCREEN: init model.base from a converged baseline; M3 params stay zero-init
    if args.init_from:
        w = torch.load(args.init_from, map_location="cpu")
        w = w.get("model", w.get("state_dict", w)) if isinstance(w, dict) else w
        rep = model.base.load_state_dict(w, strict=False)
        logger.info(f"[n7c4] init_from={args.init_from}: base load missing={len(rep.missing_keys)} "
                    f"unexpected={len(rep.unexpected_keys)} (M3 boundary_head+adapter keep zero-init)")

    base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model.base, BatchNorm2d, base_lr)
    params_list.append({"params": new_params, "lr": base_lr})
    optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999),
                                  weight_decay=config.weight_decay)

    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)
    val_loader, val_sampler = get_val_loader(engine, RGBXDataset, config, val_batch_size=int(config.batch_size))

    if engine.distributed:
        model.cuda()
        model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                        output_device=engine.local_rank, find_unused_parameters=True)
    else:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    niters = config.niters_per_epoch
    total_iters = niters * args.epochs
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iters, niters * config.warm_up_epoch)
    optimizer.zero_grad()
    logger.info(f"[n7c4] start: {args.epochs} ep, {niters} iters/ep, dropout ramp 0->{args.dropout_max} "
                f"over {args.dropout_ramp_epochs} ep (use_dropout={args.use_dropout})")

    best_miou = 0
    # --- epoch-0 identity check: zero-init adapter => M3 forward == baseline (~55.72) ---
    if engine.state.epoch == 0:
        model.eval()
        with torch.no_grad():
            m0 = evaluate(model, val_loader, config, torch.device("cuda"), engine)
        if engine.local_rank == 0:
            if engine.distributed:
                met0 = m0[0]
                for m in m0[1:]:
                    met0.update_hist(m.hist)
            else:
                met0 = m0
            _, miou0 = met0.compute_iou()
            logger.info(f"[n7c4] EPOCH0_IDENTITY val_mIoU={miou0:.4f} (expect ~55.72 +/-0.3 from init_from)")
    for epoch in range(engine.state.epoch, args.epochs):
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)
        p_drop = dropout_prob(epoch, args.dropout_max, args.dropout_ramp_epochs) if args.use_dropout else 0.0
        sum_loss = 0.0
        dataloader = iter(train_loader)
        for idx in range(niters):
            cur = epoch * niters + idx
            lr = lr_policy.get_lr(cur)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            mb = next(dataloader)
            imgs = mb["data"].cuda(non_blocking=True)
            gts = mb["label"].cuda(non_blocking=True)
            modal_xs = mb["modal_x"].cuda(non_blocking=True)

            # scheduled modality dropout: with prob p_drop, drop depth for the whole batch
            if p_drop > 0 and random.random() < p_drop:
                modal_in = torch.zeros_like(modal_xs)
                depth_avail = 0.0
            else:
                modal_in = modal_xs
                depth_avail = 1.0

            loss = model(imgs, modal_in, gts, depth_available=depth_avail)
            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            sum_loss += loss.item()

            if idx % 50 == 0 and engine.local_rank == 0:
                logger.info(f"[n7c4] epoch={epoch} iter={idx}/{niters} lr={lr:.6f} "
                            f"loss={loss.item():.4f} p_drop={p_drop:.3f}")

        if engine.local_rank == 0:
            logger.info(f"[n7c4] epoch={epoch} avg_loss={sum_loss/niters:.4f}")

        if (epoch + 1) % config.eval_iter == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                all_metrics = evaluate(model, val_loader, config, torch.device("cuda"), engine)
            if engine.local_rank == 0:
                if engine.distributed:
                    metric = all_metrics[0]
                    for m in all_metrics[1:]:
                        metric.update_hist(m.hist)
                else:
                    metric = all_metrics
                _, miou = metric.compute_iou()
                logger.info(f"[n7c4] epoch={epoch} val_mIoU={miou:.4f}")
                if miou > best_miou:
                    best_miou = miou
                    ckpt = os.path.join(config.log_dir, f"best_n7c4_{args.tag}.pth")
                    state = model.state_dict() if not engine.distributed else model.module.state_dict()
                    torch.save({"state_dict": state, "epoch": epoch, "miou": miou, "tag": args.tag}, ckpt)
                    logger.info(f"[n7c4] New best: {miou:.4f} -> {ckpt}")

        if engine.local_rank == 0 and (epoch + 1) % 50 == 0:
            ckpt = os.path.join(config.log_dir, f"checkpoint_ep{epoch+1}_n7c4_{args.tag}.pth")
            state = model.state_dict() if not engine.distributed else model.module.state_dict()
            torch.save({"state_dict": state, "epoch": epoch}, ckpt)
            logger.info(f"[n7c4] checkpoint saved: {ckpt}")

    if engine.local_rank == 0:
        logger.info(f"[n7c4] Training complete. Best mIoU: {best_miou:.4f}")
        final = os.path.join(config.log_dir, f"final_n7c4_{args.tag}.pth")
        state = model.state_dict() if not engine.distributed else model.module.state_dict()
        torch.save({"state_dict": state, "epoch": args.epochs - 1}, final)
        logger.info(f"[n7c4] Final model saved: {final}")
