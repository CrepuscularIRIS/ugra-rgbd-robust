"""n7 training: DFormerv2-S + RGB Decomp Adapter + Depth Witness.

Minimal modification of the standard DFormer training script. Changes:
  1. Wraps the base segmodel with DFormerv2WithAdapter
  2. Adds adapter/witness/seg_aux params to the optimizer
  3. Disables AMP (USE_AMP=0 — gradient overflow on this architecture)
  4. Everything else (data, optimizer, scheduler, val, checkpointing) is standard

Usage (from repos/DFormer/):
  torchrun --nproc_per_node=2 ../../experiments/n7_decomp_witness/train_n7.py \
    --config=local_configs.NYUDepthv2.DFormerv2_S --gpus=2 --epochs=500 \
    --no-amp --syncbn --no-compile \
    --continue_fpath=checkpoints/pretrained/DFormerv2_Small_pretrained.pth
"""
import argparse
import os
import sys
import random
import time
import datetime
import pprint

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
from model_wrapper import DFormerv2WithAdapter
from val_mm import evaluate, evaluate_msf
from utils.dataloader.dataloader import get_train_loader, get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset
from utils.engine.engine import Engine
from utils.engine.logger import get_logger
from utils.init_func import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.pyt_utils import all_reduce_tensor

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="train config file path")
parser.add_argument("--gpus", default=2, type=int)
parser.add_argument("-v", "--verbose", default=False, action="store_true")
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--show_image", "-s", default=False, action="store_true")
parser.add_argument("--save_path", default=None)
parser.add_argument("--checkpoint_dir")
parser.add_argument("--continue_fpath")
parser.add_argument("--sliding", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--compile", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--syncbn", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--mst", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--amp", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--pad_SUNRGBD", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--use_seed", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--local-rank", default=0)
parser.add_argument("--abort_on_nan", default=True, action=argparse.BooleanOptionalAction)
parser.add_argument("--clip_grad_norm", type=float, default=1.0)
parser.add_argument("--witness_weight", type=float, default=0.05)
parser.add_argument("--seg_aux_weight", type=float, default=0.2)
parser.add_argument("--adapter_stage_idx", type=int, default=2)
parser.add_argument("--log_dir", type=str, default=None)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

with Engine(custom_parser=parser) as engine:
    args = parser.parse_args()
    config = getattr(import_module(args.config), "C")
    if not hasattr(config, 'pad'):
        config.pad = False
    logger = get_logger(config.log_dir, config.log_file, rank=engine.local_rank)
    logger.info(f"[n7] Training with RGB Decomp Adapter + Depth Witness")
    logger.info(f"[n7] witness_weight={args.witness_weight}, seg_aux_weight={args.seg_aux_weight}")
    logger.info(f"[n7] AMP={args.amp} (should be False for this architecture)")

    if args.use_seed:
        seed = 20260721
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    BatchNorm2d = nn.SyncBatchNorm if args.syncbn else nn.BatchNorm2d

    base_model = segmodel(cfg=config, criterion=criterion, norm_layer=BatchNorm2d, syncbn=args.syncbn)
    model = DFormerv2WithAdapter(
        base_model,
        num_classes=config.num_classes,
        adapter_stage_idx=args.adapter_stage_idx,
        witness_weight=args.witness_weight,
        seg_aux_weight=args.seg_aux_weight,
    )
    adapter_params = sum(p.numel() for p in model.adapter.parameters()) + \
                     sum(p.numel() for p in model.witness_head.parameters()) + \
                     sum(p.numel() for p in model.seg_aux_head.parameters())
    logger.info(f"[n7] Added {adapter_params:,} adapter+aux params ({adapter_params/1e6:.2f}M)")

    base_lr = config.lr
    params_list = []
    params_list = group_weight(params_list, model.base, BatchNorm2d, base_lr)
    params_list.append({"params": list(model.adapter.parameters()) +
                                  list(model.witness_head.parameters()) +
                                  list(model.seg_aux_head.parameters()),
                        "lr": base_lr})

    optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999),
                                  weight_decay=config.weight_decay)

    val_batch_size = int(config.batch_size)
    train_loader, train_sampler = get_train_loader(engine, RGBXDataset, config)
    val_loader, val_sampler = get_val_loader(engine, RGBXDataset, config, val_batch_size=val_batch_size)

    if engine.distributed:
        model.cuda()
        model = DistributedDataParallel(model, device_ids=[engine.local_rank],
                                        output_device=engine.local_rank,
                                        find_unused_parameters=True)
    else:
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    engine.register_state(dataloader=train_loader, model=model, optimizer=optimizer)
    if engine.continue_state_object:
        engine.restore_checkpoint()

    niters_per_epoch = config.niters_per_epoch
    total_iters = niters_per_epoch * args.epochs
    lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iters, niters_per_epoch * config.warm_up_epoch)

    optimizer.zero_grad()
    logger.info(f"[n7] Starting training: {args.epochs} epochs, {niters_per_epoch} iters/epoch")

    best_miou = 0
    for epoch in range(engine.state.epoch, args.epochs):
        model.train()
        if engine.distributed:
            train_sampler.set_epoch(epoch)

        bar_format = "{desc}[{elapsed}<{remaining},{rate_fmt}]"
        sum_loss = 0
        dataloader = iter(train_loader)
        for idx in range(niters_per_epoch):
            current_idx = epoch * niters_per_epoch + idx
            lr = lr_policy.get_lr(current_idx)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            minibatch = next(dataloader)
            imgs = minibatch["data"]
            gts = minibatch["label"]
            modal_xs = minibatch["modal_x"]

            imgs = imgs.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            modal_xs = modal_xs.cuda(non_blocking=True)

            loss = model(imgs, modal_xs, gts)
            loss.backward()

            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            sum_loss += loss.item()

            if idx % 50 == 0 and engine.local_rank == 0:
                logger.info(f"[n7] epoch={epoch} iter={idx}/{niters_per_epoch} lr={lr:.6f} loss={loss.item():.4f}")

        avg_loss = sum_loss / niters_per_epoch
        if engine.local_rank == 0:
            logger.info(f"[n7] epoch={epoch} avg_loss={avg_loss:.4f}")

        if (epoch + 1) % config.eval_iter == 0 or epoch == args.epochs - 1:
            model.eval()
            device = torch.device("cuda")
            with torch.no_grad():
                all_metrics = evaluate(model, val_loader, config, device, engine)

            if engine.local_rank == 0:
                if engine.distributed:
                    metric = all_metrics[0]
                    for other_metric in all_metrics[1:]:
                        metric.update_hist(other_metric.hist)
                else:
                    metric = all_metrics
                ious, miou = metric.compute_iou()
                logger.info(f"[n7] epoch={epoch} val_mIoU={miou:.4f}")
                if miou > best_miou:
                    best_miou = miou
                    ckpt_path = os.path.join(config.log_dir, "best_n7.pth")
                    state = model.state_dict() if not engine.distributed else model.module.state_dict()
                    torch.save({"state_dict": state, "epoch": epoch, "miou": miou}, ckpt_path)
                    logger.info(f"[n7] New best: {miou:.4f} saved to {ckpt_path}")

        if engine.local_rank == 0 and (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(config.log_dir, f"checkpoint_ep{epoch+1}_n7.pth")
            state = model.state_dict() if not engine.distributed else model.module.state_dict()
            torch.save({"state_dict": state, "epoch": epoch}, ckpt_path)
            logger.info(f"[n7] Checkpoint saved: {ckpt_path}")

    if engine.local_rank == 0:
        logger.info(f"[n7] Training complete. Best mIoU: {best_miou:.4f}")
        final_path = os.path.join(config.log_dir, "final_n7.pth")
        state = model.state_dict() if not engine.distributed else model.module.state_dict()
        torch.save({"state_dict": state, "epoch": args.epochs - 1}, final_path)
        logger.info(f"[n7] Final model saved: {final_path}")
