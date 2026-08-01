#!/usr/bin/env python3
"""Run Phase-0 + D1 + D2 + D3 + D5 (+ optional C1 coefficient fit) end-to-end.

Usage (from ugra-rgbd-robust, GPU1):
  CUDA_VISIBLE_DEVICES=1 conda run -n dformer python diagnostics/pipeline_all.py --num 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))
os.chdir(ROOT / "repos" / "DFormer")

from diagnostics.corruptions.depth_corruptions import apply_corruption, default_corruption_suite
from diagnostics.hooks.data_utils import get_nyu_val_loader, iter_limited
from diagnostics.hooks.model_utils import load_dformerv2_s_nyu, logits_to_miou


def decode_feats(model, feats, out_hw) -> torch.Tensor:
    out = model.decode_head.forward(feats)
    logits = out["pred"] if isinstance(out, dict) and "pred" in out else out
    return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def backbone_feats(model, rgb, depth):
    feats = model.backbone(rgb, depth)
    return list(feats) if isinstance(feats, (list, tuple)) else [feats]


def accumulate_confmat(pred, label, conf, num_classes=40, ignore=255):
    pred = pred.view(-1)
    label = label.view(-1)
    valid = label != ignore
    pred = pred[valid]
    label = label[valid]
    idx = label * num_classes + pred
    conf += torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def miou_from_conf(conf: torch.Tensor) -> float:
    conf = conf.float()
    inter = conf.diag()
    union = conf.sum(0) + conf.sum(1) - inter
    valid = union > 0
    if valid.sum() == 0:
        return 0.0
    return float((inter[valid] / union[valid]).mean().item())


def pool_gap(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    return [f.float().mean(dim=(2, 3)) for f in feats]  # [B,C]


def project_delta_topk(delta: torch.Tensor, k: int) -> torch.Tensor:
    """delta [B,C,H,W] -> rank-k reconstruction via per-batch SVD on C x (HW)."""
    b, c, h, w = delta.shape
    out = []
    for i in range(b):
        x = delta[i].reshape(c, -1)  # C x HW
        # center optional
        try:
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
        except RuntimeError:
            out.append(delta[i])
            continue
        kk = min(k, S.numel())
        recon = (U[:, :kk] * S[:kk]) @ Vh[:kk, :]
        out.append(recon.reshape(c, h, w))
    return torch.stack(out, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--out", type=str, default=str(ROOT / "results" / "diagnostics"))
    ap.add_argument("--ranks", type=str, default="1,2,4,8,16,32,64")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    ranks = [int(x) for x in args.ranks.split(",")]

    t0 = time.time()
    print("==> load model")
    model, config = load_dformerv2_s_nyu(device=device)
    loader = get_nyu_val_loader(config, batch_size=args.batch_size)
    suite = default_corruption_suite(seed=20260721)
    corruptions = list(suite.keys())  # entire_missing, holes, gaussian
    conditions = ["clean"] + corruptions
    num_classes = config.num_classes
    num_stages = 4

    # Accumulators
    conf_base = {c: torch.zeros(num_classes, num_classes, device=device) for c in conditions}
    # D1 denoising: patch clean stage s into corrupted run
    conf_d1 = {
        corr: {s: torch.zeros(num_classes, num_classes, device=device) for s in range(num_stages)}
        for corr in corruptions
    }
    conf_d1_all = {corr: torch.zeros(num_classes, num_classes, device=device) for corr in corruptions}
    # D1 noising: patch corrupted stage s into clean run
    conf_d1_noise = {
        corr: {s: torch.zeros(num_classes, num_classes, device=device) for s in range(num_stages)}
        for corr in corruptions
    }
    # D3 rank-restricted for entire_missing on best stage later; compute for all stages x ranks on entire_missing
    conf_d3 = {
        s: {k: torch.zeros(num_classes, num_classes, device=device) for k in ranks} for s in range(num_stages)
    }

    # D2 lens: collect GAP feats + labels (downsampled) — store lists then fit linear probes
    # For efficiency store per-stage GAP and global label histogram is not enough; use linear probe on GAP only
    # (coarse lens). Also store stage-mean logits probe via 1x1 on full features is heavy.
    # Use GAP + linear classifier trained on clean, eval on all conditions.
    gap_store = {c: [[] for _ in range(num_stages)] for c in conditions}
    label_store = []  # one global label id via majority? better: store per-image mean one-hot of classes present
    # For GAP linear probe we need a single target per image — use multi-label BCE of class presence
    presence_store = []  # list of [K] float vectors

    # D5: collect GAP mean clean vs entire_missing at each stage
    gap_clean_sum = [None] * num_stages
    gap_miss_sum = [None] * num_stages
    gap_count = 0

    n_done = 0
    print(f"==> streaming {args.num} images")
    for rgb, depth, label in iter_limited(loader, args.num):
        rgb = rgb.float().to(device)
        depth = depth.float().to(device)
        label = label.long().to(device)
        b = rgb.shape[0]
        out_hw = rgb.shape[-2:]

        with torch.no_grad():
            feats_clean = backbone_feats(model, rgb, depth)
            logits_clean = decode_feats(model, feats_clean, out_hw)
            pred_clean = logits_clean.argmax(1)
            accumulate_confmat(pred_clean, label, conf_base["clean"], num_classes)

            # class presence for multi-label probe
            for i in range(b):
                pres = torch.zeros(num_classes, device=device)
                labs = label[i].unique()
                for c in labs.tolist():
                    if 0 <= c < num_classes:
                        pres[c] = 1.0
                presence_store.append(pres.cpu())

            gaps_c = pool_gap(feats_clean)
            for s in range(num_stages):
                gap_store["clean"][s].append(gaps_c[s].cpu())
                if gap_clean_sum[s] is None:
                    gap_clean_sum[s] = gaps_c[s].sum(0)
                else:
                    gap_clean_sum[s] += gaps_c[s].sum(0)

            feats_corr_all = {}
            for corr in corruptions:
                d_c = apply_corruption(depth, corr, **suite[corr])
                feats_c = backbone_feats(model, rgb, d_c)
                feats_corr_all[corr] = feats_c
                logits_c = decode_feats(model, feats_c, out_hw)
                accumulate_confmat(logits_c.argmax(1), label, conf_base[corr], num_classes)
                gaps = pool_gap(feats_c)
                for s in range(num_stages):
                    gap_store[corr][s].append(gaps[s].cpu())

                # D1 denoising: replace stage s with clean
                for s in range(num_stages):
                    hybrid = [feats_c[j] if j != s else feats_clean[s] for j in range(num_stages)]
                    logits_h = decode_feats(model, hybrid, out_hw)
                    accumulate_confmat(logits_h.argmax(1), label, conf_d1[corr][s], num_classes)
                # all stages clean feats (oracle upper for encoder)
                logits_all = decode_feats(model, feats_clean, out_hw)
                accumulate_confmat(logits_all.argmax(1), label, conf_d1_all[corr], num_classes)

                # D1 noising
                for s in range(num_stages):
                    hybrid = [feats_clean[j] if j != s else feats_c[s] for j in range(num_stages)]
                    logits_h = decode_feats(model, hybrid, out_hw)
                    accumulate_confmat(logits_h.argmax(1), label, conf_d1_noise[corr][s], num_classes)

            # D3 on entire_missing
            feats_m = feats_corr_all["entire_missing"]
            for s in range(num_stages):
                delta = feats_clean[s] - feats_m[s]
                for k in ranks:
                    delta_k = project_delta_topk(delta, k)
                    hybrid = list(feats_m)
                    hybrid[s] = feats_m[s] + delta_k
                    logits_h = decode_feats(model, hybrid, out_hw)
                    accumulate_confmat(logits_h.argmax(1), label, conf_d3[s][k], num_classes)

            # D5 sums for missing
            gaps_m = pool_gap(feats_corr_all["entire_missing"])
            for s in range(num_stages):
                if gap_miss_sum[s] is None:
                    gap_miss_sum[s] = gaps_m[s].sum(0)
                else:
                    gap_miss_sum[s] += gaps_m[s].sum(0)
            gap_count += b

        n_done += b
        if n_done % 20 == 0 or n_done >= args.num:
            print(f"  progress {min(n_done, args.num)}/{args.num}  elapsed {time.time()-t0:.1f}s")

    # ----- summarize base + D1 -----
    base_miou = {c: miou_from_conf(conf_base[c]) for c in conditions}
    results = {"base_miou": base_miou, "num_images": min(n_done, args.num), "seconds": time.time() - t0}

    def recovery(m_patch, m_corr, m_clean):
        den = m_clean - m_corr
        if abs(den) < 1e-6:
            return 0.0
        return float((m_patch - m_corr) / den)

    d1 = {}
    for corr in corruptions:
        m_corr = base_miou[corr]
        m_clean = base_miou["clean"]
        d1[corr] = {
            "corrupted_miou": m_corr,
            "clean_miou": m_clean,
            "denoising_recovery_by_stage": {},
            "noising_damage_by_stage": {},
            "oracle_clean_feats_miou": miou_from_conf(conf_d1_all[corr]),
        }
        for s in range(num_stages):
            m_dn = miou_from_conf(conf_d1[corr][s])
            m_ns = miou_from_conf(conf_d1_noise[corr][s])
            d1[corr]["denoising_recovery_by_stage"][str(s)] = {
                "miou": m_dn,
                "recovery": recovery(m_dn, m_corr, m_clean),
            }
            d1[corr]["noising_damage_by_stage"][str(s)] = {
                "miou": m_ns,
                "relative_drop": float((m_clean - m_ns) / max(m_clean, 1e-6)),
            }
    results["D1"] = d1

    # best stage for entire_missing by denoising recovery
    best_stage = max(
        range(num_stages),
        key=lambda s: d1["entire_missing"]["denoising_recovery_by_stage"][str(s)]["recovery"],
    )
    results["best_stage_entire_missing"] = best_stage

    # ----- D3 curves -----
    d3 = {}
    for s in range(num_stages):
        d3[str(s)] = {}
        m_corr = base_miou["entire_missing"]
        m_clean = base_miou["clean"]
        for k in ranks:
            m_k = miou_from_conf(conf_d3[s][k])
            d3[str(s)][str(k)] = {
                "miou": m_k,
                "recovery": recovery(m_k, m_corr, m_clean),
            }
    results["D3_rank_restricted_entire_missing"] = d3

    # causal low-rank? recovery at k=16 vs k=64 at best stage
    r16 = d3[str(best_stage)].get("16", {}).get("recovery", 0)
    r64 = d3[str(best_stage)].get("64", {}).get("recovery", 0)
    r8 = d3[str(best_stage)].get("8", {}).get("recovery", 0)
    low_rank_ok = r16 >= 0.5 * max(r64, 1e-6) and r8 >= 0.3 * max(r64, 1e-6)
    results["D3_low_rank_support_C1"] = {
        "best_stage": best_stage,
        "recovery_k8": r8,
        "recovery_k16": r16,
        "recovery_k64": r64,
        "go_c1": bool(low_rank_ok or r16 >= 0.15),
    }

    # ----- D2 multi-label linear probes on GAP -----
    print("==> D2 linear multi-label probes on GAP")
    presence = torch.stack(presence_store, dim=0).float()  # [N,K]
    d2 = {}
    for cond in conditions:
        d2[cond] = {}
    for s in range(num_stages):
        # train on clean only
        Xc = torch.cat(gap_store["clean"][s], dim=0).float()  # [N,C]
        N = min(Xc.shape[0], presence.shape[0])
        Xc = Xc[:N]
        Y = presence[:N]
        # ridge multi-label
        Cdim = Xc.shape[1]
        reg = 1e-2
        # solve (X^T X + reg I) W = X^T Y
        XtX = Xc.T @ Xc + reg * torch.eye(Cdim)
        XtY = Xc.T @ Y
        W = torch.linalg.solve(XtX, XtY)  # [C,K]
        b = Y.mean(0) - Xc.mean(0) @ W
        for cond in conditions:
            X = torch.cat(gap_store[cond][s], dim=0).float()[:N]
            logits = X @ W + b
            pred = (torch.sigmoid(logits) > 0.5).float()
            # macro F1-ish: per-class IoU of presence
            inter = (pred * Y).sum(0)
            union = ((pred + Y) > 0).float().sum(0).clamp_min(1)
            score = float((inter / union).mean().item())
            d2[cond][str(s)] = score
    results["D2_gap_multilabel_macro_iou"] = d2
    # fork stage: max clean-corrupt gap for entire_missing
    fork = max(
        range(num_stages),
        key=lambda s: d2["clean"][str(s)] - d2["entire_missing"][str(s)],
    )
    results["D2_fork_stage_entire_missing"] = fork

    # ----- D5 trust direction -----
    print("==> D5 trust directions")
    d5 = {}
    for s in range(num_stages):
        mean_c = gap_clean_sum[s] / max(gap_count, 1)
        mean_m = gap_miss_sum[s] / max(gap_count, 1)
        direction = mean_c - mean_m
        direction = direction / (direction.norm() + 1e-8)
        d5[str(s)] = {
            "direction_norm": float((mean_c - mean_m).norm().item()),
            "cosine_clean_vs_miss": float(
                F.cosine_similarity(mean_c.unsqueeze(0), mean_m.unsqueeze(0)).item()
            ),
        }
        # save direction
        torch.save(direction.cpu(), out_dir / f"trust_dir_stage{s}.pt")
    results["D5"] = d5

    # ----- C1 light coefficient fit (if go) -----
    c1_report = {"skipped": True}
    if results["D3_low_rank_support_C1"]["go_c1"]:
        print("==> C1: fit linear coeff predictor from RGB GAP (clean stage) to top-k coeffs")
        s = best_stage
        k = 16
        # Rebuild deltas on a second pass is expensive; approximate using stored GAP only:
        # Not ideal for full C1 but fit predictor of (gap_clean - gap_miss) PCA coeffs from gap_clean as proxy.
        Xc = torch.cat(gap_store["clean"][s], dim=0).float()
        Xm = torch.cat(gap_store["entire_missing"][s], dim=0).float()
        N = min(Xc.shape[0], Xm.shape[0])
        Xc, Xm = Xc[:N], Xm[:N]
        Delta = Xc - Xm
        # PCA via SVD
        U, S, Vh = torch.linalg.svd(Delta - Delta.mean(0, keepdim=True), full_matrices=False)
        kk = min(k, Vh.shape[0])
        B = Vh[:kk].T  # [C,k]
        coeffs = Delta @ B  # [N,k]
        # predict coeffs from clean GAP (or miss GAP): use miss GAP as available at test missing
        X = Xm
        reg = 1e-2
        A = X.T @ X + reg * torch.eye(X.shape[1])
        W = torch.linalg.solve(A, X.T @ coeffs)
        pred_c = X @ W
        mse = float(((pred_c - coeffs) ** 2).mean().item())
        # reconstruction quality of delta
        recon = pred_c @ B.T
        cos = float(F.cosine_similarity(recon, Delta, dim=-1).mean().item())
        c1_report = {
            "skipped": False,
            "stage": s,
            "k": kk,
            "coeff_mse": mse,
            "delta_recon_cosine": cos,
            "note": "GAP-level linear C1 prototype; full spatial Ba(x) inject is next step",
        }
        torch.save({"B": B, "W": W, "stage": s, "k": kk}, out_dir / "c1_gap_prototype.pt")
    results["C1_prototype"] = c1_report

    results["seconds_total"] = time.time() - t0
    out_json = out_dir / "pipeline_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", out_json)
    print(f"DONE in {results['seconds_total']:.1f}s")


if __name__ == "__main__":
    main()
