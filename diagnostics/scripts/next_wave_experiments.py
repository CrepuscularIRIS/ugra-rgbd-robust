#!/usr/bin/env python3
"""Next-wave diagnostics (no method mainline):

1) Stage-3 channel-group (head-proxy) patching under entire_missing
2) D2 spatial 1x1 lens trajectories
3) Corruption severity sweeps (holes frac, gaussian sigma)
4) Optional DFormerv2-L stage recovery control
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))
os.chdir(ROOT / "repos" / "DFormer")

from diagnostics.corruptions.depth_corruptions import apply_corruption, rectangular_holes, gaussian_noise, entire_missing
from diagnostics.hooks.data_utils import get_nyu_val_loader, iter_limited
from diagnostics.hooks.model_utils import load_dformerv2_s_nyu
from diagnostics.pipeline_all import (
    accumulate_confmat,
    backbone_feats,
    decode_feats,
    miou_from_conf,
)


def load_dformerv2_l_nyu(device="cuda:0"):
    from importlib import import_module
    from models.builder import EncoderDecoder as segmodel

    config = getattr(import_module("local_configs.NYUDepthv2.DFormerv2_L"), "C")
    if not hasattr(config, "pad"):
        config.pad = False
    ckpt = ROOT / "repos" / "DFormer" / "DFormerv2_Large_NYU.pth"
    model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    state = torch.load(str(ckpt), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


def patch_channel_groups(feat_corr, feat_clean, group_ids: List[int], n_groups: int):
    """Replace selected channel groups in feat_corr with feat_clean."""
    b, c, h, w = feat_corr.shape
    assert c % n_groups == 0, f"C={c} not divisible by n_groups={n_groups}"
    gsz = c // n_groups
    out = feat_corr.clone()
    for g in group_ids:
        sl = slice(g * gsz, (g + 1) * gsz)
        out[:, sl] = feat_clean[:, sl]
    return out


@torch.no_grad()
def run_head_group_patch(model, loader, device, num=300, stage=3, n_groups=16, batch_size=2):
    K = 40
    conf_clean = torch.zeros(K, K, device=device)
    conf_miss = torch.zeros(K, K, device=device)
    conf_one = {g: torch.zeros(K, K, device=device) for g in range(n_groups)}
    conf_topm = {m: torch.zeros(K, K, device=device) for m in [1, 2, 4, 8]}

    # first pass: single-group recovery
    n = 0
    for rgb, depth, label in iter_limited(loader, num):
        rgb, depth, label = rgb.float().to(device), depth.float().to(device), label.long().to(device)
        hw = rgb.shape[-2:]
        fc = backbone_feats(model, rgb, depth)
        fm = backbone_feats(model, rgb, entire_missing(depth))
        accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf_clean, K)
        accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf_miss, K)
        for g in range(n_groups):
            hybrid = list(fm)
            hybrid[stage] = patch_channel_groups(fm[stage], fc[stage], [g], n_groups)
            accumulate_confmat(decode_feats(model, hybrid, hw).argmax(1), label, conf_one[g], K)
        n += rgb.shape[0]
        if n % 100 == 0:
            print(f"  head-group pass1 {n}/{num}")

    mc, mm = miou_from_conf(conf_clean), miou_from_conf(conf_miss)

    def rec(m):
        return float((m - mm) / max(mc - mm, 1e-6))

    one = {str(g): {"miou": miou_from_conf(conf_one[g]), "recovery": rec(miou_from_conf(conf_one[g]))} for g in range(n_groups)}
    order = sorted(range(n_groups), key=lambda g: one[str(g)]["recovery"], reverse=True)
    print("  top groups", order[:8])

    # second pass: top-m groups
    conf_topm = {m: torch.zeros(K, K, device=device) for m in [1, 2, 4, 8]}
    n = 0
    for rgb, depth, label in iter_limited(loader, num):
        rgb, depth, label = rgb.float().to(device), depth.float().to(device), label.long().to(device)
        hw = rgb.shape[-2:]
        fc = backbone_feats(model, rgb, depth)
        fm = backbone_feats(model, rgb, entire_missing(depth))
        for m in conf_topm:
            hybrid = list(fm)
            hybrid[stage] = patch_channel_groups(fm[stage], fc[stage], order[:m], n_groups)
            accumulate_confmat(decode_feats(model, hybrid, hw).argmax(1), label, conf_topm[m], K)
        n += rgb.shape[0]
        if n % 100 == 0:
            print(f"  head-group pass2 {n}/{num}")

    topm = {str(m): {"miou": miou_from_conf(conf_topm[m]), "recovery": rec(miou_from_conf(conf_topm[m]))} for m in conf_topm}
    return {
        "stage": stage,
        "n_groups": n_groups,
        "clean_miou": mc,
        "missing_miou": mm,
        "single_group": one,
        "group_order_by_recovery": order,
        "top_m_groups": topm,
    }


@torch.no_grad()
def run_severity_sweep(model, loader, device, num=300):
    K = 40
    hole_fracs = [0.05, 0.10, 0.15, 0.25, 0.40]
    sigmas = [0.05, 0.10, 0.15, 0.25, 0.40]
    conf_h = {f: torch.zeros(K, K, device=device) for f in hole_fracs}
    conf_g = {s: torch.zeros(K, K, device=device) for s in sigmas}
    conf_c = torch.zeros(K, K, device=device)
    conf_m = torch.zeros(K, K, device=device)
    n = 0
    for rgb, depth, label in iter_limited(loader, num):
        rgb, depth, label = rgb.float().to(device), depth.float().to(device), label.long().to(device)
        hw = rgb.shape[-2:]
        fc = backbone_feats(model, rgb, depth)
        accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf_c, K)
        accumulate_confmat(
            decode_feats(model, backbone_feats(model, rgb, entire_missing(depth)), hw).argmax(1),
            label,
            conf_m,
            K,
        )
        for f in hole_fracs:
            d = rectangular_holes(depth, num_holes=3, hole_frac=f, seed=0)
            accumulate_confmat(
                decode_feats(model, backbone_feats(model, rgb, d), hw).argmax(1), label, conf_h[f], K
            )
        for s in sigmas:
            d = gaussian_noise(depth, sigma=s, seed=0)
            accumulate_confmat(
                decode_feats(model, backbone_feats(model, rgb, d), hw).argmax(1), label, conf_g[s], K
            )
        n += rgb.shape[0]
        if n % 100 == 0:
            print(f"  severity {n}/{num}")
    return {
        "clean_miou": miou_from_conf(conf_c),
        "entire_missing_miou": miou_from_conf(conf_m),
        "holes_by_frac": {str(f): miou_from_conf(conf_h[f]) for f in hole_fracs},
        "gaussian_by_sigma": {str(s): miou_from_conf(conf_g[s]) for s in sigmas},
    }


def run_spatial_lens(model, loader, device, num=200, stages=(0, 1, 2, 3)):
    """Train per-stage 1x1 linear probe on clean; eval mIoU under conditions."""
    feats_train = {s: [] for s in stages}
    labels_train = {s: [] for s in stages}
    n = 0
    for rgb, depth, label in iter_limited(loader, num // 2):
        rgb, depth, label = rgb.float().to(device), depth.float().to(device), label.long().to(device)
        with torch.no_grad():
            feats = backbone_feats(model, rgb, depth)
        for s in stages:
            lab = F.interpolate(label.unsqueeze(1).float(), size=feats[s].shape[-2:], mode="nearest").squeeze(1).long()
            feats_train[s].append(feats[s].detach().cpu())
            labels_train[s].append(lab.cpu())
        n += rgb.shape[0]
    probes = {}
    for s in stages:
        X = torch.cat(feats_train[s], 0)  # N,C,H,W
        Y = torch.cat(labels_train[s], 0)  # N,H,W
        c = X.shape[1]
        xf = X.permute(0, 2, 3, 1).reshape(-1, c)
        yf = Y.reshape(-1)
        valid = yf != 255
        xf, yf = xf[valid], yf[valid]
        if xf.shape[0] > 200_000:
            idx = torch.randperm(xf.shape[0])[:200_000]
            xf, yf = xf[idx], yf[idx]
        K = 40
        Yoh = F.one_hot(yf.clamp(0, K - 1), K).float()
        reg = 1e-2
        xf = xf.float()
        A = xf.T @ xf + reg * torch.eye(c)
        W = torch.linalg.solve(A, xf.T @ Yoh)  # C x K
        b = Yoh.mean(0) - xf.mean(0) @ W
        probes[s] = (W.to(device), b.to(device))
        print(f"  lens stage{s} trained on {xf.shape[0]} pixels")

    # eval trajectories
    conditions = {
        "clean": lambda d: d,
        "entire_missing": lambda d: entire_missing(d),
        "holes": lambda d: rectangular_holes(d, 3, 0.25, seed=0),
        "gaussian": lambda d: gaussian_noise(d, 0.25, seed=0),
    }
    conf = {name: {s: torch.zeros(40, 40, device=device) for s in stages} for name in conditions}
    n = 0
    for rgb, depth, label in iter_limited(loader, num):
        rgb, depth, label = rgb.float().to(device), depth.float().to(device), label.long().to(device)
        for name, fn in conditions.items():
            with torch.no_grad():
                feats = backbone_feats(model, rgb, fn(depth))
            for s in stages:
                W, b = probes[s]
                f = feats[s]
                # logits [B,K,H,W]
                logits = torch.einsum("bchw,ck->bkhw", f.float(), W) + b.view(1, -1, 1, 1)
                lab = F.interpolate(label.unsqueeze(1).float(), size=f.shape[-2:], mode="nearest").squeeze(1).long()
                accumulate_confmat(logits.argmax(1), lab, conf[name][s], 40)
        n += rgb.shape[0]
        if n % 100 == 0:
            print(f"  lens eval {n}/{num}")
    traj = {
        name: {str(s): miou_from_conf(conf[name][s]) for s in stages} for name in conditions
    }
    # fork stage max clean-missing
    fork = max(stages, key=lambda s: traj["clean"][str(s)] - traj["entire_missing"][str(s)])
    return {"trajectories": traj, "fork_stage": fork}


@torch.no_grad()
def run_l_stage_control(device, num=200):
    print("==> load DFormerv2-L")
    model, config = load_dformerv2_l_nyu(device=device)
    loader = get_nyu_val_loader(config, batch_size=2)
    K = config.num_classes
    stages = 4
    conf_c = torch.zeros(K, K, device=device)
    conf_m = torch.zeros(K, K, device=device)
    conf_dn = {s: torch.zeros(K, K, device=device) for s in range(stages)}
    n = 0
    for rgb, depth, label in iter_limited(loader, num):
        rgb, depth, label = rgb.float().to(device), depth.float().to(device), label.long().to(device)
        hw = rgb.shape[-2:]
        fc = backbone_feats(model, rgb, depth)
        fm = backbone_feats(model, rgb, entire_missing(depth))
        accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf_c, K)
        accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf_m, K)
        for s in range(stages):
            hybrid = [fm[j] if j != s else fc[s] for j in range(stages)]
            accumulate_confmat(decode_feats(model, hybrid, hw).argmax(1), label, conf_dn[s], K)
        n += rgb.shape[0]
        if n % 50 == 0:
            print(f"  L control {n}/{num}")
    mc, mm = miou_from_conf(conf_c), miou_from_conf(conf_m)

    def rec(m):
        return float((m - mm) / max(mc - mm, 1e-6))

    return {
        "clean_miou": mc,
        "missing_miou": mm,
        "denoising_recovery_by_stage": {
            str(s): {"miou": miou_from_conf(conf_dn[s]), "recovery": rec(miou_from_conf(conf_dn[s]))}
            for s in range(stages)
        },
    }


def main():
    device = "cuda:0"
    out_dir = ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("==> load S")
    model_s, cfg_s = load_dformerv2_s_nyu(device=device)
    loader_s = get_nyu_val_loader(cfg_s, batch_size=2)

    print("==> A: head-group patching stage3")
    head = run_head_group_patch(model_s, loader_s, device, num=300, stage=3, n_groups=16)

    print("==> B: severity sweep")
    sev = run_severity_sweep(model_s, loader_s, device, num=300)

    print("==> C: spatial lens")
    lens = run_spatial_lens(model_s, loader_s, device, num=200)

    print("==> D: L control")
    try:
        lctl = run_l_stage_control(device, num=200)
    except Exception as e:
        lctl = {"error": str(e)}

    results = {
        "seconds": time.time() - t0,
        "head_group_patch_stage3": head,
        "severity_sweep": sev,
        "spatial_lens": lens,
        "L_stage_control": lctl,
    }
    path = out_dir / "next_wave_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
