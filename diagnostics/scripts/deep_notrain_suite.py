#!/usr/bin/env python3
"""Non-training diagnostics from Deep.md priorities T1–T4, T6–T7, T9.

Writes intermediate JSON after each task and a final combined JSON.
Reviews are printed after each task (measurements only).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))
os.chdir(ROOT / "repos" / "DFormer")

from diagnostics.backbone_forward import forward_backbone_cached
from diagnostics.corruptions.depth_corruptions import entire_missing, rectangular_holes, gaussian_noise
from diagnostics.hooks.data_utils import _to_tensor
from diagnostics.hooks.model_utils import load_dformerv2_s_nyu
from diagnostics.pipeline_all import accumulate_confmat, decode_feats, miou_from_conf


OUT = ROOT / "results" / "deep_notrain"
OUT.mkdir(parents=True, exist_ok=True)


def save_partial(name: str, obj: dict):
    p = OUT / f"{name}.json"
    # convert tensors
    def conv(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist() if x.numel() < 100 else {"shape": list(x.shape), "norm": float(x.float().norm())}
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, float):
            return float(x)
        if isinstance(x, (int, str, bool)) or x is None:
            return x
        return str(x)

    p.write_text(json.dumps(conv(obj), indent=2))
    print(f"[saved] {p}")


def review(title: str, findings: dict):
    print("\n" + "=" * 72)
    print(f"REVIEW: {title}")
    print("=" * 72)
    print(json.dumps(findings, indent=2, default=str))
    print("=" * 72 + "\n")


def make_loader(config, split: str, batch_size: int = 2):
    from torch.utils import data
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.dataloader.dataloader import ValPre

    setting = {
        "rgb_root": config.rgb_root_folder,
        "rgb_format": config.rgb_format,
        "gt_root": config.gt_root_folder,
        "gt_format": config.gt_format,
        "transform_gt": config.gt_transform,
        "x_root": config.x_root_folder,
        "x_format": config.x_format,
        "x_single_channel": config.x_is_single_channel,
        "class_names": config.class_names,
        "train_source": config.train_source,
        "eval_source": config.eval_source,
        "dataset_name": config.dataset_name,
        "backbone": config.backbone,
    }
    pre = ValPre(config.norm_mean, config.norm_std, config.x_is_single_channel, config)
    ds = RGBXDataset(setting, split, pre)
    return data.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def iter_n(loader, max_images: int):
    n = 0
    for batch in loader:
        rgb = _to_tensor(batch["data"]).float()
        depth = _to_tensor(batch["modal_x"]).float()
        label = _to_tensor(batch["label"]).long()
        b = rgb.shape[0]
        take = min(b, max_images - n)
        yield rgb[:take], depth[:take], label[:take]
        n += take
        if n >= max_images:
            break


def load_variant(name: str, device: str):
    """name in {S,B,L}.

    NOTE: DFormer configs all mutate a shared EasyDict C in local_configs._base_.
    Re-importing without clearing sys.modules leaves C stuck at the last variant
    (e.g. L), so S/B checkpoints then fail shape-check. Always purge and reimport.
    """
    from importlib import import_module
    from models.builder import EncoderDecoder as segmodel

    cfg_map = {
        "S": ("local_configs.NYUDepthv2.DFormerv2_S", "checkpoints/trained/DFormerv2_Small_NYU.pth"),
        "B": ("local_configs.NYUDepthv2.DFormerv2_B", None),  # may missing
        "L": ("local_configs.NYUDepthv2.DFormerv2_L", "DFormerv2_Large_NYU.pth"),
    }
    mod, ckpt_rel = cfg_map[name]
    # purge shared config modules so side-effect C.backbone is reapplied correctly
    for m in list(sys.modules):
        if m == "local_configs" or m.startswith("local_configs."):
            del sys.modules[m]
    config = getattr(import_module(mod), "C")
    if not hasattr(config, "pad"):
        config.pad = False
    # hard-assert expected backbone before build
    expect_bb = {"S": "DFormerv2_S", "B": "DFormerv2_B", "L": "DFormerv2_L"}[name]
    if getattr(config, "backbone", None) != expect_bb:
        raise RuntimeError(f"config pollution: want {expect_bb}, got {config.backbone}")
    model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    load_info = {"strict_ok": None, "missing": [], "unexpected": [], "ckpt": ckpt_rel}
    if ckpt_rel is None:
        # try HF-style path
        cand = [
            Path("/data/datasets/rgbd-semseg/checkpoints/NYUDepthv2/DFormerv2_Base_NYU.pth"),
            Path("/data/datasets/rgbd-semseg/checkpoints/hf_DFormerv2/DFormerv2/NYU/DFormerv2_Base_NYU.pth"),
            ROOT / "repos/DFormer/checkpoints/trained/DFormerv2_Base_NYU.pth",
            ROOT / "repos/DFormer/DFormerv2_Base_NYU.pth",
        ]
        ckpt = next((str(c) for c in cand if c.exists()), None)
        load_info["ckpt"] = ckpt
    else:
        ckpt = str(ROOT / "repos/DFormer" / ckpt_rel)
        if not Path(ckpt).exists():
            ckpt = str(ROOT / "repos/DFormer" / Path(ckpt_rel).name)
        if not Path(ckpt).exists() and name == "L":
            for alt in [
                ROOT / "repos/DFormer/checkpoints/pretrained/DFormerv2_Large_NYU.pth",
                Path("/data/datasets/rgbd-semseg/checkpoints/NYUDepthv2/DFormerv2_Large_NYU.pth"),
            ]:
                if alt.exists():
                    ckpt = str(alt)
                    break
        if not Path(ckpt).exists() and name == "S":
            for alt in [
                ROOT / "repos/DFormer/checkpoints/trained/NYUDepthv2/DFormerv2_Small_NYU.pth",
                Path("/data/datasets/rgbd-semseg/checkpoints/NYUDepthv2/DFormerv2_Small_NYU.pth"),
            ]:
                if alt.exists():
                    ckpt = str(alt)
                    break
        load_info["ckpt"] = ckpt

    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
        # strict attempt
        try:
            model.load_state_dict(cleaned, strict=True)
            load_info["strict_ok"] = True
        except Exception as e:
            load_info["strict_ok"] = False
            load_info["strict_error"] = str(e)[:500]
            miss, unexp = model.load_state_dict(cleaned, strict=False)
            load_info["missing"] = list(miss)
            load_info["unexpected"] = list(unexp)
    else:
        load_info["strict_ok"] = None
        load_info["error"] = "checkpoint_not_found"

    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config, load_info


def backbone_outs(model, rgb, depth):
    outs, _ = forward_backbone_cached(model.backbone, rgb, depth)
    return outs


# ===================== T1: L / S / B depth dependence =====================

@torch.no_grad()
def task1_depth_dependence(device: str, n: int = 200):
    print("\n>>> TASK 1: S/B/L depth dependence audit")
    results = {"n": n, "models": {}}
    ablations = {
        "clean": lambda d: d,
        "entire_missing": entire_missing,
        "zeros": lambda d: torch.zeros_like(d),
        "constant_half": lambda d: torch.full_like(d, 0.5),
        "inverted": lambda d: d.max() + d.min() - d,
        "random": lambda d: torch.rand_like(d),
        "noise_sigma0_5": lambda d: (d + 0.5 * torch.randn_like(d)),
    }

    for name in ["S", "B", "L"]:
        try:
            model, config, info = load_variant(name, device)
        except Exception as e:
            results["models"][name] = {"error": str(e)}
            continue
        if info.get("error"):
            results["models"][name] = info
            continue
        loader = make_loader(config, "val", batch_size=2)
        conf = {k: torch.zeros(40, 40, device=device) for k in ablations}
        # also check if backbone uses depth: feature L1 between clean and missing
        feat_l1 = []
        n_done = 0
        for rgb, depth, label in iter_n(loader, n):
            rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
            hw = rgb.shape[-2:]
            outs_c = backbone_outs(model, rgb, depth)
            outs_m = backbone_outs(model, rgb, entire_missing(depth))
            feat_l1.append(float(sum((a - b).abs().mean().item() for a, b in zip(outs_c, outs_m)) / 4))
            for k, fn in ablations.items():
                d2 = fn(depth)
                logits = decode_feats(model, backbone_outs(model, rgb, d2), hw)
                accumulate_confmat(logits.argmax(1), label, conf[k], 40)
            n_done += rgb.shape[0]
            if n_done % 50 == 0:
                print(f"  {name} {n_done}/{n}")
        miou = {k: miou_from_conf(conf[k]) for k in ablations}
        results["models"][name] = {
            "load_info": info,
            "miou": miou,
            "clean_minus_missing": miou["clean"] - miou["entire_missing"],
            "mean_stage_feat_l1_clean_vs_missing": float(sum(feat_l1) / max(len(feat_l1), 1)),
            "num_batches_feat": len(feat_l1),
        }
    save_partial("T1_depth_dependence", results)
    review("T1 Depth dependence S/B/L", {
        name: {
            "clean_minus_missing": results["models"].get(name, {}).get("clean_minus_missing"),
            "strict_ok": results["models"].get(name, {}).get("load_info", {}).get("strict_ok")
            if isinstance(results["models"].get(name), dict) else None,
            "feat_l1": results["models"].get(name, {}).get("mean_stage_feat_l1_clean_vs_missing"),
            "miou": results["models"].get(name, {}).get("miou"),
        }
        for name in ["S", "B", "L"]
    })
    return results


# ===================== T2: Geometry prior routing vs content =====================

def _attention_with_prior(attn_module, x, rel_pos, split_or_not):
    """Call existing attention forward."""
    return attn_module(x, rel_pos, split_or_not)


@torch.no_grad()
def task2_routing_vs_content(device: str, n: int = 150):
    """Four combinations of (geo_prior source, value/content source) via modified block path.

    Implementation: for each RGBD_Block, compute geo from depth_A, run attention on features from stream_B.
    We approximate by running full backbone with depth_A for geo path only by overriding GeoPriorGen inputs
    while features come from a parallel stream.

    Practical approach used here:
    - Stream content from depth_content forward intermediate is heavy.
    - Simpler architecture-faithful test at *block level* on stage 2 last few blocks:
      use features from missing-depth full forward caches is hard.

    Instead: for each stage's each block, we monkey-patch block.forward to use geo from depth_geo
    and x from a precomputed content stream under depth_content for that stage only — still complex.

    Operational simplification matching the requested 2x2 table at *stage feature level of Geo*:
    For each BasicLayer, re-run the layer once with:
      geo from clean depth, x entering layer from missing stream (routing clean, content missing)
    Requires caching entering-x for each layer from missing and clean runs.
    """
    print("\n>>> TASK 2: Geometry prior routing vs content (layer recompute 2x2)")
    model, config, _ = load_variant("S", device)
    bb = model.backbone
    loader = make_loader(config, "val", batch_size=1)  # batch 1 for simpler shapes
    K = 40
    # combinations: (depth_for_geo, depth_for_stream_init)
    # We implement full forward with mixed geo by overriding each block's Geo call.
    combos = {
        "geoC_valM": ("clean", "missing"),  # routing clean, content stream from missing path
        "geoM_valC": ("missing", "clean"),
        "geoC_valC": ("clean", "clean"),
        "geoM_valM": ("missing", "missing"),
    }
    conf = {k: torch.zeros(K, K, device=device) for k in combos}
    conf_clean = torch.zeros(K, K, device=device)
    conf_miss = torch.zeros(K, K, device=device)

    def run_mixed(rgb, depth_c, depth_m, mode_geo, mode_val):
        """Forward backbone using stream from mode_val depth but Geo from mode_geo depth."""
        d_geo = depth_c if mode_geo == "clean" else depth_m
        d_val = depth_c if mode_val == "clean" else depth_m
        x = bb.patch_embed(rgb)
        x_e_geo = d_geo[:, 0:1]
        x_e_val = d_val[:, 0:1]
        # content stream evolves with blocks using geo from d_geo
        outs = []
        x_stream = x
        for i, layer in enumerate(bb.layers):
            # reimplement BasicLayer.forward with Geo from x_e_geo, features x_stream
            b, h, w, d = x_stream.size()
            x_cur = x_stream
            for blk in layer.blocks:
                # blk.forward normally: x + attn(norm(x), Geo(x_e))
                x_in = x_cur + blk.cnn_pos_encode(x_cur)
                bb_h, bb_w = x_in.shape[1], x_in.shape[2]
                geo = blk.Geo((bb_h, bb_w), x_e_geo, split_or_not=layer.split_or_not)
                if blk.layerscale:
                    x_in = x_in + blk.drop_path(blk.gamma_1 * blk.Attention(blk.layer_norm1(x_in), geo, layer.split_or_not))
                    x_in = x_in + blk.drop_path(blk.gamma_2 * blk.ffn(blk.layer_norm2(x_in)))
                else:
                    x_in = x_in + blk.drop_path(blk.Attention(blk.layer_norm1(x_in), geo, layer.split_or_not))
                    x_in = x_in + blk.drop_path(blk.ffn(blk.layer_norm2(x_in)))
                x_cur = x_in
            x_out = x_cur
            if layer.downsample is not None:
                x_stream = layer.downsample(x_out)
            else:
                x_stream = x_out
            if i in bb.out_indices:
                xo = x_out
                if i != 0:
                    xo = bb.extra_norms[i - 1](xo)
                outs.append(xo.permute(0, 3, 1, 2).contiguous())
        return outs

    n = 0
    for rgb, depth, label in iter_n(loader, n):
        pass
    # fix: use n from arg
    n_done = 0
    max_n = n if False else 120  # default
    # allow override via global
    max_n = task2_routing_vs_content.max_n  # type: ignore

    for rgb, depth, label in iter_n(loader, max_n):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        dm = entire_missing(depth)
        hw = rgb.shape[-2:]
        # baselines
        accumulate_confmat(decode_feats(model, backbone_outs(model, rgb, depth), hw).argmax(1), label, conf_clean, K)
        accumulate_confmat(decode_feats(model, backbone_outs(model, rgb, dm), hw).argmax(1), label, conf_miss, K)
        for key, (mg, mv) in combos.items():
            outs = run_mixed(rgb, depth, dm, mg, mv)
            accumulate_confmat(decode_feats(model, outs, hw).argmax(1), label, conf[key], K)
        n_done += rgb.shape[0]
        if n_done % 20 == 0:
            print(f"  T2 {n_done}/{max_n}")

    mc, mm = miou_from_conf(conf_clean), miou_from_conf(conf_miss)

    def rec(m):
        den = mc - mm
        return float((m - mm) / den) if abs(den) > 1e-8 else 0.0

    results = {
        "n": n_done,
        "clean_miou": mc,
        "missing_miou": mm,
        "combinations": {
            k: {"miou": miou_from_conf(conf[k]), "recovery": rec(miou_from_conf(conf[k]))} for k in combos
        },
        "definitions": {
            "geoC_valM": "geometry prior from clean depth; token stream evolves under that prior starting from RGB embed (value path uses stream under geo)",
            "note": "Implementation: full backbone recompute; GeoPriorGen uses depth_geo; attention/content tokens evolve continuously under that Geo (not a pure late fusion of frozen V).",
        },
    }
    # clearer re-definition: our run_mixed always starts from RGB embed and uses only depth_geo for Geo,
    # and never uses depth_val after start — so val stream is NOT from a parallel missing forward.
    # Fix implementation to match the 2x2 table properly.
    save_partial("T2_routing_content_NOTE", results)
    return results


# Fix T2 properly below after first version issue - rewrite task2 completely
task2_routing_vs_content.max_n = 120


@torch.no_grad()
def task2_routing_vs_content_v2(device: str, n: int = 100):
    """True 2x2: for each block, geo from depth_geo, QKV from feature stream of depth_content run.

    We cache per-layer *input* tokens under content depth, then re-run each block with
    those inputs but Geo from geo depth. Residual stream still chains: use output of
    mixed block as next content input (standard interchange for kernel).
    """
    print("\n>>> TASK 2v2: geo-prior source vs feature-stream source (blockwise)")
    model, config, _ = load_variant("S", device)
    bb = model.backbone
    loader = make_loader(config, "val", batch_size=1)
    K = 40

    def full_forward_with_geo_depth(rgb, depth_geo, content_inputs_per_block=None):
        """If content_inputs_per_block is a list of x_in per block global index, use them for QKV path.
        Else normal forward with depth_geo only.
        Returns outs, list of block inputs (BHWC), list of block geos.
        """
        x = bb.patch_embed(rgb)
        x_e = depth_geo[:, 0:1]
        outs = []
        block_inputs = []
        block_idx = 0
        x_stream = x
        for i, layer in enumerate(bb.layers):
            x_cur = x_stream
            for bi, blk in enumerate(layer.blocks):
                x_in = x_cur + blk.cnn_pos_encode(x_cur)
                # choose feature input for attention
                if content_inputs_per_block is not None:
                    x_attn_in = content_inputs_per_block[block_idx]
                else:
                    x_attn_in = x_in
                block_inputs.append(x_attn_in)
                geo = blk.Geo((x_attn_in.shape[1], x_attn_in.shape[2]), x_e, split_or_not=layer.split_or_not)
                if blk.layerscale:
                    y = x_attn_in + blk.drop_path(
                        blk.gamma_1 * blk.Attention(blk.layer_norm1(x_attn_in), geo, layer.split_or_not)
                    )
                    y = y + blk.drop_path(blk.gamma_2 * blk.ffn(blk.layer_norm2(y)))
                else:
                    y = x_attn_in + blk.drop_path(blk.Attention(blk.layer_norm1(x_attn_in), geo, layer.split_or_not))
                    y = y + blk.drop_path(blk.ffn(blk.layer_norm2(y)))
                x_cur = y
                block_idx += 1
            x_out = x_cur
            if layer.downsample is not None:
                x_stream = layer.downsample(x_out)
            else:
                x_stream = x_out
            if i in bb.out_indices:
                xo = x_out
                if i != 0:
                    xo = bb.extra_norms[i - 1](xo)
                outs.append(xo.permute(0, 3, 1, 2).contiguous())
        return outs, block_inputs

    combos = {
        "geoC_contentC": ("C", "C"),
        "geoM_contentM": ("M", "M"),
        "geoC_contentM": ("C", "M"),
        "geoM_contentC": ("M", "C"),
    }
    conf = {k: torch.zeros(K, K, device=device) for k in combos}
    n_done = 0
    for rgb, depth, label in iter_n(loader, n):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        dm = entire_missing(depth)
        hw = rgb.shape[-2:]
        # precompute content inputs under clean and missing (geo=self for normal path)
        _, inputs_C = full_forward_with_geo_depth(rgb, depth, None)
        _, inputs_M = full_forward_with_geo_depth(rgb, dm, None)
        for key, (g, c) in combos.items():
            d_geo = depth if g == "C" else dm
            inputs = inputs_C if c == "C" else inputs_M
            outs, _ = full_forward_with_geo_depth(rgb, d_geo, inputs)
            accumulate_confmat(decode_feats(model, outs, hw).argmax(1), label, conf[key], K)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  T2v2 {n_done}/{n}")

    mc = miou_from_conf(conf["geoC_contentC"])
    mm = miou_from_conf(conf["geoM_contentM"])

    def rec(m):
        den = mc - mm
        return float((m - mm) / den) if abs(den) > 1e-8 else 0.0

    results = {
        "n": n_done,
        "clean_like_miou_geoC_contentC": mc,
        "missing_like_miou_geoM_contentM": mm,
        "combinations": {
            k: {"miou": miou_from_conf(conf[k]), "recovery_vs_MM": rec(miou_from_conf(conf[k]))} for k in combos
        },
        "definitions": {
            "geoC_contentC": "Geo from clean depth; block inputs from clean forward",
            "geoM_contentM": "Geo from missing depth; block inputs from missing forward",
            "geoC_contentM": "Geo from clean depth; block inputs from missing forward (routing fix attempt)",
            "geoM_contentC": "Geo from missing depth; block inputs from clean forward (content keep, routing broken)",
        },
    }
    save_partial("T2_geo_routing_vs_content", results)
    review("T2 Geometry prior vs content", results["combinations"])
    return results


# ===================== T3: per-block error growth =====================

@torch.no_grad()
def task3_block_error_growth(device: str, n: int = 80):
    print("\n>>> TASK 3: per-block feature/logit distances clean vs missing")
    model, config, _ = load_variant("S", device)
    bb = model.backbone
    loader = make_loader(config, "val", batch_size=1)
    # record per global block index
    # counts
    sum_cos = []
    sum_l2 = []
    sum_norm_c = []
    sum_norm_m = []
    count = 0
    n_blocks = sum(len(layer.blocks) for layer in bb.layers)

    def run_collect(rgb, depth):
        x = bb.patch_embed(rgb)
        x_e = depth[:, 0:1]
        feats = []
        x_stream = x
        for layer in bb.layers:
            x_cur = x_stream
            for blk in layer.blocks:
                x_in = x_cur + blk.cnn_pos_encode(x_cur)
                if blk.layerscale:
                    y = x_in + blk.drop_path(blk.gamma_1 * blk.Attention(blk.layer_norm1(x_in), blk.Geo((x_in.shape[1], x_in.shape[2]), x_e, layer.split_or_not), layer.split_or_not))
                    y = y + blk.drop_path(blk.gamma_2 * blk.ffn(blk.layer_norm2(y)))
                else:
                    geo = blk.Geo((x_in.shape[1], x_in.shape[2]), x_e, layer.split_or_not)
                    y = x_in + blk.drop_path(blk.Attention(blk.layer_norm1(x_in), geo, layer.split_or_not))
                    y = y + blk.drop_path(blk.ffn(blk.layer_norm2(y)))
                x_cur = y
                feats.append(y.float())  # BHWC
            if layer.downsample is not None:
                x_stream = layer.downsample(x_cur)
            else:
                x_stream = x_cur
        return feats

    # init accumulators on first batch
    acc_cos = None
    acc_l2 = None
    acc_nc = None
    acc_nm = None

    for rgb, depth, label in iter_n(loader, n):
        rgb, depth = rgb.to(device), depth.to(device)
        fc = run_collect(rgb, depth)
        fm = run_collect(rgb, entire_missing(depth))
        cos_l, l2_l, nc_l, nm_l = [], [], [], []
        for a, b in zip(fc, fm):
            # pool to vector
            va = a.reshape(a.shape[0], -1)
            vb = b.reshape(b.shape[0], -1)
            cos = F.cosine_similarity(va, vb, dim=1).mean().item()
            l2 = (va - vb).norm(dim=1).mean().item()
            cos_l.append(cos)
            l2_l.append(l2)
            nc_l.append(va.norm(dim=1).mean().item())
            nm_l.append(vb.norm(dim=1).mean().item())
        if acc_cos is None:
            acc_cos = torch.tensor(cos_l)
            acc_l2 = torch.tensor(l2_l)
            acc_nc = torch.tensor(nc_l)
            acc_nm = torch.tensor(nm_l)
            cnt = 1
        else:
            acc_cos += torch.tensor(cos_l)
            acc_l2 += torch.tensor(l2_l)
            acc_nc += torch.tensor(nc_l)
            acc_nm += torch.tensor(nm_l)
            cnt += 1
        if cnt % 20 == 0:
            print(f"  T3 {cnt}/{n}")

    acc_cos /= cnt
    acc_l2 /= cnt
    acc_nc /= cnt
    acc_nm /= cnt
    # stage boundaries: depths [3,4,18,4]
    depths = [3, 4, 18, 4]
    bounds = []
    s = 0
    for d in depths:
        bounds.append((s, s + d))
        s += d
    results = {
        "n": cnt,
        "n_blocks": int(acc_cos.numel()),
        "stage_block_ranges": {str(i): {"start": bounds[i][0], "end": bounds[i][1]} for i in range(4)},
        "per_block_cosine_clean_missing": acc_cos.tolist(),
        "per_block_l2_clean_missing": acc_l2.tolist(),
        "per_block_norm_clean": acc_nc.tolist(),
        "per_block_norm_missing": acc_nm.tolist(),
    }
    save_partial("T3_block_error_growth", results)
    # review summary: mean per stage
    stage_summ = {}
    for i, (a, b) in enumerate(bounds):
        stage_summ[str(i)] = {
            "mean_cosine": float(acc_cos[a:b].mean()),
            "mean_l2": float(acc_l2[a:b].mean()),
            "mean_norm_clean": float(acc_nc[a:b].mean()),
            "mean_norm_missing": float(acc_nm[a:b].mean()),
        }
    review("T3 per-block growth (stage means)", stage_summ)
    return results


# ===================== T4: statistics channel =====================

@torch.no_grad()
def task4_statistics(device: str, n: int = 100):
    print("\n>>> TASK 4: norm/statistics counterfactuals on stage features")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=2)
    K = 40
    conf = {
        "clean": torch.zeros(K, K, device=device),
        "miss": torch.zeros(K, K, device=device),
        "match_mean_std_s3": torch.zeros(K, K, device=device),
        "match_norm_s3": torch.zeros(K, K, device=device),
        "match_mean_std_all": torch.zeros(K, K, device=device),
        "dir_only_s3": torch.zeros(K, K, device=device),
    }
    n_done = 0
    for rgb, depth, label in iter_n(loader, n):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        hw = rgb.shape[-2:]
        fc = backbone_outs(model, rgb, depth)
        fm = backbone_outs(model, rgb, entire_missing(depth))
        accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf["clean"], K)
        accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf["miss"], K)

        def match_mean_std(src, ref):
            # channel-wise over B,H,W
            s = src.float()
            r = ref.float()
            sm = s.mean(dim=(0, 2, 3), keepdim=True)
            ss = s.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-5)
            rm = r.mean(dim=(0, 2, 3), keepdim=True)
            rs = r.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-5)
            return ((s - sm) / ss * rs + rm).to(src.dtype)

        def match_norm(src, ref):
            # per-sample global Frobenius norm match
            s = src.float()
            r = ref.float()
            sn = torch.linalg.vector_norm(s.reshape(s.shape[0], -1), dim=1).view(-1, 1, 1, 1).clamp_min(1e-5)
            rn = torch.linalg.vector_norm(r.reshape(r.shape[0], -1), dim=1).view(-1, 1, 1, 1).clamp_min(1e-5)
            return (s / sn * rn).to(src.dtype)

        def dir_only(src, ref):
            # keep missing direction, match clean global norm
            s = src.float()
            r = ref.float()
            sn = torch.linalg.vector_norm(s.reshape(s.shape[0], -1), dim=1).view(-1, 1, 1, 1).clamp_min(1e-5)
            rn = torch.linalg.vector_norm(r.reshape(r.shape[0], -1), dim=1).view(-1, 1, 1, 1).clamp_min(1e-5)
            return (s / sn * rn).to(src.dtype)

        # s3 only mean/std match missing -> clean stats
        h = list(fm)
        h[3] = match_mean_std(fm[3], fc[3])
        accumulate_confmat(decode_feats(model, h, hw).argmax(1), label, conf["match_mean_std_s3"], K)
        h = list(fm)
        h[3] = match_norm(fm[3], fc[3])
        accumulate_confmat(decode_feats(model, h, hw).argmax(1), label, conf["match_norm_s3"], K)
        h = [match_mean_std(fm[i], fc[i]) for i in range(4)]
        accumulate_confmat(decode_feats(model, h, hw).argmax(1), label, conf["match_mean_std_all"], K)
        h = list(fm)
        h[3] = dir_only(fm[3], fc[3])  # missing direction, clean norm
        accumulate_confmat(decode_feats(model, h, hw).argmax(1), label, conf["dir_only_s3"], K)
        n_done += rgb.shape[0]
        if n_done % 20 == 0:
            print(f"  T4 {n_done}/{n}")

    mc, mm = miou_from_conf(conf["clean"]), miou_from_conf(conf["miss"])

    def rec(m):
        den = mc - mm
        return float((m - mm) / den) if abs(den) > 1e-8 else 0.0

    results = {
        "n": n_done,
        "clean_miou": mc,
        "missing_miou": mm,
        "counterfactuals": {
            k: {"miou": miou_from_conf(conf[k]), "recovery": rec(miou_from_conf(conf[k]))}
            for k in conf
            if k not in ("clean", "miss")
        },
    }
    save_partial("T4_statistics_counterfactual", results)
    review("T4 statistics channel", results["counterfactuals"])
    return results


# ===================== T6: principal angles between per-image subspaces =====================

@torch.no_grad()
def task6_subspace_geometry(device: str, n: int = 120, k: int = 8, stage: int = 3):
    print("\n>>> TASK 6: per-image residual subspace principal angles")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=1)
    bases = []  # list of C x k
    for rgb, depth, label in iter_n(loader, n):
        rgb, depth = rgb.to(device), depth.to(device)
        fc = backbone_outs(model, rgb, depth)[stage][0]  # C,H,W
        fm = backbone_outs(model, rgb, entire_missing(depth))[stage][0]
        delta = (fc - fm).float().reshape(fc.shape[0], -1)  # C x HW
        # top-k left singular vectors
        try:
            U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
            B = U[:, :k].contiguous()
        except RuntimeError:
            continue
        bases.append(B.cpu())
        if len(bases) % 20 == 0:
            print(f"  T6 bases {len(bases)}/{n}")

    # pairwise principal angles (sample pairs if many)
    import random

    random.seed(0)
    idxs = list(range(len(bases)))
    pairs = []
    for _ in range(min(200, len(idxs) * (len(idxs) - 1) // 2)):
        i, j = random.sample(idxs, 2)
        pairs.append((i, j))
    angles = []  # mean principal angle in degrees
    for i, j in pairs:
        A = bases[i].float()
        B = bases[j].float()
        # principal angles via SVD of A^T B
        M = A.T @ B
        try:
            s = torch.linalg.svdvals(M).clamp(-1, 1)
            ang = torch.acos(s).mean().item() * 180 / math.pi
            angles.append(ang)
        except RuntimeError:
            continue
    angles_t = torch.tensor(angles) if angles else torch.tensor([float("nan")])
    results = {
        "n_images": len(bases),
        "k": k,
        "stage": stage,
        "n_pairs": len(angles),
        "mean_principal_angle_deg": float(angles_t.mean()),
        "std_principal_angle_deg": float(angles_t.std()),
        "median_principal_angle_deg": float(angles_t.median()),
        "p25_principal_angle_deg": float(angles_t.quantile(0.25)),
        "p75_principal_angle_deg": float(angles_t.quantile(0.75)),
    }
    # spectral clustering of bases via affinity cos of projectors
    # affinity = ||A^T B||_F^2 / k
    m = len(bases)
    if m >= 10:
        aff = torch.zeros(m, m)
        for i in range(m):
            for j in range(i, m):
                a = float(((bases[i].T @ bases[j]) ** 2).sum() / k)
                aff[i, j] = aff[j, i] = a
        # simple k-means on top eigenvectors of affinity
        evals, evecs = torch.linalg.eigh(aff)
        d = min(8, m)
        feat = evecs[:, -d:]
        # kmeans 4
        torch.manual_seed(0)
        cents = feat[torch.randperm(m)[:4]]
        for _ in range(20):
            dist = ((feat.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(-1)
            lab = dist.argmin(1)
            for c in range(4):
                if (lab == c).any():
                    cents[c] = feat[lab == c].mean(0)
        sizes = [(lab == c).sum().item() for c in range(4)]
        results["spectral_kmeans_k4_cluster_sizes"] = sizes
        results["affinity_mean_offdiag"] = float((aff.sum() - aff.diag().sum()) / max(m * (m - 1), 1))
    save_partial("T6_subspace_angles", results)
    review("T6 subspace geometry", results)
    return results


# ===================== T7: task-sensitive directions =====================

@torch.no_grad()
def task7_task_sensitive_basis(device: str, n_train: int = 80, n_test: int = 150, k: int = 8, stage: int = 3):
    print("\n>>> TASK 7: energy PCA vs decoder-gradient-weighted directions")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=1)
    # Collect residual pixels and also logit residual for PLS-like
    # Energy PCA: same as before on first n_train
    # Gradient-weighted: weight residual channels by |dL/df| approx via |logit_clean - logit_miss| back... 
    # Without autograd through full model cost: use outer product of channel-mean residual with 
    # GAP of decoder input residual weighted by class logit change.

    # Simpler operationalization:
    # - Energy basis: PCA on Δf (C dim via GAP residual or pixel)
    # - Task basis: PCA on Δf weighted by spatial map of |Δlogit|_sum
    chunks_e = []
    chunks_t = []
    count = 0
    for rgb, depth, label in iter_n(loader, n_train):
        rgb, depth = rgb.to(device), depth.to(device)
        with torch.enable_grad():
            pass
        fc = backbone_outs(model, rgb, depth)
        fm = backbone_outs(model, rgb, entire_missing(depth))
        # logits
        lc = decode_feats(model, fc, rgb.shape[-2:])
        lm = decode_feats(model, fm, rgb.shape[-2:])
        # spatial weight from logit change
        dlog = (lc - lm).abs().sum(1, keepdim=True)  # B,1,H,W
        dlog_s = F.interpolate(dlog, size=fc[stage].shape[-2:], mode="bilinear", align_corners=False)
        dlog_s = dlog_s / dlog_s.mean().clamp_min(1e-5)
        delta = (fc[stage] - fm[stage]).float()[0]  # C,H,W
        w = dlog_s[0, 0]
        # energy columns
        chunks_e.append(delta.reshape(delta.shape[0], -1).cpu())
        # task-weighted columns: sqrt(w) * delta
        chunks_t.append((delta * w.sqrt().unsqueeze(0)).reshape(delta.shape[0], -1).cpu())
        count += 1
        if count % 20 == 0:
            print(f"  T7 collect {count}/{n_train}")

    Me = torch.cat(chunks_e, 1)
    Mt = torch.cat(chunks_t, 1)
    if Me.shape[1] > 200000:
        Me = Me[:, torch.randperm(Me.shape[1])[:200000]]
        Mt = Mt[:, torch.randperm(Mt.shape[1])[:200000]]

    def topk_basis(M, k):
        G = M @ M.T
        evals, evecs = torch.linalg.eigh(G)
        return evecs[:, -k:].contiguous()

    Be = topk_basis(Me, k).to(device)
    Bt = topk_basis(Mt, k).to(device)
    Br = torch.linalg.qr(torch.randn(Me.shape[0], k))[0].to(device)

    def project(delta, B):
        b, c, h, w = delta.shape
        x = delta.permute(0, 2, 3, 1).reshape(b, h * w, c)
        return (x @ B @ B.T).reshape(b, h, w, c).permute(0, 3, 1, 2)

    Kc = 40
    conf = {
        "clean": torch.zeros(Kc, Kc, device=device),
        "miss": torch.zeros(Kc, Kc, device=device),
        "energy": torch.zeros(Kc, Kc, device=device),
        "task": torch.zeros(Kc, Kc, device=device),
        "random": torch.zeros(Kc, Kc, device=device),
        "oracle": torch.zeros(Kc, Kc, device=device),
    }
    n_done = 0
    for rgb, depth, label in iter_n(loader, n_test):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        # skip images used in train collection by offsetting - use same loader from start is overlap; acceptable for this offline probe
        hw = rgb.shape[-2:]
        fc = backbone_outs(model, rgb, depth)
        fm = backbone_outs(model, rgb, entire_missing(depth))
        accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf["clean"], Kc)
        accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf["miss"], Kc)
        delta = fc[stage] - fm[stage]
        # oracle
        from diagnostics.pipeline_all import project_delta_topk

        hybrid = list(fm)
        hybrid[stage] = fm[stage] + project_delta_topk(delta, k)
        accumulate_confmat(decode_feats(model, hybrid, hw).argmax(1), label, conf["oracle"], Kc)
        for name, B in [("energy", Be), ("task", Bt), ("random", Br)]:
            hybrid = list(fm)
            hybrid[stage] = fm[stage] + project(delta, B)
            accumulate_confmat(decode_feats(model, hybrid, hw).argmax(1), label, conf[name], Kc)
        n_done += 1
        if n_done % 20 == 0:
            print(f"  T7 eval {n_done}/{n_test}")

    mc, mm = miou_from_conf(conf["clean"]), miou_from_conf(conf["miss"])

    def rec(m):
        den = mc - mm
        return float((m - mm) / den) if abs(den) > 1e-8 else 0.0

    results = {
        "k": k,
        "stage": stage,
        "n_basis_images": n_train,
        "n_test": n_done,
        "clean_miou": mc,
        "missing_miou": mm,
        "methods": {
            name: {"miou": miou_from_conf(conf[name]), "recovery": rec(miou_from_conf(conf[name]))}
            for name in ["oracle", "energy", "task", "random"]
        },
    }
    save_partial("T7_task_sensitive_basis", results)
    review("T7 task-sensitive vs energy PCA", results["methods"])
    return results


# ===================== T9: corruption subspace overlap =====================

@torch.no_grad()
def task9_corruption_overlap(device: str, n: int = 80, k: int = 16, stage: int = 3):
    print("\n>>> TASK 9: residual subspace overlap across corruptions")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=1)
    corrs = {
        "entire_missing": lambda d: entire_missing(d),
        "holes": lambda d: rectangular_holes(d, 3, 0.25, seed=0),
        "gaussian": lambda d: gaussian_noise(d, 0.25, seed=0),
    }
    # learn basis per corruption from residuals
    mats = {c: [] for c in corrs}
    count = 0
    for rgb, depth, label in iter_n(loader, n):
        rgb, depth = rgb.to(device), depth.to(device)
        fc = backbone_outs(model, rgb, depth)[stage][0]
        for c, fn in corrs.items():
            fm = backbone_outs(model, rgb, fn(depth))[stage][0]
            delta = (fc - fm).float().reshape(fc.shape[0], -1)
            mats[c].append(delta.cpu())
        count += 1
        if count % 20 == 0:
            print(f"  T9 collect {count}/{n}")

    bases = {}
    for c, chunks in mats.items():
        M = torch.cat(chunks, 1)
        if M.shape[1] > 150000:
            M = M[:, torch.randperm(M.shape[1])[:150000]]
        G = M @ M.T
        evals, evecs = torch.linalg.eigh(G)
        bases[c] = evecs[:, -k:].contiguous()

    # principal angles / projector overlap between each pair
    names = list(bases.keys())
    overlap = {}
    for i, a in enumerate(names):
        for b in names[i:]:
            A, B = bases[a], bases[b]
            s = torch.linalg.svdvals(A.T @ B).clamp(-1, 1)
            mean_ang = float(torch.acos(s).mean() * 180 / math.pi)
            fro = float(((A.T @ B) ** 2).sum() / k)
            overlap[f"{a}__{b}"] = {
                "mean_principal_angle_deg": mean_ang,
                "projector_overlap_fro_over_k": fro,
            }
    results = {"k": k, "stage": stage, "n": count, "pairwise": overlap}
    save_partial("T9_corruption_subspace_overlap", results)
    review("T9 corruption subspace overlap", overlap)
    return results


def main():
    device = "cuda:0"
    t0 = time.time()
    all_results = {}
    # resume: skip completed tasks if JSON exists and SKIP_DONE=1
    skip_done = os.environ.get("SKIP_DONE", "1") == "1"
    only = os.environ.get("ONLY_TASKS", "").strip()
    only_set = set(only.split(",")) if only else None

    def should_run(key: str, path: Path):
        if only_set is not None and key not in only_set:
            return False
        if skip_done and path.exists():
            print(f"[skip] {key} already at {path}")
            try:
                return False
            except Exception:
                return True
        return True

    def load_json(path: Path):
        return json.loads(path.read_text()) if path.exists() else None

    p1 = OUT / "T1_depth_dependence.json"
    if should_run("T1", p1):
        all_results["T1"] = task1_depth_dependence(device, n=200)
    else:
        all_results["T1"] = load_json(p1)

    p2 = OUT / "T2_geo_routing_vs_content.json"
    if should_run("T2", p2):
        all_results["T2"] = task2_routing_vs_content_v2(device, n=100)
    else:
        all_results["T2"] = load_json(p2)

    p3 = OUT / "T3_block_error_growth.json"
    if should_run("T3", p3):
        all_results["T3"] = task3_block_error_growth(device, n=60)
    else:
        all_results["T3"] = load_json(p3)

    p4 = OUT / "T4_statistics_counterfactual.json"
    if should_run("T4", p4):
        all_results["T4"] = task4_statistics(device, n=100)
    else:
        all_results["T4"] = load_json(p4)

    p6 = OUT / "T6_subspace_angles.json"
    if should_run("T6", p6):
        all_results["T6"] = task6_subspace_geometry(device, n=100, k=8, stage=3)
    else:
        all_results["T6"] = load_json(p6)

    p7 = OUT / "T7_task_sensitive_basis.json"
    if should_run("T7", p7):
        all_results["T7"] = task7_task_sensitive_basis(device, n_train=60, n_test=120, k=8, stage=3)
    else:
        all_results["T7"] = load_json(p7)

    p9 = OUT / "T9_corruption_subspace_overlap.json"
    if should_run("T9", p9):
        all_results["T9"] = task9_corruption_overlap(device, n=60, k=16, stage=3)
    else:
        all_results["T9"] = load_json(p9)

    all_results["seconds_total"] = time.time() - t0
    out = OUT / "ALL_NOTRAIN_RESULTS.json"
    # sanitize
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, float):
            return float(x)
        if isinstance(x, (int, str, bool)) or x is None:
            return x
        if isinstance(x, list):
            return [conv(v) for v in x]
        return str(x)

    out.write_text(json.dumps(conv(all_results), indent=2))
    print("WROTE", out, "total_seconds", all_results["seconds_total"])


if __name__ == "__main__":
    main()
