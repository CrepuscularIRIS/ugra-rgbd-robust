#!/usr/bin/env python3
"""Phase-4 remaining gates: Q1a/b/c/f, Q3, Q4, Q5, Q2, Q6.

Writes JSON under results/phase4/ and prints a short review after each step.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))
os.chdir(ROOT / "repos" / "DFormer")

# reuse helpers
from diagnostics.scripts.deep_notrain_suite import (  # noqa: E402
    load_variant,
    make_loader,
    iter_n,
    backbone_outs,
)
from diagnostics.corruptions.depth_corruptions import entire_missing  # noqa: E402
from diagnostics.pipeline_all import accumulate_confmat, decode_feats, miou_from_conf  # noqa: E402

OUT = ROOT / "results" / "phase4"
OUT.mkdir(parents=True, exist_ok=True)
K = 40


def conv(x):
    if isinstance(x, torch.Tensor):
        if x.numel() < 200:
            return x.detach().cpu().tolist()
        return {"shape": list(x.shape), "norm": float(x.float().norm())}
    if isinstance(x, dict):
        return {k: conv(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [conv(v) for v in x]
    if isinstance(x, float):
        return float(x)
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    return str(x)


def save(name: str, obj: dict):
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(conv(obj), indent=2))
    print(f"[saved] {p}")


def review(title: str, d: dict):
    print("\n" + "=" * 72)
    print(f"REVIEW: {title}")
    print("=" * 72)
    print(json.dumps(conv(d), indent=2))
    print("=" * 72 + "\n")


def geo_modules(model) -> List[nn.Module]:
    geos = []
    for layer in model.backbone.layers:
        for blk in layer.blocks:
            geos.append(blk.Geo)
    return geos


def set_geo_depth_scale(model, scale: float, store: Optional[list] = None):
    """Multiply Geo.weight[1] (depth term) by scale relative to original.

    If store is None, captures originals first then applies scale.
    store: list of original weight tensors.
    """
    geos = geo_modules(model)
    if store is None:
        store = [g.weight.detach().clone() for g in geos]
    for g, w0 in zip(geos, store):
        with torch.no_grad():
            g.weight.copy_(w0)
            g.weight[1].mul_(scale)
    return store


def restore_geo(model, store: list):
    for g, w0 in zip(geo_modules(model), store):
        with torch.no_grad():
            g.weight.copy_(w0)


@torch.no_grad()
def eval_miou(model, loader, n: int, device: str, depth_fn: Callable, conf_out=None):
    conf = torch.zeros(K, K, device=device) if conf_out is None else conf_out
    n_done = 0
    pix_correct = 0
    pix_total = 0
    ce_sum = 0.0
    ce_count = 0
    for rgb, depth, label in iter_n(loader, n):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        d = depth_fn(depth)
        logits = decode_feats(model, backbone_outs(model, rgb, d), rgb.shape[-2:])
        pred = logits.argmax(1)
        accumulate_confmat(pred, label, conf, K)
        valid = label != 255
        pix_correct += int((pred[valid] == label[valid]).sum().item())
        pix_total += int(valid.sum().item())
        # CE
        logp = F.log_softmax(logits, dim=1)
        lab = label.clone()
        lab[~valid] = 0
        ce = F.nll_loss(logp, lab, reduction="none")
        ce_sum += float(ce[valid].sum().item())
        ce_count += int(valid.sum().item())
        n_done += rgb.shape[0]
    return {
        "miou": miou_from_conf(conf),
        "pixel_acc": pix_correct / max(pix_total, 1),
        "mean_ce": ce_sum / max(ce_count, 1),
        "n": n_done,
        "conf": conf if conf_out is not None else None,
    }


# ===================== Q1a =====================

def q1a_coupling(device: str):
    print("\n>>> Q1a coupling weights S/B/L")
    out = {"models": {}}
    for name in ["S", "B", "L"]:
        model, _, info = load_variant(name, device)
        stages = []
        all_w0, all_w1, all_decay = [], [], []
        for si, layer in enumerate(model.backbone.layers):
            sw0, sw1, sd = [], [], []
            for blk in layer.blocks:
                w = blk.Geo.weight.detach().float().cpu().view(-1)
                d = blk.Geo.decay.detach().float().cpu()
                sw0.append(float(w[0].abs()))
                sw1.append(float(w[1].abs()))
                sd.append(float(d.abs().mean()))
                all_w0.append(sw0[-1])
                all_w1.append(sw1[-1])
                all_decay.append(sd[-1])
            stages.append({
                "stage": si,
                "mean_abs_weight_pos": sum(sw0) / len(sw0),
                "mean_abs_weight_depth": sum(sw1) / len(sw1),
                "mean_abs_decay": sum(sd) / len(sd),
                "n_blocks": len(sw0),
                "weight_depth_list": sw1,
            })
        out["models"][name] = {
            "load_strict_ok": info.get("strict_ok"),
            "overall_mean_abs_weight_pos": sum(all_w0) / len(all_w0),
            "overall_mean_abs_weight_depth": sum(all_w1) / len(all_w1),
            "overall_mean_abs_decay": sum(all_decay) / len(all_decay),
            "ratio_depth_over_pos": (sum(all_w1) / len(all_w1)) / max(sum(all_w0) / len(all_w0), 1e-8),
            "stages": stages,
        }
        del model
        torch.cuda.empty_cache()
    save("Q1a_coupling_weights", out)
    review("Q1a coupling", {k: {
        "w_depth": v["overall_mean_abs_weight_depth"],
        "w_pos": v["overall_mean_abs_weight_pos"],
        "ratio_d_over_p": v["ratio_depth_over_pos"],
        "decay": v["overall_mean_abs_decay"],
    } for k, v in out["models"].items()})
    return out


# ===================== Q1b =====================

@torch.no_grad()
def q1b_prior_disabled(device: str, n: int = 150):
    print("\n>>> Q1b prior-disabled vs zeros (S)")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=2)
    store = [g.weight.detach().clone() for g in geo_modules(model)]

    results = {}
    # clean / missing / zeros with normal weights
    for key, fn in [
        ("clean", lambda d: d),
        ("entire_missing", entire_missing),
        ("zeros_depth", lambda d: torch.zeros_like(d)),
    ]:
        results[key] = eval_miou(model, loader, n, device, fn)
        print(f"  {key}: mIoU={results[key]['miou']:.4f}")

    # prior depth weight zero, clean depth input
    set_geo_depth_scale(model, 0.0, store)
    results["prior_depth_weight_0_clean_depth"] = eval_miou(model, loader, n, device, lambda d: d)
    print(f"  prior0+clean: mIoU={results['prior_depth_weight_0_clean_depth']['miou']:.4f}")
    results["prior_depth_weight_0_zeros_depth"] = eval_miou(model, loader, n, device, lambda d: torch.zeros_like(d))
    print(f"  prior0+zeros: mIoU={results['prior_depth_weight_0_zeros_depth']['miou']:.4f}")
    restore_geo(model, store)

    # strip conf tensors
    for k in results:
        results[k].pop("conf", None)
    save("Q1b_prior_disabled", {"n": n, "conditions": results})
    review("Q1b prior disabled", {k: v["miou"] for k, v in results.items()})
    return results


# ===================== Q1c =====================

@torch.no_grad()
def q1c_amplify_bl(device: str, n: int = 80):
    print("\n>>> Q1c prior amplify B/L")
    out = {}
    for name in ["B", "L"]:
        model, config, _ = load_variant(name, device)
        loader = make_loader(config, "val", batch_size=2)
        store = [g.weight.detach().clone() for g in geo_modules(model)]
        rows = {}
        for scale in [0.0, 1.0, 2.0, 4.0]:
            set_geo_depth_scale(model, scale, store)
            clean = eval_miou(model, loader, n, device, lambda d: d)
            miss = eval_miou(model, loader, n, device, entire_missing)
            rows[str(scale)] = {
                "clean_miou": clean["miou"],
                "missing_miou": miss["miou"],
                "delta": clean["miou"] - miss["miou"],
                "clean_pixacc": clean["pixel_acc"],
                "missing_pixacc": miss["pixel_acc"],
            }
            print(f"  {name} scale={scale}: clean={clean['miou']:.4f} miss={miss['miou']:.4f} d={rows[str(scale)]['delta']:.4f}")
        restore_geo(model, store)
        out[name] = rows
        del model
        torch.cuda.empty_cache()
    save("Q1c_prior_amplify_BL", {"n": n, "models": out})
    review("Q1c amplify B/L", out)
    return out


# ===================== Q1f =====================

@torch.no_grad()
def q1f_continuous(device: str, n: int = 150):
    print("\n>>> Q1f continuous metrics S/B/L")
    out = {}
    for name in ["S", "B", "L"]:
        model, config, _ = load_variant(name, device)
        loader = make_loader(config, "val", batch_size=2)
        clean = eval_miou(model, loader, n, device, lambda d: d)
        miss = eval_miou(model, loader, n, device, entire_missing)
        out[name] = {
            "clean": {k: clean[k] for k in ("miou", "pixel_acc", "mean_ce", "n")},
            "missing": {k: miss[k] for k in ("miou", "pixel_acc", "mean_ce", "n")},
            "delta_miou": clean["miou"] - miss["miou"],
            "delta_pixacc": clean["pixel_acc"] - miss["pixel_acc"],
            "delta_ce": miss["mean_ce"] - clean["mean_ce"],
        }
        print(f"  {name}: d_mIoU={out[name]['delta_miou']:.4f} d_pix={out[name]['delta_pixacc']:.4f} d_ce={out[name]['delta_ce']:.4f}")
        del model
        torch.cuda.empty_cache()
    save("Q1f_continuous_metrics", {"n": n, "models": out})
    review("Q1f continuous", out)
    return out


# ===================== Q3 anatomy =====================

@torch.no_grad()
def q3_anatomy(device: str, n: int = 200):
    print("\n>>> Q3 anatomy of gap (S)")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=1)
    conf_c = torch.zeros(K, K, device=device)
    conf_m = torch.zeros(K, K, device=device)
    # boundary / interior / far depth
    conf_c_b = torch.zeros(K, K, device=device)
    conf_m_b = torch.zeros(K, K, device=device)
    conf_c_i = torch.zeros(K, K, device=device)
    conf_m_i = torch.zeros(K, K, device=device)
    conf_c_far = torch.zeros(K, K, device=device)
    conf_m_far = torch.zeros(K, K, device=device)
    conf_c_near = torch.zeros(K, K, device=device)
    conf_m_near = torch.zeros(K, K, device=device)

    def mask_conf(pred, label, conf, mask):
        # mask: HxW bool
        p = pred.clone()
        l = label.clone()
        l = l.clone()
        l[~mask] = 255
        accumulate_confmat(p, l, conf, K)

    n_done = 0
    for rgb, depth, label in iter_n(loader, n):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        lc = decode_feats(model, backbone_outs(model, rgb, depth), rgb.shape[-2:]).argmax(1)
        lm = decode_feats(model, backbone_outs(model, rgb, entire_missing(depth)), rgb.shape[-2:]).argmax(1)
        accumulate_confmat(lc, label, conf_c, K)
        accumulate_confmat(lm, label, conf_m, K)

        # boundary: label differs from shifted self
        lab = label[0]
        boundary = torch.zeros_like(lab, dtype=torch.bool)
        boundary[1:] |= lab[1:] != lab[:-1]
        boundary[:-1] |= lab[1:] != lab[:-1]
        boundary[:, 1:] |= lab[:, 1:] != lab[:, :-1]
        boundary[:, :-1] |= lab[:, 1:] != lab[:, :-1]
        boundary &= lab != 255
        interior = (lab != 255) & (~boundary)
        mask_conf(lc[0], lab, conf_c_b, boundary)
        mask_conf(lm[0], lab, conf_m_b, boundary)
        mask_conf(lc[0], lab, conf_c_i, interior)
        mask_conf(lm[0], lab, conf_m_i, interior)

        # far depth: top 25% of valid depth (channel 0)
        d0 = depth[0, 0]
        valid = lab != 255
        if valid.any():
            thr = torch.quantile(d0[valid].float(), 0.75)
            far = valid & (d0 >= thr)
            near = valid & (d0 < thr)
            mask_conf(lc[0], lab, conf_c_far, far)
            mask_conf(lm[0], lab, conf_m_far, far)
            mask_conf(lc[0], lab, conf_c_near, near)
            mask_conf(lm[0], lab, conf_m_near, near)
        n_done += 1
        if n_done % 40 == 0:
            print(f"  Q3 {n_done}/{n}")

    def per_class(conf):
        conf = conf.float()
        inter = conf.diag()
        union = conf.sum(0) + conf.sum(1) - inter
        iou = torch.where(union > 0, inter / union.clamp_min(1e-8), torch.zeros_like(inter))
        return iou.cpu()

    iou_c = per_class(conf_c)
    iou_m = per_class(conf_m)
    delta = (iou_c - iou_m).tolist()
    # contribution approx: equal weight among classes with union>0 in either
    class_names = list(getattr(config, "class_names", [str(i) for i in range(K)]))
    if len(class_names) < K:
        class_names += [str(i) for i in range(len(class_names), K)]
    rows = []
    for i in range(K):
        rows.append({
            "class_id": i,
            "name": class_names[i],
            "iou_clean": float(iou_c[i]),
            "iou_missing": float(iou_m[i]),
            "delta": float(iou_c[i] - iou_m[i]),
        })
    rows_sorted = sorted(rows, key=lambda r: -r["delta"])
    macro_c = miou_from_conf(conf_c)
    macro_m = miou_from_conf(conf_m)
    top3 = rows_sorted[:3]
    top3_share = sum(r["delta"] for r in top3) / max(macro_c - macro_m, 1e-8) / K
    # better: share of sum of positive deltas
    pos = [r["delta"] for r in rows if r["delta"] > 0]
    top3_of_pos = sum(r["delta"] for r in top3) / max(sum(pos), 1e-8)

    out = {
        "n": n_done,
        "macro_clean": macro_c,
        "macro_missing": macro_m,
        "macro_delta": macro_c - macro_m,
        "per_class_sorted_by_delta": rows_sorted,
        "top3_classes": top3,
        "top3_share_of_positive_deltas": top3_of_pos,
        "boundary": {
            "clean_miou": miou_from_conf(conf_c_b),
            "missing_miou": miou_from_conf(conf_m_b),
            "delta": miou_from_conf(conf_c_b) - miou_from_conf(conf_m_b),
        },
        "interior": {
            "clean_miou": miou_from_conf(conf_c_i),
            "missing_miou": miou_from_conf(conf_m_i),
            "delta": miou_from_conf(conf_c_i) - miou_from_conf(conf_m_i),
        },
        "far_depth": {
            "clean_miou": miou_from_conf(conf_c_far),
            "missing_miou": miou_from_conf(conf_m_far),
            "delta": miou_from_conf(conf_c_far) - miou_from_conf(conf_m_far),
        },
        "near_depth": {
            "clean_miou": miou_from_conf(conf_c_near),
            "missing_miou": miou_from_conf(conf_m_near),
            "delta": miou_from_conf(conf_c_near) - miou_from_conf(conf_m_near),
        },
    }
    save("Q3_anatomy", out)
    review("Q3 anatomy", {
        "macro_delta": out["macro_delta"],
        "top3": top3,
        "top3_share_of_positive_deltas": top3_of_pos,
        "boundary_delta": out["boundary"]["delta"],
        "interior_delta": out["interior"]["delta"],
        "far_delta": out["far_depth"]["delta"],
        "near_delta": out["near_depth"]["delta"],
    })
    return out


# ===================== Q5 transforms =====================

def depth_transforms() -> Dict[str, Callable]:
    def invert(d):
        return d.max() + d.min() - d

    def rank_order(d):
        # per-image rank / N on first channel, broadcast
        out = []
        for i in range(d.shape[0]):
            x = d[i, 0].reshape(-1)
            order = x.argsort()
            ranks = torch.empty_like(x, dtype=torch.float32)
            ranks[order] = torch.linspace(0, 1, x.numel(), device=d.device)
            y = d[i].clone()
            y[0] = ranks.reshape_as(d[i, 0])
            if y.shape[0] > 1:
                y[1:] = y[0:1]
            out.append(y)
        return torch.stack(out, 0)

    def quantize(d, bins):
        x = d.clone()
        # assume roughly [0,1] or scale by min max per image
        for i in range(x.shape[0]):
            lo, hi = x[i].min(), x[i].max()
            t = (x[i] - lo) / (hi - lo + 1e-6)
            q = torch.floor(t * bins).clamp(0, bins - 1) / max(bins - 1, 1)
            x[i] = q * (hi - lo) + lo
        return x

    def abs_centered(d):
        out = d.clone()
        for i in range(d.shape[0]):
            med = d[i].median()
            out[i] = (d[i] - med).abs()
        return out

    def edge_sobel(d):
        # simple finite differences on ch0
        out = torch.zeros_like(d)
        x = d[:, 0:1]
        dx = F.pad(x[:, :, :, 1:] - x[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(x[:, :, 1:, :] - x[:, :, :-1, :], (0, 0, 0, 1))
        e = (dx.abs() + dy.abs())
        e = e / e.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        out[:, 0:1] = e
        if out.shape[1] > 1:
            out[:, 1:] = e
        return out

    def piecewise_4(d):
        # 2x2 tile means
        b, c, h, w = d.shape
        out = d.clone()
        hs, ws = h // 2, w // 2
        for i in range(2):
            for j in range(2):
                sl = d[:, :, i * hs:(i + 1) * hs if i == 0 else h, j * ws:(j + 1) * ws if j == 0 else w]
                # fix slices
        # simpler: avg pool then upsample
        small = F.avg_pool2d(d, kernel_size=(max(h // 2, 1), max(w // 2, 1)))
        return F.interpolate(small, size=(h, w), mode="nearest")

    return {
        "clean": lambda d: d,
        "inverted": invert,
        "rank_order": rank_order,
        "quantize_8": lambda d: quantize(d, 8),
        "quantize_4": lambda d: quantize(d, 4),
        "abs_centered": abs_centered,
        "edge_sobel": edge_sobel,
        "piecewise_4": piecewise_4,
        "entire_missing": entire_missing,
        "zeros": lambda d: torch.zeros_like(d),
    }


@torch.no_grad()
def q5_transform_sweep(device: str, n: int = 120):
    print("\n>>> Q5 transform sweep (S)")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=2)
    tfs = depth_transforms()
    results = {}
    for name, fn in tfs.items():
        r = eval_miou(model, loader, n, device, fn)
        results[name] = {k: r[k] for k in ("miou", "pixel_acc", "mean_ce", "n")}
        print(f"  {name}: mIoU={results[name]['miou']:.4f}")
    clean = results["clean"]["miou"]
    equiv = [k for k, v in results.items() if abs(v["miou"] - clean) <= 0.01]
    save("Q5_transform_sweep", {"n": n, "results": results, "clean_equivalent_within_1pt": equiv})
    review("Q5 transforms", {"miou": {k: v["miou"] for k, v in results.items()}, "clean_equiv": equiv})
    return results


# ===================== Q4 block confinement =====================

@torch.no_grad()
def forward_mixed_geo_blocks(model, rgb, depth_c, depth_m, missing_block_ids: set):
    """Use missing depth for Geo on selected global block indices; clean elsewhere."""
    bb = model.backbone
    x = bb.patch_embed(rgb)
    x_e_c = depth_c[:, 0:1]
    x_e_m = depth_m[:, 0:1]
    outs = []
    x_stream = x
    bidx = 0
    for i, layer in enumerate(bb.layers):
        x_cur = x_stream
        for blk in layer.blocks:
            x_in = x_cur + blk.cnn_pos_encode(x_cur)
            x_e = x_e_m if bidx in missing_block_ids else x_e_c
            geo = blk.Geo((x_in.shape[1], x_in.shape[2]), x_e, split_or_not=layer.split_or_not)
            if blk.layerscale:
                y = x_in + blk.drop_path(blk.gamma_1 * blk.Attention(blk.layer_norm1(x_in), geo, layer.split_or_not))
                y = y + blk.drop_path(blk.gamma_2 * blk.ffn(blk.layer_norm2(y)))
            else:
                y = x_in + blk.drop_path(blk.Attention(blk.layer_norm1(x_in), geo, layer.split_or_not))
                y = y + blk.drop_path(blk.ffn(blk.layer_norm2(y)))
            x_cur = y
            bidx += 1
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


@torch.no_grad()
def q4_block_confinement(device: str, n: int = 60):
    print("\n>>> Q4 block confinement (S)")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=1)
    # depths S: [3,4,18,4]
    ranges = {"stage0": set(range(0, 3)), "stage1": set(range(3, 7)), "stage2": set(range(7, 25)), "stage3": set(range(25, 29))}
    subsets = {
        "none_all_clean": set(),
        "all_missing": set(range(29)),
        "stage0": ranges["stage0"],
        "stage1": ranges["stage1"],
        "stage2": ranges["stage2"],
        "stage3": ranges["stage3"],
        "stage2_first_1": set(range(7, 8)),
        "stage2_first_3": set(range(7, 10)),
        "stage2_first_6": set(range(7, 13)),
        "stage2_first_12": set(range(7, 19)),
        "stage2_first_18": set(range(7, 25)),
    }
    confs = {k: torch.zeros(K, K, device=device) for k in subsets}
    n_done = 0
    for rgb, depth, label in iter_n(loader, n):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        dm = entire_missing(depth)
        hw = rgb.shape[-2:]
        for key, ids in subsets.items():
            outs = forward_mixed_geo_blocks(model, rgb, depth, dm, ids)
            accumulate_confmat(decode_feats(model, outs, hw).argmax(1), label, confs[key], K)
        n_done += 1
        if n_done % 15 == 0:
            print(f"  Q4 {n_done}/{n}")
    mc = miou_from_conf(confs["none_all_clean"])
    mm = miou_from_conf(confs["all_missing"])

    def rec(m):
        den = mc - mm
        return float((m - mm) / den) if abs(den) > 1e-8 else 0.0

    results = {
        "n": n_done,
        "clean_miou": mc,
        "all_missing_miou": mm,
        "subsets": {
            k: {
                "miou": miou_from_conf(confs[k]),
                "recovery_vs_all_missing": rec(miou_from_conf(confs[k])),
                "damage_vs_clean": mc - miou_from_conf(confs[k]),
                "n_blocks_missing_geo": len(subsets[k]),
            }
            for k in subsets
        },
    }
    save("Q4_block_confinement", results)
    review("Q4 block confinement", {k: v["damage_vs_clean"] for k, v in results["subsets"].items()})
    return results


# ===================== Q2 learnability =====================

def q2_learnability(device: str, n_train: int = 80, n_test: int = 80):
    print("\n>>> Q2 learnability probes")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=1)

    # collect
    gaps_c, gaps_m, vecs, deltas = [], [], [], []
    feats_m_full = []  # keep stage3 for translator eval later - store GAP only for memory
    count = 0
    for rgb, depth, label in iter_n(loader, n_train + n_test):
        rgb, depth = rgb.to(device), depth.to(device)
        fc = backbone_outs(model, rgb, depth)[3]
        fm = backbone_outs(model, rgb, entire_missing(depth))[3]
        gap_c = fc.mean(dim=(2, 3)).float()[0].cpu()
        gap_m = fm.mean(dim=(2, 3)).float()[0].cpu()
        delta = (fc - fm).float()[0]
        # top left singular vector
        X = delta.reshape(delta.shape[0], -1)
        try:
            U, S, Vh = torch.linalg.svd(X, full_matrices=False)
            v = U[:, 0].cpu()
        except RuntimeError:
            v = torch.zeros(delta.shape[0])
        gaps_c.append(gap_c)
        gaps_m.append(gap_m)
        vecs.append(v)
        count += 1
        if count % 30 == 0:
            print(f"  Q2 collect {count}/{n_train+n_test}")

    Gc = torch.stack(gaps_c)
    Gm = torch.stack(gaps_m)
    V = torch.stack(vecs)
    # cluster on train residual vectors (use top-8 projector affinity style: just kmeans on V)
    n_tr = min(n_train, len(Gm))
    Vtr = V[:n_tr]
    # kmeans 4
    torch.manual_seed(0)
    cents = Vtr[torch.randperm(n_tr)[:4]].clone()
    for _ in range(25):
        dist = ((Vtr.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(-1)
        lab = dist.argmin(1)
        for c in range(4):
            if (lab == c).any():
                cents[c] = Vtr[lab == c].mean(0)
    # labels for all
    dist_all = ((V.unsqueeze(1) - cents.unsqueeze(0)) ** 2).sum(-1)
    labs = dist_all.argmin(1)

    # linear probe: Gm -> cluster (ridge)
    Xtr, ytr = Gm[:n_tr], labs[:n_tr]
    Xte, yte = Gm[n_tr:], labs[n_tr:]
    # one-hot ridge
    C = 4
    Y = F.one_hot(ytr, C).float()
    # solve W = (X^T X + lI)^{-1} X^T Y
    Xt = Xtr
    A = Xt.T @ Xt + 1e-2 * torch.eye(Xt.shape[1])
    W = torch.linalg.solve(A, Xt.T @ Y)
    pred_tr = (Xt @ W).argmax(1)
    pred_te = (Xte @ W).argmax(1) if len(Xte) else pred_tr[:0]
    acc_tr = float((pred_tr == ytr).float().mean())
    acc_te = float((pred_te == yte).float().mean()) if len(Xte) else float("nan")
    chance = 0.25

    # predict top singular vector from Gm (linear)
    # V ~ Gm W
    A2 = Xt.T @ Xt + 1e-2 * torch.eye(Xt.shape[1])
    W2 = torch.linalg.solve(A2, Xt.T @ Vtr)
    Vhat_te = Xte @ W2 if len(Xte) else Xt[:0] @ W2
    Vte = V[n_tr:]
    if len(Vte):
        cos = F.cosine_similarity(Vhat_te, Vte, dim=1).mean().item()
    else:
        cos = float("nan")
    # train cos
    cos_tr = F.cosine_similarity(Xt @ W2, Vtr, dim=1).mean().item()

    # affine translator on GAP: predict gap_c from gap_m, apply as channel gain on feature map
    # better: learn A,b on GAP then scale channels of fm
    Yc = Gc[:n_tr]
    # Gm @ A.T + b ≈ Gc  -> use [Gm,1]
    ones = torch.ones(n_tr, 1)
    Xaug = torch.cat([Gm[:n_tr], ones], 1)
    # solve for each output dim
    Aaug = torch.linalg.lstsq(Xaug, Yc).solution  # (C+1, C_out) wait Yc is (n,C)
    # lstsq: Xaug (n, C+1), Yc (n, C) -> solution (C+1, C)
    # translate features and measure mIoU on test split
    conf_base = torch.zeros(K, K, device=device)
    conf_tr = torch.zeros(K, K, device=device)
    conf_clean = torch.zeros(K, K, device=device)
    # re-iterate test portion
    idx = 0
    n_eval = 0
    for rgb, depth, label in iter_n(loader, n_train + n_test):
        if idx < n_train:
            idx += 1
            continue
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        fc = backbone_outs(model, rgb, depth)
        fm = backbone_outs(model, rgb, entire_missing(depth))
        hw = rgb.shape[-2:]
        accumulate_confmat(decode_feats(model, fc, hw).argmax(1), label, conf_clean, K)
        accumulate_confmat(decode_feats(model, fm, hw).argmax(1), label, conf_base, K)
        # GAP translate -> channel scale
        g = fm[3].mean(dim=(2, 3))  # B,C
        g_aug = torch.cat([g, torch.ones(g.shape[0], 1, device=device)], 1)
        g_hat = g_aug @ Aaug.to(device)  # B,C
        # apply: replace channel means
        f = fm[3].float()
        mean = f.mean(dim=(2, 3), keepdim=True)
        f2 = f - mean + g_hat.unsqueeze(-1).unsqueeze(-1)
        hybrid = list(fm)
        hybrid[3] = f2.to(fm[3].dtype)
        accumulate_confmat(decode_feats(model, hybrid, hw).argmax(1), label, conf_tr, K)
        n_eval += 1
        idx += 1
        if n_eval >= n_test:
            break

    mc, mm, mt = miou_from_conf(conf_clean), miou_from_conf(conf_base), miou_from_conf(conf_tr)
    rec = float((mt - mm) / (mc - mm)) if abs(mc - mm) > 1e-8 else 0.0
    out = {
        "n_train": n_tr,
        "n_test": int(len(Xte)),
        "cluster_k": 4,
        "cluster_acc_train": acc_tr,
        "cluster_acc_test": acc_te,
        "cluster_chance": chance,
        "top_sv_cosine_train": cos_tr,
        "top_sv_cosine_test": cos,
        "translator_clean_miou": mc,
        "translator_missing_miou": mm,
        "translator_miou": mt,
        "translator_recovery": rec,
        "n_translator_eval": n_eval,
    }
    save("Q2_learnability", out)
    review("Q2 learnability", out)
    return out


# ===================== Q6 dual reader =====================

def q6_dual_reader(device: str, n_train: int = 100, n_test: int = 100):
    print("\n>>> Q6 dual reader probes")
    model, config, _ = load_variant("S", device)
    loader = make_loader(config, "val", batch_size=1)

    # Collect GAP features and multi-hot labels (class present)
    def collect(n):
        feats_c, feats_m, y = [], [], []
        for rgb, depth, label in iter_n(loader, n):
            rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
            fc = backbone_outs(model, rgb, depth)[3].mean(dim=(2, 3)).float()[0].cpu()
            fm = backbone_outs(model, rgb, entire_missing(depth))[3].mean(dim=(2, 3)).float()[0].cpu()
            lab = label[0]
            mh = torch.zeros(K)
            for c in range(K):
                if ((lab == c).any()):
                    mh[c] = 1.0
            feats_c.append(fc)
            feats_m.append(fm)
            y.append(mh)
        return torch.stack(feats_c), torch.stack(feats_m), torch.stack(y)

    Fc_tr, Fm_tr, Y_tr = collect(n_train)
    # test from next images - restart loader offset by n_train
    Fc_all, Fm_all, Y_all = collect(n_train + n_test)
    Fc_te, Fm_te, Y_te = Fc_all[n_train:], Fm_all[n_train:], Y_all[n_train:]
    Fc_tr, Fm_tr, Y_tr = Fc_all[:n_train], Fm_all[:n_train], Y_all[:n_train]

    def fit_ridge(X, Y, l2=1e-1):
        A = X.T @ X + l2 * torch.eye(X.shape[1])
        return torch.linalg.solve(A, X.T @ Y)

    def macro_f1(prob, Y, thr=0.5):
        pred = (prob > thr).float()
        # per-class f1 then macro
        f1s = []
        for c in range(K):
            p, t = pred[:, c], Y[:, c]
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            prec = tp / (tp + fp + 1e-6)
            rec = tp / (tp + fn + 1e-6)
            f1s.append(float((2 * prec * rec / (prec + rec + 1e-6))))
        return sum(f1s) / len(f1s)

    W_c = fit_ridge(Fc_tr, Y_tr)
    W_m = fit_ridge(Fm_tr, Y_tr)
    W_s = fit_ridge(torch.cat([Fc_tr, Fm_tr], 0), torch.cat([Y_tr, Y_tr], 0))

    matrix = {
        "clean_probe_on_clean": macro_f1(Fc_te @ W_c, Y_te),
        "clean_probe_on_missing": macro_f1(Fm_te @ W_c, Y_te),
        "missing_probe_on_clean": macro_f1(Fc_te @ W_m, Y_te),
        "missing_probe_on_missing": macro_f1(Fm_te @ W_m, Y_te),
        "shared_probe_on_clean": macro_f1(Fc_te @ W_s, Y_te),
        "shared_probe_on_missing": macro_f1(Fm_te @ W_s, Y_te),
    }
    out = {"n_train": n_train, "n_test": len(Y_te), "macro_f1": matrix}
    save("Q6_dual_reader", out)
    review("Q6 dual reader", matrix)
    return out


def main(argv: Optional[list] = None):
    import argparse

    ap = argparse.ArgumentParser(description="Phase-4 remaining diagnostic gates")
    ap.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated steps: Q1a,Q1b,Q1c,Q1f,Q3,Q5,Q4,Q2,Q6 (empty=all)",
    )
    ap.add_argument("--device", type=str, default="cuda:0")
    args = ap.parse_args(argv)

    device = args.device
    only = {x.strip().upper() for x in args.only.split(",") if x.strip()}
    order = ["Q1A", "Q1B", "Q1C", "Q1F", "Q3", "Q5", "Q4", "Q2", "Q6"]

    def want(tag: str) -> bool:
        return (not only) or (tag in only)

    t0 = time.time()
    all_r: dict = {}
    if want("Q1A"):
        all_r["Q1a"] = q1a_coupling(device)
    if want("Q1B"):
        all_r["Q1b"] = q1b_prior_disabled(device, n=150)
    if want("Q1C"):
        all_r["Q1c"] = q1c_amplify_bl(device, n=80)
    if want("Q1F"):
        all_r["Q1f"] = q1f_continuous(device, n=150)
    if want("Q3"):
        all_r["Q3"] = q3_anatomy(device, n=200)
    if want("Q5"):
        all_r["Q5"] = q5_transform_sweep(device, n=120)
    if want("Q4"):
        all_r["Q4"] = q4_block_confinement(device, n=60)
    if want("Q2"):
        all_r["Q2"] = q2_learnability(device, n_train=80, n_test=80)
    if want("Q6"):
        all_r["Q6"] = q6_dual_reader(device, n_train=100, n_test=100)
    all_r["seconds"] = time.time() - t0
    all_r["only"] = sorted(only) if only else list(order)
    save("ALL_PHASE4" if not only else f"ALL_PHASE4_{'_'.join(sorted(only))}", all_r)
    print("TOTAL seconds", all_r["seconds"])
    return all_r


if __name__ == "__main__":
    main()
