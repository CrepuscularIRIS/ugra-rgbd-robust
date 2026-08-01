#!/usr/bin/env python3
"""Two critical follow-ups:

1) Sequential recompute activation patching (backbone causal path)
2) Train-learned global channel basis vs per-image oracle vs random
   on held-out NYU test (stage 3, entire_missing)

Does NOT train a coefficient predictor — only oracle projections onto bases.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "repos" / "DFormer"))
os.chdir(ROOT / "repos" / "DFormer")

from diagnostics.backbone_forward import (
    forward_backbone_cached,
    forward_backbone_inject,
    outs_from_cache,
)
from diagnostics.corruptions.depth_corruptions import entire_missing
from diagnostics.hooks.data_utils import _to_tensor
from diagnostics.hooks.model_utils import load_dformerv2_s_nyu
from diagnostics.pipeline_all import accumulate_confmat, decode_feats, miou_from_conf, project_delta_topk


class _Eng:
    distributed = False
    local_rank = 0
    world_size = 1


def make_loader(config, split: str, batch_size: int = 2):
    """Deterministic ValPre loader for train or val split."""
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.dataloader.dataloader import ValPre
    from torch.utils import data

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
    assert split in ("train", "val")
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


def recovery(m_patch, m_corr, m_clean):
    den = m_clean - m_corr
    if abs(den) < 1e-8:
        return 0.0
    return float((m_patch - m_corr) / den)


def project_global_basis(delta: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """delta [B,C,H,W], B [C,k] orthonormal columns -> B B^T delta per location."""
    b, c, h, w = delta.shape
    # (B,HW,C) @ (C,k) -> (B,HW,k) @ (k,C) -> (B,C,H,W)
    x = delta.permute(0, 2, 3, 1).reshape(b, h * w, c)
    coef = x @ B  # B,HW,k
    recon = coef @ B.T  # B,HW,C
    return recon.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def collect_stage3_residuals(
    model, loader, device, max_images: int, stage: int = 3
) -> Tuple[torch.Tensor, int]:
    """Return matrix C x N_pixels of residuals (subsampled)."""
    chunks = []
    n_img = 0
    max_pix = 500_000
    for rgb, depth, label in iter_n(loader, max_images):
        rgb, depth = rgb.to(device), depth.to(device)
        bb = model.backbone
        outs_c, _ = forward_backbone_cached(bb, rgb, depth)
        outs_m, _ = forward_backbone_cached(bb, rgb, entire_missing(depth))
        delta = (outs_c[stage] - outs_m[stage]).float()  # B,C,H,W
        b, c, h, w = delta.shape
        chunks.append(delta.permute(1, 0, 2, 3).reshape(c, -1).cpu())
        n_img += b
        if sum(ch.shape[1] for ch in chunks) >= max_pix:
            break
        if n_img % 50 == 0:
            print(f"  collect residuals images={n_img}")
    M = torch.cat(chunks, dim=1)
    if M.shape[1] > max_pix:
        idx = torch.randperm(M.shape[1])[:max_pix]
        M = M[:, idx]
    return M, n_img


@torch.no_grad()
def learn_global_basis(M: torch.Tensor, k: int) -> torch.Tensor:
    """M: C x N -> B: C x k with orthonormal columns (top-k left singular vecs)."""
    # eigen of Gram C x C
    G = M @ M.T
    # eigh ascending
    evals, evecs = torch.linalg.eigh(G)
    B = evecs[:, -k:]  # C x k largest
    # orthonormal already
    return B.contiguous()


@torch.no_grad()
def random_basis(c: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(c, k, generator=g)
    Q, _ = torch.linalg.qr(A)
    return Q[:, :k].contiguous()


def main():
    device = "cuda:0"
    stage = 3
    k_list = [1, 2, 4, 8, 16, 32]
    n_train_basis = 400  # train images for global basis
    n_test = 500
    batch_size = 2
    out_dir = ROOT / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("==> load model")
    model, config = load_dformerv2_s_nyu(device=device)
    if not hasattr(config, "pad"):
        config.pad = False
    bb = model.backbone

    train_loader = make_loader(config, "train", batch_size=batch_size)
    test_loader = make_loader(config, "val", batch_size=batch_size)

    # ---------- Exp1: sequential recompute patching on TEST ----------
    print("==> Exp1: sequential recompute vs decoder-only patching")
    K = config.num_classes
    conf = {
        "clean": torch.zeros(K, K, device=device),
        "miss": torch.zeros(K, K, device=device),
        "dec_only": {s: torch.zeros(K, K, device=device) for s in range(4)},
        "sequential": {s: torch.zeros(K, K, device=device) for s in range(4)},
    }
    n = 0
    for rgb, depth, label in iter_n(test_loader, n_test):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        hw = rgb.shape[-2:]
        d0 = entire_missing(depth)

        outs_c, cache_c = forward_backbone_cached(bb, rgb, depth)
        outs_m, cache_m = forward_backbone_cached(bb, rgb, d0)

        accumulate_confmat(decode_feats(model, outs_c, hw).argmax(1), label, conf["clean"], K)
        accumulate_confmat(decode_feats(model, outs_m, hw).argmax(1), label, conf["miss"], K)

        for s in range(4):
            # decoder-only: swap stage s output in outs list (old method)
            hybrid = [outs_m[j] if j != s else outs_c[s] for j in range(4)]
            accumulate_confmat(
                decode_feats(model, hybrid, hw).argmax(1), label, conf["dec_only"][s], K
            )

            # sequential: inject after layer s, recompute later stages
            x_out_c, x_stream_c = cache_c[s]
            outs_seq = forward_backbone_inject(bb, rgb, d0, s, x_out_c, x_stream_c)
            accumulate_confmat(
                decode_feats(model, outs_seq, hw).argmax(1), label, conf["sequential"][s], K
            )

        n += rgb.shape[0]
        if n % 50 == 0:
            print(f"  sequential patch {n}/{n_test}")

    m_clean = miou_from_conf(conf["clean"])
    m_miss = miou_from_conf(conf["miss"])
    exp1 = {
        "n": n,
        "clean_miou": m_clean,
        "missing_miou": m_miss,
        "decoder_only_recovery": {},
        "sequential_recompute_recovery": {},
    }
    for s in range(4):
        md = miou_from_conf(conf["dec_only"][s])
        ms = miou_from_conf(conf["sequential"][s])
        exp1["decoder_only_recovery"][str(s)] = {
            "miou": md,
            "recovery": recovery(md, m_miss, m_clean),
        }
        exp1["sequential_recompute_recovery"][str(s)] = {
            "miou": ms,
            "recovery": recovery(ms, m_miss, m_clean),
        }
    print(json.dumps(exp1, indent=2))

    # ---------- Exp2: global basis from TRAIN, eval on TEST ----------
    print("==> Exp2: learn global basis on train residuals")
    M, n_tr = collect_stage3_residuals(model, train_loader, device, n_train_basis, stage=stage)
    Cdim = M.shape[0]
    print(f"  residual matrix C={Cdim}, Npix={M.shape[1]}, images~{n_tr}")

    bases = {}
    for k in k_list:
        bases[f"global_k{k}"] = learn_global_basis(M, k).to(device)
        bases[f"random_k{k}"] = random_basis(Cdim, k, seed=42 + k).to(device)

    # evaluate on test
    print("==> Exp2: test injection comparison")
    conf2 = {
        "clean": torch.zeros(K, K, device=device),
        "miss": torch.zeros(K, K, device=device),
        "oracle": {k: torch.zeros(K, K, device=device) for k in k_list},
        "global": {k: torch.zeros(K, K, device=device) for k in k_list},
        "random": {k: torch.zeros(K, K, device=device) for k in k_list},
        "full_stage_patch": torch.zeros(K, K, device=device),
    }
    n = 0
    for rgb, depth, label in iter_n(test_loader, n_test):
        rgb, depth, label = rgb.to(device), depth.to(device), label.to(device)
        hw = rgb.shape[-2:]
        d0 = entire_missing(depth)
        outs_c, _ = forward_backbone_cached(bb, rgb, depth)
        outs_m, _ = forward_backbone_cached(bb, rgb, d0)
        accumulate_confmat(decode_feats(model, outs_c, hw).argmax(1), label, conf2["clean"], K)
        accumulate_confmat(decode_feats(model, outs_m, hw).argmax(1), label, conf2["miss"], K)

        # full clean stage patch (decoder-only upper for stage)
        hybrid_full = [outs_m[j] if j != stage else outs_c[stage] for j in range(4)]
        accumulate_confmat(
            decode_feats(model, hybrid_full, hw).argmax(1), label, conf2["full_stage_patch"], K
        )

        delta = outs_c[stage] - outs_m[stage]
        for k in k_list:
            # per-image oracle
            dk = project_delta_topk(delta, k)
            hybrid = list(outs_m)
            hybrid[stage] = outs_m[stage] + dk
            accumulate_confmat(
                decode_feats(model, hybrid, hw).argmax(1), label, conf2["oracle"][k], K
            )
            # global shared basis (oracle coefficients: proj of true delta)
            dkg = project_global_basis(delta, bases[f"global_k{k}"])
            hybrid = list(outs_m)
            hybrid[stage] = outs_m[stage] + dkg
            accumulate_confmat(
                decode_feats(model, hybrid, hw).argmax(1), label, conf2["global"][k], K
            )
            # random basis + oracle coeffs on true delta
            dkr = project_global_basis(delta, bases[f"random_k{k}"])
            hybrid = list(outs_m)
            hybrid[stage] = outs_m[stage] + dkr
            accumulate_confmat(
                decode_feats(model, hybrid, hw).argmax(1), label, conf2["random"][k], K
            )

        n += rgb.shape[0]
        if n % 50 == 0:
            print(f"  global basis eval {n}/{n_test}")

    m_c = miou_from_conf(conf2["clean"])
    m_m = miou_from_conf(conf2["miss"])
    m_full = miou_from_conf(conf2["full_stage_patch"])
    exp2 = {
        "n_test": n,
        "n_train_basis_images": n_tr,
        "stage": stage,
        "clean_miou": m_c,
        "missing_miou": m_m,
        "full_stage3_clean_patch": {"miou": m_full, "recovery": recovery(m_full, m_m, m_c)},
        "per_image_oracle": {},
        "global_basis_oracle_coeffs": {},
        "random_basis_oracle_coeffs": {},
    }
    for k in k_list:
        mo = miou_from_conf(conf2["oracle"][k])
        mg = miou_from_conf(conf2["global"][k])
        mr = miou_from_conf(conf2["random"][k])
        exp2["per_image_oracle"][str(k)] = {"miou": mo, "recovery": recovery(mo, m_m, m_c)}
        exp2["global_basis_oracle_coeffs"][str(k)] = {
            "miou": mg,
            "recovery": recovery(mg, m_m, m_c),
        }
        exp2["random_basis_oracle_coeffs"][str(k)] = {
            "miou": mr,
            "recovery": recovery(mr, m_m, m_c),
        }

    # save bases
    torch.save(
        {f"global_k{k}": bases[f"global_k{k}"].cpu() for k in k_list},
        out_dir / "global_jspace_bases_stage3.pt",
    )

    results = {
        "seconds": time.time() - t0,
        "exp1_sequential_recompute_patching": exp1,
        "exp2_global_jspace": exp2,
        "interpretation_notes": {
            "exp1": "sequential recompute = true backbone path; decoder_only = previous method",
            "exp2": "global uses train-learned B + oracle projection of test delta; not a learned coeff head",
            "success_criterion_exp2": "oracle_k8 > global_k8 >> random_k8; global recovers 50-70% of gap is strong",
        },
    }
    path = out_dir / "sequential_and_global_jspace.json"
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print("WROTE", path)


if __name__ == "__main__":
    main()
