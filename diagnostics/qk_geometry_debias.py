#!/usr/bin/env python
"""Does C2 implicitly de-bias depth via Q/K GEOMETRY rather than via the bias weight?

WHY. The group_weight bug freezes `GeoPriorGen.weight = [w0, w1]`, so C2 provably cannot
shrink the depth-bias magnitude directly (verified: byte-identical across all checkpoints).
Grok's verification then narrowed the resulting claim: ruling out the direct w1 route does
NOT rule out de-biasing, because the Q/K projections can rotate/rescale the CONTENT logits
so that the same fixed bias contributes proportionally less. That implicit route was listed
as untested, and it is a candidate explanation for the ~32% of the gap that C2 recovers
beyond what global scaling (+0.58) and statistic re-estimation (+0.74) can buy.

WHY THE TEST IS CLEAN. In Decomposed_GSA.forward the pre-softmax score is
      qk_mat = (qr @ kr^T)  +  mask
where `mask = w0*mask_pos + w1*mask_d` depends ONLY on the depth map, the position grid and
the FROZEN weights. Therefore for a fixed input, **`mask` is bitwise identical between the
baseline and C2** — nothing that C2 changed can affect it. Any change in the bias's relative
influence must come entirely from the CONTENT term. So we only need to measure the content
magnitude in both models on the same images.

MEASURED, per attention module:
    content = mean |qr @ kr^T|      (the content logits)
    bias    = mean |mask|           (identical across models; reported as a sanity check)
    ratio   = bias / content        (the bias's relative influence)
A LOWER ratio in C2 (i.e. LARGER content) means implicit de-biasing via Q/K geometry is
active. An unchanged ratio means the route is inactive and the residual must be explained
some other way.

USAGE
  CUDA_VISIBLE_DEVICES=1 python diagnostics/qk_geometry_debias.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module  # noqa: E402

from models.builder import EncoderDecoder as segmodel  # noqa: E402
from models.encoders.DFormerv2 import Decomposed_GSA, angle_transform  # noqa: E402
from utils.dataloader.dataloader import get_val_loader  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.engine.engine import Engine  # noqa: E402

CONFIG_MODULE = "local_configs.NYUDepthv2.DFormerv2_S"
_FWD_SEED = 12345
_REC: list = []


def _patched_forward(self, x, rel_pos, split_or_not=False):
    """Mirror of Decomposed_GSA.forward that records content/bias magnitudes.

    Only the _w branch is instrumented; the _h branch is structurally identical and would
    double the record for no extra information.
    """
    bsz, h, w, _ = x.size()
    (sin, cos), (mask_h, mask_w) = rel_pos
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
    lepe = self.lepe(v)
    k = k * self.scaling
    q = q.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
    k = k.view(bsz, h, w, self.num_heads, self.key_dim).permute(0, 3, 1, 2, 4)
    qr = angle_transform(q, sin, cos)
    kr = angle_transform(k, sin, cos)
    qr_w = qr.transpose(1, 2)
    kr_w = kr.transpose(1, 2)
    v = v.reshape(bsz, h, w, self.num_heads, -1).permute(0, 1, 3, 2, 4)

    content_w = qr_w @ kr_w.transpose(-1, -2)
    bias_w = mask_w.transpose(1, 2)
    _REC.append({
        "content_absmean": content_w.abs().mean().item(),
        "content_std": content_w.std().item(),
        "bias_absmean": bias_w.abs().mean().item(),
        "bias_std": bias_w.std().item(),
    })

    qk_mat_w = content_w + bias_w
    qk_mat_w = torch.softmax(qk_mat_w, -1)
    v = torch.matmul(qk_mat_w, v)

    qr_h = qr.permute(0, 3, 1, 2, 4)
    kr_h = kr.permute(0, 3, 1, 2, 4)
    v = v.permute(0, 3, 2, 1, 4)
    qk_mat_h = qr_h @ kr_h.transpose(-1, -2)
    qk_mat_h = qk_mat_h + mask_h.transpose(1, 2)
    qk_mat_h = torch.softmax(qk_mat_h, -1)
    output = torch.matmul(qk_mat_h, v)
    output = output.permute(0, 3, 1, 2, 4).flatten(-2, -1)
    output = output + lepe
    output = self.out_proj(output)
    return output


def build(ckpt, config, criterion):
    os.environ["DEPTH_LAMBDA_TRAIN"] = "1.0"
    m = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
    w = torch.load(ckpt, map_location="cpu")
    w = w.get("model", w.get("state_dict", w)) if isinstance(w, dict) else w
    # Verify the load. `strict=False` will happily match NOTHING and leave an untrained model that
    # still produces plausible-looking numbers — that failure mode cost three probes and a whole
    # false finding on 2026-07-25. Every script that builds its own model must assert the load.
    rep = m.load_state_dict(w, strict=False)
    matched = len(m.state_dict()) - len(rep.missing_keys)
    if matched < 0.9 * len(m.state_dict()):
        raise SystemExit(
            f"FATAL: checkpoint load matched only {matched}/{len(m.state_dict())} tensors "
            f"— refusing to run on an untrained model ({ckpt})"
        )
    print(f"[load] {ckpt} matched {matched}/{len(m.state_dict())} tensors")
    return m.cuda().eval()


def measure(model, samples):
    global _REC
    out = []
    for rgb, dep in samples:
        _REC = []
        torch.manual_seed(_FWD_SEED)
        torch.cuda.manual_seed_all(_FWD_SEED)
        with torch.no_grad():
            model(rgb, dep)
        out.append(list(_REC))
    return out


def main() -> None:
    base_ckpt = os.environ.get("BASE_CKPT", os.path.join(_ROOT, "checkpoints/trained/DFormerv2_Small_NYU.pth"))
    c2_ckpt = os.environ.get("C2_CKPT", os.path.join(
        _PROJECT, "results/score_push/runs/l2_nof/c2_screen/ckpt_final_c2.pth"))
    n_samples = int(os.environ.get("N_SAMPLES", "3"))

    Decomposed_GSA.forward = _patched_forward   # instrument both models identically
    torch.backends.cudnn.deterministic = True

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        config.pad = False
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
        val_loader, _ = get_val_loader(engine, RGBXDataset, config, val_batch_size=1)
        samples = []
        for i, mb in enumerate(val_loader):
            if i >= n_samples:
                break
            samples.append((mb["data"].cuda().float(), mb["modal_x"].cuda().float()))

        base = build(base_ckpt, config, criterion)
        rec_b = measure(base, samples)
        del base; torch.cuda.empty_cache()
        c2 = build(c2_ckpt, config, criterion)
        rec_c = measure(c2, samples)

        nb = len(rec_b[0])
        print(f"[info] {nb} instrumented attention modules, {len(samples)} samples\n")

        # Sanity gate: the bias MUST be identical across models (frozen weights, same input).
        bias_b = np.mean([[r["bias_absmean"] for r in s] for s in rec_b], axis=0)
        bias_c = np.mean([[r["bias_absmean"] for r in s] for s in rec_c], axis=0)
        bias_rel = float(np.max(np.abs(bias_b - bias_c) / np.maximum(bias_b, 1e-12)))
        print(f"[gate] max relative bias difference between models = {bias_rel:.3e}")
        if bias_rel > 1e-5:
            print("  !! bias differs between models — the frozen-weight premise is violated;")
            print("     the ratio comparison below would be confounded. INVESTIGATE before reading on.")
        else:
            print("  bias is identical as predicted (frozen w0/w1) => only the CONTENT term can differ.\n")

        cont_b = np.mean([[r["content_absmean"] for r in s] for s in rec_b], axis=0)
        cont_c = np.mean([[r["content_absmean"] for r in s] for s in rec_c], axis=0)
        ratio_b = bias_b / np.maximum(cont_b, 1e-12)
        ratio_c = bias_c / np.maximum(cont_c, 1e-12)

        print(f"{'blk':>4} {'content base':>13} {'content C2':>12} {'d%':>8} "
              f"{'bias/cont base':>15} {'bias/cont C2':>13} {'d%':>8}")
        for i in range(nb):
            dc = 100.0 * (cont_c[i] - cont_b[i]) / max(cont_b[i], 1e-12)
            dr = 100.0 * (ratio_c[i] - ratio_b[i]) / max(ratio_b[i], 1e-12)
            print(f"{i:>4} {cont_b[i]:13.4f} {cont_c[i]:12.4f} {dc:+7.2f}% "
                  f"{ratio_b[i]:15.4f} {ratio_c[i]:13.4f} {dr:+7.2f}%")

        d_cont = 100.0 * (cont_c.mean() - cont_b.mean()) / cont_b.mean()
        d_ratio = 100.0 * (ratio_c.mean() - ratio_b.mean()) / ratio_b.mean()
        print("\n================ VERDICT ================")
        print(f"  mean content magnitude : base {cont_b.mean():.4f} -> C2 {cont_c.mean():.4f}  ({d_cont:+.2f}%)")
        print(f"  mean bias/content ratio: base {ratio_b.mean():.4f} -> C2 {ratio_c.mean():.4f}  ({d_ratio:+.2f}%)")
        if d_ratio < -5.0:
            print("  => IMPLICIT DE-BIASING VIA Q/K GEOMETRY IS ACTIVE: C2 grew the content term, so the")
            print("     same frozen depth bias exerts proportionally less influence. This is the route")
            print("     grok flagged as untested, and it is a candidate for the unexplained residual.")
        elif d_ratio > 5.0:
            print("  => OPPOSITE of implicit de-biasing: C2 made the depth bias RELATIVELY STRONGER.")
        else:
            print("  => ROUTE INACTIVE: the bias/content ratio is essentially unchanged, so C2 did NOT")
            print("     de-bias depth via Q/K geometry. The residual must be explained otherwise.")

        out_json = os.path.join(_PROJECT, "results/score_push/runs/l2_nof/qk_geometry_debias.json")
        with open(out_json, "w") as f:
            json.dump({"content_base": cont_b.tolist(), "content_c2": cont_c.tolist(),
                       "bias_base": bias_b.tolist(), "bias_c2": bias_c.tolist(),
                       "ratio_base": ratio_b.tolist(), "ratio_c2": ratio_c.tolist(),
                       "bias_identical_rel": bias_rel,
                       "d_content_pct": d_cont, "d_ratio_pct": d_ratio}, f, indent=2)
        print(f"[out] {out_json}")


if __name__ == "__main__":
    main()
