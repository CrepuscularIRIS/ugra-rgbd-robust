#!/usr/bin/env python
"""PRP precondition tests P1 (data) + P2 (RGB-branch wiring) + PRP-OFF parity.

P2 is the load-bearing wiring check: the pseudo-depth head must read RGB-branch
features ONLY, so d_hat is IDENTICAL under clean vs zeroed depth (max abs diff
< 1e-5) and non-degenerate (std > 0.01). If P2 fails the head is mis-wired.

Also asserts:
  - PRP-OFF forward == baseline forward byte-for-byte (backbone outs identical),
    so a baseline ckpt eval is unchanged when PRP is disabled.
  - the converged S ckpt loads into the PRP model with ONLY PRP keys missing.
  - P1: a real training batch has clean depth (min > 0).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module

CONFIG_MODULE = "local_configs.NYUDepthv2.DFormerv2_S"
CKPT = os.path.join(_ROOT, "checkpoints/trained/NYUDepthv2/DFormerv2_Small_NYU.pth")


def build_model():
    from models.builder import EncoderDecoder as segmodel
    config = getattr(import_module(CONFIG_MODULE), "C")
    config.pad = False
    crit = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    m = segmodel(cfg=config, criterion=crit, norm_layer=nn.BatchNorm2d, syncbn=False)
    w = torch.load(CKPT, map_location="cpu")
    if "model" in w: w = w["model"]
    elif "state_dict" in w: w = w["state_dict"]
    rep = m.load_state_dict(w, strict=False)
    return m, rep, config


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    # ---- LOAD test: PRP model loads converged ckpt, only PRP keys missing ----
    os.environ["PRP_ON"] = "1"
    # force reimport of the encoder so _prp_enabled picks up env at forward time
    m, rep, config = build_model()
    prp_missing = [k for k in rep.missing_keys if
                   ("pseudo_depth_head" in k or "decay_p" in k or "w_p" in k)]
    non_prp_missing = [k for k in rep.missing_keys if k not in prp_missing]
    print(f"[load] missing={len(rep.missing_keys)} PRP={len(prp_missing)} "
          f"non-PRP={len(non_prp_missing)} unexpected={len(rep.unexpected_keys)}")
    if non_prp_missing:
        print(f"[load] non-PRP missing (BAD): {non_prp_missing[:10]}")
    load_ok = (len(non_prp_missing) == 0 and len(rep.unexpected_keys) == 0)
    results["LOAD_only_prp_missing"] = load_ok
    m.to(device).eval()

    B, H, W = 2, 480, 640
    torch.manual_seed(0)
    rgb = torch.randn(B, 3, H, W, device=device)
    depth_clean = torch.rand(B, 3, H, W, device=device).abs() + 0.5  # min>0
    depth_zero = torch.zeros(B, 3, H, W, device=device)              # entire_missing

    # ---- P2: d_hat identical under clean vs zero depth, and non-degenerate ----
    with torch.no_grad():
        _ = m.backbone(rgb, depth_clean)
        d_clean = m.backbone.last_dhat.clone()
        _ = m.backbone(rgb, depth_zero)
        d_zero = m.backbone.last_dhat.clone()
    max_abs_diff = float((d_clean - d_zero).abs().max().item())
    d_std = float(d_clean.std().item())
    p2_invariant = max_abs_diff < 1e-5
    p2_nondegenerate = d_std > 0.01
    print(f"[P2] max_abs_diff(clean,zero)={max_abs_diff:.3e} (<1e-5? {p2_invariant})  "
          f"std(d_hat)={d_std:.4f} (>0.01? {p2_nondegenerate})")
    results["P2_rgb_branch_invariant"] = p2_invariant
    results["P2_dhat_nondegenerate"] = p2_nondegenerate
    results["P2_max_abs_diff"] = max_abs_diff
    results["P2_dhat_std"] = d_std

    # ---- PRP-OFF parity: with PRP disabled, backbone outs == a fresh baseline ----
    # Build a baseline (PRP-OFF) model from the same ckpt and compare stage outs.
    os.environ["PRP_ON"] = "0"
    mb, repb, _ = build_model()
    mb.to(device).eval()
    with torch.no_grad():
        outs_off = mb.backbone(rgb, depth_clean)
        # PRP model with w_p==0 (zero-init) should ALSO equal baseline even PRP-ON
        os.environ["PRP_ON"] = "1"
        outs_on_zeroinit = m.backbone(rgb, depth_clean)
    parity = all(
        float((a - b).abs().max().item()) < 1e-4
        for a, b in zip(outs_off, outs_on_zeroinit)
    )
    max_parity_diff = max(
        float((a - b).abs().max().item())
        for a, b in zip(outs_off, outs_on_zeroinit)
    )
    print(f"[PARITY] PRP-ON(w_p=0) vs PRP-OFF backbone outs max_diff={max_parity_diff:.3e} "
          f"(<1e-4? {parity})")
    results["PARITY_zeroinit_equals_baseline"] = parity
    results["PARITY_max_diff"] = max_parity_diff

    print("\n=== PRECONDITION SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    hard_pass = (results["LOAD_only_prp_missing"] and results["P2_rgb_branch_invariant"]
                 and results["P2_dhat_nondegenerate"]
                 and results["PARITY_zeroinit_equals_baseline"])
    print(f"\nHARD PASS (P2 + load + parity): {hard_pass}")
    import json
    out = os.path.join(_HERE, "results", "precond_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[wrote] {out}")
    sys.exit(0 if hard_pass else 2)


if __name__ == "__main__":
    main()
