#!/usr/bin/env python
"""SUFFICIENCY test for C2's mechanism, by parameter transplant.

WHY. The current mechanism claim — "C2's gain comes from gradient-learned normalization
shifts" — rests on ELIMINATION: four candidate routes were ruled out (the depth-bias weight
is frozen; Q/K geometry is unchanged; the response curve is not flattened; re-estimated
running statistics account for only ~26%) and the remainder was assigned to norm shifts by
subtraction. That argument is only as good as the enumeration, and the enumeration was not
exhaustive — LayerNorm *weights*, the value pathway, and decoder weights were never
separately considered.

Elimination establishes necessity at best. This tests SUFFICIENCY: transplant a SUBSET of
C2's parameters into the baseline and measure how much of C2's improvement transfers. If
norm shifts alone recover most of it, the claim is established positively. If the
complement (everything except norms) recovers most of it, the elimination was wrong.

WHY TRANSPLANT IS MEANINGFUL HERE. The C2 *screen* checkpoint was finetuned FROM the trained
baseline, so the two networks are in the same loss basin and their parameters are
commensurable. (This would be meaningless for the ep500 C2 arm, which was trained from the
pretrained init independently — do not use that checkpoint here.)

Writes one transplanted checkpoint per group; score each with the FROZEN harness.

USAGE
  python diagnostics/param_transplant.py --list
  python diagnostics/param_transplant.py --group norm_bias
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
BASE = os.path.join(_PROJECT, "repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth")
# TRANSPLANT_C2_OVERRIDE lets the same groups be applied to a DIFFERENT finetuned-from-baseline
# checkpoint (e.g. the positional-alpha control arm), so two deltas can be decomposed identically
# and compared. TRANSPLANT_OUTDIR keeps those artifacts out of the C2 directory — without it the
# control would silently overwrite the C2 transplants this whole analysis rests on.
C2 = os.environ.get("TRANSPLANT_C2_OVERRIDE") or os.path.join(
    _PROJECT, "results/score_push/runs/l2_nof/c2_screen/ckpt_final_c2.pth"
)
OUTDIR = os.environ.get("TRANSPLANT_OUTDIR") or os.path.join(
    _PROJECT, "results/score_push/runs/l2_nof/transplant"
)

# Each group is a predicate over parameter names. Groups are designed so that
# `norm_all` and `complement` partition the changed parameters.
GROUPS = {
    # normalization SHIFTS only (biases), the thing the checkpoint diff flagged
    "norm_bias": lambda k: k.endswith(".bias") and (".norm" in k or "extra_norms" in k or ".bn." in k),
    # normalization running STATISTICS only (no gradients involved)
    "norm_stats": lambda k: k.endswith("running_mean") or k.endswith("running_var"),
    # normalization scales (LayerNorm/BN weights) — never separately considered before
    "norm_weight": lambda k: k.endswith(".weight") and (".norm" in k or "extra_norms" in k or ".bn." in k),
    # all normalization parameters together
    "norm_all": lambda k: (".norm" in k or "extra_norms" in k or ".bn." in k),
    # everything that is NOT normalization — the complement, the falsifier
    "complement": lambda k: not (".norm" in k or "extra_norms" in k or ".bn." in k),
    # decoder only, to separate encoder recalibration from decoder recalibration
    "decoder": lambda k: k.startswith("decode_head"),
    # sanity: transplant everything (must reproduce C2 exactly)
    "all": lambda k: True,
    # --- finer localization, added after the norm hypothesis was refuted ---
    "encoder_only": lambda k: k.startswith("backbone"),
    "stage0": lambda k: k.startswith("backbone.layers.0"),
    "stage1": lambda k: k.startswith("backbone.layers.1"),
    "stage2": lambda k: k.startswith("backbone.layers.2"),
    "stage3": lambda k: k.startswith("backbone.layers.3"),
    "attn_only": lambda k: any(t in k for t in (".q_proj", ".k_proj", ".v_proj", ".out_proj", ".lepe")),
    # --- SCORE path vs VALUE path, the thesis-discriminating split ---
    # `attn_only` lumps two functionally opposite pathways. In Decomposed_GSA the depth bias
    # (mask_h/mask_w from GeoPriorGen) is ADDED to qk_mat: softmax(qr @ kr^T + mask). So q/k are
    # the only projections that can re-calibrate the corrupted OPERATOR; v/out/lepe cannot touch
    # a score and can only compensate downstream in the REPRESENTATION. Attributing the repair to
    # "attention projections" does not distinguish these, and they mean opposite things for the
    # operator-corruption thesis.
    "attn_qk": lambda k: any(t in k for t in (".q_proj", ".k_proj")),
    "attn_vo": lambda k: any(t in k for t in (".v_proj", ".out_proj", ".lepe")),
    # stage-2-restricted (stage 2 alone carries ~82% of the repair)
    "stage2_attn_qk": lambda k: k.startswith("backbone.layers.2") and any(t in k for t in (".q_proj", ".k_proj")),
    "stage2_attn_vo": lambda k: k.startswith("backbone.layers.2")
    and any(t in k for t in (".v_proj", ".out_proj", ".lepe")),
    # --- NECESSITY (leave-one-out), the complement of the sufficiency split above ---
    # Transplant EVERYTHING except one pathway. Sufficiency says a subset is ENOUGH; necessity
    # says the rest is NOT enough without it. A pathway can be sufficient yet redundant, so the
    # two together are what license a localization claim.
    # --- lepe SPLIT: `lepe` is a DEPTHWISE 5x5 CONV (DWConv2d), i.e. a LOCAL SPATIAL OPERATOR,
    # not part of the value/representation pathway at all. Bundling it into "attn_vo" mislabels the
    # group. If lepe alone carries a large share, the reading is OPERATOR SUBSTITUTION (a depth-free
    # locality prior replacing the lost depth-conditioned geometric prior), NOT representation
    # tolerance. Corroborated a priori by MP-A ("smooth-only least damaging") and POS_LAMBDA=1.6.
    "attn_lepe": lambda k: ".lepe" in k,
    "attn_vout": lambda k: any(t in k for t in (".v_proj", ".out_proj")),
    "all_minus_qk": lambda k: not any(t in k for t in (".q_proj", ".k_proj")),
    "all_minus_vo": lambda k: not any(t in k for t in (".v_proj", ".out_proj", ".lepe")),
    "ffn_only": lambda k: ".ffn." in k,
    "downsample_only": lambda k: ".downsample." in k or "downsample_layers" in k,
}


def load(p):
    w = torch.load(p, map_location="cpu")
    return w.get("model", w.get("state_dict", w)) if isinstance(w, dict) else w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()

    base, c2 = load(BASE), load(C2)
    common = [k for k in base if k in c2 and base[k].shape == c2[k].shape]
    changed = [k for k in common if not torch.equal(base[k].float(), c2[k].float())]
    print(f"[info] common tensors {len(common)}, changed by C2: {len(changed)}")

    if a.list or a.group is None:
        print(f"\n{'group':14s} {'#changed in group':>18s} {'params':>14s}")
        for name, pred in GROUPS.items():
            sel = [k for k in changed if pred(k)]
            n = sum(base[k].numel() for k in sel)
            print(f"{name:14s} {len(sel):18d} {n:14,d}")
        print("\n(norm_all + complement should partition the changed set)")
        na = len([k for k in changed if GROUPS['norm_all'](k)])
        cm = len([k for k in changed if GROUPS['complement'](k)])
        print(f"  norm_all {na} + complement {cm} = {na+cm}  (changed = {len(changed)})")
        return

    pred = GROUPS[a.group]
    out = {k: v.clone() for k, v in base.items()}
    moved = [k for k in changed if pred(k)]
    for k in moved:
        out[k] = c2[k].clone()
    os.makedirs(OUTDIR, exist_ok=True)
    dst = os.path.join(OUTDIR, f"transplant_{a.group}.pth")
    torch.save({"model": out}, dst)
    n = sum(base[k].numel() for k in moved)
    print(f"[out] group={a.group}: transplanted {len(moved)} tensors ({n:,} params) -> {dst}")


if __name__ == "__main__":
    main()
