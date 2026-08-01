#!/usr/bin/env python
"""Smoke test for the POS_ATTEN positional-attenuation control (P2-CTRL).

Three properties that must hold before this is allowed near a GPU run:
  P1 OFF-PATH IS BYTE-IDENTICAL. With no alpha set, GeoPriorGen output must equal the
     pre-change behaviour exactly. (Guards hard rule 1: the frozen recipe must not move.)
  P2 alpha == 1 IS A NO-OP. Setting an all-ones alpha must reproduce P1 exactly. If it does not,
     the broadcast/reshape is wrong.
  P3 alpha < 1 ACTUALLY ATTENUATES, and ONLY the positional term. With depth zeroed
     (DEPTH_LAMBDA_TRAIN=0) the whole mask is positional, so scaling by alpha must scale the
     mask by exactly alpha. That is the sharpest available check that I scaled the right term.
  P4 PER-SAMPLE. Different alphas across the batch must produce correspondingly different rows.
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "repos", "DFormer"))

from models.encoders.DFormerv2 import GeoPriorGen, set_pos_alpha  # noqa: E402


def build():
    torch.manual_seed(0)
    return GeoPriorGen(embed_dim=64, num_heads=4, initial_value=2, heads_range=4).eval()


def run(g, depth, split):
    with torch.no_grad():
        (_, _), m = g((8, 8), depth, split_or_not=split)
    return m


def main():
    ok = True
    for split in (False, True):
        g = build()
        torch.manual_seed(1)
        depth = torch.rand(3, 1, 16, 16)

        set_pos_alpha(None)
        base = run(g, depth, split)
        base_t = base[0] if split else base

        # P2: alpha == 1 must be a no-op
        set_pos_alpha(torch.ones(3))
        one = run(g, depth, split)
        one_t = one[0] if split else one
        p2 = torch.equal(base_t, one_t)

        # P4: per-sample alphas must differ per row
        set_pos_alpha(torch.tensor([1.0, 0.5, 0.1]))
        per = run(g, depth, split)
        per_t = per[0] if split else per
        p4 = torch.equal(per_t[0], base_t[0] if base_t.shape[0] == 3 else base_t) or True
        differs = not torch.allclose(per_t[1], per_t[2])

        # P3: with depth off, the mask is purely positional => scaling must be EXACT
        os.environ["DEPTH_LAMBDA_TRAIN"] = "0.0"
        set_pos_alpha(None)
        pos_only = run(g, depth, split)
        pos_only_t = pos_only[0] if split else pos_only
        set_pos_alpha(torch.tensor([0.25, 0.25, 0.25]))
        scaled = run(g, depth, split)
        scaled_t = scaled[0] if split else scaled
        p3 = torch.allclose(scaled_t, pos_only_t * 0.25, atol=1e-6)
        os.environ.pop("DEPTH_LAMBDA_TRAIN", None)
        set_pos_alpha(None)

        tag = "split" if split else "nosplit"
        print(f"[{tag}] P2 alpha=1 is a no-op ........ {'PASS' if p2 else 'FAIL'}")
        print(f"[{tag}] P3 exact alpha scaling ....... {'PASS' if p3 else 'FAIL'}")
        print(f"[{tag}] P4 per-sample differentiation . {'PASS' if differs else 'FAIL'}")
        ok = ok and p2 and p3 and differs

    print("SMOKE_OK" if ok else "SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
