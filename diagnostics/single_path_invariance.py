#!/usr/bin/env python
"""T3 SINGLE-PATH / NEUTRAL-ELEMENT audit for DFormerv2 (L2-NOF thesis linchpin).

WHY THIS EXISTS
---------------
The operator-corruption thesis claims depth enters the attention operator ONLY
through the additive relational bias

    mask = weight[0]*mask_pos + weight[1]*mask_d,    mask_d = |d_i - d_j| * decay

and therefore that "depth absent" is not an off-support *input* so much as a
specific, well-defined *operator* (the spatial-only one, mask_d = 0).

The standard referee objection (verbatim from the round-2 grill) is:

    "'very weak but present depth' and 'no depth input exists' are not
     automatically neighboring points. Attention normalization, biases,
     normalization statistics, positional embeddings and cross-modal routing may
     all behave discontinuously when an entire stream is removed. A continuity
     claim would require an architecture with a defined neutral element."

Two mIoU numbers agreeing to 0.05 (clean@lambda=0 = 49.12 vs entire_missing =
49.07) is WEAK evidence for that, and the 0.05 was never triaged. This script
settles it at the tensor level instead.

THE BATTERY (all on a frozen trained checkpoint, eval-only, no training)
-----------------------------------------------------------------------
  A. INVARIANCE  - at lambda=0, the logits must be BITWISE identical for real /
                   zero / random / constant depth. Any difference means a second
                   depth pathway exists and the single-path premise is FALSE.
  B. EQUIVALENCE - lambda=1 + zero depth  (zero-fill realization of absence)
                   must be BITWISE identical to
                   lambda=0 + real depth  (structural removal of the term).
                   This is the referee objection answered directly: for this
                   architecture the three ways to realize absence coincide.
  C. POSITIVE    - lambda=1 real vs lambda=1 zero MUST differ. Without this the
     CONTROL      whole battery could pass vacuously (e.g. a dead depth input).
  D. CONTINUITY  - sweep lambda and report the logit-space distance to the
                   lambda=0 endpoint, checking it goes smoothly to 0 with no
                   cliff (the continuity premise behind C2 attenuation).

Severity: if the thesis premise were false, A and B would be expected to FAIL
(nonzero diffs), and C guards against a vacuous pass. This test can therefore
kill the premise, which is what makes it worth running.

USAGE
  CKPT=checkpoints/trained/DFormerv2_Small_NYU.pth \
  CUDA_VISIBLE_DEVICES=1 N_SAMPLES=4 \
  python diagnostics/single_path_invariance.py
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
from utils.dataloader.dataloader import get_val_loader  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.engine.engine import Engine  # noqa: E402

CONFIG_MODULE = "local_configs.NYUDepthv2.DFormerv2_S"


def _set_lambda(v: float) -> None:
    """Set the forward-time depth-authority dial (see DFormerv2._depth_lambda)."""
    os.environ["DEPTH_LAMBDA_TRAIN"] = repr(float(v))


def main() -> None:
    ckpt = os.environ.get("CKPT", "checkpoints/trained/DFormerv2_Small_NYU.pth")
    n_samples = int(os.environ.get("N_SAMPLES", "4"))
    out_json = os.environ.get(
        "OUT_JSON",
        os.path.join(_PROJECT, "results", "score_push", "runs", "l2_nof", "single_path_invariance.json"),
    )

    torch.manual_seed(0)
    np.random.seed(0)
    # Determinism matters: a nondeterministic kernel would masquerade as a
    # second depth pathway in test A.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        config.pad = False
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)

        _set_lambda(1.0)
        model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
        weight = torch.load(ckpt, map_location="cpu")
        if "model" in weight:
            weight = weight["model"]
        elif "state_dict" in weight:
            weight = weight["state_dict"]
        rep = model.load_state_dict(weight, strict=False)
        print(f"[load] {ckpt}  missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        val_loader, _ = get_val_loader(engine, RGBXDataset, config, val_batch_size=1)

        # Collect a few real (rgb, normalized-depth) pairs.
        samples = []
        for i, mb in enumerate(val_loader):
            if i >= n_samples:
                break
            rgb = mb["data"].to(device).float()
            dep = mb["modal_x"].to(device).float()
            samples.append((rgb, dep))
        print(f"[data] {len(samples)} val samples, rgb={tuple(samples[0][0].shape)} depth={tuple(samples[0][1].shape)}")

        # The Ham (Hamburger/NMF) decoder re-draws its bases with torch.rand on
        # EVERY forward when RAND_INIT=True (models/decoders/ham_head.py:80), so
        # two identical calls do NOT agree bitwise. That randomness lives
        # downstream of the encoder and is unrelated to depth, but it would
        # swamp this audit. Reseeding immediately before each forward makes every
        # call draw the SAME bases, which restores determinism and lets the
        # comparison isolate the depth pathway. (It also explains the previously
        # untriaged 0.05 mIoU gap between clean@lambda=0 and entire_missing: the
        # frozen harness seeds once per run, so different conditions consume
        # different amounts of RNG and land on different bases.)
        _FWD_SEED = 12345

        def fwd(rgb, dep, lam):
            _set_lambda(lam)
            torch.manual_seed(_FWD_SEED)
            torch.cuda.manual_seed_all(_FWD_SEED)
            with torch.no_grad():
                return model(rgb, dep)

        def enc(rgb, dep, lam):
            """Backbone-only features: depth acts ONLY here (GeoPriorGen), so this
            is the sharpest place to test the single-path claim, free of decoder
            randomness."""
            _set_lambda(lam)
            torch.manual_seed(_FWD_SEED)
            torch.cuda.manual_seed_all(_FWD_SEED)
            with torch.no_grad():
                return model.backbone(rgb, dep)

        # Self-consistency gate: two identical calls MUST agree, or every other
        # number in this script is meaningless.
        _s0, _d0 = samples[0]
        _r1, _r2 = fwd(_s0, _d0, 1.0), fwd(_s0, _d0, 1.0)
        _selfdiff = (_r1 - _r2).abs().max().item()
        print(f"[gate] repeat-call self-consistency max|diff| = {_selfdiff:.3e} "
              f"({'DETERMINISTIC' if _selfdiff == 0.0 else 'STILL NONDETERMINISTIC - results invalid'})")
        if _selfdiff != 0.0:
            raise SystemExit("ABORT: forward is nondeterministic even after reseeding; fix before interpreting.")

        results = {"ckpt": ckpt, "n_samples": len(samples), "per_sample": []}

        for si, (rgb, dep) in enumerate(samples):
            zero = torch.zeros_like(dep)
            rand = torch.randn_like(dep)
            const = torch.full_like(dep, 0.37)

            # --- A: invariance at lambda = 0 -------------------------------
            a_real = fwd(rgb, dep, 0.0)
            a_zero = fwd(rgb, zero, 0.0)
            a_rand = fwd(rgb, rand, 0.0)
            a_const = fwd(rgb, const, 0.0)

            def maxdiff(x, y):
                return (x - y).abs().max().item()

            A = {
                "real_vs_zero": maxdiff(a_real, a_zero),
                "real_vs_random": maxdiff(a_real, a_rand),
                "real_vs_const": maxdiff(a_real, a_const),
                "bitwise_all_equal": bool(
                    torch.equal(a_real, a_zero)
                    and torch.equal(a_real, a_rand)
                    and torch.equal(a_real, a_const)
                ),
            }

            # A' : the same invariance at the ENCODER output, where depth acts.
            # Independent of the decoder entirely.
            e_real = enc(rgb, dep, 0.0)
            e_zero = enc(rgb, zero, 0.0)
            e_rand = enc(rgb, rand, 0.0)
            e_stage_diffs = [
                max(maxdiff(x, y), maxdiff(x, z))
                for x, y, z in zip(e_real, e_zero, e_rand)
            ]
            A["encoder_stage_maxdiffs"] = e_stage_diffs
            A["encoder_bitwise_all_equal"] = bool(
                all(torch.equal(x, y) for x, y in zip(e_real, e_zero))
                and all(torch.equal(x, z) for x, z in zip(e_real, e_rand))
            )

            # --- B: zero-fill absence  ==  structural term removal ---------
            b_zerofill = fwd(rgb, zero, 1.0)   # entire_missing, as the harness defines it
            b_removed = fwd(rgb, dep, 0.0)     # term algebraically removed
            B = {
                "maxdiff": maxdiff(b_zerofill, b_removed),
                "bitwise_equal": bool(torch.equal(b_zerofill, b_removed)),
            }

            # --- C: positive control (battery must not pass vacuously) -----
            c_real = fwd(rgb, dep, 1.0)
            C = {
                "lam1_real_vs_lam1_zero_maxdiff": maxdiff(c_real, b_zerofill),
                "differs": not torch.equal(c_real, b_zerofill),
            }

            # --- D: continuity toward the absent endpoint ------------------
            D = {}
            for lam in (0.0, 0.05, 0.25, 0.5, 0.75, 1.0):
                D[f"lam={lam}"] = maxdiff(fwd(rgb, dep, lam), b_removed)

            results["per_sample"].append({"A": A, "B": B, "C": C, "D": D})
            print(f"\n--- sample {si} ---")
            print(f"  A invariance@lam0 : logits bitwise={A['bitwise_all_equal']} "
                  f"max|d|(real,zero)={A['real_vs_zero']:.3e} rand={A['real_vs_random']:.3e}")
            print(f"  A' encoder@lam0   : bitwise={A['encoder_bitwise_all_equal']} "
                  f"per-stage max|d|={['%.2e' % v for v in A['encoder_stage_maxdiffs']]}")
            print(f"  B zerofill==removed: bitwise={B['bitwise_equal']} max|d|={B['maxdiff']:.3e}")
            print(f"  C positive control : differs={C['differs']} max|d|={C['lam1_real_vs_lam1_zero_maxdiff']:.3e}")
            print("  D continuity      : " + "  ".join(f"{k}->{v:.3e}" for k, v in D.items()))

        # ---- verdicts -------------------------------------------------------
        allA = all(s["A"]["bitwise_all_equal"] for s in results["per_sample"])
        allAe = all(s["A"]["encoder_bitwise_all_equal"] for s in results["per_sample"])
        allB = all(s["B"]["bitwise_equal"] for s in results["per_sample"])
        allC = all(s["C"]["differs"] for s in results["per_sample"])
        results["verdict"] = {
            "A_single_path_bitwise_logits": allA,
            "A_single_path_bitwise_encoder": allAe,
            "B_zerofill_equals_removal_bitwise": allB,
            "C_positive_control_differs": allC,
            "PREMISE_ESTABLISHED": bool(allA and allAe and allB and allC),
        }
        print("\n================ VERDICT ================")
        print(f"  A  single-path @lam=0, LOGITS  (bitwise)       : {'PASS' if allA else 'FAIL'}")
        print(f"  A' single-path @lam=0, ENCODER (bitwise)       : {'PASS' if allAe else 'FAIL'}")
        print(f"  B zero-fill absence == structural removal      : {'PASS' if allB else 'FAIL'}")
        print(f"  C positive control (non-vacuous)               : {'PASS' if allC else 'FAIL'}")
        print(f"  => SINGLE-PATH / NEUTRAL-ELEMENT PREMISE: "
              f"{'ESTABLISHED' if results['verdict']['PREMISE_ESTABLISHED'] else 'NOT ESTABLISHED'}")

        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[out] {out_json}")


if __name__ == "__main__":
    main()
