# The M-series probes

Eval-only diagnostics run on checkpoints that already exist. No probe in this directory trains
anything. Each writes one JSON next to its script in `scripts/mNN_*.py`.

Every probe carries two things the numbers depend on:

- a **gate** — the probe must reproduce an independently recorded reference (a clean mIoU band, or a
  previously measured per-condition value) before any internal is read;
- a **canary** — a self-comparison that must return exactly `0`, or a stability check on a
  finite-difference step. A probe that fails its canary returns no verdict.

Predictions were filed before each run. Where a prediction missed, the JSON records the miss.

## What backs the paper

| artifact | what it measures |
|---|---|
| `m37_equivalence_dformer.json`, `m37_equivalence_cmnext.json` | the removal-equivalence audit — five candidate removals per architecture, decision-level distance and mIoU |
| `m40_released_set_audit.json` | the same audit on released DFormerv2 weights across two model sizes and two datasets |
| `m30_matched_kl.json` | 45 conditions (9 families × 5 doses): divergence-to-damage within and across families |
| `m38_four_arm_n3.json` | four trained arms × 3 seeds — reliance versus the accuracy columns |
| `m34_dof_matched_and_eigen.json` | spatial versus channel structure at matched degrees of freedom |
| `m35_reff_reaudit.json` | every conclusion re-derived with the causally inert branch removed |
| `m31_ggn_identity.json` | the quadratic form checked against directly measured KL |
| `../removal_dropout/eval.json` | the two matched CMNeXt dropout arms, 3 seeds each |
| `../audit/geminifusion_fill_sweep.json` | fill-value sweep on a third architecture |

## Reading the JSONs

Keys are stable across probes: `gate` / `gate_median_ratio`, `canary*`, `rows` (per condition or per
arm), and `P_MNNx` booleans for each filed prediction. A `verdicts_withheld` key means the canary
failed and nothing downstream should be read.

## Dependencies

`m1_mig_full.py` defines the corruption suite (`COND`, `corrupt`) and is imported by most probes.
`m35_reff_reaudit.py` exports `measure()`, reused by `m38` and `m39`. `m8_local_support.py` supplies
the support-band statistic. Checkpoints and datasets are not shipped — see the top-level README.
