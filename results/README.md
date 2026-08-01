# Frozen result artifacts

**293 JSON files, ~1 MB, SHA-256 verified.** These are the evidence base: every number in
the manuscript traces to a file here.

```bash
python3 scripts/verify_results.py
```

That hashes every artifact against
[`FROZEN_20260728/FREEZE_MANIFEST.md`](FROZEN_20260728/FREEZE_MANIFEST.md) and recomputes
the paper's headline table from the per-run files. **No GPU, no datasets, no weights, no
third-party dependencies.**

---

## Policy: raw measurements only

This directory ships **measurement artifacts**, not interpretation. Interpretation lives
in the paper and in [`docs/`](../docs/).

The workspace's internal analysis memos are deliberately **not** published here. Several
carried claims that later measurements withdrew — a `6.61` baseline gap superseded by
`5.99`, and a *capacity* account refuted by a same-capacity cross-dataset control.
Shipping them next to a paper that supersedes them would lead readers to cite retracted
findings. What survived is in
[`docs/WHAT_DOESNT_WORK.md`](../docs/WHAT_DOESNT_WORK.md), each entry paired with the
measurement that withdrew it.

`FREEZE_MANIFEST.md` is the one non-JSON file kept, because it is a checksum index rather
than a narrative.

---

## Layout

```text
results/
├── FROZEN_20260728/
│   └── FREEZE_MANIFEST.md       SHA-256 of every artifact, at freeze time
├── score_push/
│   ├── baselines/               frozen eval baselines, before any method
│   └── runs/
│       ├── l2_nof/              THE PAPER'S ARMS and their diagnostics
│       │   ├── *_final.json         the four full-length arms
│       │   ├── screens/             short screens, incl. the RGB-only reference R
│       │   ├── stage_profiles/      per-stage sufficiency / necessity
│       │   ├── transplant/          parameter transplants
│       │   ├── failure_families/    corruption families at fixed severities
│       │   ├── failure_curve/       invalid-pixel-fraction sweep
│       │   ├── dose_response/       dose–response sweeps
│       │   ├── pos_control_screen/  the matched positional control
│       │   ├── emm_bench/           four-condition benchmark table
│       │   ├── e2_sun_screen/       SUN RGB-D screen
│       │   └── zeroinfo_ladder/     POS_LAMBDA ladder
│       └── <other>/             earlier / abandoned runs (see the graveyard)
├── deep_notrain/                the no-train battery, T1–T9
├── diagnostics/  phase4/  phases/   earlier diagnostic waves
└── _archive/                    superseded bundles
```

---

## Reading an artifact

Each file carries its own provenance:

```jsonc
{
  "model_tag": "...",        "checkpoint": "...",   "config": "...",
  "dataset": "NYUDepthv2",   "eval_split": "...",   "n_val_images": 654,
  "num_classes": 40,         "seed": 20260721,
  "sliding": true, "flip": true, "amp": false,      // protocol flags
  "results": {
    "single": {
      "clean":          { "miou": 56.39, "macc": 70.08, "mf1": 70.13, "ious": [ ... 40 ... ] },
      "entire_missing": { "miou": 50.40, ... },
      "holes":          { ... },
      "gaussian":       { ... }
    }
  },
  "summary": { "single": { "clean_miou": 56.39, "entire_missing_miou": 50.40,
                           "clean_minus_missing_gap": 5.99 } },
  "timestamp": "...", "torch_version": "..."
}
```

`results` holds per-class IoUs; `summary` holds the headline scalars.

---

## The paper's arms

Under `score_push/runs/l2_nof/`:

| Arm | File(s) |
|---|---|
| Baseline (no intervention) | `m1_matched_baseline_final.json` |
| **Ours**, 3 independent runs | `c2_ep500_seed1_final.json`, `c2_ep500_seed2_final.json`, `baseline_seed2_final.json` |
| Modality dropout | `dropout_ep500_final.json` |
| Constant-depth training | `oracle_ep500_final.json` |
| RGB-only reference `R` | `screens/rgb_only_finetune/eval.json` |

> **Naming wart:** the third run of *our* arm is in a file named
> `baseline_seed2_final.json`. It is a **method** run — the name is historical and the
> file is frozen, so it cannot be renamed without breaking the checksums. This is exactly
> why `scripts/verify_results.py` names its inputs explicitly instead of globbing.

Also present but **not** an arm: `method_seed3_final.json` is a run that was terminated
mid-training. It is **not a seed and must never be quoted** — see the freeze manifest.

---

## Two things to know before quoting a number

1. **Every value is a single evaluation** at one fixed harness seed. The four-seed control
   (`run_eval_noise_floor.sh`) exists to establish what that seed is worth; no arm's
   metric is an average over it.
2. **Judge training-arm comparisons against the run-to-run dispersion** (`0.29` clean /
   `0.49` entire-missing), *not* the eval dispersion (`0.045` / `0.031`). Eval-only
   diagnostics, which compare configurations of a single fixed checkpoint, use the latter.

Full context and caveats: [`docs/RESULTS.md`](../docs/RESULTS.md).

---

## Absolute paths inside these files

Some artifacts embed paths from the machine that produced them. They are **left untouched
on purpose**: these files are checksummed evidence, and rewriting them would break the
integrity check. Portability was handled in the `.py` and `.sh` files instead, via
[`scripts/env.sh`](../scripts/env.sh).
