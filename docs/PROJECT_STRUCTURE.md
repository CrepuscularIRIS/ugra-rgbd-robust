# Project structure

```text
ugra-rgbd-robust/
│
├── README.md                    start here
├── LICENSE                      MIT
├── NOTICE.md                    third-party notices (DFormer, GeminiFusion, datasets)
├── CITATION.cff                 ⚠ placeholder author block — must be replaced
├── CONTRIBUTING.md
├── requirements.txt
│
├── patches/
│   ├── dformer-ugra.patch       THE METHOD — +625 lines vs VCIP-RGBD/DFormer @ 23cd8f8
│   └── README.md                what it changes, and two defects it deliberately does not fix
│
├── scripts/
│   ├── env.sh                   shared environment contract, sourced by every script
│   ├── setup_dformer.sh         clone + pin + patch + verify        (idempotent)
│   ├── eval_robust_suite.py     THE FROZEN HARNESS — every paper number came from here
│   ├── eval_robust_suite.sh     shell wrapper
│   ├── eval_dformerv2s_nyu_gpu1.sh   single-GPU eval, upstream protocol
│   ├── verify_results.py        rebuild the headline table from artifacts (no GPU, stdlib only)
│   ├── gpu_guard.sh             refuse to start a probe that would disturb a training run
│   └── gen_failure_vis.py       qualitative failure figures
│
├── diagnostics/                 eval-only instruments — read a checkpoint, never train
│   ├── single_path_invariance.py    the bitwise neutral-element proof (run this first)
│   ├── neutral_element_survey.py    does the modality admit a neutral element at all?
│   ├── gf_fill_sensitivity.py       fill-value sensitivity on a dual-stream model
│   ├── param_transplant.py          where the repair lives — sufficiency by transplant
│   ├── stage_output_delta.py        per-stage damage
│   ├── attn_divergence_per_stage.py per-stage functional effect on attention
│   ├── stage2_linear_probe.py       the probe that returned a negative result
│   ├── adabn_statistic_instrument.py / qk_geometry_debias.py / pathway_similarity.py
│   ├── modulus_of_continuity.py / bias_to_logit_ratio.py / per_image_agreement.py
│   ├── pos_alpha_smoke.py           smoke test for the positional control
│   ├── corruptions/                 the four eval conditions + extra families
│   ├── hooks/                       model + dataloader helpers
│   └── scripts/                     multi-instrument suites (deep_notrain_suite, …)
│
├── experiments/
│   ├── l2_nof/                  the arms in the paper's table
│   ├── n5_* n6_* n7_*           the graveyard — kept as evidence, NOT recommendations
│   └── wp1_* wp1b_* wp2_*       earlier work packages
│
├── results/                     293 frozen JSON artifacts — RAW MEASUREMENTS ONLY
│   ├── FROZEN_20260728/FREEZE_MANIFEST.md   SHA-256 of every artifact
│   ├── score_push/runs/l2_nof/  the paper's arms, per-stage profiles, transplants
│   ├── deep_notrain/            the no-train battery, T1–T9
│   ├── diagnostics/  phase4/  phases/       earlier diagnostic waves
│   └── README.md
│
├── docs/
│   ├── INSTALLATION.md          three tiers, lowest needs no GPU
│   ├── DATASETS.md              layout, sources, how depth is consumed
│   ├── REPRODUCTION.md          five stages ordered by cost
│   ├── EXPERIMENTS.md           the full environment-variable interface
│   ├── DIAGNOSTICS.md           the instruments, and two non-optional rules
│   ├── RESULTS.md               every table, with its caveats attached
│   ├── WHAT_DOESNT_WORK.md      the graveyard: each claim + what withdrew it
│   ├── TROUBLESHOOTING.md
│   └── PROJECT_STRUCTURE.md     this file
│
└── tests/
```

---

## Where to start, by question

| Your question | Go to |
|---|---|
| Are the paper's numbers real? | `python3 scripts/verify_results.py` |
| What exactly is the method? | [`patches/README.md`](../patches/README.md) |
| How do I run it? | [INSTALLATION](INSTALLATION.md) → [REPRODUCTION](REPRODUCTION.md) |
| What do the numbers mean? | [RESULTS](RESULTS.md) |
| What knob does what? | [EXPERIMENTS](EXPERIMENTS.md) |
| What should I not bother trying? | [WHAT_DOESNT_WORK](WHAT_DOESNT_WORK.md) |
| How do I apply the criterion to *my* model? | [README → The criterion](../README.md#the-criterion) |

---

## Three curation decisions, and why

### `results/` ships raw measurement artifacts only

The JSONs are the evidence base; interpretation lives in the paper and in `docs/`.

The workspace's internal analysis memos are **not** published. Several carried claims that
later measurements withdrew — a `6.61` baseline gap superseded by `5.99`, and a *capacity*
account that a same-capacity cross-dataset control refuted. Shipping them beside a paper
that supersedes them would mislead readers into citing retracted findings.
[WHAT_DOESNT_WORK](WHAT_DOESNT_WORK.md) carries what was worth keeping, with the
withdrawing measurement attached to each entry.

### `experiments/` keeps the failed arms

They are the evidence base for the graveyard, and a negative is only a product if it is
general and explained. They are **not** recommendations — several are disqualified by
construction.

### Absolute paths inside `results/**/*.json` are left untouched

They are the original machine's, and they are **provenance**. Those files are checksummed
in the freeze manifest; rewriting them would break the integrity check that
`verify_results.py` performs. Every `.py` and `.sh` file, by contrast, was made portable
via [`scripts/env.sh`](../scripts/env.sh).

---

## Not in this repository

| | Why | Where |
|---|---|---|
| Model checkpoints (~7 GB) | too large for git | retrain, or the released DFormer baseline |
| NYU Depth v2 / SUN RGB-D | third-party | [DATASETS](DATASETS.md) |
| Upstream DFormer source | MIT, fetched at setup | `bash scripts/setup_dformer.sh` |
| GeminiFusion + weights | third-party, optional | [INSTALLATION](INSTALLATION.md#optional-geminifusion) |
| Manuscript source | separate tree | — |
