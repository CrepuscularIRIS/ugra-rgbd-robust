<div align="center">

# Missing Is Not Zero

**Well-Posed Modality Removal, and What It Reveals About RGB-D Segmentation Robustness**

Code, frozen result artifacts and diagnostics for a study of what
*"we set the missing modality to zero"* actually measures.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Artifacts verified](https://img.shields.io/badge/artifacts-293%20SHA--256%20verified-brightgreen.svg)](results/FROZEN_20260728/FREEZE_MANIFEST.md)

</div>

---

## TL;DR

The standard missing-modality robustness protocol sets the absent modality's input to
zero. That primitive is **not architecture-neutral**: a convolution with a bias does not
annihilate a zero input, so `conv(0) = b ≠ 0`. Zero-filling therefore replaces the
depth image with a *constant feature map* that propagates through normalization layers
calibrated on real depth. What gets measured is the cost of losing depth **plus** the
cost of processing an input drawn from no distribution the network has seen — and the
two cannot be separated after the fact.

This repository contains:

1. **A removal-equivalence criterion** and an **inference-time test** for it, which
   needs no retraining and depends on none of our trained models.
2. **The instrument**: a patch to DFormerv2 exposing the depth pathway as a single
   scalar dial, so depth can be removed *exactly* rather than substituted.
3. **293 frozen result artifacts**, SHA-256 verified, from which the paper's entire
   headline table recomputes — [in one command, with no GPU](#verify-in-30-seconds).
4. **A characterized graveyard**: what did not work, and the measurement that
   withdrew each claim ([`docs/WHAT_DOESNT_WORK.md`](docs/WHAT_DOESNT_WORK.md)).

---

## Verify in 30 seconds

No GPU, no datasets, no model weights — this reads only the committed artifacts:

```bash
git clone https://github.com/CrepuscularIRIS/ugra-rgbd-robust.git
cd ugra-rgbd-robust
python3 scripts/verify_results.py
```

<details>
<summary><b>Expected output</b> (click to expand)</summary>

```
[1] artifact integrity (SHA-256 against the freeze manifest)
  293 artifacts verified byte-identical, 0 mismatched, 0 missing

[2] headline table, recomputed from the frozen per-run artifacts
  arm                            clean  entire-missing    diff  runs
  ---------------------------- ------- --------------- ------- -----
  Baseline (no intervention)     56.39           50.40    5.99     1
  Modality dropout               54.50           52.91    1.59     1
  Constant-depth training        53.13           53.03    0.10     1
  Ours (alpha-attenuation)       55.86           53.05    2.81     3
                               sd 0.29            0.49   (run-to-run, n=3)

  RGB-only finetune (reference R)   54.93           53.26

  gap partition against R:  5.99  =  3.13 (complete-modality advantage over R)
                                  +  2.86 (reference-relative depth-absent deficit)

OK — repository matches the manuscript.
```
</details>

Every number the manuscript reports for a training arm is recomputed here from the
per-run file that produced it. Nothing is transcribed by hand.

---

## The problem, in one figure's worth of text

Measured on released weights, varying **nothing but the constant** used to fill the
depth slot, a dual-stream encoder's stage-3 features move by:

| fill value | stage-3 feature displacement |
|---|---|
| zero-fill (the convention) | — (the reference point) |
| dataset-mean depth | **2.63 %** |
| high constant | **3.75 %** |
| random | **28.26 %** |
| *neutral element of the operator* | **0.00 %** (bitwise identical) |

These are four different "absences" of one modality in an architecture that published
robustness benchmarks evaluate. The fill value that sets the mixture is a free
parameter that published protocols do not report.

> The displacement above is **representational**. What it costs in mIoU would require
> that model's own evaluation pipeline, which we did not run. This is stated as a
> measurement, not as a claimed accuracy penalty.

Reproduce it: [`diagnostics/gf_fill_sensitivity.py`](diagnostics/gf_fill_sensitivity.py).

---

## The criterion

> A substituted value is **removal-equivalent** only relative to a *specified
> model-level removal*. Supplying the **neutral element** at every modality-injection
> operator is a sufficient — not necessary — construction for this equivalence.

The test is checkable at inference time, without retraining, in any architecture
admitting such a removal: **evaluate once with the modality zero-filled, once under the
named removal, and compare at the logits and at every stage output.**

- Disagreement is a measured discrepancy between two protocols. It belongs *beside* the
  robustness figure, not inside it.
- Agreement is evidence of equivalence **on the inputs tested** — not a proof of it.

A negative verdict is **not** a finding of error against a published result. It
establishes that the reported number is a composite whose parts were not separated, and
that the separation is available cheaply.

### Why DFormerv2 is the instrument

In DFormerv2 depth has no stream of its own. It enters as a single additive term in the
attention score, built from the pairwise depth difference `|d_i − d_j|`, which **vanishes
identically for any constant depth field**. Zeroing depth is therefore not a substitution
but a *limit of the trained computation*.

We verify this bitwise — with the depth term gated off, logits and all four encoder
stage outputs are identical whether the depth input is real, zero, random or constant:

```bash
python diagnostics/single_path_invariance.py
```

The identity is forced by the algebra rather than discovered, so the check establishes
that **the implementation realizes it**.

---

## Results

Four full-length trainings (500 epochs from the released ImageNet-pretrained backbone),
differing **only** in the training-time intervention. DFormerv2-Small, NYU Depth v2, 40
classes, single-scale + flip, sliding-window, one frozen harness.

| Arm | clean ↑ | entire-missing ↑ | within-arm diff. | runs |
|---|---:|---:|---:|---:|
| *Published checkpoint (reproduction reference)* | *55.70* | *49.11* | — | *1* |
| Baseline (no intervention) | 56.39 | 50.40 | 5.99 | 1 |
| Modality dropout | 54.50 | 52.91 | 1.59 | 1 |
| Constant-depth training | 53.13 | 53.03 | 0.10 | 1 |
| **Ours** (α-attenuation) | **55.86 ± 0.29** | **53.05 ± 0.49** | 2.81 | 3 |

**No entry is emphasised as a winner, because the design does not support a ranking.**
Read this table with four qualifications, all of which are the authors':

1. The last column is each arm's *own* clean-minus-absent difference and is
   **descriptive only**. A small value need not indicate robustness — constant-depth
   training reaches `0.10` largely because its clean score is low.
2. Only the "ours" arm was replicated. The comparators are **single runs** of unknown
   variability, and were **not tuned to a common budget**.
3. One of our three runs was evaluated at its best checkpoint rather than its last,
   worth roughly `0.3` mIoU. It cannot be repaired after the fact.
4. The same validation split was used for screening, checkpoint selection and final
   reporting, so point estimates may carry selection bias.

### The two dispersions, which are not interchangeable

| dispersion | clean | entire-missing | what it bounds |
|---|---:|---:|---|
| 4 evaluation seeds, one checkpoint | 0.045 | 0.031 | measurement noise only |
| 3 independent trainings | 0.29 | 0.49 | **the standard every arm comparison is judged against** |

The second is about 6× the first on clean and 16× on entire-missing. The `0.14` mIoU
interval separating the three recipes' depth-absent endpoints lies **inside** that
scale — which is a statement about our resolution, **not** a demonstration of
equivalence.

### The partition

Where removal is exact, the gap can be partitioned against a same-capacity reference
that does not use the modality at test time:

```
5.99  =  3.13  (complete-modality advantage over R)  +  2.86  (reference-relative depth-absent deficit)
```

The identity is **algebraic and holds for any reference**, so it is the reference that
carries the content. Ours is an RGB-only finetune whose weights began from a
depth-trained checkpoint, so it establishes what a same-capacity model attains *without
depth at test time* — **not** what a model with no depth in its history could attain,
and **not** a ceiling. (An earlier version of this work used causal names for the two
terms; they are not licensed, and the descriptive names are used throughout.)

Full detail and per-condition tables: [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Installation

```bash
# 1. environment
conda create -n dformer python=3.10 -y && conda activate dformer
pip install -r requirements.txt

# 2. fetch upstream DFormer and apply the method patch
bash scripts/setup_dformer.sh

# 3. confirm the checkout is patched
bash scripts/setup_dformer.sh --check
```

The method is a **625-line patch** against upstream
[`VCIP-RGBD/DFormer`](https://github.com/VCIP-RGBD/DFormer) at commit `23cd8f8`,
verified to apply cleanly and to reproduce our working tree byte-for-byte. See
[`patches/README.md`](patches/README.md) for what it changes and why.

Datasets and pretrained weights: [`docs/DATASETS.md`](docs/DATASETS.md).
Full instructions, including the optional second architecture:
[`docs/INSTALLATION.md`](docs/INSTALLATION.md).

---

## Usage

### Evaluate a checkpoint under all four conditions

```bash
CKPT=checkpoints/trained/DFormerv2_Small_NYU.pth \
OUT_JSON=results/my_eval.json \
EVAL_CONDITIONS=clean,entire_missing,holes,gaussian \
python scripts/eval_robust_suite.py
```

### Train the three recipes

All three are **three distributions over one scalar α** in this architecture — which is
why their agreement is less surprising than it first appears:

```bash
# ours: alpha ~ U(0.05, 1) — never reaches 0, so training never visits the absent operator
C2_ATTEN=1 C2_DIST=u005_1  bash experiments/l2_nof/run_c2_fulltrain.sh

# modality dropout: alpha in {0,1}
C2_ATTEN=1 C2_DIST=bernoulli bash experiments/l2_nof/run_c2_fulltrain.sh

# constant-depth training reference
DEPTH_LAMBDA_TRAIN=0 bash experiments/l2_nof/run_spatial_oracle.sh
```

> **`USE_AMP=0` is mandatory.** fp16 is unstable on this architecture and every
> training run in the paper used fp32. `scripts/env.sh` enforces the default.

The complete environment-variable interface — training dials, eval dials, per-stage
depth gates and the positional control — is documented in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

---

## Repository structure

```text
ugra-rgbd-robust/
├── patches/
│   └── dformer-ugra.patch      the method: +625 lines against VCIP-RGBD/DFormer @ 23cd8f8
├── scripts/
│   ├── setup_dformer.sh        clone + patch + verify upstream          (idempotent)
│   ├── env.sh                  shared environment contract, sourced by every script
│   ├── eval_robust_suite.py    THE FROZEN HARNESS — every number came from here
│   ├── verify_results.py       rebuild the headline table from artifacts (no GPU)
│   └── gpu_guard.sh            refuse to start a probe that would disturb a training run
├── diagnostics/                eval-only instruments; no training required
│   ├── single_path_invariance.py   the bitwise neutral-element proof
│   ├── gf_fill_sensitivity.py      fill-value sensitivity on a dual-stream model
│   ├── param_transplant.py         where the repair lives (transplant, not checkpoint diff)
│   ├── stage_output_delta.py       per-stage damage profile
│   └── ...                         see docs/DIAGNOSTICS.md
├── experiments/                training recipes, including the dead ends
│   ├── l2_nof/                 the arms in the paper's table
│   └── n5_* n6_* n7_*          the graveyard — kept deliberately, see below
├── results/                    293 frozen JSON artifacts (raw measurements only)
│   └── FROZEN_20260728/FREEZE_MANIFEST.md    SHA-256 of every one of them
├── docs/                       installation, datasets, reproduction, experiments, results
└── tests/
```

Detail: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

### Two deliberate curation decisions

- **`results/` ships raw measurement artifacts only.** Interpretation lives in the paper
  and in `docs/`. The workspace's internal analysis memos are *not* published, because
  several carried claims that later measurements withdrew, and shipping them beside a
  paper that supersedes them would mislead.
- **`experiments/` keeps the failed arms.** They are the evidence base for
  [`docs/WHAT_DOESNT_WORK.md`](docs/WHAT_DOESNT_WORK.md). A negative is only a product
  if it is *general* and *explained*.

---

## What is **not** here

Stated plainly, so nobody wastes time looking:

| Not shipped | Why | Where to get it |
|---|---|---|
| Model checkpoints (`.pth`, ~7 GB) | Too large for git | Retrain via `docs/REPRODUCTION.md`, or the released baseline from [DFormer](https://github.com/VCIP-RGBD/DFormer) |
| NYU Depth v2 / SUN RGB-D | Third-party datasets | [`docs/DATASETS.md`](docs/DATASETS.md) |
| Upstream DFormer source | MIT, fetched + patched on setup | `bash scripts/setup_dformer.sh` |
| GeminiFusion weights | Third-party | [`docs/INSTALLATION.md`](docs/INSTALLATION.md) |
| The manuscript source | Separate tree | See [Citation](#citation) |

Absolute paths inside `results/**/*.json` are the original machine's and are left
untouched **on purpose** — those files are checksummed evidence, and rewriting them
would break provenance.

---

## Known defects that change how these numbers should be read

Three implementation defects are disclosed in the paper and reproduced here so that
anyone building on this code inherits the caveats rather than rediscovering them:

1. **The optimizer grouping helper never adds bare parameter tensors to the optimizer**,
   so the 29 positional/depth mixture weights **do not update during training**. Every
   principal experiment is therefore a *frozen-mixture* variant of DFormerv2. This is a
   shared condition across all arms, so between-arm comparisons remain internally
   matched; a single corrected training gave similar point estimates, which is a
   sensitivity check rather than a replication.
2. **The Hamburger/NMF decoder redraws its factorization bases on every forward pass**,
   so inference is nondeterministic. This is the entire source of the harness
   dispersion above. **Any tensor-level diagnostic must reseed before every forward.**
3. **The training loop retains only the best checkpoint**, which is why one run's
   final-epoch weights were not kept and checkpoint selection is not uniform.

A fourth, methodological, is worth more than the three: **a checkpoint diff shows where
parameters *moved*, not where the function *lives*.** Reading mechanism off relative
per-tensor change gave us a normalization account that a transplant test then destroyed,
at 8 % against 91 %. Use transplant or ablation, never a parameter diff. See
[`docs/WHAT_DOESNT_WORK.md`](docs/WHAT_DOESNT_WORK.md).

---

## Scope and limitations

The partition and the mechanism analysis are measured **inside one architecture family,
at one capacity, and — of the two datasets tested — on NYU Depth v2**, with
short-schedule diagnostics on SUN RGB-D. Every arm except ours is a single run,
checkpoint selection is not uniform, and the comparator recipes were not given a matched
tuning budget.

The **criterion and its test carry none of those conditions**, and they are the part we
would defend first. We have not exercised the audit on anyone else's model: the one
external architecture for which we hold weights is a dual-stream design, where no
removal is canonical.

---

## Citation

> [!IMPORTANT]
> **TODO — the author block is a placeholder and must be replaced before circulation.**
> The manuscript currently carries stand-in names and affiliations. The structure below
> is final; the author content is **not**. Update both this block and
> [`CITATION.cff`](CITATION.cff) before sharing this repository or submitting.

```bibtex
@article{missingisnotzero,
  title   = {Missing Is Not Zero: Well-Posed Modality Removal, and What It
             Reveals About RGB-D Segmentation Robustness},
  author  = {TODO: replace placeholder author block before circulation},
  year    = {2026},
  note    = {Preprint. Author list pending.}
}
```

If you use the **criterion or the audit procedure** rather than the models, that part
depends on none of our trained checkpoints and is reusable independently.

### Upstream work this builds on

```bibtex
@inproceedings{yin2025dformerv2,
  title     = {DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation},
  author    = {Yin, Bo-Wen and others},
  booktitle = {CVPR}, year = {2025}
}
```

Please also cite [DFormer](https://github.com/VCIP-RGBD/DFormer) (ICLR 2024),
[NYU Depth v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) and
[SUN RGB-D](https://rgbd.cs.princeton.edu/) as appropriate.

---

## License

[MIT](LICENSE) — matching upstream DFormer, so the patch and everything derived from it
stay license-compatible.

Third-party components keep their own licenses and are **not** redistributed here:
**DFormer** (MIT, fetched at setup), **GeminiFusion**, and the datasets. See
[`NOTICE.md`](NOTICE.md).

---

## Acknowledgements

Built on [DFormer / DFormerv2](https://github.com/VCIP-RGBD/DFormer) by VCIP-RGBD, whose
MIT-licensed release made the instrument possible. The fill-sensitivity measurement uses
[GeminiFusion](https://github.com/JiaDingCN/GeminiFusion).
