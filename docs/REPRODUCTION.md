# Reproduction

Ordered by cost. Each stage is independently useful, so stop wherever your question is
answered.

| Stage | Cost | Reproduces |
|---|---|---|
| [0. Artifacts](#0-artifacts--no-gpu) | seconds | every number in the headline table |
| [1. Harness parity](#1-harness-parity--minutes) | minutes | the released checkpoint at 55.70 |
| [2. Neutral element](#2-the-neutral-element--minutes) | minutes | the bitwise removal-equivalence proof |
| [3. Eval-only diagnostics](#3-eval-only-diagnostics--gpu-minutes-to-hours) | GPU-minutes | the mechanism analysis |
| [4. Training arms](#4-the-training-arms--gpu-days) | GPU-days | Table 1 from scratch |

---

## 0. Artifacts — no GPU

```bash
python3 scripts/verify_results.py
```

Hashes all 293 committed result JSONs against
[`results/FROZEN_20260728/FREEZE_MANIFEST.md`](../results/FROZEN_20260728/FREEZE_MANIFEST.md)
and recomputes the four arms, the run-to-run spread, and the gap partition from the
per-run files. Standard library only.

This is the strongest cheap check available: it proves the paper's numbers are the
numbers on disk, independent of whether you can run the model at all.

---

## 1. Harness parity — minutes

**Do this before anything else.** If your harness does not reproduce the released
checkpoint, no later number is comparable.

```bash
CKPT=checkpoints/trained/DFormerv2_Small_NYU.pth \
OUT_JSON=results/parity.json \
EVAL_MODES=single \
EVAL_CONDITIONS=clean,entire_missing \
python scripts/eval_robust_suite.py
```

Expected:

| | clean | entire-missing |
|---|---:|---:|
| released checkpoint, single-scale | **55.70** | **49.11** |

Multi-scale (`EVAL_MODES=ms`) should give `56.19` against the published `56.0`.

### The frozen harness

`scripts/eval_robust_suite.py` is the single script every number in the paper came from:
single-scale inference with horizontal-flip augmentation and sliding-window aggregation
over the full 654-image validation split, at a fixed seed. It was fixed before the
experiments and has not been modified since. It applies this protocol uniformly and
**overrides per-dataset defaults where they differ**.

```
CKPT             checkpoint path, relative to $DFORMER_ROOT        [required]
OUT_JSON         output path                                       [required]
EVAL_MODES       single,ms                                         [single]
EVAL_CONDITIONS  clean,entire_missing,holes,gaussian               [all four]
SEED             corruption + torch seed                           [20260721]
VAL_BATCH        validation batch size                             [config default]
SLIDING          1/0 sliding-window inference                      [1]
MODEL_TAG        provenance label                                  [derived from CKPT]
CONFIG_MODULE    DFormer config module                             [DFormerv2_S]
```

### Two dispersions — do not confuse them

| dispersion | clean | entire-missing | judge what against it |
|---|---:|---:|---|
| 4 eval seeds, one checkpoint | 0.045 | 0.031 | eval-only diagnostics on a fixed checkpoint |
| 3 independent trainings | 0.29 | 0.49 | **any comparison between training arms** |

Every reported number is a **single** evaluation at the one fixed harness seed. The
four-seed control exists to establish what that seed is worth; no arm's metric is an
average over it. The sole source of the eval dispersion is the nondeterministic decoder.

---

## 2. The neutral element — minutes

```bash
python diagnostics/single_path_invariance.py
```

With the depth term gated off, logits and **all four encoder stage outputs** must be
bitwise identical whether the depth input is real, zero, random or constant — and
zero-fill must be identical to structural removal.

The identity is forced by the algebra (`mask_depth = |d_i − d_j|` vanishes for any
constant field), so this check establishes that **the implementation realizes it**, not
that the mathematics is true.

If it fails: the checkpoint did not load, the patch is not applied, or the decoder was
not reseeded. See [TROUBLESHOOTING](TROUBLESHOOTING.md).

---

## 3. Eval-only diagnostics — GPU-minutes to hours

These read a fixed checkpoint and never train. Judge every result against the **eval**
dispersion (0.045 / 0.031), not the training one.

```bash
# per-stage damage profile: which encoder stage carries the loss
DEPTH_LAMBDA_STAGES=0,0,1,0 python scripts/eval_robust_suite.py   # stage 2 only

# where the repair lives — transplant, NOT a checkpoint diff
python diagnostics/param_transplant.py

# fill-value sensitivity on a dual-stream model (needs GeminiFusion)
python diagnostics/gf_fill_sensitivity.py
```

> **`DEPTH_LAMBDA` and `DEPTH_LAMBDA_TRAIN` must never both be set.** They multiply, and
> the result is λ². The harness aborts with a fatal error if it detects both — that guard
> exists because a training shell once exported `DEPTH_LAMBDA_TRAIN=0` and silently
> zeroed an entire profile.

Full instrument list: [`docs/DIAGNOSTICS.md`](DIAGNOSTICS.md).

---

## 4. The training arms — GPU-days

Every arm: **500 epochs from `DFormerv2_Small_pretrained.pth`**, AdamW, base LR
`6e-5`, weight decay `0.01`, batch size 16 over 2 GPUs, 50 iterations/epoch, linear
warmup over 10 epochs, polynomial decay at power `0.9`, per-pixel cross-entropy with
index 255 ignored. Augmentation is random horizontal flip, random rescale from
`{0.5, 0.75, 1, 1.25, 1.5, 1.75}`, and random crop. **No photometric augmentation.**

The arms differ in **exactly one thing** — the training-time intervention.

```bash
# 1. baseline (no intervention)
C2_ATTEN=0 bash experiments/l2_nof/run_c2_fulltrain.sh

# 2. ours — alpha ~ U(0.05, 1); never reaches 0, so training never visits the absent operator
C2_ATTEN=1 C2_DIST=u005_1 C2_PCLEAN=0.5 C2_ALPHAMIN=0.05 \
  bash experiments/l2_nof/run_c2_fulltrain.sh

# 3. modality dropout — alpha in {0,1}
C2_ATTEN=1 C2_DIST=bernoulli bash experiments/l2_nof/run_c2_fulltrain.sh

# 4. constant-depth training reference
DEPTH_LAMBDA_TRAIN=0 bash experiments/l2_nof/run_spatial_oracle.sh
```

Then evaluate each under the frozen harness and compare against `results/`.

### Non-negotiables

- **`USE_AMP=0`.** fp16 is unstable on this architecture and every paper run was fp32.
  `scripts/env.sh` sets the default. A run under AMP is not a reproduction.
- **Set `C2_ATTEN` and `C2_DIST` explicitly, every time.** `C2_ATTEN` was once
  hard-coded to `1` in the launcher, which silently overrode a caller passing `0` and
  turned a baseline replication into a third method run — 21 GPU-hours lost.
  `C2_DIST` was not exported and leaked in from the caller's environment, turning a
  method run into a dropout run. Both are now explicit with defaults, and both are
  echoed at launch. **Read the echoed line before walking away.**
- **Launch detached** (`setsid`, `nohup`) — a session teardown once killed a 4.5-hour run.
- **Gate concurrent probes** through `scripts/gpu_guard.sh`: a concurrent eval costs
  training ~19 %, and an OOM costs hours.

### What you will and will not reproduce

Reproducible: the recipes, the harness, the conditions, the diagnostics.

**Not exactly reproducible**, and stated in the paper:

- Our three method runs were launched **without an explicitly set seed**. Their variation
  comes from nondeterministic initialization and data ordering. "Three independent
  training runs" is accurate; "three random seeds" would overstate the control.
- One of the three was evaluated at its **best** checkpoint rather than its last (worth
  roughly `0.3` mIoU), because a save-only-best loop did not retain its final weights.
- The **frozen-mixture defect**: the 29 positional/depth mixture weights never reach the
  optimizer, so every arm is a frozen-mixture variant. Shared across arms, so
  between-arm comparisons stay matched — but it is not stock DFormerv2 training.

Expect your numbers to land within the run-to-run spread (`0.29` clean, `0.49`
entire-missing), not on the printed digits.

---

## Reporting a reproduction

Useful reports include: the parity number for the released checkpoint (stage 1), which
stage you reached, your torch/CUDA versions, and — for training — the **echoed
`C2_ATTEN` / `C2_DIST` line**. Please open an issue; a failure to reproduce stage 0 or 2
is a bug in this repository and we want to know.
