# Troubleshooting

---

## Setup

### `env.sh: could not locate conda.sh`

Probed locations are `$CONDA_EXE`, `~/miniconda3`, `~/anaconda3`, `~/miniforge3`,
`~/mambaforge`, `/opt/conda`, `/usr/local/conda`. If yours is elsewhere:

```bash
export CONDA_SH=/path/to/etc/profile.d/conda.sh
```

### `env.sh: DFormer not found at ...`

```bash
bash scripts/setup_dformer.sh
```

Or point `DFORMER_ROOT` at an existing patched checkout.

### `patch does not apply cleanly at 23cd8f8`

The checkout is not at the pinned commit, or it has other local modifications. Check:

```bash
cd "$DFORMER_ROOT" && git status && git log --oneline -1
```

If you have your own changes, `setup_dformer.sh` deliberately refuses to touch the tree.
Stash them first.

If you *regenerated* the patch yourself and it no longer applies, you probably used
`git diff -w`. Don't — the fork is reformatted relative to upstream, so a
whitespace-insensitive diff produces an invalid patch. See
[`patches/README.md`](../patches/README.md#regenerating-the-patch).

### `setup_dformer.sh` reports `UNKNOWN`

The checkout matches neither the stock nor the patched state. It is either locally
modified or at a different upstream commit. Inspect with `git status`; the clean fix is
to remove the directory and re-run.

---

## Results don't match

### `verify_results.py` reports a mismatch

An artifact was modified after the freeze. `git checkout -- results/` restores it. If it
persists on a fresh clone, that is a bug in this repository — please open an issue.

### My eval doesn't reproduce 55.70 on the released checkpoint

Fix this **before** interpreting any other number.

- Confirm **single-scale**. `55.70` is single-scale; `56.19` is multi-scale. The paper's
  protocol is single-scale throughout.
- Confirm sliding-window (`SLIDING=1`) and flip augmentation are on.
- Confirm the full 654-image validation split.
- Confirm `USE_AMP=0`.
- Confirm you are using `scripts/eval_robust_suite.py`, not upstream's `utils/eval.py` —
  the harness overrides per-dataset defaults where they differ.

### My retrained arm is off by ~0.3–0.5 mIoU

Probably expected. Judge against the **run-to-run** dispersion (`0.29` clean / `0.49`
entire-missing), not the eval dispersion (`0.045` / `0.031`). Contributors:

- Our three method runs were launched **without an explicitly set seed**.
- One of the three was evaluated at its **best** checkpoint rather than its last (~`0.3`).
- Data ordering and initialization are nondeterministic.

### Two evaluations of the *same* checkpoint differ

Expected, and bounded by `0.045` / `0.031`. The Ham/NMF decoder redraws its factorization
bases from `torch.rand` on every forward pass. This is the entire source of the harness
dispersion.

For tensor-level work, **reseed before every forward**.

---

## Diagnostics

### `single_path_invariance.py` reports non-identical outputs

The strongest signal that something is wrong upstream of any measurement. In order of
likelihood:

1. **The patch is not applied.** `bash scripts/setup_dformer.sh --check`
2. **The checkpoint did not load.** See below — this is the most dangerous failure,
   because it is silent.
3. **The decoder was not reseeded** between the two forwards.
4. You are using a **non-sensor depth representation** (HHA, disparity). The neutral
   element argument is about `|d_i − d_j|` vanishing for a constant *sensor depth* field.

### A probe prints plausible numbers that are wrong

Assume this until proven otherwise. A `load_state_dict(w.get("model", w), strict=False)`
once matched **0 of 877 tensors** — the checkpoint's top key is `state_dict` — and ran an
**untrained** model at 1.6 % pixel accuracy while printing entirely reasonable-looking
numbers. Three probes and a published finding were built on it.

**Any probe that builds its own model must assert ≥ 90 % tensor match and reproduce a
known reference number before reading any internal quantity.** Never pair `strict=False`
with an unchecked `.get()` chain.

### A per-stage profile is all zeros, or implausibly flat

Almost certainly `DEPTH_LAMBDA_TRAIN` is set in your shell while `DEPTH_LAMBDA*` is also
active. They **multiply**. The harness aborts with a fatal error when it detects both —
that guard exists because exactly this silently zeroed a full profile.

```bash
unset DEPTH_LAMBDA_TRAIN
```

### The profile's endpoints don't self-validate

All-on must reproduce clean; all-off must reproduce entire-missing. If they don't, **the
run is invalid** — do not interpret the interior. That is the control whose failure
invalidates everything else.

### `gf_fill_sensitivity.py` fails its load gate

Expected if GeminiFusion is absent — it is optional. See
[INSTALLATION](INSTALLATION.md#optional-geminifusion). The `mit-b3` encoder checkpoint
must be at `$GEMINIFUSION_ROOT/checkpoints/mit-b3.pth.tar`.

The assertion firing is the gate **working**. Do not lower the threshold.

---

## Training

### Mixed precision

**`USE_AMP=0`. Always.** fp16 is unstable on this architecture and silently overflows
gradients; every training run in the paper was fp32. A run under AMP is not a
reproduction of anything here. `scripts/env.sh` sets the default — don't override it.

### My "baseline" came out looking like a method run

Check the launcher's echoed line. `C2_ATTEN` was once hard-coded to `1`, silently
overriding a caller passing `0` — 21 GPU-hours lost. `C2_DIST` was not exported and
leaked in from the caller's environment, turning a method run into a dropout run.

Both are now explicit with defaults and echoed at launch:

```
[C2-full] C2_ATTEN=1 C2_DIST=u005_1 pclean=0.5 alphamin=0.05 gpus=2
```

**Read that line before walking away from a two-day run.**

### The run died when my session ended

Launch detached — `setsid` or `nohup`. A session teardown once killed a 4.5-hour run.

### Training slowed down sharply

A concurrent evaluation costs training about **19 %**. Gate probes through
[`scripts/gpu_guard.sh`](../scripts/gpu_guard.sh). An OOM costs hours.

### The final-epoch checkpoint is missing

The training loop retains only the best checkpoint by validation metric. This is the
known defect that made checkpoint selection non-uniform across the paper's arms. If you
need final-epoch weights, change the save policy before launching — it cannot be repaired
afterwards.

---

## Still stuck?

Open an issue with: what you ran, the full command including environment variables,
`torch.__version__` and CUDA version, the output of
`bash scripts/setup_dformer.sh --check`, and — for training — the echoed `C2_ATTEN` /
`C2_DIST` line.

A failure to reproduce `scripts/verify_results.py` or
`diagnostics/single_path_invariance.py` is a bug in this repository, and we want to know.
