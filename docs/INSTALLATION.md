# Installation

Three tiers, depending on what you actually want to do. Start at the lowest tier that
covers it — the first needs no GPU, no datasets and no weights.

| Tier | You want to | Needs |
|---|---|---|
| **1** | Check the paper's numbers against the artifacts | Python 3.9+, nothing else |
| **2** | Run eval-only diagnostics on a checkpoint | + GPU, DFormer, datasets, weights |
| **3** | Retrain an arm | + 2 GPUs, ~1–2 days per 500-epoch run |

---

## Tier 1 — verify the artifacts (no GPU)

```bash
git clone https://github.com/CrepuscularIRIS/ugra-rgbd-robust.git
cd ugra-rgbd-robust
python3 scripts/verify_results.py
```

`verify_results.py` uses only the standard library. It hashes all 293 committed result
JSONs against the freeze manifest and recomputes the paper's headline table from them.

---

## Tier 2 — the full environment

### 1. Python environment

```bash
conda create -n dformer python=3.10 -y
conda activate dformer
```

Install the PyTorch build matching **your** CUDA toolkit first — installing it via
`requirements.txt` will pull a default build that may not match your driver:

```bash
# example: CUDA 12.8, the reference environment
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

Reference environment for every number in the paper: **Python 3.10.18, torch 2.8.0+cu128,
numpy 1.26.4, mmcv 2.1.0, timm 1.0.19**.

> `numpy` is pinned `<2.0`: upstream DFormer is not numpy-2 clean.

### 2. Fetch and patch DFormer

```bash
bash scripts/setup_dformer.sh
```

This clones [`VCIP-RGBD/DFormer`](https://github.com/VCIP-RGBD/DFormer) into
`repos/DFormer`, pins it to commit `23cd8f8`, and applies
`patches/dformer-ugra.patch`. It is **idempotent** — re-running on a patched checkout is
a no-op. Confirm with:

```bash
bash scripts/setup_dformer.sh --check     # -> PATCHED
```

The location matters: several scripts resolve `repos/DFormer` relative to the repository
root. Override with `DFORMER_ROOT` if you must put it elsewhere.

See [`patches/README.md`](../patches/README.md) for what the patch does.

### 3. Datasets and pretrained weights

See [`docs/DATASETS.md`](DATASETS.md). Datasets stay **by path** and are never copied
into the repository.

### 4. Smoke test

```bash
# the bitwise neutral-element proof — the cheapest end-to-end check that
# environment + patch + weights are all wired correctly
python diagnostics/single_path_invariance.py
```

It should report that logits and all four encoder stage outputs are **identical** across
real / zero / random / constant depth when the depth term is gated off. If they are not
identical, something is wrong before any measurement is meaningful — see
[TROUBLESHOOTING](TROUBLESHOOTING.md).

---

## Tier 3 — training

Add nothing to the software environment; you need hardware and time.

- **2 GPUs** for the full-length arms (batch size 16 over two devices).
- **~1–2 days** per 500-epoch NYU Depth v2 run.
- `USE_AMP=0` is **mandatory** — fp16 is unstable on this architecture. `scripts/env.sh`
  sets the default; do not override it.

See [`docs/REPRODUCTION.md`](REPRODUCTION.md) and [`docs/EXPERIMENTS.md`](EXPERIMENTS.md).

---

## The environment contract

Every script sources [`scripts/env.sh`](../scripts/env.sh), which derives everything from
one variable and lets you override any of it. **Nothing in this repository needs editing
to run it on a different machine.**

| Variable | Default | Meaning |
|---|---|---|
| `UGRA_ROOT` | auto-detected from the script's own path | repository root |
| `DFORMER_ROOT` | `$UGRA_ROOT/repos/DFormer` | patched DFormer checkout |
| `GEMINIFUSION_ROOT` | `$UGRA_ROOT/repos/GeminiFusion` | optional second architecture |
| `UGRA_CONDA_ENV` | `dformer` | conda environment name |
| `CONDA_SH` | auto-detected | path to `etc/profile.d/conda.sh` |
| `UGRA_GPU` | `0` | default CUDA device |
| `USE_AMP` | `0` | **do not change** — fp16 is broken here |

`CONDA_SH` is probed across `$CONDA_EXE`, `~/miniconda3`, `~/anaconda3`, `~/miniforge3`,
`~/mambaforge`, `/opt/conda` and `/usr/local/conda`. If your install is elsewhere:

```bash
export CONDA_SH=/path/to/etc/profile.d/conda.sh
```

---

## Optional: GeminiFusion

Needed **only** by `diagnostics/gf_fill_sensitivity.py`, the fill-value sensitivity
measurement on a dual-stream architecture (the one for which no canonical removal
exists).

```bash
git clone https://github.com/JiaDingCN/GeminiFusion.git repos/GeminiFusion
# place the mit-b3 encoder checkpoint at:
#   repos/GeminiFusion/checkpoints/mit-b3.pth.tar
```

The diagnostic fills the depth slot with constants **by construction**, so the native
depth encoding of either codebase is irrelevant — which is exactly what makes it runnable
without GeminiFusion's own dataset. It needs only NYU Depth v2 RGB images.

It gates on the encoder loading ≥ 90 % of its tensors before reading any number. That
gate exists for a reason: a silent `strict=False` load once ran three probes on an
untrained model and produced plausible-looking numbers that were entirely false.

---

## Verifying the install

```bash
bash scripts/setup_dformer.sh --check      # PATCHED
python3 scripts/verify_results.py          # OK — repository matches the manuscript
python  diagnostics/single_path_invariance.py   # bitwise identity holds
pytest tests/ -q                            # unit tests
```
