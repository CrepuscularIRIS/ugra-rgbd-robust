# The method patch

`dformer-ugra.patch` is the method. Everything the paper measures is either produced by
this patch or by the eval-only diagnostics that read a model it built.

| | |
|---|---|
| **Upstream** | [`VCIP-RGBD/DFormer`](https://github.com/VCIP-RGBD/DFormer) |
| **Base commit** | `23cd8f833df12c74af45bc671e75891e48f8ff1b` |
| **Upstream license** | MIT — this patch is a derivative work under the same terms |
| **Size** | +625 / −30 lines across 5 files |
| **Applied by** | `bash scripts/setup_dformer.sh` |
| **Verified** | applies cleanly at the base commit, and reproduces the authors' working tree **byte-for-byte** |

```bash
bash scripts/setup_dformer.sh          # clone, pin to 23cd8f8, apply  (idempotent)
bash scripts/setup_dformer.sh --check  # report PATCHED / STOCK / UNKNOWN
```

The patch is deliberately distributed as a patch rather than a fork: it keeps upstream's
history and license intact, makes every one of our changes reviewable in isolation, and
keeps this repository small enough to clone in seconds.

---

## What it changes, and why

### `models/encoders/DFormerv2.py` (+189 / −10)

The core. Depth enters DFormerv2 in exactly one place — an additive term in the attention
score built from the pairwise depth difference:

```
mask = weight[0] * mask_positional  +  weight[1] * mask_depth
                                                   └── mask_depth = |d_i − d_j|
```

The patch exposes three dials on that expression.

- **`DEPTH_LAMBDA_TRAIN`** — a global scale on the depth term *applied in the forward
  pass*:

  ```
  mask = weight[0] * mask_pos  +  (DEPTH_LAMBDA_TRAIN * weight[1]) * mask_depth
  ```

  At `1.0` (default) the backbone is byte-identical to the standard recipe. At `0.0` the
  depth pathway is **removed**, leaving the spatial-only operator — which, *trained* at
  `0.0`, yields the in-family RGB-only reference at identical parameter count and
  capacity. That reference is the control the gap partition needs.

  > It is deliberately **not** named `DEPTH_LAMBDA`. The frozen eval harness already
  > implements an eval-only `DEPTH_LAMBDA` by scaling `weight[1]` in place at load time.
  > Sharing one name would apply the factor twice (λ²) whenever both paths were active.
  > The harness refuses to run if both are set — see `eval_robust_suite.py`.

- **`set_pos_alpha()` / the positional-alpha control** — the matched negative control.
  Our method scales the depth input by α, which scales `mask_depth` by exactly α. Two
  rival accounts of its gain: depth-*specific* repair, or a generic noise-injection
  regularizer on attention logits. They separate by applying the same augmentation family
  to a **depth-irrelevant** term of the same module and functional form — the positional
  bias `weight[0] * mask_pos`.

  Matching matters: our method draws **one α per sample**, shared across all 29
  `GeoPriorGen` modules, because it scales the shared depth input. An independent draw
  inside each module would be a different, weaker perturbation — so α is set once per
  forward by the trainer and read by every module.

- **`PRP_ON` / `PseudoDepthHead`** — a pseudo-depth head predicting depth from RGB
  features, feeding a parallel zero-initialized bias. **This is a graveyard entry**, kept
  because it is instructive: the zero-init gate never left zero. See
  [`docs/WHAT_DOESNT_WORK.md`](../docs/WHAT_DOESNT_WORK.md).

### `utils/dataloader/dataloader.py` (+33)

The α-attenuation augmentation, and with it all three training recipes — which in this
architecture are **three distributions over one scalar**:

| `C2_DIST` | α | what it is |
|---|---|---|
| `u005_1` *(default)* | `U(C2_ALPHAMIN, 1)` | **ours** — α > 0 always, so training *never* visits the absent operator |
| `bernoulli` | `{0, 1}`, `P(0) = 0.475` | ordinary modality dropout |
| `uniform01` | `U(0, 1)` | approaches the absent state |
| `constant` | `0.525` | separates stochasticity from mean shrinkage |

Gated by `C2_ATTEN=1`; **off by default**, so the standard recipe is byte-identical.
`C2_PCLEAN` (default `0.5`) is the probability of keeping α = 1. `E[α]` is matched across
arms; only ours never reaches 0.

### `utils/train.py` (+215 / −4)

Training-loop wiring: the positional-alpha draw (`POS_ATTEN`, `POS_PCLEAN`,
`POS_ALPHAMIN`), periodic in-training evaluation (`EVAL_EVERY`), and determinism controls.

### `models/builder.py` (+169 / −11) and `utils/engine/engine.py` (+14)

Plumbing to route the auxiliary outputs and diagnostic logging through the existing
model/engine interfaces without changing their signatures for unpatched callers.

---

## Two upstream defects the patch does **not** fix

Both are disclosed in the paper because they change how the numbers should be read, and
both are left in place so that this code reproduces the published results exactly.

1. **`utils/init_func.py::group_weight` never adds bare `nn.Parameter`s to the
   optimizer.** `.modules()` yields `Module`s, so its `isinstance(m, nn.Parameter)`
   branch is dead code. The 29 `GeoPriorGen.weight` tensors — the positional/depth
   mixture authority — therefore **never train**. Every principal experiment is a
   *frozen-mixture* variant of DFormerv2. It is a shared condition across all arms, so
   between-arm comparisons stay internally matched.

2. **The Ham/NMF decoder redraws `torch.rand` bases every forward**, making inference
   nondeterministic. This is the entire source of the harness dispersion (0.045 clean /
   0.031 entire-missing). **Reseed before every forward** in any tensor-level diagnostic.

If you fix either, you are no longer reproducing this paper — you are running a variant,
and it should be reported as one.

---

## Regenerating the patch

If you modify the patched checkout and want to refresh the patch:

```bash
cd "$DFORMER_ROOT"
git diff --no-color --binary -- \
    models/builder.py models/encoders/DFormerv2.py \
    utils/train.py utils/dataloader/dataloader.py utils/engine/engine.py \
  > "$UGRA_ROOT/patches/dformer-ugra.patch"
```

Do **not** pass `git diff -w`. The fork is reformatted relative to upstream, so a
whitespace-insensitive diff produces a patch that does not apply.
