# Experiments and the environment-variable interface

Everything in this codebase is driven by environment variables rather than argparse,
because the training entry point is upstream DFormer's and its `Engine` owns argparse.

---

## Training dials

Read by the patched `utils/train.py` and `utils/dataloader/dataloader.py`. See
[`patches/README.md`](../patches/README.md) for what each one does to the model.

### α-attenuation — the three recipes

| Variable | Default | Meaning |
|---|---|---|
| `C2_ATTEN` | `0` (**off**) | enable α-attenuation. Off ⇒ the standard recipe is byte-identical. |
| `C2_DIST` | `u005_1` | the α distribution — this is what selects the recipe |
| `C2_PCLEAN` | `0.5` | probability of keeping α = 1 |
| `C2_ALPHAMIN` | `0.05` | lower bound of `U(α_min, 1)` |

| `C2_DIST` | α | recipe |
|---|---|---|
| `u005_1` | `U(α_min, 1)` | **ours** — α > 0 always, so training never visits the absent operator |
| `bernoulli` | `{0,1}`, `P(0)=0.475` | ordinary modality dropout |
| `uniform01` | `U(0,1)` | approaches the absent state |
| `constant` | `0.525` | separates stochasticity from mean shrinkage |

`E[α]` is matched across arms; only ours never reaches 0.

> **Always set `C2_ATTEN` and `C2_DIST` explicitly.** `C2_ATTEN` was once hard-coded to
> `1` inside the launcher, silently overriding a caller passing `0` and turning a
> baseline replication into a third method run — **21 GPU-hours lost.** `C2_DIST` was not
> exported and leaked in from the caller's environment, turning a method run into a
> dropout run. Both now have explicit defaults and are **echoed at launch**. Read the
> echoed line before walking away.

### Depth authority

| Variable | Default | Meaning |
|---|---|---|
| `DEPTH_LAMBDA_TRAIN` | `1.0` | global scale on the depth term, applied **in the forward pass** |

`1.0` is byte-identical to standard. `0.0` **removes** the depth pathway, leaving the
spatial-only operator — and *training* at `0.0` produces the constant-depth reference at
identical parameter count and capacity.

> **Never set this together with the eval-only `DEPTH_LAMBDA`.** They multiply, giving
> λ². The harness aborts with a fatal error if both are active — a guard added after a
> training shell exported `DEPTH_LAMBDA_TRAIN=0` and silently zeroed an entire profile.

> The arm labelled *constant-depth training* is **not a depth-blind model**. Its depth
> pathway is active at test time, and under the modality-complete condition it receives
> real depth — an input its training never contained. It is not exactly flat across
> conditions and should not be expected to be.

### The positional control — the matched negative control

| Variable | Default | Meaning |
|---|---|---|
| `POS_ATTEN` | unset | apply the same augmentation family to the **positional** bias |
| `POS_PCLEAN` | `0.5` | as `C2_PCLEAN` |
| `POS_ALPHAMIN` | `0.05` | as `C2_ALPHAMIN` |
| `POS_LAMBDA` | `1.0` | eval-only scale on the surviving positional bias |

This separates *depth-specific repair* from *generic attention-logit regularization*, by
applying the identical perturbation to a depth-**irrelevant** term of the same module and
functional form. Without it, the two accounts are indistinguishable.

Evaluation never removes the positional term, so this arm does **not** train on the
evaluation failure condition.

### Other

| Variable | Default | Meaning |
|---|---|---|
| `USE_AMP` | `0` | **do not change.** fp16 is unstable here; every paper run was fp32. |
| `EVAL_EVERY` | `0` | periodic in-training evaluation |
| `PRP_ON` | `0` | pseudo-depth head — a [graveyard](WHAT_DOESNT_WORK.md) entry |
| `CUDA_VISIBLE_DEVICES` | — | 2 GPUs for full-length arms |

---

## Evaluation dials

Read by [`scripts/eval_robust_suite.py`](../scripts/eval_robust_suite.py), the frozen
harness.

| Variable | Default | Meaning |
|---|---|---|
| `CKPT` | **required** | checkpoint path, relative to `$DFORMER_ROOT` |
| `OUT_JSON` | **required** | output path |
| `EVAL_MODES` | `single` | `single`, `ms`, or both |
| `EVAL_CONDITIONS` | all four | `clean,entire_missing,holes,gaussian` |
| `SEED` | `20260721` | corruption + torch seed |
| `VAL_BATCH` | config default | validation batch size |
| `SLIDING` | `1` | sliding-window inference |
| `MODEL_TAG` | derived | provenance label written into the JSON |
| `CONFIG_MODULE` | `DFormerv2_S` | DFormer config module |
| `EVAL_PAD` | dataset default | padding (SUN RGB-D) |
| `MAX_ITERS` | `0` | cap iterations, for smoke tests |

### Eval-only depth gates

| Variable | Meaning |
|---|---|
| `DEPTH_LAMBDA` | scale `weight[1]` in place at load time — a global eval-only authority dial |
| `DEPTH_LAMBDA_STAGES` | per-stage gate, e.g. `0,0,1,0` = depth at stage 2 only |
| `DEPTH_LAMBDA_BLOCKS` | per-block gate within a stage |
| `IMPUTE_DEPTH_PT` | inject a `.pt` depth field (the refuted imputation family) |

**Endpoints are self-validating and must be checked:** all-on must reproduce clean,
all-off must reproduce entire-missing. If they do not, the run is invalid — that is the
control whose failure invalidates everything else in the profile.

---

## Experiment directories

### `experiments/l2_nof/` — the arms in the paper

| Script | Purpose |
|---|---|
| `run_c2_fulltrain.sh` | **the main launcher.** 500 ep from pretrained; `C2_DIST` selects the recipe |
| `run_c2_fulltrain_resume.sh` | resume after an interruption |
| `run_spatial_oracle.sh` | the constant-depth / depth-free reference (`DEPTH_LAMBDA_TRAIN=0`) |
| `run_pos_control_screen.sh` | the matched positional control |
| `run_stage_profiles.sh`, `run_stage_necessity.sh`, `run_within_stage2.sh` | per-stage sufficiency / necessity |
| `run_qk_vo_transplant_eval.sh` | parameter transplants |
| `run_alphamin_sweep.sh`, `run_lambda_sweep.sh`, `run_dose_response_control.sh` | dose–response |
| `run_eval_noise_floor.sh` | the four-eval-seed dispersion control |
| `run_sunrgbd_capacity_arm.sh` | the SUN RGB-D screen |
| `run_zeroinfo_ladder.sh` | the `POS_LAMBDA` ladder |
| `run_c2_screen.sh`, `run_seed_screens.sh`, `run_screen_queue.sh` | short screens |

### The graveyard — kept on purpose

`n5_dual_kd/` (Large→Small distillation, abandoned at 3× budget) · `n6_adqg/`,
`n6_hhd_grp/` (coupling parameters + auxiliary losses) · `n7_cycle4/`, `n7_cycle5_mpa/`
(edge-only oracle, falsified), `n7_cycle5_mpb*/` (depth-dropout screen),
`n7_cycle5_prp/` (pseudo-depth head), `n7_decomp_witness/` · `wp1_reliance/`,
`wp1b_adsd/`, `wp2_stage2/`.

These are the evidence base for [WHAT_DOESNT_WORK](WHAT_DOESNT_WORK.md). **Do not read
them as recommended methods.** Several are disqualified by construction (they train on
the evaluation failure condition), and several were falsified.

---

## Screens vs full-length arms

**A screen is not a result.** Screens are 40-epoch continuations from an already-trained
checkpoint under the same optimizer and warmup — which is exactly the property that makes
them **systematically flattering**. Only a full 500-epoch run from the pretrained
backbone is promotable.

Screens exist to kill candidates cheaply, not to rank survivors.

---

## Operational rules

- **Launch detached** — `setsid` or `nohup`. A session teardown once killed a 4.5-hour run.
- **Gate concurrent probes** through [`scripts/gpu_guard.sh`](../scripts/gpu_guard.sh):
  a concurrent eval costs training ~19 %, and an OOM costs hours.
- **Never promote an effect smaller than the training-seed σ** (`0.49` on entire-missing).
- **One full-length training at a time.**
