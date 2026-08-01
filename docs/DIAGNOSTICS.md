# Diagnostics

Eval-only instruments. They read a fixed checkpoint and never train, so they are cheap —
GPU-minutes, not GPU-days — and they are judged against the **evaluation** dispersion
(0.045 clean / 0.031 entire-missing), not the training one.

```bash
python diagnostics/<instrument>.py
```

---

## Two rules that are not optional

Both were learned by publishing something false.

### 1. A probe that builds its own model must prove it loaded

Assert **≥ 90 % tensor match** *and* reproduce a known reference number before reading any
internal quantity.

> A `load_state_dict(w.get("model", w), strict=False)` matched **0 of 877 tensors** —
> the checkpoint's top-level key was `state_dict` — and ran an **untrained** model at
> 1.6 % pixel accuracy while printing entirely plausible numbers. Three probes and a
> published finding were built on it before a self-consistency gate caught the
> arithmetic. Never pair `strict=False` with an unchecked `.get()` chain.

### 2. Reseed before every forward

The Ham/NMF decoder redraws its factorization bases from `torch.rand` on every forward
pass. Without reseeding, every small delta you measure is noise. This is also the entire
source of the harness dispersion.

**And a corollary rule:** no diagnostic ships without a control whose failure invalidates
the run. We were burned both ways in one session — a self-consistency gate caught a false
FAIL, and a *missing* positive control produced a false PASS.

---

## The instruments

### Removal equivalence — the criterion

| Script | What it answers |
|---|---|
| **`single_path_invariance.py`** | **The linchpin.** With the depth term gated off, are the logits and all four encoder stage outputs bitwise identical across real / zero / random / constant depth — and is zero-fill identical to structural removal? |
| `neutral_element_survey.py` | Generalization: does the auxiliary modality admit a neutral element *at all*, in a given architecture? |
| `gf_fill_sensitivity.py` | Where it does **not**: how far does a dual-stream model's surviving RGB stream move as the depth fill value changes? (2.63 / 3.75 / 28.26 % vs 0.00 bitwise) |

`single_path_invariance.py` is the cheapest end-to-end check that environment, patch and
weights are correctly wired. Run it first.

### Where the damage is

| Script | What it answers |
|---|---|
| `stage_output_delta.py` | Per-stage: where does the depth-driven change to attention die out? Two forwards on identical inputs with `DEPTH_LAMBDA_TRAIN` toggled 1 → 0. |
| `attn_divergence_per_stage.py` | Per-stage functional effect — how much does the depth bias move the attention *itself*, as opposed to the logits? |
| `bias_to_logit_ratio.py` | Per-stage bias-to-logit ratio. |
| `per_image_agreement.py` | Task-space agreement, replacing a CKA instrument that saturated. |

Damage localizes overwhelmingly to **encoder stage 2**, under a *sufficiency*
intervention. Read the caveat in [RESULTS](RESULTS.md#per-stage-profile) before treating
that as a localization.

### Where the repair is

| Script | What it answers |
|---|---|
| **`param_transplant.py`** | **Sufficiency by transplant.** Move a parameter group from the repaired model into the baseline and measure how much of the gain comes with it. |
| `adabn_statistic_instrument.py` | How much of the gap is pure normalization-*statistic* miscalibration (as opposed to learned shifts)? |
| `qk_geometry_debias.py` | Does the repair act implicitly through Q/K geometry rather than through the bias weight? |
| `pathway_similarity.py` | Does the method's primary pathway resemble the depth-free reference's, or the baseline's? |
| `stage2_linear_probe.py` | A linear probe on stage-3 features. Returned a **modest** change — a measurement, not an exclusion of representational damage. |

> **Use transplant or ablation. Never a checkpoint diff.** A checkpoint diff shows where
> parameters *moved*, not where the function *lives*. Normalization biases moved ~21 %
> but hold 2.4k parameters; encoder/decoder weights moved ~5.6 % but hold 26.7M. Reading
> mechanism off the former gave a normalization account that transplant destroyed at
> **8 % against 91 %** (sanity: transplanting everything recovers 101 %).

### Controls

| Script | What it controls for |
|---|---|
| `pos_alpha_smoke.py` | Smoke test for the positional-alpha control — the matched negative control that separates depth-*specific* repair from generic attention-logit regularization. |
| `modulus_of_continuity.py` | Is the α → 0 limit a *regular* perturbation, and how much of the excursion lies inside the untrained interval? |

The positional control is the one that matters most: it applies the *same* augmentation
family to a depth-**irrelevant** term of the same module and functional form. Without it,
"our method repairs the depth operator" and "our method is a noise regularizer" are
indistinguishable.

### Suites

| Script | Scope |
|---|---|
| `pipeline_all.py` | Phase-0 + D1/D2/D3/D5 end-to-end |
| `scripts/deep_notrain_suite.py` | The no-train battery, T1–T9 |
| `scripts/verify_patching_integrity.py` | Activation-patching integrity |
| `scripts/phase4_remaining_suite.py` | Remaining phase-4 gates |

---

## Per-stage depth gating

The patch exposes a per-stage gate, used eval-only:

```bash
# enable depth at stage 2 only; measure how much of the gap it restores
DEPTH_LAMBDA_STAGES=0,0,1,0 python scripts/eval_robust_suite.py

# per-block, within a stage
DEPTH_LAMBDA_BLOCKS=... python scripts/eval_robust_suite.py
```

The endpoints are self-validating: all-on must reproduce clean, all-off must reproduce
the depth-absent condition. **If they do not, the run is invalid** — that is the control
whose failure invalidates everything else.

> Never set `DEPTH_LAMBDA*` and `DEPTH_LAMBDA_TRAIN` together. They multiply. The harness
> aborts with a fatal error if both are active.

---

## Interpreting a profile: a warning

The per-stage profile is **superadditive** — stage shares do not sum to the whole, and
stages 0 + 2 together reproduce clean exactly while neither does alone. That is **joint
sufficiency, not an additive localization**, and the difference matters.

Two invalid arguments, both of which we made and withdrew:

1. **Per-parameter efficiency comparisons across functionally different modules.** They
   assume linear additivity, which does not hold here.
2. **"Capacity-matched by parameter count."** False in function space — q/k enter
   bilinearly, are rank-limited by `head_dim`, and softmax shift-invariance nulls a
   subspace of q/k deltas. Equal parameter count is not equal expressive capacity.

**A share is not an attribution when the shares do not sum.** We reported that the repair
is "forced out of the attention score path" on the strength of a contrast between two
transplant shares at equal parameter count; training the score path alone recovers most
of the gain, so the contrast was a **denominator artifact**. See
[WHAT_DOESNT_WORK](WHAT_DOESNT_WORK.md).
