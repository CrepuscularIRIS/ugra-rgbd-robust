# Diagnostics

Eval-only instruments. They read a fixed checkpoint and never train, so they cost
GPU-minutes rather than GPU-days.

**Full annotated index — what each instrument answers, and the two non-optional rules for
writing a new one — is in [`docs/DIAGNOSTICS.md`](../docs/DIAGNOSTICS.md).**

Start here:

```bash
python diagnostics/single_path_invariance.py
```

That is the bitwise neutral-element proof and the cheapest end-to-end check that
environment, patch and weights are correctly wired.

---

## Layout

```text
diagnostics/
├── single_path_invariance.py     the neutral-element proof            ← run first
├── neutral_element_survey.py     does the modality admit one at all?
├── gf_fill_sensitivity.py        fill-value sensitivity, dual-stream (needs GeminiFusion)
├── param_transplant.py           where the repair lives — sufficiency by transplant
├── stage_output_delta.py         per-stage damage
├── attn_divergence_per_stage.py  per-stage functional effect on attention
├── stage2_linear_probe.py        the probe that returned a negative result
├── adabn_statistic_instrument.py normalization-statistic miscalibration
├── qk_geometry_debias.py         implicit Q/K-rotation de-biasing
├── pathway_similarity.py         method vs depth-free reference vs baseline
├── modulus_of_continuity.py      regularity of the alpha -> 0 limit
├── bias_to_logit_ratio.py        per-stage bias-to-logit ratio
├── per_image_agreement.py        task-space agreement (replaces a saturated CKA probe)
├── pos_alpha_smoke.py            smoke test for the positional control
├── pipeline_all.py               early wave: D1/D2/D3/D5
├── backbone_forward.py           layer-wise forward with cache + sequential inject
├── corruptions/                  clean, entire_missing, holes, gaussian, + extra families
├── hooks/                        model + dataloader helpers
└── scripts/                      multi-instrument suites
    ├── deep_notrain_suite.py     the no-train battery, T1-T9
    ├── verify_patching_integrity.py
    ├── phase4_remaining_suite.py
    └── ...
```

## Output paths

Script output paths are **stable**. Do not rename them without updating the code.

| Script family | Output |
|---|---|
| `pipeline_all`, sequential, next_wave | `results/diagnostics/` |
| `deep_notrain_suite` | `results/deep_notrain/` |
| harness-driven evals | wherever `OUT_JSON` points |

---

## Three traps specific to this codebase

1. **Config pollution.** DFormer's `local_configs` mutate a **shared EasyDict**. Loading
   the Small config after the Large one requires purging the `local_configs*` modules —
   see `load_variant` in `scripts/deep_notrain_suite.py`.

2. **The decoder is nondeterministic.** `ham_head.py` redraws `torch.rand` bases on every
   forward. **Reseed before every forward** or every small delta you measure is noise.

3. **A silent load failure looks like a result.** Any probe that builds its own model must
   assert **≥ 90 % tensor match** *and* reproduce a known reference number before reading
   any internal quantity. A `strict=False` load paired with an unchecked `.get()` chain
   once matched 0 of 877 tensors and ran an untrained model while printing entirely
   plausible numbers.

And one interface rule: **never set `DEPTH_LAMBDA*` and `DEPTH_LAMBDA_TRAIN` together.**
They multiply. The harness aborts if both are active.
