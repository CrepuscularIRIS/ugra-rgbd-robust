# Results

Every number here is recomputed from the artifacts in [`results/`](../results/):

```bash
python3 scripts/verify_results.py
```

All numbers are mIoU on the NYU Depth v2 validation split (654 images, 40 classes),
single-scale + flip, sliding-window, under the one frozen harness described in
[REPRODUCTION](REPRODUCTION.md#the-frozen-harness).

---

## Read this before reading any table

Four conditions govern how every number below should be interpreted.

1. **No arm is emphasised as a winner, because the design does not support a ranking.**
2. **Only the "ours" arm was replicated.** The comparators are single runs of unknown
   variability, and were **not tuned to a common budget**.
3. **One of our three runs was evaluated at its best checkpoint** rather than its last,
   worth roughly `0.3` mIoU. Checkpoint selection is therefore not uniform.
4. **The same validation split was used for screening, checkpoint selection and final
   reporting**, so point estimates may carry selection bias and should not be read as
   performance on an untouched test set.

No *p*-values are reported, no multiplicity correction is applied across conditions, and
no claim of statistical significance is made. Every comparison is stated as a difference
against a measured dispersion, and any difference smaller than that dispersion is
reported as one we cannot resolve.

---

## The two dispersions

They are **not** interchangeable, and conflating them is the single easiest way to
misread this work.

| dispersion | clean | entire-missing | what it may be used to judge |
|---|---:|---:|---|
| 4 evaluation seeds, one fixed checkpoint | 0.045 | 0.031 | eval-only diagnostics on a fixed checkpoint |
| 3 independent training runs | **0.29** | **0.49** | **any comparison between training arms** |

The second is about **6×** the first on clean and **16×** on entire-missing. Its only
source is the nondeterministic Ham/NMF decoder.

Every reported number is a *single* evaluation at the one fixed harness seed. The
four-seed control exists solely to establish what that seed is worth; **no arm's metric
is an average over it.**

A standard deviation estimated from three runs is itself imprecise. Treat it as an
order-of-magnitude scale, not a calibrated interval.

---

## The four arms

500 epochs from the released ImageNet-pretrained backbone, differing **only** in the
training-time intervention.

| Arm | clean ↑ | entire-missing ↑ | within-arm diff. | runs |
|---|---:|---:|---:|---:|
| *Published checkpoint (reproduction reference)* | *55.70* | *49.11* | — | *1* |
| Baseline (no intervention) | 56.39 | 50.40 | 5.99 | 1 |
| Modality dropout | 54.50 | 52.91 | 1.59 | 1 |
| Constant-depth training | 53.13 | 53.03 | 0.10 | 1 |
| **Ours** (α-attenuation) | **55.86 ± 0.29** | **53.05 ± 0.49** | 2.81 | 3 |

> The "within-arm diff." column is each arm's **own** clean-minus-absent difference and is
> **descriptive only**. A small value need not indicate robustness: constant-depth
> training reaches `0.10` largely because its clean score is low. These are two arms'
> within-arm differences, **not** a gap that shrank inside one model.

Against the baseline, our intervention moves entire-missing by `+2.65` and clean by
`−0.53`. Both exceed the run-to-run scale measured for that arm — though the variability
of the single-run comparators is unknown.

### All four conditions

| Arm | clean | entire-missing | holes | gaussian |
|---|---:|---:|---:|---:|
| Baseline | 56.39 | 50.40 | 56.10 | 55.22 |
| Modality dropout | 54.50 | 52.91 | 54.36 | 54.24 |
| Constant-depth training | 53.13 | 53.03 | — | — |
| Ours (mean of 3) | 55.86 | 53.05 | 55.58 | 55.14 |

The repair is large on corruptions that drive the depth term toward its neutral element
and small on those that leave it intact — which is the behaviour the mechanism predicts.

### Per-class

The median class loses `5.1` percentage points of IoU when depth is removed from the
baseline, and `2.3` after repair, with **36 of 40 classes improving**. The repair is
broad rather than concentrated. By image region, the damage is **not** confined to object
boundaries — contrary to what a purely edge-related account of the geometry prior would
predict.

Both are descriptive observations. Neither is an independent test of the mechanism.

---

## The gap partition

Where removal is exact, the gap partitions against a same-capacity reference that does
not use the modality at test time:

```
5.99  =  3.13  +  2.86
 │        │        └── reference-relative depth-absent deficit   (R − baseline_absent)
 │        └── complete-modality advantage over R                 (baseline_clean − R)
 └── the baseline's own within-arm difference
```

with `R = 53.26`, the depth-absent score of an RGB-only finetune.

**Three things this is not:**

- **Not a causal decomposition.** The identity is algebraic and holds for *any* reference,
  so **it is the reference that carries all the content**. An earlier version used causal
  names for the two terms; they are not licensed and were withdrawn.
- **Not a ceiling.** `R` is an RGB-only finetune whose weights began from a depth-trained
  checkpoint. It establishes what a same-capacity model attains *without depth at test
  time* — not what a model with no depth in its history could attain. The constant-depth
  arm scores `53.03`, *below* `R`, which is how "depth-free is a ceiling" was falsified
  from inside our own arms.
- **Not transferable as-is.** The second term is a difference of two depth-absent
  quantities in which the modality-complete column cancels.

---

## Cost of robustness

Three recipes the literature treats as unrelated give depth-absent point estimates
within a **`0.14` mIoU interval**:

| recipe | depth-absent | clean cost vs baseline |
|---|---:|---:|
| α-attenuation (ours) | 53.05 | **−0.53** |
| Modality dropout | 52.91 | −1.89 |
| Constant-depth training | 53.03 | −3.26 |

That `0.14` interval is **smaller than the run-to-run standard deviation (`0.49`) of the
one arm we replicated.** Because the comparators were not replicated, this supports
**neither an equivalence claim nor a formal test of difference** — it is a statement about
our resolution.

One consequence of the neutral element explains why the agreement is less surprising than
it looks: in this architecture these three are **three distributions over one scalar α**
(see [`patches/README.md`](../patches/README.md)). Where they differ is what they charge
on the modality-complete condition — and there they differ by several points.

> The three configurations were **not tuned to a common budget.** The clean-cost column
> is the comparison that carries the argument, and it is also the one most exposed to
> that caveat.

---

## Fill-value sensitivity

Measured on released GeminiFusion weights, varying **nothing but the constant** filling
the depth slot. Stage-3 encoder feature displacement relative to the zero-filled forward:

| fill | displacement |
|---|---:|
| dataset-mean depth | 2.63 % |
| high constant | 3.75 % |
| random | 28.26 % |
| **neutral element** (DFormerv2) | **0.00 %** — bitwise |

These are four different "absences" of one modality in an architecture that published
robustness benchmarks evaluate.

> **This is a representational quantity.** What it costs in mIoU would require that
> model's own evaluation pipeline, which we did not run. It is not a claimed accuracy
> penalty.

Reproduce: `python diagnostics/gf_fill_sensitivity.py`.

---

## Per-stage profile

Damage localizes overwhelmingly to **encoder stage 2** under a *sufficiency* intervention
(enable depth at stage `s` only, eval-only, endpoints self-validating).

**Read the caveat before using this.** The profile is **superadditive**: stages 0 + 2
together reproduce clean exactly, while neither does alone, and the shares do not sum.
That is **joint sufficiency, not an additive localization.** Within stage 2 the effect is
distributed and redundant — do not subdivide further.

See [DIAGNOSTICS](DIAGNOSTICS.md#interpreting-a-profile-a-warning) for two arguments about
this profile that we made and withdrew.

---

## Cross-dataset

| model / dataset | gap |
|---|---:|
| Small / NYU Depth v2 | large |
| Small / SUN RGB-D | ~0.07 |

**Same capacity, opposite outcome.** The phenomenon is **NYU-Depth-v2-specific of the two
datasets tested**, and the paper states that as a limitation up front. What tracks the gap
is learned depth **reliance**, not capacity — see
[WHAT_DOESNT_WORK](WHAT_DOESNT_WORK.md#capacity-controls-the-gap).

SUN RGB-D numbers come from **short-schedule** diagnostics (300 epochs / 40-epoch screens),
not full-length arms. Screens continue from an already-trained checkpoint under the same
optimizer and warmup, which makes them **systematically flattering**. They are screening
evidence only.

---

## Artifact map

| Quantity | File under `results/score_push/runs/l2_nof/` |
|---|---|
| Baseline | `m1_matched_baseline_final.json` |
| Ours, run 1 / 2 / 3 | `c2_ep500_seed1_final.json`, `c2_ep500_seed2_final.json`, `baseline_seed2_final.json` |
| Modality dropout | `dropout_ep500_final.json` |
| Constant-depth training | `oracle_ep500_final.json` |
| RGB-only reference `R` | `screens/rgb_only_finetune/eval.json` |
| Per-stage profile | `stage_profiles/*.json` |
| Transplants | `transplant/*.json` |
| Corruption families | `failure_families/*.json`, `failure_curve/*.json` |
| Positional control | `pos_control_screen/*.json` |
| SUN RGB-D screen | `e2_sun_screen/*.json` |

> **Naming wart:** the third run of *our* arm lives in a file named
> `baseline_seed2_final.json`. It is a **method** run; the name is historical. This is
> exactly why `scripts/verify_results.py` names its inputs explicitly rather than globbing.

Each JSON carries its own provenance: checkpoint path, config module, split, seed,
protocol flags, per-class IoUs, and a timestamp. Absolute paths inside are the original
machine's and are deliberately left untouched — these files are checksummed evidence.
