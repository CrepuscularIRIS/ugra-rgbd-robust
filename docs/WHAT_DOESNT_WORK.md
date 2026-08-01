# What didn't work, and what withdrew it

A negative is only a product if it is **general** and **explained**. Every entry below
states the mechanism of failure, not merely the outcome, and — where a claim of ours was
withdrawn — the measurement that withdrew it.

> **Nothing in the "withdrawn claim" column is asserted anywhere in the paper.** Several
> are attractive enough that a reader could reconstruct them from our own results. That
> is precisely why the list exists.

> [!WARNING]
> This document supersedes any earlier internal analysis notes. Older drafts of this
> project circulated a *capacity* framing and a `6.61` baseline gap; **both are
> withdrawn**. The current baseline gap is `5.99` (our own retrain), and the capacity
> account is refuted below. If you find a memo quoting `6.61` or "capacity dominates
> depth", it predates this file.

---

## 1. Withdrawn claims of ours

### "Depth-free is a ceiling"

**Withdrawn by:** our own arms. An RGB-only finetune reached `53.26` depth-absent, *above*
the constant-depth arm's `53.03`.

A depth-free model is a **reference**, not a bound. This is why the paper calls `R` an
RGB-only training *reference* and never a ceiling, and why the two partition terms carry
descriptive rather than causal names.

### "The gap is not information loss, because the Large model has no gap"

**Withdrawn by:** inspection of what Large actually does. DFormerv2-L **ignores** depth
(stage-2 cosine `0.9995`), so it supplies the RGB-only reference at *L's* capacity — not
at Small's. The comparison never controlled capacity.

Only a **same-capacity** depth-free reference can decompose Small's gap, which is what
the paper uses.

### "Capacity controls the gap"

**Withdrawn by:** the same architecture at the same capacity on a second dataset.

| model / dataset | normalized reliance | gap |
|---|---:|---:|
| Small / NYU Depth v2 | 0.372 | **6.61**\* |
| Small / SUN RGB-D | 0.050 | **0.07** |
| Large / NYU Depth v2 | 0.044 | **0.11** |

\* *this table is from the earlier eval convention; see the warning above.*

**Same capacity gives a large gap on one dataset and none on the other.** Capacity is
therefore not the operative variable, and the phenomenon is **NYU-Depth-v2-specific of
the two datasets tested** — stated as a limitation, up front. What tracks the gap is
learned depth **reliance**, a joint property of training-depth quality and capacity.

### "C2 re-calibrates the attention operator (normalization)"

**Withdrawn by:** a parameter-transplant sufficiency test. Transplanting the repaired
model's normalization parameters into the baseline recovers **8 %** of the gain; the
non-normalization complement recovers **91 %** (sanity: everything recovers 101 %).

**The transferable lesson — the most transferable thing in this document:**

> **A checkpoint diff shows where parameters *moved*, not where the function *lives*.**

Normalization biases moved ~21 % but hold 2.4k parameters. Encoder/decoder weights moved
~5.6 % but hold 26.7M. Reading mechanism off *relative per-tensor change* is reading the
wrong quantity. Use **transplant or ablation**.

### "The repair is forced out of the attention score path"

**Withdrawn by:** training the score path alone, which recovers most of the gain.

We had rested this on a contrast between two transplant shares at equal parameter count
(q/k `21.6 %` vs v/out `38.0 %`). The contrast was a **denominator artifact**, and the
accompanying algebraic claim — that the depth difference is not expressible in the
query/key geometry — is simply **false**, since both projections are full rank.

> **A share is not an attribution when the shares do not sum.**

Related: the value path is **not diagnostic**. It carries ~50 % of *both* a depth and a
positional perturbation, so that share is a property of the site, not of depth. Discard
any conclusion resting on it.

### "Damage is operator corruption" — true but nearly vacuous

Depth has exactly **one** pathway in this architecture, so when it vanishes nothing else
*can* change. The claim is **architecturally tautological here** and a reviewer will say
"of course". The non-trivial half is *"not information loss"*, whose load-bearing
evidence is the same-capacity depth-free reference.

### "37 % of the gain is generic"

Superseded by the corrected partition. Values like this survive **only** inside a claim
this document reports as withdrawn — they are legitimate there and nowhere else.

---

## 2. Restoring the information does not work

Because information is not what is missing.

### Generic / expected-statistic depth imputation

Replacing absent depth with the train-set mean depth map recovered `+0.10` mIoU;
separately, injecting the closed-form train-set expectation of the attention bias
`E[mask_depth]` gave `−0.21`. **Refuted twice.**

**Why it generalizes:** the recoverable quantity is **per-instance**, not a population
statistic. The dose-response curve on the bias scale is smooth, which proves the
*magnitude* of the bias is recoverable — **not** that any single global statistic carries
it. A smooth dose-response licenses continuity arguments; it never licenses
statistic-substitution ones. This kills the whole "substitute the operator with its
expectation" family for about 0.3 GPU-hours.

> "Inject the closed-form `E[mask_depth]`" is this same idea under a different name. Do
> not resurrect it.

### Predicting depth from RGB (PRP)

An RGB-branch pseudo-depth head feeding a parallel, zero-init gated bias. Clean `+0.28`,
but the stage-2 feature cosine stayed flat (`0.9370 → 0.9365`) and `final_wp_abs_sum` was
exactly `0.0` — **the zero-init gate never left zero** in 5000 finetune steps.

**Why it generalizes:** a zero-init-gated mechanism is **structurally un-screenable by a
finetune proxy.** At gate 0 the segmentation gradient into the gate is ≈ 0, and weight
decay plus a decaying learning rate give it no escape velocity. Such mechanisms need warm
init, a no-decay higher-LR parameter group, or a from-scratch full train. The `+0.28`
came from the auxiliary loss reshaping shared features, not from the mechanism under
test.

> **Correction:** this was first diagnosed as "the gate could not move." The real cause
> was worse — `w_p` was **never in the optimizer at all** (see §5). PRP is *untested*,
> not refuted.

### Edge-only priors

A ground-truth-depth-edge oracle **falsified** the "gradient-discontinuity conflation"
account: edge-only injection is *worse* than full injection (clean `−1.60`), and
smooth-only is the **least** damaging — the opposite of the prediction. The τ sweep is
monotone with no interior peak.

The depth-decay bias is a broadly-useful prior across **all** depth structure, not a
discontinuity detector. "Inject only at edges" is a dead direction.

---

## 3. Adding capacity or surface parameters does not work

| Approach | Result | Why |
|---|---|---|
| **Adapter-class modules** (two independent designs) | `−0.21` and `+0.09` — both at baseline | An adapter bolted onto a frozen backbone cannot manufacture capacity the model lacks. |
| **Per-block coupling parameters + auxiliary loss** | ~3 % of the gap | The coupling *parameters* were never the bottleneck — and see §5: the depth-bias authority is not even trainable in this codebase, so this was tuning around a frozen surface. |
| **Large → Small distillation** | Abandoned at 12–18 GPU-h (3× budget) | **Infeasible at budget, not disproven.** That is the honest label. |

---

## 4. Methods that train on the evaluation condition

**Disqualified by construction**, not by measurement: modality dropout to zeros,
availability flags, CFG-style null-conditioning, and RPD-Drop all train the model in the
configuration the evaluation failure produces.

Verified bitwise that gate-off ≡ zeros-depth forward, so these are literally training on
the test condition.

They still appear in this repository **as comparators** — the paper reports modality
dropout as one of the three recipes, because comparing against it is the point. What is
disqualified is *promoting* one as our method.

A depth-dropout screen did establish something real and cheap: the gap is
**removable**, not a capacity wall. But it buys that by trading clean accuracy, so it
proves a ceiling and is not itself the method.

---

## 5. Implementation defects that change how the numbers read

Disclosed in the paper, preserved in the code, so that anyone building on this inherits
the caveats.

### `group_weight` never adds bare `nn.Parameter`s to the optimizer

`.modules()` yields `Module`s, so the `isinstance(m, nn.Parameter)` branch is **dead
code**. All 29 `GeoPriorGen.weight` tensors — the positional/depth mixture authority —
**never train**, plus a handful of other core parameters.

Every principal experiment is therefore a **frozen-mixture** variant of DFormerv2. It is
a *shared* condition across arms, so between-arm comparisons remain internally matched,
and a single corrected training gave similar point estimates (a sensitivity check, not a
replication).

**Narrow consequence:** it rules out only the *direct bias-magnitude* de-biasing route.
Implicit Q/K-rotation de-biasing is untested. And it narrows an earlier claim: "SGD does
not repair the operator" describes repairs that were **available**, not **preferred**.

### The Ham/NMF decoder is nondeterministic

`ham_head.py` redraws `torch.rand` bases on every forward. **Reseed before every forward**
for any tensor-level work, and put a self-consistency gate in every diagnostic. This is
the entire source of the harness dispersion.

### Save-only-best lost a final checkpoint

The training loop retains only the best checkpoint by validation metric, so one run's
converged final-epoch weights were not kept. That is why checkpoint selection is not
uniform across the arms, worth roughly `0.3` mIoU on that run. It cannot be repaired
after the fact.

### A silent `strict=False` load ran three probes on an untrained model

`load_state_dict(w.get("model", w), strict=False)` — the checkpoint's top-level key is
`state_dict`, so this matched **0 of 877 tensors** and ran an untrained model at 1.6 %
pixel accuracy **while printing plausible numbers.** It produced a published false
finding. Caught only by a self-consistency gate added after an arithmetic inconsistency
forced a withdrawal.

> **Rule:** a probe that builds its own model must assert **≥ 90 % tensor match** *and*
> reproduce a known reference number before any internal quantity is read. Never pair
> `strict=False` with an unchecked `.get()` chain.

---

## 6. Method lessons worth more than the results

- **No diagnostic ships without a control whose failure invalidates the run.** Burned
  both ways in one session: a self-consistency gate caught a false FAIL; a *missing*
  positive control produced a false PASS.
- **Never promote an effect smaller than the training-seed σ** (`0.49` on
  entire-missing).
- **A passed test proves nothing without severity.** Before any expensive test: *if the
  hypothesis were false, how likely is it this test still passes?* If "fairly likely",
  the test is ceremonial.
- **Triage an anomaly before chasing it:** artifact → bug → noise → known → real. The
  first three rungs are eval-only minutes, and most surprises die there.
- **Query the arXiv API directly for novelty on recent preprints.** Web search has a
  ~2-week blind spot; it false-negatived a directly relevant preprint three days old.
