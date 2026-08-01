# Contributing

This is a research repository attached to a specific paper, so "contributing" mostly
means one of four things. Each has a different bar.

---

## 1. Reporting a reproduction — most valuable

Especially a **failed** one. Two failures are bugs in this repository and we want to know
immediately:

- `python3 scripts/verify_results.py` does not print `OK` on a fresh clone.
- `python diagnostics/single_path_invariance.py` does not report bitwise identity on a
  correctly patched checkout with the released weights.

Please include: the exact command with all environment variables, `torch.__version__` and
CUDA version, the output of `bash scripts/setup_dformer.sh --check`, and — for training —
the launcher's echoed `C2_ATTEN` / `C2_DIST` line.

Numbers landing within the run-to-run dispersion (`0.29` clean / `0.49` entire-missing)
are a **successful** reproduction, not a failed one. See
[TROUBLESHOOTING](docs/TROUBLESHOOTING.md#my-retrained-arm-is-off-by-03-05-miou).

---

## 2. Running the audit on another architecture — most wanted

The paper's criterion is the part that depends on none of our trained models, and **we
have not exercised it on anyone else's**. The one external architecture for which we hold
weights is a dual-stream design, where no removal is canonical.

The procedure, for a model you have:

1. **Identify the pathway** through which the auxiliary modality enters — its encoder
   branch, its token set, or its bias term. This is stated in the architecture's own
   paper.
2. **Evaluate twice**: once zero-filled, once with that pathway structurally deleted.
3. **Compare** at the logits and at each stage output.

Report the verdict as one of three states:

| state | meaning |
|---|---|
| **Exact** | the substituted value *is* the operator's neutral element, verified. The number measures absence. |
| **Approximate, artifact quantified** | not neutral; the named removal, the fill value, and the measured discrepancy are all reported *beside* the robustness figure. |
| **Approximate, artifact unquantified** | not neutral and the test was not run. The number differs from the cost of absence by an unknown amount **and unknown sign**. |

Two caveats that must travel with any result:

- Bitwise equality is usable only when the two paths execute algebraically identical
  operations under deterministic kernels. Otherwise report a **pre-specified numerical
  tolerance** under a stated reseeding protocol.
- **A negative verdict is not a finding of error** against a published result. It
  establishes that the number is a composite whose parts were not separated.

Open an issue — results on families we could only make predictions about are exactly what
this work needs.

---

## 3. Code changes

Modest bar, but two hard rules:

- **Do not "fix" the two disclosed upstream defects** in a way that silently changes the
  reproduction. The frozen-mixture optimizer bug and the nondeterministic decoder are
  preserved deliberately, because the paper's numbers were produced with them. A fix is
  welcome as an **opt-in flag** with a clear note that enabling it means you are running a
  variant, not reproducing the paper.
- **Never modify anything under `results/`.** Those files are checksummed evidence.
  `verify_results.py` will catch it, which is the point.

Style: match the surrounding code. Python is PEP 8 with type hints on new function
signatures. Keep the comment density — this codebase explains *why* far more than *what*,
because most of its traps are invisible otherwise.

Portability: nothing may hardcode an absolute path. Use the
[`scripts/env.sh`](scripts/env.sh) contract.

---

## 4. Adding a diagnostic

Three requirements, all learned by publishing something false.

1. **Prove the model loaded.** Assert ≥ 90 % tensor match **and** reproduce a known
   reference number before reading any internal quantity. A silent
   `load_state_dict(w.get("model", w), strict=False)` once matched 0 of 877 tensors and
   ran an untrained model while printing plausible numbers.
2. **Reseed before every forward.** The decoder is nondeterministic; without this, every
   small delta is noise.
3. **Ship a control whose failure invalidates the run.** Self-validating endpoints, a
   negative control, or both. We were burned in both directions in one session: a
   self-consistency gate caught a false FAIL, and a *missing* positive control produced a
   false PASS.

Before proposing an expensive test, state its **severity**: *if the hypothesis were
false, how likely is it this test still passes?* If "fairly likely", the test is
ceremonial — add a negative control that should fail, or pick the adversarial setting.

---

## What will be declined

- Methods that **train on the evaluation failure condition** (modality dropout to zeros,
  availability flags, CFG-style null-conditioning) proposed *as our method*. They are
  disqualified by construction. They remain welcome as **comparators** — the paper reports
  modality dropout as one of the three recipes.
- Directions already falsified in [WHAT_DOESNT_WORK](docs/WHAT_DOESNT_WORK.md), unless you
  are challenging the measurement that falsified them — in which case, please do.
- Mechanism claims resting on a **checkpoint diff**. A parameter diff shows where
  parameters moved, not where the function lives. Use transplant or ablation.
- Results promoted from **screens**. Screens are 40-epoch continuations from a trained
  checkpoint and are systematically flattering. Only full 500-epoch runs from the
  pretrained backbone are promotable.
- Effects smaller than the training-seed σ (`0.49` on entire-missing) presented as wins.

---

## Before circulating this repository

> [!IMPORTANT]
> [`CITATION.cff`](CITATION.cff) and the BibTeX block in [`README.md`](README.md) carry a
> **placeholder author list**. Replace both before sharing or submitting.

---

## Code of conduct

Be straightforward and accurate. Report what you measured, including what did not work
and what it cost. If you withdraw a claim, say what withdrew it — that convention is why
[WHAT_DOESNT_WORK](docs/WHAT_DOESNT_WORK.md) is useful rather than embarrassing.
