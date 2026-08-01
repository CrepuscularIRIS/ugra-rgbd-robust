# Documentation

| Document | What it covers |
|---|---|
| [INSTALLATION](INSTALLATION.md) | Three tiers of setup. The lowest needs no GPU, datasets or weights. |
| [DATASETS](DATASETS.md) | Layout, sources, pretrained weights, and how depth is consumed. |
| [REPRODUCTION](REPRODUCTION.md) | Five stages ordered by cost, from seconds to GPU-days. |
| [EXPERIMENTS](EXPERIMENTS.md) | The complete environment-variable interface, training and eval. |
| [DIAGNOSTICS](DIAGNOSTICS.md) | The eval-only instruments, and two non-optional rules. |
| [RESULTS](RESULTS.md) | Every table, with its caveats attached rather than in a footnote. |
| [WHAT_DOESNT_WORK](WHAT_DOESNT_WORK.md) | The graveyard: each withdrawn claim, and what withdrew it. |
| [TROUBLESHOOTING](TROUBLESHOOTING.md) | Symptom → cause, for the failures that actually happen here. |
| [PROJECT_STRUCTURE](PROJECT_STRUCTURE.md) | The tree, and three curation decisions. |

Also: [`../patches/README.md`](../patches/README.md) — what the method patch changes, and
the two upstream defects it deliberately does **not** fix.

---

## Suggested reading order

**Evaluating whether this work is sound** →
[RESULTS](RESULTS.md) → [WHAT_DOESNT_WORK](WHAT_DOESNT_WORK.md) →
`python3 scripts/verify_results.py`

**Reusing the criterion on your own model** →
[README → The criterion](../README.md#the-criterion) →
[CONTRIBUTING → Running the audit](../CONTRIBUTING.md#2-running-the-audit-on-another-architecture--most-wanted).
This part depends on none of our trained models.

**Reproducing the paper** →
[INSTALLATION](INSTALLATION.md) → [DATASETS](DATASETS.md) → [REPRODUCTION](REPRODUCTION.md)

**Building on the code** →
[PROJECT_STRUCTURE](PROJECT_STRUCTURE.md) → [`patches/README.md`](../patches/README.md) →
[EXPERIMENTS](EXPERIMENTS.md)

---

## The three things most likely to trip you up

1. **The two dispersions are not interchangeable.** Eval noise is `0.045` / `0.031`;
   run-to-run is `0.29` / `0.49`. Comparisons between *training arms* are judged against
   the second. ([RESULTS](RESULTS.md#the-two-dispersions))
2. **`DEPTH_LAMBDA` and `DEPTH_LAMBDA_TRAIN` multiply.** Setting both gives λ² and
   silently zeroes profiles. The harness now aborts if it sees both.
   ([EXPERIMENTS](EXPERIMENTS.md#depth-authority))
3. **`USE_AMP=0`, always.** fp16 is unstable on this architecture. A run under AMP
   reproduces nothing here. ([TROUBLESHOOTING](TROUBLESHOOTING.md#mixed-precision))
