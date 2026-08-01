# Experiments

Training recipes and the launchers that ran them. The full environment-variable interface
is in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md).

> **Read this before running anything here.** Most of these directories are **failed
> arms**, kept deliberately as the evidence base for
> [`docs/WHAT_DOESNT_WORK.md`](../docs/WHAT_DOESNT_WORK.md). They are **not**
> recommendations. Several are disqualified by construction — they train on the very
> condition the evaluation uses as its failure mode — and several were falsified outright.

---

## `l2_nof/` — the arms in the paper

This is the directory you want.

```bash
# baseline (no intervention)
C2_ATTEN=0 bash experiments/l2_nof/run_c2_fulltrain.sh

# ours: alpha ~ U(0.05, 1) — never reaches 0, so training never visits the absent operator
C2_ATTEN=1 C2_DIST=u005_1 bash experiments/l2_nof/run_c2_fulltrain.sh

# modality dropout: alpha in {0,1}
C2_ATTEN=1 C2_DIST=bernoulli bash experiments/l2_nof/run_c2_fulltrain.sh

# constant-depth reference
DEPTH_LAMBDA_TRAIN=0 bash experiments/l2_nof/run_spatial_oracle.sh
```

All four arms are 500 epochs from `DFormerv2_Small_pretrained.pth` and differ in
**exactly one thing**. In this architecture the three recipes are three distributions
over one scalar α — see [`patches/README.md`](../patches/README.md).

Also here: the matched positional control (`run_pos_control_screen.sh`), per-stage
profiles, transplants, dose–response sweeps, the four-eval-seed noise floor, and the
SUN RGB-D screen.

---

## The graveyard

| Directory | What it tried | Outcome |
|---|---|---|
| `n5_dual_kd/` | Large→Small distillation | abandoned at 3× budget — **infeasible, not disproven** |
| `n6_adqg/`, `n6_hhd_grp/` | per-block coupling parameters + auxiliary losses | ~3 % of the gap |
| `n7_cycle4/` | adapter-class modules | at baseline (`−0.21`, `+0.09`) |
| `n7_cycle5_mpa/` | edge-only depth prior | **falsified** by a GT-edge oracle — edge-only is *worse* than full |
| `n7_cycle5_mpb*/` | depth-dropout screen | proved the gap is removable, but trades clean accuracy |
| `n7_cycle5_prp/` | RGB-predicted pseudo-depth | gate never left zero — **untested, not refuted** |
| `n7_decomp_witness/` | decomposition witness adapter | superseded |
| `wp1_reliance/`, `wp1b_adsd/`, `wp2_stage2/` | earlier work packages | superseded |

Each entry's mechanism of failure — which is the part worth reading — is in
[`docs/WHAT_DOESNT_WORK.md`](../docs/WHAT_DOESNT_WORK.md).

---

## Screens are not results

A **screen** is a 40-epoch continuation from an already-trained checkpoint under the same
optimizer and warmup. That is precisely the property that makes screens **systematically
flattering**. They exist to kill candidates cheaply, not to rank survivors.

Only a full 500-epoch run from the pretrained backbone is promotable.

---

## Operational rules

- **`USE_AMP=0`.** Always. fp16 is unstable on this architecture.
- **Set `C2_ATTEN` and `C2_DIST` explicitly** and read the echoed launch line. A
  hard-coded `C2_ATTEN=1` once turned a baseline replication into a third method run,
  costing 21 GPU-hours.
- **Launch detached** (`setsid`, `nohup`). A session teardown killed a 4.5-hour run.
- **Gate concurrent probes** through [`scripts/gpu_guard.sh`](../scripts/gpu_guard.sh) —
  a concurrent eval costs training ~19 %.
- **One full-length training at a time.**
