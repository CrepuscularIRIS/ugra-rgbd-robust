#!/usr/bin/env bash
# Wait for the in-flight C2 seed-2 training to exit, then run the positional-alpha control screen.
# The control is a 28 GB finetune and MUST NOT run concurrently with training (it would be denied
# by the guard, or worse, OOM the run). Chaining avoids both idle GPU and contention.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

PROJ=${UGRA_ROOT}
cd "$PROJ"
LOG="$PROJ/results/score_push/runs/l2_nof/pos_control_chain.log"

echo "[chain] waiting for utils/train.py to exit ..." | tee -a "$LOG"
# NOTE: `pgrep -f "utils/train.py"` ALSO matches any monitoring shell whose command line contains
# that string (the agent harness wraps commands in `bash -c '...'`), so a plain -f match can be held
# true indefinitely by the act of watching it. Require an actual python/torchrun process instead.
while pgrep -f "python.*utils/train\.py" > /dev/null 2>&1; do sleep 120; done
echo "[chain] training exited at $(date -Is)" | tee -a "$LOG"

# Settle, then require real free memory before a 28 GB finetune.
sleep 60
source scripts/gpu_guard.sh
GPU=""
for attempt in $(seq 1 30); do
  GPU=$(pick_probe_gpu "$FOOTPRINT_FINETUNE") && break
  echo "[chain] no 28 GB capacity yet (attempt $attempt)" | tee -a "$LOG"
  sleep 120
done
if [ -z "$GPU" ]; then
  echo "[chain] ABORT — never got finetune capacity" | tee -a "$LOG"
  exit 1
fi

# --- FIRST: evaluate seed 2 itself on the frozen harness. This is the result the 5-hour run
# exists to produce; the control must not jump the queue and occupy the GPU ahead of it. Use the
# rolling latest.pth when present (added this session) so the FINAL epoch is scored, not merely
# the best-validation one; fall back to the highest epoch-*.pth otherwise.
# Derive RUNDIR from the TRAINING LOG, not from `ls -td`. Diagnostic probes that instantiate the
# model also create timestamped checkpoint dirs (two appeared during this session), so a newest-mtime
# guess is only correct while training is still writing — exactly the assumption that breaks the
# moment training exits, which is when this chain runs.
TRAINLOG="$PROJ/results/score_push/runs/l2_nof/c2_full_seed2_resume.log"
RUNDIR="$PROJ/repos/DFormer/$(grep -oP 'checkpoints/NYUDepthv2_DFormerv2_S_[0-9-]+' "$TRAINLOG" | tail -1)"
if [ ! -d "$RUNDIR" ]; then
  echo "[chain] WARNING: could not derive RUNDIR from the training log; falling back to newest dir" | tee -a "$LOG"
  RUNDIR=$(ls -td "$PROJ"/repos/DFormer/checkpoints/NYUDepthv2_DFormerv2_S_* 2>/dev/null | head -1)
fi
echo "[chain] RUNDIR=$RUNDIR" | tee -a "$LOG"
CK="$RUNDIR/latest.pth"
if [ ! -f "$CK" ]; then
  CK=$(ls "$RUNDIR"/epoch-*.pth 2>/dev/null | sed -E 's/.*epoch-([0-9]+)_.*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
fi
if [ -n "${CK:-}" ] && [ -f "$CK" ]; then
  echo "[chain] evaluating seed-2 final checkpoint: $CK" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$GPU" CKPT="$CK" EVAL_MODES=single \
    EVAL_CONDITIONS=clean,entire_missing,holes,gaussian SEED=20260721 \
    OUT_JSON="$PROJ/results/score_push/runs/l2_nof/c2_ep500_seed2_final.json" \
    MODEL_TAG=C2_ep500_seed2 \
    bash scripts/eval_robust_suite.sh 2>&1 | tee -a "$LOG" | grep -E "miou=|Traceback|out of memory" || true
  echo "[chain] SEED2_EVAL_DONE" | tee -a "$LOG"
else
  echo "[chain] WARNING: no seed-2 checkpoint found under $RUNDIR — skipping its eval" | tee -a "$LOG"
fi

echo "[chain] launching P2-CTRL on GPU$GPU at $(date -Is)" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES="$GPU" bash experiments/l2_nof/run_pos_control_screen.sh >> "$LOG" 2>&1
rc=$?
# Never trust a completion marker without checking the log (standing rule).
if [ $rc -eq 0 ] && ! grep -qE "Traceback|out of memory|CUDA error" "$LOG"; then
  echo "[chain] P2CTRL_OK" | tee -a "$LOG"
else
  echo "[chain] P2CTRL_FAILED rc=$rc" | tee -a "$LOG"
fi
