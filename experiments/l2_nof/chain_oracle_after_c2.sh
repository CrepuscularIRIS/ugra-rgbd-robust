#!/usr/bin/env bash
# Chain: wait for the in-flight C2 ep500 promotion run to finish, then launch the
# S-SPATIAL-ORACLE (the selected missing point, prediction row written 2026-07-25).
#
# Pipelining rule: WIP is one T4+ build at a time, so the oracle must not start while
# the promotion arm holds the GPUs. This waits for the training processes to disappear,
# confirms the GPUs are actually free, smoke-tests the oracle for a couple of epochs,
# and only then commits the full 500-epoch run.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

ROOT=${UGRA_ROOT}
cd "$ROOT"
RUN=results/score_push/runs/l2_nof
LOG="$RUN/oracle_chain.log"

log() { echo "[chain $(date -u +%H:%M:%S)] $*" >> "$LOG"; }

log "waiting for the C2 promotion run (utils/train.py) to exit ..."
while pgrep -f "utils/train.py" > /dev/null 2>&1; do sleep 60; done
log "no train.py processes remain."

# Confirm the GPUs are genuinely free before committing.
for i in $(seq 1 10); do
  USED=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | wc -l)
  [ "$USED" -eq 0 ] && break
  log "still $USED compute app(s) on the GPUs; waiting ..."
  sleep 60
done

# Record how the promotion arm ended, so a crash is visible rather than silently
# inherited by the next stage.
if grep -q "C2_FULLTRAIN_RESUME_DONE" "$RUN/c2_full_seed1_ep500_resume.log" 2>/dev/null; then
  log "promotion arm COMPLETED normally."
else
  log "WARNING: promotion arm log has no completion marker - it may have died. Launching the"
  log "         oracle anyway (the GPUs are free and the oracle is independently gated), but the"
  log "         promotion arm needs inspection."
fi
log "final promotion evals:"
grep "validation result" "$RUN/c2_full_seed1_ep500_resume.log" 2>/dev/null | tail -5 >> "$LOG"

# ---- SMOKE: bounded run of the real recipe; verify it starts, trains and stays finite ----
# nepochs comes from the Hydra-style config, so there is no epoch-count override to pass.
# Instead run the real command under a short timeout and require observable progress with
# no error signatures, then kill it and commit to the full run.
log "SMOKE: bounded oracle dry run (must reach epoch 2 with finite loss) ..."
SMOKE_LOG="$RUN/oracle_smoke.log"
PORT=29613 timeout 900 bash experiments/l2_nof/run_spatial_oracle.sh > "$SMOKE_LOG" 2>&1
pkill -f "master_port=29613" 2>/dev/null

if grep -qE "Traceback|CUDA out of memory|Non-finite|ChildFailedError" "$SMOKE_LOG"; then
  log "SMOKE FAILED (error signature present) - aborting before the full run. See $SMOKE_LOG"
  exit 1
fi
if ! grep -q "Epoch 2/" "$SMOKE_LOG"; then
  log "SMOKE FAILED (never reached epoch 2) - aborting. See $SMOKE_LOG"
  exit 1
fi
# Guard the one thing that would silently invalidate the oracle: the depth dial must be 0.
if ! grep -q "DEPTH_LAMBDA_TRAIN=0" "$SMOKE_LOG"; then
  log "SMOKE FAILED (DEPTH_LAMBDA_TRAIN=0 not confirmed in the log) - aborting."
  exit 1
fi
log "SMOKE ok (reached epoch 2, no error signatures, depth dial confirmed 0)."
sleep 30   # let the GPUs drain before the real launch

# ---- FULL: 500 epochs from pretrained, depth bias permanently removed ----
log "launching FULL oracle (500ep, DEPTH_LAMBDA_TRAIN=0) ..."
EVAL_EVERY=25 PORT=29612 \
  bash experiments/l2_nof/run_spatial_oracle.sh > "$RUN/oracle_ep500_seed1.log" 2>&1
log "oracle run exited. tail:"
tail -3 "$RUN/oracle_ep500_seed1.log" >> "$LOG"
log "final oracle evals:"
grep "validation result" "$RUN/oracle_ep500_seed1.log" 2>/dev/null | tail -5 >> "$LOG"
log "CHAIN DONE"
