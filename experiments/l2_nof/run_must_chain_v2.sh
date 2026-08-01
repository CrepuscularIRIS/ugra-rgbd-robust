#!/usr/bin/env bash
# MUST chain, RELAUNCH after the 2026-07-27 env-leak incident.
# Each arm now ASSERTS its own configuration from the launcher's own echo line before training,
# so a silent override cannot waste another 10 GPU-h.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}; conda activate "$UGRA_CONDA_ENV"
P=${UGRA_ROOT}; cd $P
L=$P/results/score_push/runs/l2_nof/must_chain_v2.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }
wait_gpus () { while pgrep -f "python.*utils/train\.py" >/dev/null 2>&1; do sleep 120; done; sleep 30; }

run_arm () {  # name expected_atten expected_dist logfile port
  local name=$1 xa=$2 xd=$3 log=$4 port=$5
  say "$name START (expect C2_ATTEN=$xa C2_DIST=$xd)"
  env C2_ATTEN=$xa C2_DIST=$xd CUDA_VISIBLE_DEVICES=0,1 PORT=$port \
    bash experiments/l2_nof/run_c2_fulltrain.sh > "$log" 2>&1 &
  local pid=$!
  # ASSERT the actual config from the launcher's echo, then let it run.
  for i in $(seq 1 60); do grep -q "\[C2-full\]" "$log" 2>/dev/null && break; sleep 5; done
  local line; line=$(grep "\[C2-full\]" "$log" | head -1)
  say "  actual: $line"
  if ! echo "$line" | grep -q "C2_ATTEN=$xa" || ! echo "$line" | grep -q "C2_DIST=$xd"; then
    say "  CONFIG MISMATCH — killing $name"; kill $pid 2>/dev/null; pkill -f "utils/train.py"; return 1
  fi
  say "  config verified; training"
  wait $pid
  say "$name DONE ($(tail -1 "$log"))"
}

run_arm "M1 baseline-replication" 0 u005_1 "$P/results/score_push/runs/l2_nof/baseline_seed2_v2.log" 29630
wait_gpus
run_arm "M2 method-seed3"         1 u005_1 "$P/results/score_push/runs/l2_nof/method_seed3_v2.log"   29631
say "MUST_CHAIN_V2_COMPLETE"
