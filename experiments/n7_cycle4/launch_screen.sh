#!/usr/bin/env bash
# n7 cycle-4 (M3) finetune-proxy SCREEN launcher.
# Runs torchrun DETACHED under the wrapper, tees to a log, and maintains state.json for the
# Monitor heartbeat. This is a SCREEN (kill-gate), NOT a promotable full-train. No ccnative_ingest.
set -u
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"


RUNDIR="$1"; PORT="$2"
ABS_RUNDIR="$(readlink -f "$RUNDIR")"
LOG="$ABS_RUNDIR/train.log"
STATE="$ABS_RUNDIR/state.json"

PROJ=${UGRA_ROOT}
source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONUNBUFFERED=1

write_state () {
  local status="$1"; local note="$2"
  local ep0 last_miou epoch
  ep0="$(grep -oE 'EPOCH0_IDENTITY val_mIoU=[0-9.]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')"
  last_miou="$(grep -oE 'val_mIoU=[0-9.]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+$')"
  epoch="$(grep -oE 'epoch=[0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+$')"
  cat > "$STATE" <<EOF
{
  "node": "n7",
  "cycle": 4,
  "method": "M3_boundary_gate",
  "kind": "finetune_proxy_SCREEN",
  "reservation": "rsv_00c161fd",
  "status": "$status",
  "note": "$note",
  "pid": ${TRAIN_PID:-null},
  "port": $PORT,
  "log": "$LOG",
  "epoch0_identity_miou": ${ep0:-null},
  "last_val_miou": ${last_miou:-null},
  "last_epoch": ${epoch:-null},
  "kill_switch_miou": 56.0,
  "updated": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
}

cd "$PROJ/repos/DFormer" || { echo "cd DFormer failed"; exit 3; }

torchrun --nproc_per_node=2 --master_port="$PORT" \
  "$PROJ/experiments/n7_cycle4/train_n7_cycle4.py" \
  --config=local_configs.NYUDepthv2.DFormerv2_S --gpus=2 --epochs=50 --no-amp --syncbn \
  --eval_iter=5 \
  --init_from=checkpoints/trained/DFormerv2_Small_NYU.pth \
  --tag=m3_screen --seed=20260721 >> "$LOG" 2>&1 &
TRAIN_PID=$!
write_state "RUNNING" "torchrun launched"

# maintain state.json while the run lives
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  write_state "RUNNING" "training"
  sleep 30
done
wait "$TRAIN_PID"
RC=$?
if [ "$RC" -eq 0 ] && grep -q "Training complete" "$LOG"; then
  write_state "DONE" "training complete rc=0"
else
  if grep -qiE "Traceback|CUDA out of memory|RuntimeError|FATAL|Error" "$LOG"; then
    write_state "CRASHED" "rc=$RC (error signature in log)"
  else
    write_state "EXITED" "rc=$RC (no completion marker)"
  fi
fi
