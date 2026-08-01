#!/bin/bash
# n7 training launcher -- fully daemonized (double-fork, nohup, signal traps)
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"


# Ignore hangup and terminal signals
trap '' HUP TERM INT

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"

cd ${UGRA_ROOT}/repos/DFormer

export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${UGRA_ROOT}/repos/DFormer:${UGRA_ROOT}/repos/DFormer/utils:${UGRA_ROOT}/experiments/n7_decomp_witness"
export PYTHONUNBUFFERED=1
export USE_AMP=0

LOG_DIR="${UGRA_ROOT}/experiments/n7_decomp_witness/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_n7_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="$LOG_DIR/train_n7.pid"

echo "[n7] Starting training at $(date -Iseconds)" >> "$LOG_FILE"
echo "[n7] Log: $LOG_FILE" >> "$LOG_FILE"

# Write PID file for monitoring
echo $$ > "$PID_FILE"

exec nohup torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_port=29573 \
  ${UGRA_ROOT}/experiments/n7_decomp_witness/train_n7.py \
  --config=local_configs.NYUDepthv2.DFormerv2_S \
  --gpus=2 \
  --epochs=500 \
  --no-amp \
  --syncbn \
  --no-compile \
  --witness_weight=0.05 \
  --seg_aux_weight=0.2 \
  --adapter_stage_idx=2 \
  >> "$LOG_FILE" 2>&1
