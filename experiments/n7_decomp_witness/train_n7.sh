#!/usr/bin/env bash
# n7: Full 500-ep train with RGB Decomposition Adapter + Depth Witness
# Recipe: DFormerv2_Small_pretrained.pth -> 500 epochs, AdamW, USE_AMP=0
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"

ROOT="$(cd "$(dirname "$0")/../../repos/DFormer" && pwd)"
EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${ROOT}:${EXP_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export USE_AMP=0

CKPT="${1:-checkpoints/pretrained/DFormerv2_Small_pretrained.pth}"
EPOCHS="${2:-500}"
PORT="${MASTER_PORT:-29573}"
LOG_DIR="${3:-${EXP_DIR}/logs}"

mkdir -p "$LOG_DIR"

echo "[n7] Starting training: epochs=$EPOCHS, ckpt=$CKPT, amp=OFF, gpus=$CUDA_VISIBLE_DEVICES"
echo "[n7] Log dir: $LOG_DIR"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=2 --master_port="$PORT" \
  "$EXP_DIR/train_n7.py" \
  --config=local_configs.NYUDepthv2.DFormerv2_S \
  --gpus=2 \
  --epochs="$EPOCHS" \
  --no-amp \
  --syncbn \
  --no-compile \
  --continue_fpath="$CKPT" \
  --witness_weight=0.05 \
  --seg_aux_weight=0.2 \
  --adapter_stage_idx=2 \
  --log_dir="$LOG_DIR" \
  2>&1 | tee "$LOG_DIR/train_n7_$(date +%Y%m%d_%H%M%S).log"
