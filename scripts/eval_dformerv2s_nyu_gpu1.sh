#!/usr/bin/env bash
# Single-GPU eval: DFormerv2-S on NYU Depth V2 (GPU 1 by default)
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"

ROOT="$(cd "$(dirname "$0")/../repos/DFormer" && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TORCHDYNAMO_VERBOSE=0

CKPT="${1:-checkpoints/trained/DFormerv2_Small_NYU.pth}"
PORT="${MASTER_PORT:-29572}"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_port="$PORT" \
  utils/eval.py \
  --config=local_configs.NYUDepthv2.DFormerv2_S \
  --gpus=1 \
  --sliding \
  --no-compile \
  --syncbn \
  --mst \
  --amp \
  --continue_fpath="$CKPT"
