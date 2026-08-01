#!/usr/bin/env bash
# RPD-Drop finetune (WP1/P1), single-GPU. Env-driven (see train_reliance.py).
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCHDYNAMO_VERBOSE=0
export PYTHONUNBUFFERED=1
export LOCAL_RANK=0
export CKPT_INIT="${CKPT_INIT:-checkpoints/trained/DFormerv2_Small_NYU.pth}"
export SAVE_DIR="${SAVE_DIR:?set SAVE_DIR}"
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
python "${PROJECT}/experiments/wp1_reliance/train_reliance.py"
