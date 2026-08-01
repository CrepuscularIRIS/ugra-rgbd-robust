#!/usr/bin/env bash
# MP-A oracle KILL TEST driver (EVAL-ONLY). Runs one (mode, tau, conditions) eval
# via the frozen harness and writes a JSON. GPU is passed in. NO training.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"
cd ${UGRA_ROOT}

GPU="$1"; MODE="$2"; TAU="$3"; CONDS="$4"; TAG="$5"; OUT="$6"

export CUDA_VISIBLE_DEVICES="$GPU"
export LOCAL_RANK=0 TORCHDYNAMO_VERBOSE=0
export CKPT="checkpoints/trained/DFormerv2_Small_NYU.pth"
export EVAL_MODES=single
export EVAL_CONDITIONS="$CONDS"
export SLIDING=1 EVAL_SPLIT=full SEED=20260721
export MPA_ORACLE="$MODE" MPA_TAU="$TAU" MPA_STAGES=2
export MODEL_TAG="$TAG"
export OUT_JSON="$OUT"
python scripts/eval_robust_suite.py
