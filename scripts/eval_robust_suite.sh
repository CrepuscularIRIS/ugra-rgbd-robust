#!/usr/bin/env bash
# Frozen robustness eval harness (WP0) for DFormerv2-S on NYUDepthv2.
# Runs the OFFICIAL protocol (sliding + flip + amp; single-scale and/or MS) over
# clean / entire_missing / holes / gaussian via the Phase-1 corruption registry.
#
# Usage:
#   CKPT=... EVAL_MODES=single,ms EVAL_CONDITIONS=clean,entire_missing,holes,gaussian \
#   OUT_JSON=... MODEL_TAG=... bash scripts/eval_robust_suite.sh
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TORCHDYNAMO_VERBOSE=0
export LOCAL_RANK=0

# defaults (override via env)
export CKPT="${CKPT:-checkpoints/trained/DFormerv2_Small_NYU.pth}"
export EVAL_MODES="${EVAL_MODES:-single}"
export EVAL_CONDITIONS="${EVAL_CONDITIONS:-clean,entire_missing,holes,gaussian}"
export SEED="${SEED:-20260721}"
export SLIDING="${SLIDING:-1}"
export EVAL_SPLIT="${EVAL_SPLIT:-full}"
export OUT_JSON="${OUT_JSON:?set OUT_JSON}"

PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
python "${PROJECT}/scripts/eval_robust_suite.py"
