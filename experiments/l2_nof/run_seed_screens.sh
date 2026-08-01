#!/usr/bin/env bash
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}; conda activate "$UGRA_CONDA_ENV"
PROJ=${UGRA_ROOT}; cd "$PROJ"
export CUDA_VISIBLE_DEVICES=0
for SD in 20260722 20260723; do
  SAVE="$PROJ/results/score_push/runs/l2_nof/c2_seed${SD}"; mkdir -p "$SAVE"
  echo "=== [seed=$SD] TRAIN ==="
  CKPT_INIT="checkpoints/trained/DFormerv2_Small_NYU.pth" SAVE_DIR="$SAVE" \
    P_CLEAN=0.5 ALPHA_MIN=0.05 EPOCHS=40 LR=6e-5 TRAIN_BATCH=16 NUM_WORKERS=4 SEED=$SD \
    python experiments/l2_nof/train_c2_attenuate.py
  CK="$SAVE/ckpt_final_c2.pth"
  echo "=== [seed=$SD] EVAL ==="
  CKPT="$CK" EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing,holes,gaussian \
    OUT_JSON="$SAVE/eval.json" MODEL_TAG="c2_seed${SD}" bash scripts/eval_robust_suite.sh
  echo "SEED_${SD}_DONE"
done
echo SEED_SCREENS_DONE
