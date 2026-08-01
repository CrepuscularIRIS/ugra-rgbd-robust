#!/usr/bin/env bash
# C2 SCREEN: legal depth-authority attenuation finetune (alpha~U(0.05,1), prob 0.5 clean),
# then full metric-table eval. Finetune from the trained S ckpt. USE_AMP=0 (fp32) hard.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"
PROJ=${UGRA_ROOT}
cd "$PROJ"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SAVE="$PROJ/results/score_push/runs/l2_nof/c2_screen"
mkdir -p "$SAVE"

echo "=== [C2] TRAIN attenuate screen (epochs=${EPOCHS:-40}) ==="
CKPT_INIT="checkpoints/trained/DFormerv2_Small_NYU.pth" SAVE_DIR="$SAVE" \
  P_CLEAN="${P_CLEAN:-0.5}" ALPHA_MIN="${ALPHA_MIN:-0.05}" EPOCHS="${EPOCHS:-40}" \
  LR="${LR:-6e-5}" TRAIN_BATCH="${TRAIN_BATCH:-16}" NUM_WORKERS=4 SEED=20260721 \
  python experiments/l2_nof/train_c2_attenuate.py

CKPT="$SAVE/ckpt_final_c2.pth"
echo "=== [C2] EVAL full metric table ==="
CKPT="$CKPT" EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing,holes,gaussian \
  OUT_JSON="$SAVE/c2_eval.json" MODEL_TAG=c2_screen \
  bash scripts/eval_robust_suite.sh
echo "C2_SCREEN_DONE"
