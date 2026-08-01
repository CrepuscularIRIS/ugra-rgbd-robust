#!/usr/bin/env bash
# MP-B removability SCREEN driver (DIAGNOSTIC ONLY; NOT promotable).
# For each p in {0.25, 0.5}: depth-dropout finetune (USE_AMP=0) -> eval clean+entire_missing
# (+holes+gaussian) on the FROZEN harness -> stage-2 drift probe on the finetuned ckpt.
# All numbers from real evals; no fabrication. Single GPU (GPU1), deterministic-ish.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"


source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"

PROJ=${UGRA_ROOT}
SCR=$PROJ/experiments/n7_cycle5_mpb_screen
RES=$SCR/results
CKPT_S=$PROJ/repos/DFormer/checkpoints/trained/NYUDepthv2/DFormerv2_Small_NYU.pth
SEED=20260721
export CUDA_VISIBLE_DEVICES=1
export USE_AMP=0

echo "=== MP-B REMOVABILITY SCREEN START $(date) ==="

for P in 0.25 0.5; do
  TAG="p${P}"
  SAVE=$RES/ft_${TAG}
  CKPT_FT=$SAVE/ckpt_final_p${P}.pth
  echo ">>> [${TAG}] FINETUNE (depth-dropout p=${P}, 15 ep, fp32)"
  CKPT_INIT=$CKPT_S SAVE_DIR=$SAVE P_DROP=$P EPOCHS=15 TRAIN_BATCH=16 LR=6e-5 \
    NUM_WORKERS=4 SEED=$SEED \
    python "$SCR/train_depthdropout_screen.py" 2>&1 | grep -E '\[ep|\[cfg|\[load|\[save|\[done' || true

  echo ">>> [${TAG}] EVAL on frozen harness (clean, entire_missing, holes, gaussian)"
  CKPT="$CKPT_FT" \
    EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing,holes,gaussian \
    SEED=$SEED SLIDING=1 EVAL_SPLIT=full \
    MODEL_TAG="mpb_screen_${TAG}" \
    OUT_JSON="$RES/eval_${TAG}.json" \
    python "$PROJ/scripts/eval_robust_suite.py" 2>&1 | grep -E '\[single|\[summary|\[done|\[load' || true

  echo ">>> [${TAG}] STAGE-2 DRIFT PROBE on finetuned ckpt"
  MODEL=S CKPT="$CKPT_FT" N_IMAGES=200 SEED=$SEED \
    OUT_JSON="$RES/drift_${TAG}.json" \
    python "$PROJ/experiments/n7_cycle5_mpb/feature_drift_probe.py" 2>&1 | grep -E '\[done|\[load' || true
done

echo "=== MP-B REMOVABILITY SCREEN DONE $(date) ==="
