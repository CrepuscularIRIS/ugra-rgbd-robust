#!/usr/bin/env bash
# Tier-1 screen block: 9 x 40-epoch screens, TWO LANES (one per GPU), each lane sequential.
# All share train_c2_attenuate.py so the harness is identical across arms.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

# The trainer needs the dformer env; without this the queue silently runs the SYSTEM python
# and every arm dies in ~1 s on a missing module.
source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"
PROJ=${UGRA_ROOT}
cd "$PROJ"
OUT=results/score_push/runs/l2_nof/screens
mkdir -p "$OUT"

run_one () {   # gpu tag env-assignments...
  local gpu="$1" tag="$2"; shift 2
  local dir="$OUT/$tag"; mkdir -p "$dir"
  echo "[$(date +%H:%M:%S)] START $tag on GPU$gpu ($*)" | tee -a "$OUT/queue.log"
  env CUDA_VISIBLE_DEVICES="$gpu" "$@" \
      CKPT_INIT="checkpoints/trained/DFormerv2_Small_NYU.pth" SAVE_DIR="$PROJ/$dir" \
      P_CLEAN=0.5 ALPHA_MIN=0.05 EPOCHS=40 LR=6e-5 TRAIN_BATCH=16 NUM_WORKERS=4 SEED=20260721 \
      python experiments/l2_nof/train_c2_attenuate.py > "$dir/train.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ] || grep -qE "Traceback|out of memory|CUDA error" "$dir/train.log"; then
    echo "[$(date +%H:%M:%S)] TRAIN_FAILED $tag rc=$rc" | tee -a "$OUT/queue.log"; return
  fi
  env CUDA_VISIBLE_DEVICES="$gpu" CKPT="$PROJ/$dir/ckpt_final_c2.pth" EVAL_MODES=single \
      EVAL_CONDITIONS=clean,entire_missing EVAL_SPLIT=full SEED=20260721 \
      OUT_JSON="$PROJ/$dir/eval.json" MODEL_TAG="$tag" \
      bash scripts/eval_robust_suite.sh > "$dir/eval.log" 2>&1
  if grep -qE "Traceback|out of memory" "$dir/eval.log"; then
    echo "[$(date +%H:%M:%S)] EVAL_FAILED $tag" | tee -a "$OUT/queue.log"; return
  fi
  local c=$(grep -oP 'clean\s+miou=\K[0-9.]+' "$dir/eval.log" | tail -1)
  local e=$(grep -oP 'entire_missing\s+miou=\K[0-9.]+' "$dir/eval.log" | tail -1)
  echo "[$(date +%H:%M:%S)] RESULT $tag clean=$c entire_missing=$e" | tee -a "$OUT/queue.log"
}

# ---- LANE 0 (GPU0): alternative baselines + corrected-optimizer arms ----
lane0 () {
  run_one 0 alt_bernoulli  ATTEN_DIST=bernoulli
  run_one 0 alt_uniform01  ATTEN_DIST=uniform01
  run_one 0 alt_constant   ATTEN_DIST=constant
  run_one 0 fix_depth      GROUP_WEIGHT_FIX=1
  run_one 0 fix_pos        GROUP_WEIGHT_FIX=1 ATTEN_TARGET=pos
}
# ---- LANE 1 (GPU1): constrained-update (score path vs representation path) ----
lane1 () {
  run_one 1 cu_depth_qk    FREEZE_MODE=qk_only
  run_one 1 cu_depth_vout  FREEZE_MODE=vout_only
  run_one 1 cu_pos_qk      ATTEN_TARGET=pos FREEZE_MODE=qk_only
  run_one 1 cu_pos_vout    ATTEN_TARGET=pos FREEZE_MODE=vout_only
}
lane0 & lane1 & wait
echo "SCREEN_QUEUE_COMPLETE" | tee -a "$OUT/queue.log"
