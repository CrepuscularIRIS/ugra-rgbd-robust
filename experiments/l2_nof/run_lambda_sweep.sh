#!/usr/bin/env bash
# L2-NOF T1+T2 diagnostic: equivalence audit + depth-authority lambda dose-response.
# EVAL-ONLY, frozen ckpt. Scales weight[1]*mask_d by lambda on CLEAN input.
#   T1 equivalence: clean@lambda=0 should match entire_missing baseline.
#   T2 dose-response: clean@lambda in {0,.25,.5,.75,1,1.25}; lambda=1 == clean baseline.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

OUTDIR="${UGRA_ROOT}/results/score_push/runs/l2_nof"
mkdir -p "$OUTDIR"
cd ${UGRA_ROOT}

# Reference baselines (no lambda): clean + entire_missing, single-scale full val.
CUDA_VISIBLE_DEVICES=1 EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing \
  OUT_JSON="$OUTDIR/ref_baselines.json" MODEL_TAG=l2_ref \
  bash scripts/eval_robust_suite.sh

# Lambda dose-response on CLEAN input.
for LAM in 0.0 0.25 0.5 0.75 1.0 1.25; do
  CUDA_VISIBLE_DEVICES=1 EVAL_MODES=single EVAL_CONDITIONS=clean DEPTH_LAMBDA="$LAM" \
    OUT_JSON="$OUTDIR/clean_lambda_${LAM}.json" MODEL_TAG="l2_clean_lam${LAM}" \
    bash scripts/eval_robust_suite.sh
done
echo "L2_SWEEP_DONE"
