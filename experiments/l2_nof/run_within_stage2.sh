#!/usr/bin/env bash
# (a) ADDITIVITY CHECK: stages 0+2 together should ~ reproduce all-on (stages 1,3 measured dead).
# (b) WITHIN-STAGE-2 profile: stage 2 carries 86.9% of the gap across 18 blocks — is that a few
#     blocks or all of them? Thirds first; refine only if a third dominates.
# Eval-only, guard-gated against the in-flight training.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

PROJECT=${UGRA_ROOT}
cd "$PROJECT"
source scripts/gpu_guard.sh
OUT="$PROJECT/results/score_push/runs/l2_nof/stage_profiles"
BASE="$PROJECT/repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth"
mkdir -p "$OUT"

run_eval () {  # tag env_name env_val
  local tag="$1" n="$2" v="$3" GPU=""
  for attempt in $(seq 1 90); do GPU=$(pick_probe_gpu "$FOOTPRINT_EVAL") && break; sleep 60; done
  [ -n "$GPU" ] || { echo "GUARD_DEFER $tag"; return; }
  env CUDA_VISIBLE_DEVICES="$GPU" CKPT="$BASE" "$n"="$v" \
    EVAL_MODES=single EVAL_CONDITIONS=clean EVAL_SPLIT=full SEED=20260721 \
    OUT_JSON="$OUT/eval_${tag}.json" MODEL_TAG="$tag" \
    bash scripts/eval_robust_suite.sh > "$OUT/eval_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && ! grep -qE "Traceback|out of memory|CUDA error" "$OUT/eval_${tag}.log"; then
    echo "RESULT $tag $(grep -oP 'miou=\K[0-9.]+' "$OUT/eval_${tag}.log" | tail -1)"
  else
    echo "EVAL_FAILED $tag rc=$rc"
  fi
}

echo "--- additivity: stages 0+2 vs all-on 55.75 ---"
run_eval "damage_s0s2" DEPTH_LAMBDA_STAGES "1,0,1,0"
echo "--- within stage 2, thirds (18 blocks) ---"
run_eval "s2_blk00_05" DEPTH_LAMBDA_BLOCKS "2:0-5"
run_eval "s2_blk06_11" DEPTH_LAMBDA_BLOCKS "2:6-11"
run_eval "s2_blk12_17" DEPTH_LAMBDA_BLOCKS "2:12-17"
echo "WITHIN_STAGE2_COMPLETE"
