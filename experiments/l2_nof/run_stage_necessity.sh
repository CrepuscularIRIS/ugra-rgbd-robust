#!/usr/bin/env bash
# Stage NECESSITY + REDUNDANCY-UNMASKING. Pre-registered .grill/prediction-ledger.md.
# Completes the causal battery: sufficiency (done) + necessity (leave-one-out) + unmasking.
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
for spec in "s0s1:1,1,0,0" "s2s3:0,0,1,1" "loo_s1:1,0,1,1" "loo_s3:1,1,1,0"; do
  tag="${spec%%:*}"; vec="${spec#*:}"
  GPU=""
  for a in $(seq 1 90); do GPU=$(pick_probe_gpu "$FOOTPRINT_EVAL") && break; sleep 60; done
  [ -n "$GPU" ] || { echo "GUARD_DEFER $tag"; continue; }
  env CUDA_VISIBLE_DEVICES="$GPU" CKPT="$BASE" DEPTH_LAMBDA_STAGES="$vec" \
    EVAL_MODES=single EVAL_CONDITIONS=clean EVAL_SPLIT=full SEED=20260721 \
    OUT_JSON="$OUT/eval_nec_${tag}.json" MODEL_TAG="nec_${tag}" \
    bash scripts/eval_robust_suite.sh > "$OUT/eval_nec_${tag}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && ! grep -qE "Traceback|out of memory|CUDA error" "$OUT/eval_nec_${tag}.log"; then
    echo "RESULT $tag ($vec) $(grep -oP 'miou=\K[0-9.]+' "$OUT/eval_nec_${tag}.log" | tail -1)"
  else
    echo "EVAL_FAILED $tag rc=$rc"
  fi
done
echo "STAGE_NECESSITY_COMPLETE"
