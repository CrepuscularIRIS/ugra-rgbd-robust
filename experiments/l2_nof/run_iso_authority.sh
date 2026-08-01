#!/usr/bin/env bash
# ISO-AUTHORITY test: trace stage 2's own authority->damage curve and ask whether stage 1's
# point lies on it. lambda=0.187 puts stage 2 at stage 1's exact total authority (0.42).
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
for L in "${@:-0.5 0.25 0.187}"; do
  tag="s2_lam${L}"
  GPU=""
  for a in $(seq 1 90); do GPU=$(pick_probe_gpu "$FOOTPRINT_EVAL") && break; sleep 60; done
  [ -n "$GPU" ] || { echo "GUARD_DEFER $tag"; continue; }
  env CUDA_VISIBLE_DEVICES="$GPU" CKPT="$BASE" DEPTH_LAMBDA_STAGES="0,0,${L},0" \
    EVAL_MODES=single EVAL_CONDITIONS=clean EVAL_SPLIT=full SEED=20260721 \
    OUT_JSON="$OUT/eval_${tag}.json" MODEL_TAG="$tag" \
    bash scripts/eval_robust_suite.sh > "$OUT/eval_${tag}.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ] && ! grep -qE "Traceback|out of memory|CUDA error" "$OUT/eval_${tag}.log"; then
    echo "RESULT $tag $(grep -oP 'miou=\K[0-9.]+' "$OUT/eval_${tag}.log" | tail -1)"
  else
    echo "EVAL_FAILED $tag rc=$rc"
  fi
done
echo "ISO_AUTHORITY_COMPLETE"
