#!/usr/bin/env bash
# DOSE-RESPONSE CONTROL for the transplant-leverage confound.
# Pre-registered in .grill/prediction-ledger.md (2026-07-25).
#
# lambda = eval-time DEPTH_LAMBDA = the operator-corruption dose.
#   lambda=0 -> operator fully corrupted (equivalent to entire_missing; T1-verified)
#   lambda=1 -> operator intact (ordinary clean)
# If the value-path advantage is TOLERANCE to a corrupted operator it must fade with dose;
# if it is generic transplant leverage it must persist at every dose.
#
# Eval-only. Each eval gated by the GPU admission guard so it cannot OOM the in-flight training.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

PROJECT=${UGRA_ROOT}
cd "$PROJECT"
source scripts/gpu_guard.sh

OUT="$PROJECT/results/score_push/runs/l2_nof/dose_response"
TP="$PROJECT/results/score_push/runs/l2_nof/transplant"
mkdir -p "$OUT"

# name:absolute_checkpoint
CKPTS=(
  "baseline:$PROJECT/repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth"
  "attn_qk:$TP/transplant_attn_qk.pth"
  "attn_vo:$TP/transplant_attn_vo.pth"
  "c2screen:$PROJECT/results/score_push/runs/l2_nof/c2_screen/ckpt_final_c2.pth"
)
LAMBDAS=(0 0.5 1.0)

for entry in "${CKPTS[@]}"; do
  name="${entry%%:*}"; ck="${entry#*:}"
  [ -f "$ck" ] || { echo "MISSING_CKPT $name $ck"; continue; }
  for L in "${LAMBDAS[@]}"; do
    tag="${name}_L${L}"
    GPU=""
    for attempt in $(seq 1 60); do
      GPU=$(pick_probe_gpu "$FOOTPRINT_EVAL") && break
      echo "[guard] no capacity for $tag (attempt $attempt); waiting 60s"
      sleep 60
    done
    [ -n "$GPU" ] || { echo "GUARD_DEFER $tag"; continue; }

    CUDA_VISIBLE_DEVICES=$GPU \
    CKPT="$ck" \
    DEPTH_LAMBDA="$L" \
    EVAL_MODES=single \
    EVAL_CONDITIONS=clean \
    EVAL_SPLIT=full \
    SEED=20260721 \
    OUT_JSON="$OUT/eval_${tag}.json" \
    MODEL_TAG="$tag" \
    bash scripts/eval_robust_suite.sh > "$OUT/eval_${tag}.log" 2>&1
    rc=$?

    # Never trust a completion marker without checking the log (standing campaign rule).
    if [ $rc -eq 0 ] && [ -f "$OUT/eval_${tag}.json" ] \
       && ! grep -qE "Traceback|out of memory|CUDA error" "$OUT/eval_${tag}.log"; then
      miou=$(grep -oP 'miou=\K[0-9.]+' "$OUT/eval_${tag}.log" | tail -1)
      echo "RESULT $name lambda=$L miou=$miou"
    else
      echo "EVAL_FAILED $tag rc=$rc — see $OUT/eval_${tag}.log"
    fi
  done
done
echo "DOSE_RESPONSE_COMPLETE"
