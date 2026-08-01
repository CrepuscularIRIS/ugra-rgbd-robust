#!/usr/bin/env bash
# DAMAGE profile vs REPAIR profile, per encoder stage. Pre-registered .grill/prediction-ledger.md.
#
# DAMAGE D(s): on the trained BASELINE, condition=clean, restore the true depth bias at stage s
#   ONLY (DEPTH_LAMBDA_STAGES one-hot). Measures where operator corruption actually costs mIoU.
#   Same intervention TYPE at every location => no site-leverage confound.
# REPAIR R(s): transplant C2's parameters for stage s only. Already have stage2; add 0,1,3.
#
# CONTROLS whose failure invalidates the whole run (standing rule: no diagnostic without one):
#   all-off "0,0,0,0" must reproduce entire_missing ~49.07-49.12
#   all-on  "1,1,1,1" must reproduce clean          ~55.72-55.75
# Eval-only, gated so it cannot OOM the in-flight training.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

PROJECT=${UGRA_ROOT}
cd "$PROJECT"
source scripts/gpu_guard.sh

OUT="$PROJECT/results/score_push/runs/l2_nof/stage_profiles"
TP="$PROJECT/results/score_push/runs/l2_nof/transplant"
BASE="$PROJECT/repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth"
mkdir -p "$OUT"

run_eval () {  # tag ckpt extra_env_name extra_env_val condition
  local tag="$1" ck="$2" ev_n="$3" ev_v="$4" cond="$5"
  local GPU=""
  for attempt in $(seq 1 90); do
    GPU=$(pick_probe_gpu "$FOOTPRINT_EVAL") && break
    sleep 60
  done
  [ -n "$GPU" ] || { echo "GUARD_DEFER $tag"; return; }
  env CUDA_VISIBLE_DEVICES="$GPU" CKPT="$ck" "$ev_n"="$ev_v" \
    EVAL_MODES=single EVAL_CONDITIONS="$cond" EVAL_SPLIT=full SEED=20260721 \
    OUT_JSON="$OUT/eval_${tag}.json" MODEL_TAG="$tag" \
    bash scripts/eval_robust_suite.sh > "$OUT/eval_${tag}.log" 2>&1
  local rc=$?
  if [ $rc -eq 0 ] && ! grep -qE "Traceback|out of memory|CUDA error" "$OUT/eval_${tag}.log"; then
    echo "RESULT $tag $(grep -oP 'miou=\K[0-9.]+' "$OUT/eval_${tag}.log" | tail -1)"
  else
    echo "EVAL_FAILED $tag rc=$rc — see $OUT/eval_${tag}.log"
  fi
}

echo "--- CONTROLS (must bracket: all_off~49.1, all_on~55.7) ---"
run_eval "ctrl_all_off" "$BASE" DEPTH_LAMBDA_STAGES "0,0,0,0" clean
run_eval "ctrl_all_on"  "$BASE" DEPTH_LAMBDA_STAGES "1,1,1,1" clean

echo "--- DAMAGE profile: depth restored at ONE stage only ---"
run_eval "damage_s0" "$BASE" DEPTH_LAMBDA_STAGES "1,0,0,0" clean
run_eval "damage_s1" "$BASE" DEPTH_LAMBDA_STAGES "0,1,0,0" clean
run_eval "damage_s2" "$BASE" DEPTH_LAMBDA_STAGES "0,0,1,0" clean
run_eval "damage_s3" "$BASE" DEPTH_LAMBDA_STAGES "0,0,0,1" clean

echo "--- REPAIR profile: C2 params transplanted at ONE stage only ---"
for s in 0 1 3; do
  [ -f "$TP/transplant_stage${s}.pth" ] && run_eval "repair_s${s}" "$TP/transplant_stage${s}.pth" SEED 20260721 entire_missing
done
echo "STAGE_PROFILES_COMPLETE"
