#!/usr/bin/env bash
# SCORE-path vs VALUE-path transplant evals (pre-registered .grill/prediction-ledger.md 2026-07-25).
# Eval-only. Runs sequentially, each gated by the GPU admission guard so a probe cannot OOM
# the in-flight seed-2 training run. Protocol identical to the sweep it extends:
# single-scale, full split, seed 20260721, metric entire_missing.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

PROJECT=${UGRA_ROOT}
cd "$PROJECT"
# Capture argv BEFORE sourcing. NOTE: the array is TGROUPS, never GROUPS — GROUPS is a bash
# special builtin array (the user's GIDs) and assigning to it is SILENTLY IGNORED, which
# yields a loop over group IDs instead of over the requested transplant groups.
TGROUPS=("$@")
source scripts/gpu_guard.sh

# ABSOLUTE — eval_robust_suite.py resolves CKPT from its own working directory, not this one.
OUT="$PROJECT/results/score_push/runs/l2_nof/transplant"
mkdir -p "$OUT"

[ ${#TGROUPS[@]} -gt 0 ] || TGROUPS=(attn_qk attn_vo stage2_attn_qk stage2_attn_vo)

for g in "${TGROUPS[@]}"; do
  CK="$OUT/transplant_${g}.pth"
  [ -f "$CK" ] || { echo "MISSING_CKPT $g"; continue; }

  # Re-check admission before EVERY eval: the training job's footprint changes across
  # epochs (it evaluates at epoch ends), so one early check would not stay valid.
  GPU=""
  for attempt in $(seq 1 60); do
    GPU=$(pick_probe_gpu "$FOOTPRINT_EVAL") && break
    echo "[guard] no capacity for $g (attempt $attempt); waiting 60s"
    sleep 60
  done
  [ -n "$GPU" ] || { echo "GUARD_DEFER $g — never admitted"; continue; }

  echo "[eval] group=$g on GPU$GPU"
  CUDA_VISIBLE_DEVICES=$GPU \
  CKPT="$CK" \
  EVAL_MODES=single \
  EVAL_CONDITIONS=entire_missing \
  EVAL_SPLIT=full \
  SEED=20260721 \
  OUT_JSON="$OUT/eval_${g}.json" \
  MODEL_TAG="transplant_${g}" \
  bash scripts/eval_robust_suite.sh > "$OUT/eval_${g}.log" 2>&1

  if [ $? -eq 0 ] && [ -f "$OUT/eval_${g}.json" ]; then
    echo "RESULT $g $(python -c "
import json;d=json.load(open('$OUT/eval_${g}.json'))
def f(o):
    if isinstance(o,dict):
        if 'miou' in o and not isinstance(o['miou'],(dict,list)): return o['miou']
        for v in o.values():
            r=f(v)
            if r is not None: return r
    if isinstance(o,list):
        for v in o:
            r=f(v)
            if r is not None: return r
print(f(d))
" 2>/dev/null)"
  else
    echo "EVAL_FAILED $g — see $OUT/eval_${g}.log"
  fi
done
echo "QK_VO_TRANSPLANT_SWEEP_COMPLETE"
