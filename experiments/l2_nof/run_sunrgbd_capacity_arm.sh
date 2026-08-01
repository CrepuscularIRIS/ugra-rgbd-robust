#!/usr/bin/env bash
# SECOND-DATASET generalization arm: does the capacity-dependence of the missing-depth
# gap replicate on SUN-RGBD?
#
# On NYUv2 the gap collapses from 6.61 (S, dim 256) to 0.11 (L, dim 448) -- a 60x
# reduction at 2x width, which is the strongest evidence against the information-loss
# account (depth carries the same information whichever model reads it). This runs the
# same 2x2 cell on a different dataset with locally available trained checkpoints.
#
# SUN-RGBD has 5050 test images vs NYU's 654, so each condition takes far longer;
# this is why it runs detached.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

ROOT=${UGRA_ROOT}
cd "$ROOT"
source scripts/gpu_guard.sh
OUT=results/score_push/runs/l2_nof

run_arm() {
  local tag="$1" cfg="$2" ck="$3"
  local G
  if ! G=$(pick_probe_gpu 16000); then
    echo "=== SUNRGBD $tag: guard DENIED (insufficient free GPU memory) - skipping ==="
    return 0
  fi
  echo "=== SUNRGBD $tag on GPU$G ==="
  CUDA_VISIBLE_DEVICES="$G" \
  CONFIG_MODULE="$cfg" \
  CKPT="$ck" \
  EVAL_MODES=single \
  EVAL_CONDITIONS=clean,entire_missing \
  SEED=20260721 \
  OUT_JSON="$ROOT/$OUT/SUN_gap_${tag}.json" \
  MODEL_TAG="DFormerv2_${tag}_SUNRGBD" \
  bash scripts/eval_robust_suite.sh 2>&1 | grep -E "\[cfg\]|miou=|summary|Traceback|Error" | tail -5
}

run_arm S local_configs.SUNRGBD.DFormerv2_S checkpoints/trained/SUNRGBD/DFormerv2_Small_SUNRGBD.pth
run_arm L local_configs.SUNRGBD.DFormerv2_L DFormerv2_Large_SUNRGBD.pth
echo "SUNRGBD_CAPACITY_ARM_DONE"
