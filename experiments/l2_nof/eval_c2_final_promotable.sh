#!/usr/bin/env bash
# PROMOTABLE evaluation of the C2 ep500 seed-1 arm across the full corruption suite.
#
# The in-training val numbers are NOT the promotable figures — those must come from the
# frozen harness. This scores the final checkpoint on the full CAS-Zone-2 condition set:
# clean / entire_missing / holes / gaussian, plus the OOD pair half_missing / salt_pepper.
#
# PRE-REGISTERED GATE (from .grill/prediction-ledger.md, C2 PROMOTION row):
#   PASS  = clean >= 55.4 AND entire_missing >= 51.5   -> launch seeds 2 and 3
#   clean < 55.0                                       -> kills "clean-preserving at full-train";
#                                                          sweep p_clean before spending more seeds
#   entire_missing < 51.0                              -> kills "the screen gain transfers"
# Baselines to compare against (4 eval seeds): clean 55.72 +/- 0.045, entire_missing 49.11 +/- 0.031.
#
# Picks the highest-epoch checkpoint in the resumed run's directory. Gated through the GPU
# guard so it cannot OOM whatever else is running.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

ROOT=${UGRA_ROOT}
cd "$ROOT"
source scripts/gpu_guard.sh

RUNDIR="${RUNDIR:-repos/DFormer/checkpoints/NYUDepthv2_DFormerv2_S_20260725-030349}"
# highest epoch number present
CK=$(ls "$RUNDIR"/epoch-*.pth 2>/dev/null | sed -E 's/.*epoch-([0-9]+)_.*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "${CK:-}" ] || { echo "FATAL: no epoch checkpoint found in $RUNDIR"; exit 1; }
echo "[eval] final checkpoint: $CK"

G=$(pick_probe_gpu 11000) || { echo "[guard] no capacity right now; retry when a GPU frees."; exit 0; }
echo "[guard] admitted GPU$G; $(report_gpu_state)"

CUDA_VISIBLE_DEVICES="$G" \
CKPT="$ROOT/$CK" \
EVAL_MODES=single \
EVAL_CONDITIONS="${CONDS:-clean,entire_missing,holes,gaussian}" \
SEED=20260721 \
OUT_JSON="$ROOT/results/score_push/runs/l2_nof/c2_ep500_seed1_final.json" \
MODEL_TAG=C2_ep500_seed1 \
bash scripts/eval_robust_suite.sh 2>&1 | grep -E "\[cfg\]|miou=|summary|Traceback|Error"

echo "C2_FINAL_EVAL_DONE"
