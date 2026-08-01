#!/usr/bin/env bash
# Promotable evaluation of the S-SPATIAL-ORACLE (the same-capacity depth-free control).
#
# CRITICAL DETAIL. The oracle trained with DEPTH_LAMBDA_TRAIN=0, so its weight[1] received no
# gradient and never adapted. Evaluating it WITHOUT disabling the depth term would switch on
# a bias the model never trained with, and the number would be meaningless. We therefore pass
# the frozen harness its OWN eval-side DEPTH_LAMBDA=0 (which scales weight[1] in place).
#
# BUILT-IN SANITY CHECK: for this model, `clean` and `entire_missing` MUST agree, because depth
# is ignored either way. If they differ by more than the eval noise floor (~0.05), something is
# wrong and the number must not be used.
#
# Interpretation (pre-registered, .grill/prediction-ledger.md):
#   gap decomposition:  information value = clean_baseline(55.72) - A_oracle
#                       brittleness       = A_oracle - absent_baseline(49.11)
#   outcome cells: <=50.5 falsifies the strong thesis | ~51.5-52.0 limit agreement with C2 |
#                  52.0-54.5 thesis quantified, C2 bounded | >=55.4 thesis radicalized |
#                  < C2@absent(52.50) => C2 beats the endpoint-trained depth-free model
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

ROOT=${UGRA_ROOT}
cd "$ROOT"
source scripts/gpu_guard.sh

RUNDIR="${RUNDIR:?set RUNDIR to the oracle checkpoint directory}"
CK=$(ls "$RUNDIR"/epoch-*.pth 2>/dev/null | sed -E 's/.*epoch-([0-9]+)_.*/\1 &/' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "${CK:-}" ] || { echo "FATAL: no epoch checkpoint in $RUNDIR"; exit 1; }
echo "[oracle-eval] checkpoint: $CK"

G=$(pick_probe_gpu 11000) || { echo "[guard] no capacity; retry later."; exit 0; }

CUDA_VISIBLE_DEVICES="$G" \
CKPT="$ROOT/$CK" \
DEPTH_LAMBDA=0 \
EVAL_MODES=single \
EVAL_CONDITIONS=clean,entire_missing \
SEED=20260721 \
OUT_JSON="$ROOT/results/score_push/runs/l2_nof/oracle_final.json" \
MODEL_TAG=S_SPATIAL_ORACLE \
bash scripts/eval_robust_suite.sh 2>&1 | grep -E "\[l2nof\]|miou=|summary|Traceback|Error"

echo "ORACLE_EVAL_DONE"
