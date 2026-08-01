#!/usr/bin/env bash
# L3-ID: ZERO-INFORMATION recalibration ladder -> a LOWER BOUND on excess brittleness.
#
# Identifiability problem: the observed gap (6.61 +/- 0.07) mixes two unknowns,
#   gap = (genuine depth information value) + (excess brittleness),
# and we have one equation. Any intervention that carries ZERO bits about the absent
# depth but still recovers mIoU proves brittleness by construction -- it cannot be
# recovering information it never received. So this sweep lower-bounds brittleness
# WITHOUT training and WITHOUT the oracle.
#
# Knob: POS_LAMBDA scales weight[0] (the POSITIONAL bias coefficient). A single scalar,
# no depth used, no per-instance information. Swept on the entire_missing condition.
#
# Also run on the C2 checkpoint: if C2 already absorbed the statistic-level
# miscalibration, it should show LESS headroom than baseline (mechanistic confirmation
# of what C2 did); equal headroom would mean the two effects are orthogonal/additive.
#
# HONEST PROTOCOL NOTE: a scalar chosen by looking at entire_missing on the val set is
# tuned on the test condition. The sweep below is therefore an UPPER bound on what the
# knob can do (an oracle-tuned knob). To claim it as a method, the scalar must be
# selected on train/held-out data. For the identifiability argument the oracle-tuned
# version is exactly what we want -- it bounds the family's ceiling -- but it MUST be
# labelled as oracle-tuned, never as a deployable result.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

ROOT=${UGRA_ROOT}
cd "$ROOT"
OUT=results/score_push/runs/l2_nof/zeroinfo_ladder
mkdir -p "$OUT"

CKPTS="${CKPTS:-baseline:checkpoints/trained/DFormerv2_Small_NYU.pth}"
LAMS="${LAMS:-0.6 0.8 1.0 1.25 1.6 2.0 2.5}"
CONDS="${CONDS:-entire_missing}"

# Admission guard: a full eval needs ~10-12 GB. Never let a probe OOM a live training
# run -- 19% slowdown is an acceptable cost, losing 4h of training is not.
source "$ROOT/scripts/gpu_guard.sh"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if PROBE_GPU=$(pick_probe_gpu 12000); then
    export CUDA_VISIBLE_DEVICES="$PROBE_GPU"
    echo "[guard] probe admitted to GPU${PROBE_GPU}; $(report_gpu_state)"
  else
    echo "[guard] DEFERRING: insufficient GPU memory for a probe without risking the training run."
    exit 0
  fi
fi

for entry in $CKPTS; do
  tag="${entry%%:*}"; ck="${entry#*:}"
  for pl in $LAMS; do
    echo "=== ${tag} POS_LAMBDA=${pl} ==="
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    CKPT="$ck" \
    EVAL_MODES=single \
    EVAL_CONDITIONS="$CONDS" \
    SEED=20260721 \
    POS_LAMBDA="$pl" \
    OUT_JSON="$ROOT/$OUT/${tag}_pos${pl}_$(echo "$CONDS" | tr ',' '-').json" \
    MODEL_TAG="${tag}_pos${pl}" \
    bash scripts/eval_robust_suite.sh 2>&1 | grep -E "\[l3id\]|miou=|summary" | tail -3
  done
done
echo "ZEROINFO_LADDER_DONE"
