#!/usr/bin/env bash
# P2-CTRL: POSITIONAL-alpha screen — the MATCHED NEGATIVE CONTROL for C2.
#
# Identical to run_c2_screen.sh in every respect (same trainer, same init ckpt, same epochs, LR,
# batch, seed, p_clean, alpha_min, USE_AMP=0) EXCEPT ATTEN_TARGET=pos, which attenuates the
# POSITIONAL bias term instead of the DEPTH bias term.
#
# WHAT IT DECIDES. Two live accounts of C2's transplant split (score q/k 21.6% vs value v/out
# 38.0% vs lepe 14.6%):
#   (i)  MECHANISM — the split reflects where depth-specific repair lives.
#   (ii) SITE ARTIFACT — v/out is simply where ~half of ANY finetune delta in this architecture
#        lands (supported by the dose-response, where the value path carried the same ~48% share
#        of the depth-IRRELEVANT effect as of the depth-specific one).
# This arm produces a genuinely helpful delta in the SAME loss basin with NO depth involvement.
# Transplant-decomposing it separates (i) from (ii).
#
# Requires: GPU free (28 GB finetune footprint — do NOT run concurrently with a training job).
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"
PROJ=${UGRA_ROOT}
cd "$PROJ"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
SAVE="$PROJ/results/score_push/runs/l2_nof/pos_control_screen"
mkdir -p "$SAVE"

echo "=== [P2-CTRL] positional-alpha screen (epochs=${EPOCHS:-40}) ==="
ATTEN_TARGET=pos \
CKPT_INIT="checkpoints/trained/DFormerv2_Small_NYU.pth" SAVE_DIR="$SAVE" \
  P_CLEAN="${P_CLEAN:-0.5}" ALPHA_MIN="${ALPHA_MIN:-0.05}" EPOCHS="${EPOCHS:-40}" \
  LR="${LR:-6e-5}" TRAIN_BATCH="${TRAIN_BATCH:-16}" NUM_WORKERS=4 SEED=20260721 \
  python experiments/l2_nof/train_c2_attenuate.py

CKPT="$SAVE/ckpt_final_c2.pth"
[ -f "$CKPT" ] || { echo "P2CTRL_FAILED: no checkpoint at $CKPT"; exit 1; }

echo "=== [P2-CTRL] full metric table ==="
CKPT="$CKPT" EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing,holes,gaussian \
  OUT_JSON="$SAVE/pos_control_eval.json" MODEL_TAG=pos_control_screen \
  bash scripts/eval_robust_suite.sh

echo "=== [P2-CTRL] transplant-decompose THIS delta (the actual point of the arm) ==="
POS_CKPT="$CKPT" python - <<'PY'
import os, subprocess, sys
# Re-point param_transplant at the positional-control checkpoint, then build the same groups
# used for C2 so the two decompositions are directly comparable.
os.environ["TRANSPLANT_C2_OVERRIDE"] = os.environ["POS_CKPT"]
os.environ["TRANSPLANT_OUTDIR"] = os.path.join(
    os.path.dirname(os.environ["POS_CKPT"]), "transplant"
)
for g in ("all", "attn_qk", "attn_vout", "attn_lepe", "attn_vo"):
    r = subprocess.run([sys.executable, "diagnostics/param_transplant.py", "--group", g],
                       capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip())
PY
echo "P2CTRL_DONE"
