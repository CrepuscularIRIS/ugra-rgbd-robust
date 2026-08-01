#!/usr/bin/env bash
# PRP 100-ep finetune-proxy SCREEN driver (DIAGNOSTIC ONLY; NOT promotable).
# Pipeline: PRP finetune (100 ep, PRP_ON=1, USE_AMP=0) from the converged S ckpt
#   -> eval clean + entire_missing on the FROZEN harness (PRP_ON=1 pinned)
#   -> post-train stage-2 clean-vs-zeros cosine probe (PRP_ON=1) on ~50 val imgs.
# All numbers from real evals; no fabrication. Single GPU (GPU1) per the mpb
# screen precedent (~0.76s/step @ b16 -> ~1h for 100 ep). Writes state.json for
# the heartbeat/monitor. Reservation rsv_5b0aa768.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"


source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"

PROJ=${UGRA_ROOT}
SCR=$PROJ/experiments/n7_cycle5_prp
RES=$SCR/results
CKPT_S=$PROJ/repos/DFormer/checkpoints/trained/NYUDepthv2/DFormerv2_Small_NYU.pth
SEED=20260721
LAMBDA_AUX=${LAMBDA_AUX:-0.1}
EPOCHS=${EPOCHS:-100}
export CUDA_VISIBLE_DEVICES=1
export USE_AMP=0

SAVE=$RES/ft_prp
CKPT_FT=$SAVE/ckpt_final_prp.pth
STATE=$SCR/state.json
mkdir -p "$RES" "$SAVE"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
write_state() {
  cat > "$STATE" <<JSON
{"node":"n7","method":"PRP","phase":"$1","recipe":"finetune-proxy-screen","promotable":false,
 "reservation":"rsv_5b0aa768","gpu":"1","epochs":${EPOCHS},"lambda_aux":${LAMBDA_AUX},
 "seed":${SEED},"updated":"$(ts)","note":"$2"}
JSON
}

echo "=== PRP SCREEN START $(ts) ==="
write_state "train" "finetune 100ep PRP_ON=1 USE_AMP=0 GPU1"

# ---- 1) FINETUNE (PRP_ON=1 set inside the trainer) ----
CKPT_INIT=$CKPT_S SAVE_DIR=$SAVE EPOCHS=$EPOCHS TRAIN_BATCH=16 LR=6e-5 \
  LAMBDA_AUX=$LAMBDA_AUX NUM_WORKERS=4 SEED=$SEED \
  python "$SCR/train_prp_screen.py" 2>&1 | tee "$RES/train_prp.log" | \
  grep -E '\[ep|\[cfg|\[load init|\[prp|\[save|\[done|FAILED' || true

if [ ! -f "$CKPT_FT" ]; then
  echo "!!! FINETUNE FAILED: no ckpt at $CKPT_FT"; write_state "error" "finetune produced no ckpt"; exit 2
fi

# ---- 2) EVAL on frozen harness (PRP_ON=1 pinned; clean + entire_missing) ----
write_state "eval" "frozen harness clean+entire_missing PRP_ON=1"
PRP_ON=1 CKPT="$CKPT_FT" \
  EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing \
  SEED=$SEED SLIDING=1 EVAL_SPLIT=full \
  MODEL_TAG="prp_screen" \
  OUT_JSON="$RES/eval_prp_screen.json" \
  python "$PROJ/scripts/eval_robust_suite.py" 2>&1 | tee "$RES/eval_prp.log" | \
  grep -E '\[single|\[summary|\[done|\[load' || true

# ---- 3) POST-TRAIN stage-2 cosine probe (PRP_ON=1) on ~50 val imgs ----
write_state "probe" "stage-2 clean-vs-zeros cosine post-train PRP_ON=1"
PRP_ON=1 MODEL=S CKPT="$CKPT_FT" N_IMAGES=50 SEED=$SEED \
  OUT_JSON="$RES/cosine_post_prp.json" \
  python "$PROJ/experiments/n7_cycle5_mpb/feature_drift_probe.py" 2>&1 | tee "$RES/cosine_post.log" | \
  grep -E '\[done|\[load' || true

write_state "done" "screen complete"
echo "=== PRP SCREEN DONE $(ts) ==="
