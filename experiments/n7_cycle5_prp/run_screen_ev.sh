#!/usr/bin/env bash
# PRP ESCAPE-VELOCITY re-screen driver (DIAGNOSTIC ONLY; NOT promotable).
# Gives w_p escape velocity (aux-warmup + gate ramp + warm-init + no-WD LRx8
# group) so mask_p ACTIVELY engages, then a HARD kill rule that can refute the
# mechanism itself.
#   1) EV finetune 100ep (PRP_ON=1, USE_AMP=0) from converged S ckpt
#   2) eval clean + entire_missing on the FROZEN harness (PRP_ON=1 pinned)
#   3) post-train stage-2 clean-vs-zeros cosine probe (PRP_ON=1) on 50 val imgs
# Single GPU (GPU1). Reservation rsv_5b0aa768.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"


source ${CONDA_SH}
conda activate "$UGRA_CONDA_ENV"

PROJ=${UGRA_ROOT}
SCR=$PROJ/experiments/n7_cycle5_prp
RES=$SCR/results_ev
CKPT_S=$PROJ/repos/DFormer/checkpoints/trained/NYUDepthv2/DFormerv2_Small_NYU.pth
SEED=20260721
LAMBDA_AUX=${LAMBDA_AUX:-0.1}
EPOCHS=${EPOCHS:-100}
WARMUP_EP=${WARMUP_EP:-10}
RAMP_EP=${RAMP_EP:-20}
WP_WARM=${WP_WARM:-0.2}
LR_MULT=${LR_MULT:-8.0}
export CUDA_VISIBLE_DEVICES=1
export USE_AMP=0

SAVE=$RES/ft_prp_ev
CKPT_FT=$SAVE/ckpt_final_prp_ev.pth
STATE=$SCR/state_ev.json
mkdir -p "$RES" "$SAVE"

ts() { date "+%Y-%m-%d %H:%M:%S"; }
write_state() {
  cat > "$STATE" <<JSON
{"node":"n7","method":"PRP-EV","phase":"$1","recipe":"finetune-proxy-rescreen","promotable":false,
 "reservation":"rsv_5b0aa768","gpu":"1","epochs":${EPOCHS},"warmup_ep":${WARMUP_EP},
 "ramp_ep":${RAMP_EP},"wp_warm":${WP_WARM},"lr_mult":${LR_MULT},"lambda_aux":${LAMBDA_AUX},
 "seed":${SEED},"updated":"$(ts)","note":"$2"}
JSON
}

echo "=== PRP-EV RE-SCREEN START $(ts) ==="
write_state "train" "EV finetune 100ep PRP_ON=1 USE_AMP=0 GPU1 warmup=${WARMUP_EP} ramp=${RAMP_EP} wp_warm=${WP_WARM} lrmult=${LR_MULT}"

# ---- 1) EV FINETUNE ----
CKPT_INIT=$CKPT_S SAVE_DIR=$SAVE EPOCHS=$EPOCHS WARMUP_EP=$WARMUP_EP RAMP_EP=$RAMP_EP \
  WP_WARM=$WP_WARM LR_MULT=$LR_MULT TRAIN_BATCH=16 LR=6e-5 \
  LAMBDA_AUX=$LAMBDA_AUX NUM_WORKERS=4 SEED=$SEED \
  python "$SCR/train_prp_screen_ev.py" 2>&1 | tee "$RES/train_prp_ev.log" | \
  grep -E '\[ep|\[cfg|\[load|\[prp|\[ev|\[save|\[done|FAILED|Error|Traceback' || true

if [ ! -f "$CKPT_FT" ]; then
  echo "!!! FINETUNE FAILED: no ckpt at $CKPT_FT"; write_state "error" "finetune produced no ckpt"; exit 2
fi

# ---- 2) EVAL on frozen harness (PRP_ON=1 pinned; clean + entire_missing) ----
write_state "eval" "frozen harness clean+entire_missing PRP_ON=1"
PRP_ON=1 CKPT="$CKPT_FT" \
  EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing \
  SEED=$SEED SLIDING=1 EVAL_SPLIT=full \
  MODEL_TAG="prp_ev_screen" \
  OUT_JSON="$RES/eval_prp_ev_screen.json" \
  python "$PROJ/scripts/eval_robust_suite.py" 2>&1 | tee "$RES/eval_prp_ev.log" | \
  grep -E '\[single|\[summary|\[done|\[load' || true

# ---- 3) POST-TRAIN stage-2 cosine probe (PRP_ON=1) on 50 val imgs ----
write_state "probe" "stage-2 clean-vs-zeros cosine post-train PRP_ON=1"
PRP_ON=1 MODEL=S CKPT="$CKPT_FT" N_IMAGES=50 SEED=$SEED \
  OUT_JSON="$RES/cosine_post_prp_ev.json" \
  python "$PROJ/experiments/n7_cycle5_mpb/feature_drift_probe.py" 2>&1 | tee "$RES/cosine_post_ev.log" | \
  grep -E '\[done|\[load' || true

write_state "done" "EV re-screen complete"
echo "=== PRP-EV RE-SCREEN DONE $(ts) ==="
