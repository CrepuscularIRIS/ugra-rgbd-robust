#!/usr/bin/env bash
# FINALIZED MUST-COMPLETE CHAIN (V3 §8.1). Sequential: the promotable recipe is 2-GPU and
# splitting to 1 GPU/run would change batch/syncbn semantics and break comparability with
# seeds 1-2. E2 runs first: it is the only pre-registered falsification test and the cheapest.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}; conda activate "$UGRA_CONDA_ENV"
P=${UGRA_ROOT}; cd $P
L=$P/results/score_push/runs/l2_nof/must_chain.log
say () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L"; }

wait_gpus () { while pgrep -f "python.*utils/train\.py|train_c2_attenuate" >/dev/null 2>&1; do sleep 120; done; sleep 30; }

# ---------- M3 / E2: SUN-RGBD falsification screen (matched 40-ep protocol) ----------
say "M3/E2 START — SUN-RGBD alpha-attenuation screen (matched to the NYU screen that gave +2.87)"
D=$P/results/score_push/runs/l2_nof/e2_sun_screen; mkdir -p $D
CUDA_VISIBLE_DEVICES=1 CONFIG_MODULE=local_configs.SUNRGBD.DFormerv2_S \
  CKPT_INIT="$P/repos/DFormer/checkpoints/trained/SUNRGBD/DFormerv2_Small_SUNRGBD.pth" \
  SAVE_DIR="$D" P_CLEAN=0.5 ALPHA_MIN=0.05 EPOCHS=40 LR=6e-5 TRAIN_BATCH=16 \
  NUM_WORKERS=4 SEED=20260721 \
  python experiments/l2_nof/train_c2_attenuate.py > $D/train.log 2>&1
if [ -f "$D/ckpt_final_c2.pth" ]; then
  for tag in sun_baseline sun_method; do
    CK=$P/repos/DFormer/checkpoints/trained/SUNRGBD/DFormerv2_Small_SUNRGBD.pth
    [ "$tag" = "sun_method" ] && CK=$D/ckpt_final_c2.pth
    CUDA_VISIBLE_DEVICES=1 CONFIG_MODULE=local_configs.SUNRGBD.DFormerv2_S CKPT="$CK" \
      EVAL_MODES=single EVAL_CONDITIONS=clean,entire_missing EVAL_SPLIT=full SEED=20260721 \
      OUT_JSON="$D/${tag}.json" MODEL_TAG="$tag" \
      bash scripts/eval_robust_suite.sh > "$D/${tag}_eval.log" 2>&1
    say "E2 $tag clean=$(grep -oP 'clean\s+miou=\K[0-9.]+' $D/${tag}_eval.log|tail -1) EM=$(grep -oP 'entire_missing\s+miou=\K[0-9.]+' $D/${tag}_eval.log|tail -1)"
  done
  say "M3/E2 DONE"
else
  say "M3/E2 FAILED — see $D/train.log"
fi

# ---------- M1: baseline 2nd full-training seed (standard recipe, C2 OFF) ----------
wait_gpus
say "M1 START — baseline replication, 500 ep, C2_ATTEN=0, 2 GPUs"
C2_ATTEN=0 CUDA_VISIBLE_DEVICES=0,1 PORT=29620 \
  bash experiments/l2_nof/run_c2_fulltrain.sh > $P/results/score_push/runs/l2_nof/baseline_seed2.log 2>&1
say "M1 DONE ($(tail -1 $P/results/score_push/runs/l2_nof/baseline_seed2.log))"

# ---------- M2: method 3rd full-training seed ----------
wait_gpus
say "M2 START — method seed 3, 500 ep, C2_ATTEN=1, 2 GPUs"
C2_ATTEN=1 CUDA_VISIBLE_DEVICES=0,1 PORT=29621 \
  bash experiments/l2_nof/run_c2_fulltrain.sh > $P/results/score_push/runs/l2_nof/method_seed3.log 2>&1
say "M2 DONE ($(tail -1 $P/results/score_push/runs/l2_nof/method_seed3.log))"
say "MUST_CHAIN_COMPLETE"
