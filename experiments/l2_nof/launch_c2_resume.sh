#!/usr/bin/env bash
# Detached launcher for the C2 seed-1 ep500 resume.
# setsid puts the job in its own session+process group so a Claude Code session exit
# cannot SIGTERM it (that is what killed the first attempt at epoch 135).
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

ROOT=${UGRA_ROOT}
cd "$ROOT"

RUN=results/score_push/runs/l2_nof
LOG="$RUN/c2_full_seed1_ep500_resume.log"
PIDF="$RUN/c2_full_seed1_resume.pid"
CKPT="$ROOT/repos/DFormer/checkpoints/NYUDepthv2_DFormerv2_S_20260725-000211/epoch-125_miou_54.63.pth"

setsid nohup env RESUME_CKPT="$CKPT" EVAL_EVERY=25 C2_ATTEN=1 C2_PCLEAN=0.5 C2_ALPHAMIN=0.05 \
  CUDA_VISIBLE_DEVICES=0,1 PORT=29611 \
  bash "$ROOT/experiments/l2_nof/run_c2_fulltrain_resume.sh" \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "$PIDF"
echo "launched pid=$(cat "$PIDF") log=$LOG"
