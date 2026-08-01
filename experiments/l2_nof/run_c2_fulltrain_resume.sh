#!/usr/bin/env bash
# C2 PROMOTABLE full-train — RESUME arm.
# The seed-1 ep500 gate was SIGTERM'd at epoch 135/500 (session teardown, not a crash).
# This resumes from the last saved checkpoint. LR is get_lr((epoch-1)*niters+idx), a pure
# function of epoch/step, so the resumed schedule is identical to an uninterrupted run.
# Launch with setsid so a session exit cannot signal the process group again.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}; conda activate "$UGRA_CONDA_ENV"
cd ${UGRA_ROOT}/repos/DFormer

# C2 aug config — must match the interrupted run exactly
export C2_ATTEN=1 C2_PCLEAN="${C2_PCLEAN:-0.5}" C2_ALPHAMIN="${C2_ALPHAMIN:-0.05}"
export EVAL_EVERY="${EVAL_EVERY:-25}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCHDYNAMO_VERBOSE=0
GPUS=2; PORT="${PORT:-29611}"
export PYTHONPATH="${UGRA_ROOT}/repos/DFormer:${UGRA_ROOT}/repos:${PYTHONPATH:-}"

RESUME_CKPT="${RESUME_CKPT:?set RESUME_CKPT to the .pth to resume from}"
[ -f "$RESUME_CKPT" ] || { echo "FATAL: resume ckpt not found: $RESUME_CKPT"; exit 1; }

echo "[C2-full-resume] C2_ATTEN=$C2_ATTEN pclean=$C2_PCLEAN alphamin=$C2_ALPHAMIN gpus=$GPUS eval_every=$EVAL_EVERY"
echo "[C2-full-resume] resuming from: $RESUME_CKPT"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$GPUS --master_port="$PORT" \
  utils/train.py --config=local_configs.NYUDepthv2.DFormerv2_S --gpus=$GPUS \
  --continue_fpath="$RESUME_CKPT" \
  --no-sliding --no-compile --syncbn --mst --compile_mode="default" --no-amp --val_amp --pad_SUNRGBD --no-use_seed
echo "C2_FULLTRAIN_RESUME_DONE"
