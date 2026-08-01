#!/usr/bin/env bash
# C2 PROMOTABLE full-train: 500ep from pretrained, standard recipe + legal alpha-attenuation aug.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}; conda activate "$UGRA_CONDA_ENV"
cd ${UGRA_ROOT}/repos/DFormer
# C2_ATTEN was hard-coded to 1 here, which SILENTLY OVERRODE a caller passing C2_ATTEN=0 and
# turned a baseline-replication run into a third method run (21 GPU-h lost, 2026-07-27).
# C2_DIST was never exported either, so it leaked in from the caller's environment and turned a
# method run into a dropout run. Both are now explicit with defaults, and both are echoed below.
export C2_ATTEN="${C2_ATTEN:-1}" C2_PCLEAN="${C2_PCLEAN:-0.5}" C2_ALPHAMIN="${C2_ALPHAMIN:-0.05}"
export C2_DIST="${C2_DIST:-u005_1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCHDYNAMO_VERBOSE=0
GPUS=2; PORT="${PORT:-29610}"
export PYTHONPATH="${UGRA_ROOT}/repos/DFormer:${UGRA_ROOT}/repos:${PYTHONPATH:-}"
echo "[C2-full] C2_ATTEN=$C2_ATTEN C2_DIST=$C2_DIST pclean=$C2_PCLEAN alphamin=$C2_ALPHAMIN gpus=$GPUS"
torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$GPUS --master_port="$PORT" \
  utils/train.py --config=local_configs.NYUDepthv2.DFormerv2_S --gpus=$GPUS \
  --no-sliding --no-compile --syncbn --mst --compile_mode="default" --no-amp --val_amp --pad_SUNRGBD --no-use_seed
echo "C2_FULLTRAIN_DONE"
