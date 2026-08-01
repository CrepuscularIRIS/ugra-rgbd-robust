#!/usr/bin/env bash
# S-SPATIAL-ORACLE: the missing A_R^oracle control (in-family, capacity-matched).
#
# WHY. Every campaign number so far compares depth-absent (49.07) against
# depth-present (55.75) and calls the 6.68 difference "the missing-depth gap".
# The field's own decomposition splits that into
#     observed drop = (genuine depth information value)  +  (excess brittleness)
#                     [A_full - A_oracle]                   [A_oracle - A_absent]
# where A_oracle is a model of the SAME architecture trained without depth.
# We have never measured A_oracle, so we cannot say WHICH component C2's +3.0
# recovered -- and that is precisely the thesis claim.
#
# HOW. DEPTH_LAMBDA_TRAIN=0 zeroes the weight[1]*mask_d term in the forward.
# Since depth enters ONLY through that term (proven bitwise by
# diagnostics/single_path_invariance.py), this trains a genuinely RGB-only model
# at IDENTICAL parameter count, FLOPs and capacity -- so the comparison isolates
# operator repair from any capacity/architecture difference.
#
# READING THE RESULT (both outcomes are publishable):
#   C2 entire_missing (~52.1) >= oracle  => the operator is essentially REPAIRED;
#       the residual gap to clean is genuine depth information. THESIS ESTABLISHED + BOUNDED.
#   oracle >> C2 entire_missing          => large excess brittleness REMAINS after C2;
#       attenuation only partially repairs the operator. THESIS BOUNDED/WEAKENED,
#       and a characterized negative for the "what doesn't work and why" section.
#
# NOTE ON EVAL. weight[1] gets no gradient here (its contribution is multiplied by
# 0), so it keeps its pretrained value while being irrelevant. The trained model
# must therefore ALSO be evaluated with the depth term off -- use the FROZEN
# harness with its own eval-side DEPTH_LAMBDA=0 (which scales weight[1] in place).
# Built-in sanity check: for this model, condition=clean and condition=entire_missing
# must yield the SAME mIoU, because depth is ignored either way.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

source ${CONDA_SH}; conda activate "$UGRA_CONDA_ENV"
cd ${UGRA_ROOT}/repos/DFormer

# Depth pathway OFF for training. C2 attenuation explicitly OFF (this is the plain
# RGB-only reference, not an attenuated model); with lambda=0 it would be a no-op anyway.
export DEPTH_LAMBDA_TRAIN=0
export C2_ATTEN=0
export EVAL_EVERY="${EVAL_EVERY:-25}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TORCHDYNAMO_VERBOSE=0
GPUS=2; PORT="${PORT:-29612}"
export PYTHONPATH="${UGRA_ROOT}/repos/DFormer:${UGRA_ROOT}/repos:${PYTHONPATH:-}"

echo "[oracle] DEPTH_LAMBDA_TRAIN=$DEPTH_LAMBDA_TRAIN C2_ATTEN=$C2_ATTEN gpus=$GPUS eval_every=$EVAL_EVERY"
echo "[oracle] 500ep from pretrained init, identical recipe to the C2 promotion arm"

torchrun --nnodes=1 --node_rank=0 --nproc_per_node=$GPUS --master_port="$PORT" \
  utils/train.py --config=local_configs.NYUDepthv2.DFormerv2_S --gpus=$GPUS \
  --no-sliding --no-compile --syncbn --mst --compile_mode="default" --no-amp --val_amp --pad_SUNRGBD --no-use_seed
echo "SPATIAL_ORACLE_DONE"
