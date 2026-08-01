#!/usr/bin/env bash
# GPU admission guard for eval-only probes.
#
# WHY. Measured this session: a concurrent eval costs the 2-GPU training run ~19%
# per-epoch wall time (37.8s -> 44-47s). That is an acceptable price. What is NOT
# acceptable is a probe OOM-ing a 5-hour training run at epoch 400 -- the asymmetry
# is ~12 minutes of slowdown versus ~4 hours of lost work. This guard makes the
# catastrophic case structurally impossible instead of merely unlikely.
#
# Usage (source it, then call):
#   source scripts/gpu_guard.sh
#   GPU=$(pick_probe_gpu 12000) || { echo "no capacity; deferring"; exit 0; }
#   CUDA_VISIBLE_DEVICES=$GPU ... run the probe ...
#
# MEASURED FOOTPRINTS — pass the RIGHT one. Guessing defeats the guard.
#   eval / frozen-harness pass ......... ~10-12 GB   ($FOOTPRINT_EVAL)
#   forward-only diagnostic probe ...... ~3-6 GB     ($FOOTPRINT_PROBE)
#   FINETUNE, batch 16, single GPU ..... ~28 GB      ($FOOTPRINT_FINETUNE)
#   full train, batch 16, 2 GPUs ....... ~19 GB/GPU  ($FOOTPRINT_TRAIN_PER_GPU)
# 2026-07-25 incident: a batch-16 finetune was admitted using the 12 GB EVAL figure while a
# 2-GPU training run held 19 GB/GPU. It OOM'd on launch. The training run survived only
# because PyTorch failed the NEW allocation rather than the established one — luck, not
# design. Use the constants below; do not pass an ad-hoc number.
FOOTPRINT_PROBE=6000
FOOTPRINT_EVAL=12000
FOOTPRINT_FINETUNE=28000
FOOTPRINT_TRAIN_PER_GPU=19000

# pick_probe_gpu <needed_MiB> [reserve_MiB]
#   Prints the index of the GPU with the most free memory, but ONLY if
#   free - reserve >= needed. Otherwise prints nothing and returns 1.
#   reserve defaults to 4096 MiB of headroom left untouched for the training job's
#   own transient allocations (activation spikes, eval-at-epoch-end, cudnn workspace).

pick_probe_gpu() {
  local needed="${1:?needed MiB required}"
  local reserve="${2:-4096}"
  local best_idx="" best_free=-1

  while IFS=, read -r idx total used; do
    idx=$(echo "$idx" | tr -d ' '); total=$(echo "$total" | tr -d ' MiB'); used=$(echo "$used" | tr -d ' MiB')
    local free=$(( total - used ))
    if [ "$free" -gt "$best_free" ]; then best_free=$free; best_idx=$idx; fi
  done < <(nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader)

  if [ $(( best_free - reserve )) -ge "$needed" ]; then
    echo "$best_idx"
    return 0
  fi
  echo "gpu_guard: DENIED - best free=${best_free}MiB on GPU${best_idx}, need ${needed}MiB + ${reserve}MiB reserve" >&2
  return 1
}

# report_gpu_state — one-line snapshot for logs.
report_gpu_state() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader \
    | awk -F',' '{printf "GPU%s: %s/%s util%s  ", $1, $2, $3, $4}'
  echo
}
