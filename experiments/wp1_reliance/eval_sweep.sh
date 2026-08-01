#!/usr/bin/env bash
# Eval a sweep of finetune checkpoints on a given split with the FROZEN harness.
# Usage: RUN_DIR=results/score_push/runs/rpd_drop_c1 SPLIT=dev MODES=single bash eval_sweep.sh
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
SPLIT="${SPLIT:-dev}"
MODES="${MODES:-single}"
CONDS="${CONDS:-clean,entire_missing}"
ABS_RUN="${PROJECT}/${RUN_DIR}"
[ -d "$ABS_RUN" ] || ABS_RUN="$RUN_DIR"

for ck in "$ABS_RUN"/ckpt_ep*.pth; do
  [ -e "$ck" ] || { echo "no checkpoints in $ABS_RUN"; exit 1; }
  # harness CKPT is relative to repos/DFormer; give it an absolute path instead
  tag="$(basename "$ck" .pth)"
  out="${ABS_RUN}/eval_${SPLIT}_${tag}.json"
  echo "=== eval $tag on split=$SPLIT modes=$MODES ==="
  CKPT="$ck" EVAL_MODES="$MODES" EVAL_CONDITIONS="$CONDS" EVAL_SPLIT="$SPLIT" \
  VAL_BATCH="${VAL_BATCH:-8}" SEED="${SEED:-20260721}" MODEL_TAG="rpd_${tag}" \
  OUT_JSON="$out" \
  bash "${PROJECT}/scripts/eval_robust_suite.sh"
done

echo && echo "=== SUMMARY (split=$SPLIT) ==="
python3 - "$ABS_RUN" "$SPLIT" "$MODES" <<'PY'
import glob, json, os, sys
run, split, modes = sys.argv[1], sys.argv[2], sys.argv[3].split(",")[0]
rows=[]
for f in sorted(glob.glob(os.path.join(run, f"eval_{split}_ckpt_ep*.json"))):
    d=json.load(open(f)); r=d["results"].get(modes,{})
    ep=int(os.path.basename(f).split("ep")[-1].split(".")[0])
    cl=r.get("clean",{}).get("miou"); mi=r.get("entire_missing",{}).get("miou")
    rows.append((ep,cl,mi))
rows.sort()
print(f"{'epoch':>6} {'clean':>7} {'missing':>8}")
for ep,cl,mi in rows:
    print(f"{ep:>6} {cl:>7.2f} {mi:>8.2f}")
PY
