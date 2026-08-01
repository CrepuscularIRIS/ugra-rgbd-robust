#!/usr/bin/env bash
# EVAL NOISE FLOOR for the frozen harness.
#
# The Ham/NMF decoder re-draws random bases every forward (ham_head.py:80), so the
# harness is reproducible-given-SEED but not insensitive to it. Every campaign
# number therefore carries decoder noise of unknown magnitude. This measures it:
# identical checkpoint, identical condition, only SEED varies.
#
# The resulting spread is the error bar that any claimed delta must clear -- in
# particular the C2 clean gain (+0.23/+0.39/+0.46), whose smallest value would not
# be separable from noise if sigma_eval were ~0.2.
set -uo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

ROOT=${UGRA_ROOT}
cd "$ROOT"
OUT=results/score_push/runs/l2_nof/noise_floor
mkdir -p "$OUT"

CKPT_REL="${CKPT_REL:-checkpoints/trained/DFormerv2_Small_NYU.pth}"
CONDS="${CONDS:-clean,entire_missing}"
SEEDS="${SEEDS:-20260721 20260722 20260723 20260724}"

for s in $SEEDS; do
  echo "=== SEED=$s ==="
  CUDA_VISIBLE_DEVICES=1 \
  CKPT="$CKPT_REL" \
  EVAL_MODES=single \
  EVAL_CONDITIONS="$CONDS" \
  SEED="$s" \
  OUT_JSON="$ROOT/$OUT/seed_${s}.json" \
  MODEL_TAG="noisefloor_seed${s}" \
  bash scripts/eval_robust_suite.sh 2>&1 | grep -E "miou|gap|clean=|missing=" | tail -5
done

echo "=== SUMMARY ==="
python3 - <<'PY'
import json, glob, statistics, os
rows = {}
for f in sorted(glob.glob("results/score_push/runs/l2_nof/noise_floor/seed_*.json")):
    d = json.load(open(f))
    seed = os.path.basename(f).replace("seed_","").replace(".json","")
    res = d.get("single") or d.get("results") or d
    for cond in ("clean","entire_missing"):
        v = None
        # tolerate a couple of plausible layouts
        if isinstance(res, dict):
            if cond in res and isinstance(res[cond], dict):
                v = res[cond].get("miou")
            elif "single" in d and cond in d["single"]:
                v = d["single"][cond].get("miou")
        if v is not None:
            rows.setdefault(cond, []).append((seed, float(v)))
for cond, vals in rows.items():
    xs = [v for _, v in vals]
    print(f"\n{cond}: n={len(xs)}")
    for s, v in vals:
        print(f"   seed {s}: {v:.2f}")
    if len(xs) > 1:
        print(f"   mean={statistics.mean(xs):.3f} sd={statistics.stdev(xs):.3f} "
              f"min={min(xs):.2f} max={max(xs):.2f} range={max(xs)-min(xs):.3f}")
PY
echo "NOISE_FLOOR_DONE"
