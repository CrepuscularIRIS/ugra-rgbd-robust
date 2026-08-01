#!/usr/bin/env bash
# WP1 confirmation eval: baseline(test) + RPD ep40 (test/full/MS/secondary) + ablation ep40.
# Assumes rpd_drop_c1/ckpt_ep40.pth and rpd_ablate_c1/ckpt_ep40.pth exist.
set -euo pipefail
# --- portable bootstrap (see scripts/env.sh) ---
UGRA_ROOT="${UGRA_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
# shellcheck source=/dev/null
source "$UGRA_ROOT/scripts/env.sh"

PROJECT="${UGRA_ROOT}"
BASE_CKPT="$PROJECT/repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth"
RPD="$PROJECT/results/score_push/runs/rpd_drop_c1/ckpt_ep40.pth"
ABL="$PROJECT/results/score_push/runs/rpd_ablate_c1/ckpt_ep40.pth"
OUT="$PROJECT/results/score_push/runs/rpd_drop_c1"
H="$PROJECT/scripts/eval_robust_suite.sh"

run () { # ckpt tag modes conds split out
  CKPT="$1" MODEL_TAG="$2" EVAL_MODES="$3" EVAL_CONDITIONS="$4" EVAL_SPLIT="$5" \
  VAL_BATCH=8 SEED=20260721 OUT_JSON="$6" bash "$H" >/dev/null 2>&1 && echo "  ok: $6"
}

echo "== baseline test =="
run "$BASE_CKPT" base_test single clean,entire_missing test "$PROJECT/results/score_push/baselines/baseline_single_test.json"
echo "== RPD ep40 test =="
run "$RPD" rpd_test single clean,entire_missing test "$OUT/eval_test_ckpt_ep40.json"
echo "== RPD ep40 full (4 conds, single) =="
run "$RPD" rpd_full_ss single clean,entire_missing,holes,gaussian full "$OUT/eval_full_ss_ckpt_ep40.json"
if [ "${DO_MS:-0}" = "1" ]; then
  echo "== RPD ep40 full (MS, clean+missing) =="
  run "$RPD" rpd_full_ms ms clean,entire_missing full "$OUT/eval_full_ms_ckpt_ep40.json"
fi
echo "== ablation ep40 dev+full =="
run "$ABL" abl_dev single clean,entire_missing dev "$PROJECT/results/score_push/runs/rpd_ablate_c1/eval_dev_ckpt_ep40.json"
run "$ABL" abl_full single clean,entire_missing full "$PROJECT/results/score_push/runs/rpd_ablate_c1/eval_full_ckpt_ep40.json"

echo && echo "===== WP1 CONSOLIDATED TABLE ====="
python3 - <<'PY'
import json, os
P="${UGRA_ROOT}/results/score_push"
def g(f, mode="single"):
    try:
        d=json.load(open(f))["results"][mode]
        return d
    except Exception as e:
        return None
def row(name, d):
    if not d: print(f"  {name:28s}  (missing)"); return
    cl=d.get("clean",{}).get("miou"); mi=d.get("entire_missing",{}).get("miou")
    ho=d.get("holes",{}).get("miou"); ga=d.get("gaussian",{}).get("miou")
    def f(x): return f"{x:6.2f}" if isinstance(x,(int,float)) else "   -  "
    print(f"  {name:28s} clean={f(cl)} miss={f(mi)} holes={f(ho)} gauss={f(ga)}")
print("--- single-scale ---")
row("baseline full",          g(f"{P}/baselines/baseline_single.json"))
row("baseline dev",           g(f"{P}/baselines/baseline_single_dev.json"))
row("baseline test",          g(f"{P}/baselines/baseline_single_test.json"))
row("RPD ep40 full",          g(f"{P}/runs/rpd_drop_c1/eval_full_ss_ckpt_ep40.json"))
row("RPD ep40 dev",           g(f"{P}/runs/rpd_drop_c1/eval_dev_ckpt_ep40.json"))
row("RPD ep40 test",          g(f"{P}/runs/rpd_drop_c1/eval_test_ckpt_ep40.json"))
row("ablation(plain) ep40 dev",  g(f"{P}/runs/rpd_ablate_c1/eval_dev_ckpt_ep40.json"))
row("ablation(plain) ep40 full", g(f"{P}/runs/rpd_ablate_c1/eval_full_ckpt_ep40.json"))
print("--- official multi-scale ---")
row("baseline full MS",       g(f"{P}/baselines/baseline_ms.json","ms"))
row("RPD ep40 full MS",       g(f"{P}/runs/rpd_drop_c1/eval_full_ms_ckpt_ep40.json","ms"))
PY
