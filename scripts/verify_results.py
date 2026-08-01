#!/usr/bin/env python3
"""Rebuild the paper's headline table from the frozen artifacts, and check their integrity.

This script needs no GPU, no datasets and no model weights. It reads only the result
JSONs committed to this repository, so anyone can run it immediately after cloning and
confirm that the numbers printed in the paper are the numbers on disk.

Two independent checks:

  1. INTEGRITY  -- every result JSON is hashed and compared against the SHA-256 recorded
                   in results/FROZEN_20260728/FREEZE_MANIFEST.md at freeze time. This
                   catches silent corruption or an edited artifact.
  2. TABLE      -- the four training arms are recomputed from their per-run files. The
                   "ours" arm is a mean over three independent training runs and its
                   spread is recomputed rather than transcribed.

Usage:
    python scripts/verify_results.py
    python scripts/verify_results.py --no-integrity     # table only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "score_push" / "runs" / "l2_nof"
MANIFEST = ROOT / "results" / "FROZEN_20260728" / "FREEZE_MANIFEST.md"

# One entry per full-length training arm. Every arm is 500 epochs from the released
# ImageNet-pretrained backbone and differs ONLY in the training-time intervention.
ARMS = [
    ("Baseline (no intervention)", ["m1_matched_baseline_final.json"]),
    ("Modality dropout", ["dropout_ep500_final.json"]),
    ("Constant-depth training", ["oracle_ep500_final.json"]),
    # Three INDEPENDENT training runs. Note the third file is named "baseline_seed2"
    # for historical reasons; it is a method run. See docs/RESULTS.md.
    ("Ours (alpha-attenuation)", [
        "c2_ep500_seed1_final.json",
        "c2_ep500_seed2_final.json",
        "baseline_seed2_final.json",
    ]),
]

# The RGB-only finetune is the reference R that the gap is partitioned against. It is
# NOT one of the four arms: its schedule differs, so only its depth-absent value is quoted.
REFERENCE = ("RGB-only finetune (reference R)", "screens/rgb_only_finetune/eval.json")

# Values as they appear in the manuscript, so a drift between repo and paper is loud.
PAPER = {
    "Baseline (no intervention)": (56.39, 50.40),
    "Modality dropout": (54.50, 52.91),
    "Constant-depth training": (53.13, 53.03),
    "Ours (alpha-attenuation)": (55.86, 53.05),
}
PAPER_SD = (0.29, 0.49)          # run-to-run sd of the replicated arm
PAPER_REFERENCE_EM = 53.26
TOL = 0.005                       # the paper quotes 2 decimal places


def read_arm(fname: str) -> tuple[float, float]:
    single = json.loads((RUNS / fname).read_text(encoding="utf-8"))["summary"]["single"]
    return single["clean_miou"], single["entire_missing_miou"]


def check_integrity() -> int:
    if not MANIFEST.exists():
        print(f"  manifest not found: {MANIFEST}", file=sys.stderr)
        return 1
    entries = re.findall(r"^([0-9a-f]{64})  (\S+)$", MANIFEST.read_text(encoding="utf-8"), re.M)
    ok = bad = missing = 0
    for sha, rel in entries:
        path = ROOT / rel
        if not path.is_file():
            missing += 1
            print(f"  MISSING   {rel}")
            continue
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got == sha:
            ok += 1
        else:
            bad += 1
            print(f"  MISMATCH  {rel}")
    print(f"  {ok} artifacts verified byte-identical, {bad} mismatched, {missing} missing")
    return 0 if (bad == 0 and missing == 0) else 1


def check_table() -> int:
    print(f"  {'arm':<28} {'clean':>7} {'entire-missing':>15} {'diff':>7} {'runs':>5}")
    print(f"  {'-'*28} {'-'*7} {'-'*15} {'-'*7} {'-'*5}")
    failures = 0
    for arm, files in ARMS:
        cleans, ems = zip(*(read_arm(f) for f in files))
        clean, em = statistics.fmean(cleans), statistics.fmean(ems)
        line = f"  {arm:<28} {clean:7.2f} {em:15.2f} {clean - em:7.2f} {len(files):5d}"
        exp_c, exp_e = PAPER[arm]
        if abs(round(clean, 2) - exp_c) > TOL or abs(round(em, 2) - exp_e) > TOL:
            line += f"   <-- DIFFERS from paper ({exp_c}/{exp_e})"
            failures += 1
        print(line)
        if len(files) > 1:
            sd_c, sd_e = statistics.stdev(cleans), statistics.stdev(ems)
            print(f"  {'':<28} {'sd ' + format(sd_c, '.2f'):>7} {format(sd_e, '.2f'):>15}"
                  f"   (run-to-run, n={len(files)})")
            if (abs(round(sd_c, 2) - PAPER_SD[0]) > TOL
                    or abs(round(sd_e, 2) - PAPER_SD[1]) > TOL):
                print(f"  {'':<28} <-- sd DIFFERS from paper {PAPER_SD}")
                failures += 1

    name, rel = REFERENCE
    ref = json.loads((RUNS / rel).read_text(encoding="utf-8"))["summary"]["single"]
    flag = "" if abs(round(ref["entire_missing_miou"], 2) - PAPER_REFERENCE_EM) <= TOL \
        else f"   <-- DIFFERS from paper ({PAPER_REFERENCE_EM})"
    failures += 1 if flag else 0
    print(f"\n  {name:<28} {ref['clean_miou']:7.2f} {ref['entire_missing_miou']:15.2f}{flag}")
    print("  (reference R is not an arm: its schedule differs, so only its"
          " depth-absent value is quoted)")

    # The partition of the baseline's gap against that reference (paper: 5.99 = 3.13 + 2.86).
    b_clean, b_em = read_arm(ARMS[0][1][0])
    r_em = ref["entire_missing_miou"]
    print(f"\n  gap partition against R:  {b_clean - b_em:.2f}"
          f"  =  {b_clean - r_em:.2f} (complete-modality advantage over R)"
          f"  +  {r_em - b_em:.2f} (reference-relative depth-absent deficit)")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-integrity", action="store_true", help="skip the SHA-256 check")
    args = ap.parse_args()

    rc = 0
    if not args.no_integrity:
        print("\n[1] artifact integrity (SHA-256 against the freeze manifest)")
        rc |= check_integrity()
    print("\n[2] headline table, recomputed from the frozen per-run artifacts")
    rc |= check_table()
    print("\nOK — repository matches the manuscript." if rc == 0
          else "\nFAILED — see the lines marked above.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
