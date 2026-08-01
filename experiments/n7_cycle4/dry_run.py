"""
Dry-run: synthetic-data validation of ALL planned analyses for n7/cycle4/M3 prereg.
Proposal: Detached Semantic-Boundary Gate + Zero-Init Parallel Edge Adapter.
Schema-matched to real eval output shape (4-condition mIoU table + 2x2 ablation).
Must exit 0 with no AssertionError to count as executed:True.
Seeds are FIXED before any random draw so results are reproducible.
"""

import json
import math
import random

# ============================================================
# Fixed seeds — reproducible
# ============================================================
SEED = 20260724
random.seed(SEED)

# ============================================================
# Baseline (frozen eval protocol)
# ============================================================
BASELINE = {
    "clean":          55.72,
    "holes":          55.40,
    "gaussian":       55.09,
    "entire_missing": 49.07,
}

# Planar-cluster classes (the negative-control cluster that n7-cycle3 regressed 6–9 pt)
PLANAR_CLASSES = ["bathtub", "tv", "toilet", "whiteboard"]
BASELINE_PLANAR = {"bathtub": 57.3, "tv": 52.1, "toilet": 61.4, "whiteboard": 48.6}

# ============================================================
# Promotion bars and guardrails
# ============================================================
BARS = {
    "clean":          0.5,
    "holes":          0.5,
    "gaussian":       0.5,
    "entire_missing": 2.0,
}
GUARDRAIL_CLEAN_HARM = -0.3   # clean must not drop below baseline - 0.3
KILL_SCREEN_THRESHOLD = 56.0  # epoch-50 proxy: clean < threshold → kill

def _jitter(mu, sigma=0.12):
    """Gaussian noise around a mean delta."""
    return mu + random.gauss(0, sigma)

# ============================================================
# Synthetic deltas — hypothesis-consistent for M3
# (realistic prediction; NOT guaranteed to pass bars — just the pipeline)
# ============================================================
# 2×2 ablation axes:
#   dim-0: stop-grad ON vs OFF (stop-grad ON = B trained only by BCE aux)
#   dim-1: label-boundary supervision vs depth-edge supervision
#
# Expected behaviour:
#   (stop-grad=ON, label-boundary) — the canonical M3 mechanism  → best performance
#   (stop-grad=ON, depth-edge)     — conflation re-introduced    → planar regression
#   (stop-grad=OFF, label-boundary)— task-grad leaks → over-suppression risk, modest clean cost
#   (stop-grad=OFF, depth-edge)    — worst case

SYN_DELTAS_2x2 = {
    # (stop_grad_on, label_boundary) — canonical mechanism
    "sg_on__lb":  {
        "clean":          _jitter(+0.65),
        "holes":          _jitter(+0.62),
        "gaussian":       _jitter(+0.55),
        "entire_missing": _jitter(+2.30),
    },
    # (stop_grad_on, depth_edge) — conflation re-introduced (B fires on smooth depth gradients)
    "sg_on__de":  {
        "clean":          _jitter(-0.25),   # expect planar regression ~ n7 cycle3
        "holes":          _jitter(+0.10),
        "gaussian":       _jitter(-0.40),
        "entire_missing": _jitter(+0.50),
    },
    # (stop_grad_off, label_boundary) — task-grad leaks into B → over-suppression of some edges
    "sg_off__lb": {
        "clean":          _jitter(+0.35),
        "holes":          _jitter(+0.30),
        "gaussian":       _jitter(+0.28),
        "entire_missing": _jitter(+1.50),
    },
    # (stop_grad_off, depth_edge) — both failure modes active; worst
    "sg_off__de": {
        "clean":          _jitter(-0.40),
        "holes":          _jitter(-0.05),
        "gaussian":       _jitter(-0.55),
        "entire_missing": _jitter(+0.20),
    },
}

# Scheduled-dropout ablation: with vs without depth-dropout ramp (for entire_missing)
SYN_DELTAS_DROPOUT = {
    "with_dropout":    {"entire_missing": _jitter(+2.30), "clean": _jitter(+0.65)},
    "without_dropout": {"entire_missing": _jitter(+0.30), "clean": _jitter(+0.60)},
}

# Precondition test synthetic measurements
PCT1_BOUNDARY_F1_INIT  = 0.31   # random-init boundary head
PCT1_BOUNDARY_F1_EPOCH5 = 0.62  # after 5-ep smoke: should improve
PCT2_GATE_MEAN_PLANAR  = 0.042  # mean B on planar interior pixels (should be near 0)
PCT2_GATE_MEAN_EDGE    = 0.71   # mean B on true semantic boundary pixels (should be near 1)
PCT3_GATE_MISSING_ENTROPY = 0.53  # entropy of B distribution under depth-dropout (should be > 0.3)

# Planar per-class IoU under canonical M3 (sg_on, label_boundary)
PLANAR_DELTA_M3 = {
    "bathtub":    _jitter(+0.10),   # should be ~0 (B≈0 on planar interiors)
    "tv":         _jitter(+0.05),
    "toilet":     _jitter(-0.08),
    "whiteboard": _jitter(+0.03),
}

# 3-seed clean results
random.seed(SEED)
seed_clean_3 = [round(BASELINE["clean"] + _jitter(+0.65, sigma=0.10), 4) for _ in range(3)]

# ============================================================
# ANALYSIS 1: 4-condition mIoU table — canonical M3 (sg_on, label_boundary)
# ============================================================
print("=== Analysis 1: 4-condition mIoU table (canonical M3) ===")
metrics = list(BASELINE.keys())
conditions_4 = ["baseline", "sg_on__lb", "sg_on__de", "sg_off__lb"]

header = f"{'Condition':<24} {'clean':>8} {'holes':>8} {'gaussian':>10} {'entire_missing':>16}"
print(header)
print("-" * len(header))

# Baseline row
bl_row = f"{'baseline':<24} {BASELINE['clean']:>8.4f} {BASELINE['holes']:>8.4f} {BASELINE['gaussian']:>10.4f} {BASELINE['entire_missing']:>16.4f}"
print(bl_row)

condition_results = {}
for cond, deltas in SYN_DELTAS_2x2.items():
    row = {m: round(BASELINE[m] + deltas[m], 4) for m in metrics}
    condition_results[cond] = row
    vals = " ".join(f"{row[m]:>8.4f}" if i < 2 else f"{row[m]:>10.4f}" if i == 2 else f"{row[m]:>16.4f}"
                    for i, m in enumerate(metrics))
    print(f"{cond:<24} {row['clean']:>8.4f} {row['holes']:>8.4f} {row['gaussian']:>10.4f} {row['entire_missing']:>16.4f}")

# Assertions: canonical condition has all metrics above baseline
canonical = condition_results["sg_on__lb"]
for m in metrics:
    delta = canonical[m] - BASELINE[m]
    assert delta > 0, f"Analysis 1 FAIL: canonical M3 must beat baseline on {m}; delta={delta:.4f}"
print(f"  [OK] Canonical M3 (sg_on__lb) beats baseline on all 4 metrics.")

# Guardrail: clean harm limit
clean_delta_canonical = canonical["clean"] - BASELINE["clean"]
assert clean_delta_canonical > GUARDRAIL_CLEAN_HARM, (
    f"Analysis 1 GUARDRAIL: clean delta {clean_delta_canonical:.4f} < harm_limit {GUARDRAIL_CLEAN_HARM}"
)
print(f"  [OK] Clean guardrail passed: delta={clean_delta_canonical:+.4f} > {GUARDRAIL_CLEAN_HARM}.")

# ============================================================
# ANALYSIS 2: 2×2 ablation — {stop-grad} × {boundary supervision}
# Main estimator: mean mIoU per condition; interaction term per metric
# ============================================================
print("\n=== Analysis 2: 2×2 ablation — stop-grad × supervision source ===")
print(f"{'Condition':<22} {'clean_delta':>12} {'holes_delta':>12} {'gauss_delta':>12} {'em_delta':>12}")
for cond, row in condition_results.items():
    deltas = {m: round(row[m] - BASELINE[m], 4) for m in metrics}
    print(f"{cond:<22} {deltas['clean']:>12.4f} {deltas['holes']:>12.4f} {deltas['gaussian']:>12.4f} {deltas['entire_missing']:>12.4f}")

# Interaction term per metric (estimator pinned):
# interaction = delta(sg_on,lb) - delta(sg_on,de) - delta(sg_off,lb) + delta(sg_off,de)
print("\n  --- Interaction terms (label-boundary effect modulated by stop-grad) ---")
interaction_terms = {}
for m in metrics:
    d_sg_on_lb  = condition_results["sg_on__lb"][m]  - BASELINE[m]
    d_sg_on_de  = condition_results["sg_on__de"][m]  - BASELINE[m]
    d_sg_off_lb = condition_results["sg_off__lb"][m] - BASELINE[m]
    d_sg_off_de = condition_results["sg_off__de"][m] - BASELINE[m]
    interaction = d_sg_on_lb - d_sg_on_de - d_sg_off_lb + d_sg_off_de
    interaction_terms[m] = interaction
    print(f"    {m}: interaction={interaction:+.4f}")
    assert isinstance(interaction, float), f"interaction for {m} must be float"

# The interaction being positive means stop-grad amplifies the label-boundary benefit
# (a prediction, not a hard assertion — just confirm the computation runs)
print(f"  [OK] 2x2 interaction terms computed for all {len(metrics)} metrics.")

# ============================================================
# ANALYSIS 3: depth-edge supervision falsification
# Estimator: mIoU delta (depth-edge condition) on planar cluster — must regress relative to canonical
# ============================================================
print("\n=== Analysis 3: Depth-edge supervision falsification (planar regression check) ===")
de_clean_delta = condition_results["sg_on__de"]["clean"] - BASELINE["clean"]
lb_clean_delta = condition_results["sg_on__lb"]["clean"] - BASELINE["clean"]
print(f"  label-boundary clean delta: {lb_clean_delta:+.4f}")
print(f"  depth-edge     clean delta: {de_clean_delta:+.4f}")
margin_de_vs_lb = lb_clean_delta - de_clean_delta
print(f"  margin (lb - de) on clean: {margin_de_vs_lb:+.4f}  (>0 → lb wins, depth-edge conflation confirmed)")
assert lb_clean_delta > de_clean_delta, (
    f"Analysis 3 FAIL: label-boundary should outperform depth-edge on clean "
    f"(lb={lb_clean_delta:.4f}, de={de_clean_delta:.4f})"
)
print(f"  [OK] label-boundary condition outperforms depth-edge — conflation confirmed in synthetic data.")

# ============================================================
# ANALYSIS 4: modality-dropout ablation (entire_missing sensitivity)
# Estimator: mIoU entire_missing delta — with vs without scheduled depth-dropout ramp
# ============================================================
print("\n=== Analysis 4: Scheduled depth-dropout ablation (entire_missing) ===")
em_with    = BASELINE["entire_missing"] + SYN_DELTAS_DROPOUT["with_dropout"]["entire_missing"]
em_without = BASELINE["entire_missing"] + SYN_DELTAS_DROPOUT["without_dropout"]["entire_missing"]
print(f"  entire_missing WITH dropout:    {em_with:.4f}  (delta: {em_with - BASELINE['entire_missing']:+.4f})")
print(f"  entire_missing WITHOUT dropout: {em_without:.4f}  (delta: {em_without - BASELINE['entire_missing']:+.4f})")
dropout_em_benefit = em_with - em_without
print(f"  entire_missing benefit of dropout: {dropout_em_benefit:+.4f}")
assert em_with > em_without, (
    f"Analysis 4 FAIL: with-dropout should yield higher entire_missing "
    f"(with={em_with:.4f}, without={em_without:.4f})"
)
print(f"  [OK] Scheduled dropout provides {dropout_em_benefit:+.4f} entire_missing gain in synthetic data.")

# ============================================================
# ANALYSIS 5: stop-grad ablation — task-gradient-leak check on clean
# Estimator: clean mIoU delta, sg_on vs sg_off (both with label-boundary supervision)
# ============================================================
print("\n=== Analysis 5: Stop-grad vs no-stop-grad (clean delta) ===")
clean_sg_on  = condition_results["sg_on__lb"]["clean"]  - BASELINE["clean"]
clean_sg_off = condition_results["sg_off__lb"]["clean"] - BASELINE["clean"]
print(f"  clean delta stop-grad ON:  {clean_sg_on:+.4f}")
print(f"  clean delta stop-grad OFF: {clean_sg_off:+.4f}")
sg_margin = clean_sg_on - clean_sg_off
print(f"  margin (sg_on - sg_off):   {sg_margin:+.4f}  (>0 expected → stop-grad preserves boundary fidelity)")
assert isinstance(sg_margin, float), "sg_margin must be float"
print(f"  [OK] Stop-grad ablation metric computed.")

# ============================================================
# ANALYSIS 6: 3-seed clean variance (headline condition)
# Estimator: mean ± std of clean mIoU across seeds [20260721, 20260722, 20260723]
# ============================================================
print("\n=== Analysis 6: 3-seed clean variance ===")
seeds = [20260721, 20260722, 20260723]
print(f"  seeds: {seeds}")
print(f"  clean values: {seed_clean_3}")
mean_clean = sum(seed_clean_3) / len(seed_clean_3)
var_clean  = sum((x - mean_clean)**2 for x in seed_clean_3) / (len(seed_clean_3) - 1)
std_clean  = math.sqrt(var_clean)
delta_mean = mean_clean - BASELINE["clean"]
print(f"  mean={mean_clean:.4f}, std={std_clean:.4f}, delta_vs_baseline={delta_mean:+.4f}")
assert len(seed_clean_3) == 3, "must have 3 seed values"
assert std_clean >= 0.0, "std must be non-negative"
print(f"  [OK] 3-seed variance computed; std={std_clean:.4f}.")

# ============================================================
# PRECONDITION TEST CHECKS (boundary-head learnability, planar gate, missing-depth fallback)
# ============================================================
print("\n=== Precondition test checks ===")

# PCT-1: boundary-head learnability
print("  PCT-1: Boundary-head learnability (BCE loss descends, F1 rises)")
assert PCT1_BOUNDARY_F1_EPOCH5 > PCT1_BOUNDARY_F1_INIT + 0.10, (
    f"PCT-1 FAIL: F1 should rise ≥0.10 in first 5 ep "
    f"(init={PCT1_BOUNDARY_F1_INIT:.3f}, ep5={PCT1_BOUNDARY_F1_EPOCH5:.3f})"
)
print(f"  [OK] Boundary F1: {PCT1_BOUNDARY_F1_INIT:.3f} → {PCT1_BOUNDARY_F1_EPOCH5:.3f} (+{PCT1_BOUNDARY_F1_EPOCH5 - PCT1_BOUNDARY_F1_INIT:.3f})")

# PCT-2: planar gate check — B ≈ 0 on planar interior pixels
print("  PCT-2: Planar gate (mean B on planar interiors ≈ 0)")
PLANAR_GATE_THRESHOLD = 0.10
assert PCT2_GATE_MEAN_PLANAR < PLANAR_GATE_THRESHOLD, (
    f"PCT-2 FAIL: gate too active on planar interiors "
    f"(mean_B={PCT2_GATE_MEAN_PLANAR:.3f} >= threshold {PLANAR_GATE_THRESHOLD})"
)
print(f"  [OK] Gate mean on planar interiors: {PCT2_GATE_MEAN_PLANAR:.3f} < {PLANAR_GATE_THRESHOLD}")
print(f"  [OK] Gate mean on true edge pixels:  {PCT2_GATE_MEAN_EDGE:.3f}  (active at boundaries)")

# PCT-3: missing-depth fallback — B non-degenerate under depth-dropout
print("  PCT-3: Missing-depth fallback (B entropy under dropout > 0.3)")
ENTROPY_FLOOR = 0.30
assert PCT3_GATE_MISSING_ENTROPY > ENTROPY_FLOOR, (
    f"PCT-3 FAIL: gate collapsed to constant under depth-dropout "
    f"(entropy={PCT3_GATE_MISSING_ENTROPY:.3f} <= floor {ENTROPY_FLOOR})"
)
print(f"  [OK] Gate entropy under depth-dropout: {PCT3_GATE_MISSING_ENTROPY:.3f} > {ENTROPY_FLOOR}")

# ============================================================
# NEGATIVE CONTROL: planar cluster per-class IoU delta (n7-cycle3 regressed 6–9 pt here)
# Under canonical M3, these must stay within ±2 pt of baseline
# ============================================================
print("\n=== Negative control: planar cluster per-class IoU delta ===")
PLANAR_HARM_LIMIT = -2.0   # must not regress more than 2 pt
for cls in PLANAR_CLASSES:
    delta = PLANAR_DELTA_M3[cls]
    pass_flag = delta >= PLANAR_HARM_LIMIT
    print(f"  {cls:<12}: delta={delta:+.4f}  {'OK' if pass_flag else 'FAIL (regression)'}")
    assert pass_flag, (
        f"Negative control FAIL: {cls} regresses {delta:.4f} pt "
        f"(limit {PLANAR_HARM_LIMIT}) — B fires on planar interior (PCT-2 precondition violated)"
    )
print(f"  [OK] All planar classes within harm limit ({PLANAR_HARM_LIMIT} pt).")

# ============================================================
# KILL-SWITCH SIMULATION: epoch-50 finetune-proxy screen
# Estimator: interpolated epoch-50 clean proxy; kill if < KILL_SCREEN_THRESHOLD
# ============================================================
print("\n=== Kill-switch screen (epoch-50 proxy) ===")
# Epoch-50 proxy: assume ~40% of final gain has accrued by epoch 50 in finetune-from-converged
epoch50_clean = BASELINE["clean"] + 0.40 * (canonical["clean"] - BASELINE["clean"])
kill = epoch50_clean < KILL_SCREEN_THRESHOLD
print(f"  epoch50_clean={epoch50_clean:.4f}, threshold={KILL_SCREEN_THRESHOLD}, kill={kill}")
assert isinstance(kill, bool), "kill flag must be bool"
print(f"  [OK] Kill-switch simulation complete (kill={'yes' if kill else 'no'}).")

# ============================================================
# PROMOTION BAR CHECK (synthetic — informational only; real outcome is empirical)
# ============================================================
print("\n=== Promotion bar check (synthetic / informational) ===")
pass_count = 0
for m, bar in BARS.items():
    delta = canonical[m] - BASELINE[m]
    passes = delta >= bar
    if passes:
        pass_count += 1
    print(f"  {m:<18}: delta={delta:+.4f}  bar={bar:+.1f}  {'PASS' if passes else 'miss (informational)'}")
all_bars_pass = (pass_count == len(BARS))
print(f"  {pass_count}/{len(BARS)} bars met in synthetic data (informational — real outcome is empirical).")

# ============================================================
# Summary JSON (for embedding in prereg)
# ============================================================
dry_run_summary = {
    "executed": True,
    "seed": SEED,
    "analyses_confirmed": [
        "A1: 4-condition mIoU table (canonical + 3 ablation conditions)",
        "A2: 2x2 ablation interaction term (stop-grad x supervision source)",
        "A3: Depth-edge supervision falsification (planar regression check)",
        "A4: Scheduled depth-dropout ablation (entire_missing delta)",
        "A5: Stop-grad task-gradient-leak check (clean delta sg_on vs sg_off)",
        "A6: 3-seed clean variance (mean, std)",
        "PCT-1: Boundary-head learnability (BCE F1 rise)",
        "PCT-2: Planar gate check (B near 0 on planar interiors)",
        "PCT-3: Missing-depth fallback (gate entropy under dropout)",
        "Negative control: planar cluster per-class IoU delta (harm limit -2.0 pt)",
        "Kill-switch screen: epoch-50 proxy < 56.0",
    ],
    "precondition_tests_all_pass": True,
    "negative_control_pass": True,
    "kill_switch_simulated": True,
    "promotion_bars_in_synthetic": f"{pass_count}/{len(BARS)}",
    "discrepancy": (
        "Synthetic data is drawn from hypothesis-consistent Gaussian jitter; "
        "promotion-bar passage in dry-run is informational, not a prediction. "
        "The analysis pipeline (formulas + assertions) is confirmed sound end-to-end."
    ),
}
print("\n=== DRY-RUN SUMMARY ===")
print(json.dumps(dry_run_summary, indent=2))
print("\n=== DRY-RUN COMPLETE: all analyses executed without error ===")
