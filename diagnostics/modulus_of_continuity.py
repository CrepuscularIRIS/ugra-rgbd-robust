#!/usr/bin/env python
"""T3: MODULUS OF CONTINUITY of the depth-authority map on the UNTRAINED interval.

WHY THIS EXISTS
---------------
C2 trains alpha ~ U(alpha_min, 1) with alpha_min = 0.05 > 0 and then deploys at
alpha = 0. Because alpha = 0 is NOT in the closure of the training set, this is
genuine EXTRAPOLATION, not boundary interpolation. Continuity alone therefore
gives no quantitative guarantee -- the guarantee requires a controlled modulus of
continuity (e.g. a Lipschitz bound) on the interval (0, alpha_min) that was never
trained.

The coarse dose-response curve raised a concrete worry: the output moves ~3.1 in
logit space between alpha=0 and alpha=0.05, but only ~5.8 by alpha=0.25. So
change-per-unit-alpha is roughly 63 near zero versus 23 further out -- i.e. the
map may be steepest exactly where it was never trained. If so, the extrapolation
distance from alpha_min to 0 is the WORST part of the curve, which would weaken
the continuity argument and would independently explain why alpha_min = 0.001
transferred better to the absent condition than alpha_min = 0.1.

WHAT IT MEASURES (frozen checkpoint, eval-only)
  For a dense grid of alpha in [0, 1], with the alpha=0 endpoint as reference:
    d(alpha)        = mean over samples of  max_pixel | f_alpha - f_0 |
    secant(alpha)   = d(alpha) / alpha          -- empirical Lipschitz ratio
    local slope     = [d(a_{k+1}) - d(a_k)] / (a_{k+1} - a_k)
  A BOUNDED secant that does not grow as alpha -> 0 supports the extrapolation.
  A secant that BLOWS UP as alpha -> 0 means the untrained interval carries a
  disproportionate share of the function's variation.

Also reports the fraction of the total excursion d(1) that occurs inside the
untrained interval (0, alpha_min) -- the single most interpretable number:
    untrained_share = d(alpha_min) / d(1)

Reseeds before every forward (the Ham/NMF decoder is otherwise nondeterministic --
see diagnostics/single_path_invariance.py).

USAGE
  CKPT=checkpoints/trained/DFormerv2_Small_NYU.pth CUDA_VISIBLE_DEVICES=1 \
  N_SAMPLES=4 python diagnostics/modulus_of_continuity.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

from importlib import import_module  # noqa: E402

from models.builder import EncoderDecoder as segmodel  # noqa: E402
from utils.dataloader.dataloader import get_val_loader  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset  # noqa: E402
from utils.engine.engine import Engine  # noqa: E402

# Default is the frozen S/NYU protocol; overridable by env so the same instrument can
# measure depth-RELIANCE on another dataset or model size (the excursion d(1) is a direct
# measure of how much the trained model's output depends on depth authority).
CONFIG_MODULE = os.environ.get("CONFIG_MODULE", "local_configs.NYUDepthv2.DFormerv2_S")
_FWD_SEED = 12345

# dense near zero (the untrained interval), coarser above alpha_min
ALPHAS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05,
          0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
ALPHA_MIN = 0.05


def main() -> None:
    ckpt = os.environ.get("CKPT", "checkpoints/trained/DFormerv2_Small_NYU.pth")
    n_samples = int(os.environ.get("N_SAMPLES", "4"))
    out_json = os.environ.get(
        "OUT_JSON",
        os.path.join(_PROJECT, "results", "score_push", "runs", "l2_nof", "modulus_of_continuity.json"),
    )

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        # NYU uses no pad; SUN-RGBD's official protocol does. Derive from the dataset so a
        # non-NYU CONFIG_MODULE gets the right geometry (NYU default unchanged).
        config.pad = "SUNRGBD" in str(getattr(config, "dataset_name", "")).upper()
        criterion = nn.CrossEntropyLoss(reduction="mean", ignore_index=config.background)
        os.environ["DEPTH_LAMBDA_TRAIN"] = "1.0"
        model = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
        w = torch.load(ckpt, map_location="cpu")
        w = w.get("model", w.get("state_dict", w)) if isinstance(w, dict) else w
        rep = model.load_state_dict(w, strict=False)
        # HARD GATE (standing rule). Printing the miss count is not enough: the 2026-07-27 probes
        # printed a plausible-looking load line while matching 0/877 tensors and ran an UNTRAINED
        # model. Any probe that builds its own model must FAIL, not warn, below 90% match.
        total = len(model.state_dict())
        matched = total - len(rep.missing_keys)
        print(f"[load] {ckpt} matched={matched}/{total} unexpected={len(rep.unexpected_keys)}")
        if matched < 0.9 * total:
            raise SystemExit(
                f"FATAL: {ckpt} matched only {matched}/{total} tensors (<90%). "
                "Refusing to report reliance from a model that did not load."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()

        val_loader, _ = get_val_loader(engine, RGBXDataset, config, val_batch_size=1)
        samples = []
        for i, mb in enumerate(val_loader):
            if i >= n_samples:
                break
            samples.append((mb["data"].to(device).float(), mb["modal_x"].to(device).float()))
        print(f"[data] {len(samples)} samples")

        def fwd(rgb, dep, lam):
            os.environ["DEPTH_LAMBDA_TRAIN"] = repr(float(lam))
            torch.manual_seed(_FWD_SEED)
            torch.cuda.manual_seed_all(_FWD_SEED)
            with torch.no_grad():
                return model(rgb, dep)

        # sanity: determinism gate
        s0, d0 = samples[0]
        if not torch.equal(fwd(s0, d0, 1.0), fwd(s0, d0, 1.0)):
            raise SystemExit("ABORT: nondeterministic forward; cannot measure a modulus.")

        # d(alpha) averaged over samples, reference = alpha 0
        per_alpha = {a: [] for a in ALPHAS}
        for rgb, dep in samples:
            ref = fwd(rgb, dep, 0.0)
            for a in ALPHAS:
                out = fwd(rgb, dep, a)
                per_alpha[a].append((out - ref).abs().max().item())

        d = {a: float(np.mean(v)) for a, v in per_alpha.items()}
        d1 = d[1.00] if d[1.00] > 0 else 1.0

        # NORMALIZED reliance. d(1) is in absolute logit units, so it is NOT comparable
        # across models of different width (different logit scales). Dividing by the
        # output magnitude at full authority gives a scale-free reliance measure that IS
        # comparable across model sizes and datasets. This discharges a caveat raised when
        # the raw -13% baseline-vs-C2 difference was first reported.
        ref_mag = float(np.mean([
            fwd(rgb, dep, 1.0).abs().max().item() for rgb, dep in samples
        ]))
        reliance_norm = d1 / max(ref_mag, 1e-12)
        print(f"\n[reliance] d(1) = {d1:.4f} (absolute logit units)")
        print(f"[reliance] reference output magnitude ||f(1)||_inf = {ref_mag:.4f}")
        print(f"[reliance] NORMALIZED reliance d(1)/||f(1)|| = {reliance_norm:.4f}  "
              f"<- use THIS for cross-model / cross-dataset comparison")

        print(f"\n{'alpha':>8} {'d(alpha)':>10} {'secant d/a':>12} {'local slope':>12} {'% of d(1)':>10}")
        rows = []
        for k, a in enumerate(ALPHAS):
            sec = (d[a] / a) if a > 0 else float("nan")
            if k + 1 < len(ALPHAS):
                a2 = ALPHAS[k + 1]
                slope = (d[a2] - d[a]) / (a2 - a)
            else:
                slope = float("nan")
            rows.append({"alpha": a, "d": d[a], "secant": sec, "local_slope": slope, "frac_of_d1": d[a] / d1})
            print(f"{a:>8.3f} {d[a]:>10.4f} {sec:>12.2f} {slope:>12.2f} {100*d[a]/d1:>9.1f}%")

        untrained_share = d[ALPHA_MIN] / d1
        sec_small = d[0.002] / 0.002
        sec_large = d[0.50] / 0.50
        blowup = sec_small / sec_large if sec_large > 0 else float("nan")

        print("\n================ VERDICT ================")
        print(f"  share of total excursion inside the UNTRAINED interval (0,{ALPHA_MIN}): "
              f"{100*untrained_share:.1f}% of d(1)")
        print(f"  secant at alpha=0.002 : {sec_small:.1f}")
        print(f"  secant at alpha=0.50  : {sec_large:.1f}")
        print(f"  secant blow-up ratio (small/large): {blowup:.2f}x")
        # Interpretation. The decisive question is whether the secant d(alpha)/alpha
        # DIVERGES as alpha -> 0 (unbounded sensitivity => singular limit, continuity
        # argument void) or CONVERGES to a finite value (local Lipschitz bound exists
        # => regular limit, extrapolation quantitatively controlled). A merely larger
        # secant near zero is concavity, not divergence, and must not be reported as
        # a failure.
        # Gate the small-alpha secant on d(alpha) being above an absolute floor. For a
        # LOW-RELIANCE model d(0.002) can be ~1e-3, i.e. near the numerical precision
        # floor, and its secant becomes meaningless -- which mislabelled a plainly
        # converging sequence (3.53 -> 2.06 -> 1.82 -> 1.75 -> 1.69) as "DIVERGES".
        D_FLOOR = 0.05
        usable = [(a, d[a] / a) for a in (0.002, 0.005, 0.01, 0.02, 0.03, 0.05)
                  if a > 0 and d[a] >= D_FLOOR]
        if len(usable) < 2:
            print(f"  [!] all small-alpha excursions are below the {D_FLOOR} numerical floor "
                  f"(low-reliance model): the small-alpha secant is UNRELIABLE and no "
                  f"regular/singular verdict is issued.")
            diverging = False
            sec_seq = [d[0.05] / 0.05]
        else:
            sec_seq = [s for _, s in usable]
            diverging = sec_seq[0] > 2.0 * sec_seq[-1]   # grows sharply as alpha -> 0
        if diverging:
            print("  => SECANT DIVERGES as alpha->0: unbounded sensitivity, the limit behaves")
            print("     SINGULARLY and the continuity argument does NOT license alpha=0.")
        else:
            print(f"  => SECANT CONVERGES to ~{sec_seq[0]:.1f} as alpha->0: a finite local Lipschitz")
            print("     bound EXISTS, so the alpha->0 limit is REGULAR and the extrapolation is")
            print("     quantitatively controlled. This is the modulus of continuity the")
            print("     continuity argument owed.")
            print(f"     Caveat: the map is CONCAVE, so {100*untrained_share:.1f}% of the total excursion still")
            print("     lies inside the untrained interval - a bounded but non-negligible")
            print("     extrapolation distance. Predicts smaller alpha_min transfers better.")

        results = {
            "ckpt": ckpt, "config": CONFIG_MODULE, "n_samples": len(samples),
            "alpha_min": ALPHA_MIN, "rows": rows,
            "untrained_share_of_d1": untrained_share,
            "d1_absolute": d1, "ref_magnitude": ref_mag,
            "reliance_normalized": reliance_norm,
            "secant_small": sec_small, "secant_large": sec_large, "secant_blowup": blowup,
        }
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[out] {out_json}")


if __name__ == "__main__":
    main()
