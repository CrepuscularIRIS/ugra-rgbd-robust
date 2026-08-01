#!/usr/bin/env python
"""Frozen robustness eval harness for DFormerv2-S on NYUDepthv2 (WP0).

Produces clean / entire_missing / holes / gaussian mIoU under the OFFICIAL
evaluation protocol (sliding + flip + amp; single-scale via `evaluate`, official
multi-scale via `evaluate_msf`), by injecting the Phase-1 depth-corruption
registry through a single forward_pre_hook on the model. Every model call in the
eval path -- evaluate(), slide_inference(), evaluate_msf() -- routes through
`model(rgb, modal_x)`, so one hook corrupts the *normalized 3-ch depth tensor*
exactly as Phases 1-4 defined it, while the sliding/MS/flip logic stays identical
to `scripts/eval_dformerv2s_nyu_gpu1.sh`.

All parameters are passed via environment variables (Engine owns argparse):
  CKPT           checkpoint path (relative to repos/DFormer)   [required]
  EVAL_MODES     comma list of {single,ms}                     [default: single]
  EVAL_CONDITIONS comma list of {clean,entire_missing,holes,gaussian}
                                                               [default: all four]
  SEED           corruption + torch seed                       [default: 20260721]
  VAL_BATCH      val batch size                                [default: config.batch_size]
  SLIDING        "1"/"0" sliding-window inference              [default: 1]
  OUT_JSON       output JSON path                              [required]
  MODEL_TAG      label for provenance (e.g. DFormerv2_S_baseline) [default: derived]

Grounding gate (GOAL_SCORE_PUSH WP0): this is the fixed script; every number
after WP0 must come from it with the same seed.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

# --- path setup: mirror how repos/DFormer/utils/eval.py resolves imports -------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)                       # ugra-rgbd-robust
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")      # vendored repo
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)                                         # config uses root_dir="datasets"
os.environ.setdefault("LOCAL_RANK", "0")                # Engine reads this unconditionally

from importlib import import_module

from models.builder import EncoderDecoder as segmodel   # noqa: E402
from val_mm import evaluate, evaluate_msf               # noqa: E402
from utils.dataloader.dataloader import get_val_loader  # noqa: E402
from utils.dataloader.RGBXDataset import RGBXDataset    # noqa: E402
from utils.engine.engine import Engine                  # noqa: E402
from diagnostics.corruptions.depth_corruptions import (  # noqa: E402
    apply_corruption,
    default_corruption_suite,
)

# Default is the frozen DFormerv2-S protocol; every campaign number to date comes from
# it and that must not change. CONFIG_MODULE may be overridden by env ONLY to evaluate a
# different architecture (e.g. DFormerv2_L for the capacity-generalization arm), and the
# chosen module is already recorded in the output JSON under "config" for provenance.
CONFIG_MODULE = os.environ.get("CONFIG_MODULE", "local_configs.NYUDepthv2.DFormerv2_S")
MS_SCALES = [0.5, 0.75, 1.0, 1.25, 1.5]


class _TruncatedLoader:
    """Wrap a DataLoader to expose only the first ``k`` batches (smoke tests).

    Supports the two things val_mm.evaluate()/evaluate_msf() use: iteration and
    ``len()``; forwards ``.dataset`` for bookkeeping.
    """

    def __init__(self, loader, k):
        self._loader = loader
        self._k = max(1, int(k))

    def __iter__(self):
        import itertools

        return iter(itertools.islice(self._loader, self._k))

    def __len__(self):
        return self._k

    @property
    def dataset(self):
        return self._loader.dataset

# --- global corruption spec toggled per condition; read by the pre-hook --------
_CORR = {"name": "clean", "kwargs": {}}
# L2-NOF C1: eval-only expected-bias imputation. When set, `mean_impute` replaces
# the (absent) depth with a fixed train-set-average depth map instead of zeros.
_IMPUTE_MAP = None  # torch.Tensor [C,H,W] normalized, or None


def _corruption_pre_hook(module, args):
    """Replace the normalized depth tensor (args[1]) just before model forward."""
    name = _CORR["name"]
    if not name or name == "clean":
        return None
    if len(args) < 2 or args[1] is None:
        return None
    rgb, modal_x = args[0], args[1]
    if name == "rgb_missing":
        # The OTHER damaged combination required by the EMM/RMM benchmark protocol
        # (arXiv:2503.18445). With two modalities the damaged set is {depth-missing,
        # rgb-missing}; we already measure the first as `entire_missing`. Zero-filling RGB
        # is the benchmark's own convention for an absent modality. NOTE this is NOT a
        # neutral element for the RGB pathway (unlike depth, RGB feeds biased convolutions),
        # so it removes the modality in the benchmark's sense, not in the exact sense of
        # Section "Where depth enters".
        return (torch.zeros_like(rgb), modal_x) + tuple(args[2:])
    if name == "mean_impute":
        # C1 rung-a: feed E_train[d] (broadcast to batch, resized to modal_x HxW).
        import torch.nn.functional as _Fh
        m = _IMPUTE_MAP.to(modal_x.device, modal_x.dtype)
        if m.dim() == 3:
            m = m.unsqueeze(0)
        if m.shape[-2:] != modal_x.shape[-2:]:
            m = _Fh.interpolate(m, size=modal_x.shape[-2:], mode="bilinear", align_corners=False)
        modal_x = m.expand(modal_x.shape[0], -1, -1, -1).contiguous()
        return (rgb, modal_x) + tuple(args[2:])
    modal_x = apply_corruption(modal_x, name, **_CORR["kwargs"])
    return (rgb, modal_x) + tuple(args[2:])


def _env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def main():
    ckpt = _env("CKPT")
    if not ckpt:
        raise SystemExit("CKPT env var is required")
    out_json = _env("OUT_JSON")
    if not out_json:
        raise SystemExit("OUT_JSON env var is required")
    modes = [m.strip() for m in _env("EVAL_MODES", "single").split(",") if m.strip()]
    conditions = [
        c.strip()
        for c in _env(
            "EVAL_CONDITIONS", "clean,entire_missing,holes,gaussian"
        ).split(",")
        if c.strip()
    ]
    seed = int(_env("SEED", "20260721"))
    sliding = _env("SLIDING", "1") == "1"
    model_tag = _env("MODEL_TAG", os.path.splitext(os.path.basename(ckpt))[0])

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    with Engine(custom_parser=None) as engine:
        config = getattr(import_module(CONFIG_MODULE), "C")
        # eval.py sets this from the --pad_SUNRGBD flag: NYU uses no pad, SUN-RGBD does.
        # Derive it from the dataset so a non-NYU CONFIG_MODULE gets the right protocol,
        # with an explicit env override. Default for NYU is unchanged (False), so the
        # frozen S/NYU numbers are untouched.
        _ds = str(getattr(config, "dataset_name", ""))
        _pad_default = "1" if "SUNRGBD" in _ds.upper() else "0"
        config.pad = _env("EVAL_PAD", _pad_default) == "1"
        print(f"[cfg] dataset={_ds} config.pad={config.pad} (module={CONFIG_MODULE})")
        val_batch = int(_env("VAL_BATCH", str(int(config.batch_size))))

        criterion = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=config.background
        )
        # eval-mode BatchNorm uses stored running stats -> identical to SyncBN,
        # and avoids needing a process group on a single process.
        model = segmodel(
            cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False
        )
        # --- n7 model-LOAD adaptation (env-gated; corruption hook + metric path
        # below are UNCHANGED) ---------------------------------------------------
        # The n7 checkpoint is a DFormerv2WithAdapter.state_dict() with prefixed
        # keys (base./adapter./witness_head./seg_aux_head.). A plain segmodel load
        # would reject every key. When N7_ADAPTER=1 we wrap the base model exactly
        # as train_n7.py does and load the full dict. At eval self.training=False,
        # so the adapter's forward returns logits and the witness/seg-aux heads are
        # inert (loaded but unused). The forward(rgb, modal_x, label=None) contract
        # is identical to segmodel, so the pre-hook corruption + evaluate_msf path
        # are byte-identical to the frozen harness.
        if _env("N7_ADAPTER", "0") == "1":
            _EXP = os.path.join(
                _PROJECT, "experiments", "n7_decomp_witness"
            )
            if _EXP not in sys.path:
                sys.path.insert(0, _EXP)
            from model_wrapper import DFormerv2WithAdapter  # noqa: E402
            model = DFormerv2WithAdapter(
                model,
                num_classes=int(config.num_classes),
                adapter_stage_idx=int(_env("N7_ADAPTER_STAGE_IDX", "2")),
                witness_weight=float(_env("N7_WITNESS_WEIGHT", "0.05")),
                seg_aux_weight=float(_env("N7_SEG_AUX_WEIGHT", "0.2")),
                n_bins=int(_env("N7_N_BINS", "8")),
            )
        # --- n7 cycle-4 (M3) model-LOAD adaptation (env-gated; corruption + metric
        # path UNCHANGED). Checkpoint is a DFormerv2WithBoundaryGate.state_dict()
        # with prefixed keys (base./boundary_head./adapter.). At eval training=False
        # so forward(rgb, modal_x, label=None) returns logits; the boundary BCE path
        # is skipped and the wrapper detects structureless (zeroed) depth under the
        # entire_missing corruption via its own per-sample present-check. ---
        if _env("N7C4_ADAPTER", "0") == "1":
            _EXP4 = os.path.join(_PROJECT, "experiments", "n7_cycle4")
            if _EXP4 not in sys.path:
                sys.path.insert(0, _EXP4)
            from model_wrapper_m3 import DFormerv2WithBoundaryGate  # noqa: E402
            model = DFormerv2WithBoundaryGate(
                model,
                num_classes=int(config.num_classes),
                adapter_stage_idx=int(_env("N7C4_ADAPTER_STAGE_IDX", "2")),
                boundary_weight=float(_env("N7C4_BOUNDARY_WEIGHT", "0.10")),
                boundary_pos_weight=float(_env("N7C4_BOUNDARY_POS_WEIGHT", "30.0")),
                supervision=_env("N7C4_SUPERVISION", "label"),
                stop_grad=(_env("N7C4_STOP_GRAD", "1") == "1"),
            )
        weight = torch.load(ckpt, map_location=torch.device("cpu"))
        if "model" in weight:
            weight = weight["model"]
        elif "state_dict" in weight:
            weight = weight["state_dict"]
        load_report = model.load_state_dict(weight, strict=False)
        missing = list(load_report.missing_keys)
        unexpected = list(load_report.unexpected_keys)
        if missing or unexpected:
            print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
            if missing:
                print(f"[load] first missing: {missing[:5]}")
            if unexpected:
                print(f"[load] first unexpected: {unexpected[:5]}")

        # --- n7 cycle-5 MP-A oracle gate (EVAL-ONLY, env-gated; corruption hook +
        # metric path below UNCHANGED). Installs a class-level monkeypatch on
        # GeoPriorGen.forward that gates ONLY the weight[1]*mask_d depth-prior
        # term by a GT-depth-discontinuity mask. NO parameter changes, NO
        # training. When MPA_ORACLE is unset/off this block is a no-op and the
        # frozen forward is retained byte-for-byte. See
        # experiments/n7_cycle5_mpa/oracle_gate.py. ---
        if _env("MPA_ORACLE", "off") not in ("off", "", None):
            _EXP5 = os.path.join(_PROJECT, "experiments", "n7_cycle5_mpa")
            if _EXP5 not in sys.path:
                sys.path.insert(0, _EXP5)
            import oracle_gate  # noqa: E402
            from models.encoders.DFormerv2 import GeoPriorGen  # noqa: E402
            oracle_gate.install(GeoPriorGen)
            oracle_gate.tag_stages(model)
            print(
                f"[mpa] oracle={_env('MPA_ORACLE')} tau={_env('MPA_TAU','0.15')} "
                f"stages={_env('MPA_STAGES','2')}"
            )

        # --- L2-NOF diagnostic: env-gated scalar lambda on the weight[1]*mask_d
        # depth-prior term (EVAL-ONLY, default off => byte-identical frozen
        # forward). weight[1] is the depth-decay coefficient in GeoPriorGen; the
        # positional term weight[0]*mask_pos is untouched. Uses:
        #   T1 equivalence audit: DEPTH_LAMBDA=0 on condition=clean should match
        #       condition=entire_missing (both zero the depth prior).
        #   T2 dose-response: sweep DEPTH_LAMBDA in {0,.25,.5,.75,1,1.25} on clean.
        # No parameter training; scaling is applied once on the frozen weights. ---
        # MUTUAL EXCLUSION. These three knobs each `mul_` weight[1] independently, so setting two
        # would apply their PRODUCT (e.g. lambda^2) with no error — and a leftover
        # DEPTH_LAMBDA_TRAIN=0 exported by a training shell would silently ZERO every profile,
        # producing a table of 49.1s that looks like a finding. Fail loudly instead.
        _active = [n for n in ("DEPTH_LAMBDA", "DEPTH_LAMBDA_STAGES", "DEPTH_LAMBDA_BLOCKS") if _env(n) is not None]
        if len(_active) > 1:
            raise SystemExit(f"FATAL: {_active} are mutually exclusive — each scales weight[1]; together they multiply.")
        _dlt = float(os.environ.get("DEPTH_LAMBDA_TRAIN", "1.0"))
        if _active and _dlt != 1.0:
            raise SystemExit(
                f"FATAL: DEPTH_LAMBDA_TRAIN={_dlt} is set while {_active} is active; the depth bias would be "
                f"scaled twice. Unset it (this is a TRAINING-side knob that leaks via exported env)."
            )

        _dl = _env("DEPTH_LAMBDA")
        if _dl is not None:
            from models.encoders.DFormerv2 import GeoPriorGen  # noqa: E402
            _lam = float(_dl)
            _n_scaled = 0
            for _m in model.modules():
                if isinstance(_m, GeoPriorGen):
                    with torch.no_grad():
                        _m.weight[1].mul_(_lam)
                    _n_scaled += 1
            print(f"[l2nof] DEPTH_LAMBDA={_lam} scaled weight[1] on {_n_scaled} GeoPriorGen modules")

        # --- L3-ID: DEPTH_LAMBDA_STAGES, a PER-STAGE version of the above (EVAL-ONLY,
        # default off => byte-identical frozen forward). Spec is 4 comma-separated
        # scalars, one per encoder stage, e.g. "0,0,1,0" = restore the true depth bias
        # at stage 2 ONLY and leave it zeroed everywhere else.
        #
        # WHY. The repair profile (which parameters carry C2's fix) is measured in a
        # parameter basis I chose, so it inherits a site-leverage confound. The DAMAGE
        # profile does not: depth enters through exactly ONE tensor per block (proven
        # bitwise), so gating that tensor per stage is an EXACT and COMPLETE intervention
        # basis chosen by the architecture, with the SAME intervention type at every
        # location. Overlap between the damage profile and the repair profile is then a
        # confound-free test of the "damage-side and repair-side are different pathways"
        # claim.
        # SELF-VALIDATING ENDPOINTS: "0,0,0,0" must reproduce entire_missing (49.07-49.12)
        # and "1,1,1,1" must reproduce clean (55.72-55.75). If either endpoint misses, the
        # harness is wrong, not the hypothesis.
        _dls = _env("DEPTH_LAMBDA_STAGES")
        if _dls is not None:
            import re as _re

            from models.encoders.DFormerv2 import GeoPriorGen  # noqa: E402

            _per_stage = [float(x) for x in str(_dls).split(",")]
            if len(_per_stage) != 4:
                raise ValueError(f"DEPTH_LAMBDA_STAGES needs 4 comma-separated values, got {_dls!r}")
            _counts = [0, 0, 0, 0]
            for _name, _m in model.named_modules():
                if not isinstance(_m, GeoPriorGen):
                    continue
                _hit = _re.search(r"layers\.(\d+)\.", _name)
                if _hit is None:
                    raise RuntimeError(f"GeoPriorGen at unexpected path {_name!r}; cannot assign a stage")
                _si = int(_hit.group(1))
                with torch.no_grad():
                    _m.weight[1].mul_(_per_stage[_si])
                _counts[_si] += 1
            if sum(_counts) == 0:
                raise RuntimeError("DEPTH_LAMBDA_STAGES matched no GeoPriorGen modules")
            print(f"[l2nof] DEPTH_LAMBDA_STAGES={_per_stage} applied to stages, counts={_counts}")

        # --- L3-ID: DEPTH_LAMBDA_BLOCKS="s:a-b" — everything OFF except blocks a..b of stage s.
        # Localizes WITHIN a stage. Stage 2 carries 86.9% of the gap across 18 blocks, so the
        # stage-level profile does not say whether that is a few blocks or all of them. ---
        _dlb = _env("DEPTH_LAMBDA_BLOCKS")
        if _dlb is not None:
            import re as _re2

            from models.encoders.DFormerv2 import GeoPriorGen  # noqa: E402

            _m2 = _re2.match(r"^(\d+):(\d+)-(\d+)$", str(_dlb).strip())
            if _m2 is None:
                raise ValueError(f"DEPTH_LAMBDA_BLOCKS must look like 's:a-b', got {_dlb!r}")
            _ts, _ba, _bb = int(_m2.group(1)), int(_m2.group(2)), int(_m2.group(3))
            _kept = 0
            _zeroed = 0
            for _name, _mod in model.named_modules():
                if not isinstance(_mod, GeoPriorGen):
                    continue
                _h = _re2.search(r"layers\.(\d+)\.blocks\.(\d+)\.", _name)
                if _h is None:
                    raise RuntimeError(f"GeoPriorGen at unexpected path {_name!r}")
                _si, _bi = int(_h.group(1)), int(_h.group(2))
                _on = (_si == _ts) and (_ba <= _bi <= _bb)
                with torch.no_grad():
                    _mod.weight[1].mul_(1.0 if _on else 0.0)
                _kept += int(_on)
                _zeroed += int(not _on)
            print(f"[l2nof] DEPTH_LAMBDA_BLOCKS={_dlb} kept={_kept} zeroed={_zeroed}")

        # --- L3-ID: POS_LAMBDA, a ZERO-INFORMATION recalibration knob (EVAL-ONLY,
        # default off => byte-identical frozen forward). Scales weight[0], the
        # POSITIONAL bias coefficient, leaving the depth term alone.
        #
        # Why this is the identifiability instrument. When depth vanishes, mask_d ->
        # 0 and the score matrix loses one of its two bias terms, so the surviving
        # positional term is left mis-calibrated relative to the regime the model was
        # fitted in. Rescaling weight[0] compensates that using a single scalar tuned
        # on TRAIN data only -- it carries ZERO bits of per-instance information about
        # the absent depth. Therefore ANY mIoU it recovers at entire_missing is, by
        # construction, "excess brittleness" (a never-trained operator configuration)
        # and CANNOT be information recovery. That yields a rigorous LOWER BOUND on
        # the brittleness component of the observed gap with NO training at all.
        #
        # Note this is a DIFFERENT mechanism from C1/IMPUTE_DEPTH_PT: C1 substituted
        # the missing bias with its train-set expectation (refuted, +0.10). This does
        # not substitute anything -- it recalibrates the terms that survive.
        _pl = _env("POS_LAMBDA")
        if _pl is not None:
            from models.encoders.DFormerv2 import GeoPriorGen  # noqa: E402
            _plam = float(_pl)
            _n_pos = 0
            for _m in model.modules():
                if isinstance(_m, GeoPriorGen):
                    with torch.no_grad():
                        _m.weight[0].mul_(_plam)
                    _n_pos += 1
            print(f"[l3id] POS_LAMBDA={_plam} scaled weight[0] on {_n_pos} GeoPriorGen modules")

        # L2-NOF C1: load the expected-bias imputation map (eval-only). When
        # IMPUTE_DEPTH_PT is set, the `mean_impute` condition feeds this fixed
        # train-set-average depth instead of zeros. No training; frozen weights.
        _imp = _env("IMPUTE_DEPTH_PT")
        if _imp is not None:
            global _IMPUTE_MAP
            _IMPUTE_MAP = torch.load(_imp, map_location="cpu")
            print(f"[l2nof] loaded impute map {tuple(_IMPUTE_MAP.shape)} from {_imp}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        model.register_forward_pre_hook(_corruption_pre_hook)

        val_loader, _ = get_val_loader(
            engine, RGBXDataset, config, val_batch_size=val_batch
        )
        # Deterministic dev/test split (prereg): dev=even, test=odd image indices.
        split = _env("EVAL_SPLIT", "full")
        base_ds = val_loader.dataset
        n_eval = len(base_ds)
        if split in ("dev", "test"):
            from torch.utils.data import DataLoader as _DL
            from torch.utils.data import Subset as _Subset

            n = len(base_ds)
            idxs = list(range(0, n, 2)) if split == "dev" else list(range(1, n, 2))
            sub = _Subset(base_ds, idxs)
            n_eval = len(sub)
            val_loader = _DL(
                sub, batch_size=val_batch, num_workers=config.num_workers,
                drop_last=False, shuffle=False, pin_memory=True,
            )
            print(f"[split] EVAL_SPLIT={split} -> {n_eval}/{n} images")
        elif split != "full":
            raise SystemExit(f"unknown EVAL_SPLIT {split}; choose full|dev|test")
        max_iters = _env("MAX_ITERS")
        if max_iters:
            val_loader = _TruncatedLoader(val_loader, int(max_iters))
            print(f"[smoke] truncated to {len(val_loader)} batches")
        n_val = n_eval

        suite = default_corruption_suite(seed=seed)
        cond_kwargs = {
            "clean": {},
            "entire_missing": {},
            "rgb_missing": {},   # EMM/RMM benchmark: the second damaged combination
            "holes": suite["holes"],
            "gaussian": suite["gaussian"],
            "mean_impute": {},  # L2-NOF C1 rung-a: feed E_train[d] when depth absent
            "half_missing": {"seed": seed},   # Gate-4 OOD held-out
            "salt_pepper": {"frac": 0.2, "seed": seed},  # Gate-4 OOD held-out
            # --- STRUCTURE-type corruptions (deployment failure modes: miscalibration,
            # RGB-D misregistration, sensor drift). Severity is env-tunable so a single
            # condition name can be swept. Defaults chosen to be clearly non-trivial. ---
            "depth_offset": {"delta": float(_env("CORR_DELTA", "0.2"))},
            "depth_gain": {"gain": float(_env("CORR_GAIN", "1.5"))},
            "depth_gamma": {"gamma": float(_env("CORR_GAMMA", "1.6"))},
            "depth_shift": {"dx": int(_env("CORR_DX", "4")), "dy": int(_env("CORR_DY", "0"))},
            # realistic GLOBAL-DEGENERACY modes — see depth_corruptions.py
            "depth_constant": {},
            "depth_saturate": {"q": float(_env("CORR_Q", "0.5"))},
            "depth_invalid": {"frac": float(_env("CORR_INVALID", "0.95")), "seed": seed},
            # decisive geometry-vs-scaffold test (see depth_corruptions.py)
            "depth_flip": {"axis": _env("CORR_AXIS", "h")},
            "depth_smooth_random": {"scale": int(_env("CORR_SCALE", "16")), "seed": seed},
        }
        for c in conditions:
            if c not in cond_kwargs:
                raise SystemExit(f"unknown condition {c}; choose from {list(cond_kwargs)}")

        results = {}
        for mode in modes:
            if mode not in ("single", "ms"):
                raise SystemExit(f"unknown mode {mode}; choose single|ms")
            results[mode] = {}
            for cond in conditions:
                _CORR["name"] = cond
                _CORR["kwargs"] = cond_kwargs[cond]
                t0 = time.time()
                # Both modes route through evaluate_msf (the correct sliding path);
                # single-scale == scales=[1.0]. evaluate()'s sliding branch passes a
                # list to slide_inference (vendored bug), so we avoid it.
                scales = MS_SCALES if mode == "ms" else [1.0]
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.float16
                ):
                    metric = evaluate_msf(
                        model, val_loader, config, device,
                        scales, True, engine, sliding=sliding,
                    )
                ious, miou = metric.compute_iou()
                acc, macc = metric.compute_pixel_acc()
                f1, mf1 = metric.compute_f1()
                dt = round(time.time() - t0, 1)
                _gate_frac = None
                if _env("MPA_ORACLE", "off") not in ("off", "", None):
                    try:
                        import oracle_gate  # already on sys.path if gate active
                        _gate_frac = oracle_gate.LAST_GATE_STATS.get("pixel_gated_frac")
                    except Exception:
                        _gate_frac = None
                results[mode][cond] = {
                    "miou": miou,
                    "macc": macc,
                    "mf1": mf1,
                    "ious": ious,
                    "seconds": dt,
                    "mpa_pixel_gated_frac": _gate_frac,
                }
                print(
                    f"[{mode:6s}] {cond:15s} miou={miou:.2f} macc={macc:.2f} "
                    f"mf1={mf1:.2f} ({dt}s)"
                )
                _CORR["name"] = "clean"  # reset between conditions

        # --- gap / recovery summary (per mode, if clean+entire_missing present) ---
        summary = {}
        for mode, res in results.items():
            if "clean" in res and "entire_missing" in res:
                gap = round(res["clean"]["miou"] - res["entire_missing"]["miou"], 2)
                summary[mode] = {
                    "clean_miou": res["clean"]["miou"],
                    "entire_missing_miou": res["entire_missing"]["miou"],
                    "clean_minus_missing_gap": gap,
                }

        payload = {
            "model_tag": model_tag,
            "checkpoint": ckpt,
            "config": CONFIG_MODULE,
            "dataset": "NYUDepthv2",
            "eval_split": split,
            "n_val_images": n_val,
            "num_classes": int(config.num_classes),
            "eval_crop_size": list(config.eval_crop_size),
            "seed": seed,
            "sliding": sliding,
            "flip": True,
            "amp": "fp16",
            "val_batch_size": val_batch,
            "ms_scales": MS_SCALES,
            "protocol": {
                "single": "evaluate_msf(scales=[1.0]): single-scale, flip, sliding, amp fp16",
                "ms": f"evaluate_msf(scales={MS_SCALES}): flip, sliding, amp fp16",
            },
            "corruption_kwargs": {
                "entire_missing": "zeros_like(normalized_depth)",
                "holes": suite["holes"],
                "gaussian": suite["gaussian"],
            },
            "results": results,
            "summary": summary,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "torch_version": torch.__version__,
        }
        os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[done] wrote {out_json}")
        for mode, s in summary.items():
            print(
                f"[summary:{mode}] clean={s['clean_miou']:.2f} "
                f"missing={s['entire_missing_miou']:.2f} gap={s['clean_minus_missing_gap']:.2f}"
            )


if __name__ == "__main__":
    main()
