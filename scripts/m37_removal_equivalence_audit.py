"""M37 — THE REMOVAL-EQUIVALENCE AUDIT, run identically on two architectures. Paper Section 5.

THE QUESTION. Missing-modality evaluation assumes the evaluation condition removes the modality
*from the model*. That is testable. An evaluation condition E is **removal-equivalent** to a
reference removal R iff it drives the model to the same decision-relevant internal state.

THE OPERATOR-LEVEL TEST, which is what this script runs. If the modality's pathway carries a neutral
element, then EVERY condition attaining that element is the same removal — however different the
inputs look. DFormerv2 injects depth solely through an attention bias `mask_d = |d_i - d_j|`, which
vanishes for a zero field AND for ANY constant field. CMNeXt-B4 consumes HHA-encoded 3-channel depth
through a parallel branch, where no such element is expected.

So the audit sweeps a family of candidate removals — zero, and constants at several levels — and
asks whether they land on the same state as the reference removal (zero), at three depths of
evidence:

    ||Z_E - Z_R|| / ||Z_R - Z_clean||   representation-level distance, RELATIVE to the removal itself
    KL(P_R || P_E)                      decision-level distance
    mIoU(E) - mIoU(R)                   the practical consequence

A condition passes iff the decision-level distance is at the self-comparison floor. **CANARY:** the
self-comparison arm (the reference against itself) must give exactly 0; if it does not, the probe
cannot resolve equivalence and no verdict is read.

GATE: each architecture must reproduce its own recorded clean mIoU band before any internal is read.

Prereg: .grill/prediction-ledger.md (2026-08-10, M37).

    python3 scripts/m37_removal_equivalence_audit.py --arch dformer --gpu 1 --n-eval 654
    python3 scripts/m37_removal_equivalence_audit.py --arch cmnext  --gpu 1 --n-eval 654
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
_CAP = {"store": None, "mode": None}

GATES = {"dformer": (53.0, 57.0), "cmnext": (52.0, 57.0)}
NCLS = 40


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def miou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    return float((tp / u)[u > 0].mean()) * 100.0


def build_conditions(depth):
    """Candidate removals: the standard protocol (zero) and constants at several levels.

    If the modality pathway has a neutral element, ALL of these are the same removal.
    """
    d = depth
    lo = float(torch.quantile(d.float().flatten(), 0.1))
    hi = float(torch.quantile(d.float().flatten(), 0.9))
    mn = float(d.float().mean())
    return {
        "zero(reference)": torch.zeros_like(d),
        "self-comparison": torch.zeros_like(d),     # CANARY: identical to the reference
        "const_mean": torch.full_like(d, mn),
        "const_q10": torch.full_like(d, lo),
        "const_q90": torch.full_like(d, hi),
        "const_+1.0": torch.full_like(d, mn + 1.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["dformer", "cmnext"], required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=654)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu

    if a.arch == "dformer":
        ROOT = PROJECT / "repos" / "DFormer"
        for _p in (ROOT, ROOT / "utils", PROJECT / "scripts"):
            sys.path.insert(0, str(_p))
        os.chdir(ROOT)
        from importlib import import_module
        cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
        cfg.pad = False
        from models.builder import EncoderDecoder
        model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
        ck = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
        sd = torch.load(ck, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd.get("state_dict", sd))
        msg = model.load_state_dict(sd, strict=False)
        got = len(sd) - len(msg.missing_keys)
        print(f"checkpoint: {got}/{len(sd)} loaded")
        if got < 0.9 * len(sd):
            raise SystemExit("REFUSING: <90% loaded")
        model.eval().cuda()
        from utils.dataloader.dataloader import get_val_loader
        from utils.dataloader.RGBXDataset import RGBXDataset
        from utils.engine.engine import Engine
        sys.argv = [sys.argv[0]]
        os.environ.setdefault("LOCAL_RANK", "0")
        engine = Engine(custom_parser=None)
        loader, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

        def unpack(b):
            return b["data"].cuda(), b["modal_x"].cuda(), b["label"].cuda().squeeze(0).long()

        def run(rgb, dep):
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            with torch.no_grad():
                return model(rgb, dep)
    else:
        ROOT = PROJECT / "repos" / "DELIVER"
        sys.path.insert(0, str(ROOT))
        os.chdir(ROOT)
        from semseg.models.cmnext import CMNeXt
        from semseg.datasets.nyu import NYU
        from semseg.augmentations_mm import get_val_augmentation
        from torch.utils.data import DataLoader
        model = CMNeXt("CMNeXt-B4", NCLS, ["img", "depth"])
        ck = ROOT / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
        sd = torch.load(str(ck), map_location="cpu")
        sd = sd.get("state_dict", sd.get("model", sd))
        msg = model.load_state_dict(sd, strict=False)
        got = len(sd) - len(msg.missing_keys)
        print(f"checkpoint: {got}/{len(sd)} loaded")
        if got < 0.9 * len(sd):
            raise SystemExit("REFUSING: <90% loaded")
        model.eval().cuda()
        ds = NYU(str(ROOT / "data" / "NYUDepthv2"), "val",
                 get_val_augmentation([480, 640]), ["img", "depth"], None)
        loader = DataLoader(ds, batch_size=1, num_workers=2, pin_memory=False)

        def unpack(b):
            sample, lbl = b
            mods = [t.cuda() for t in sample]
            return mods[0], mods[1], lbl.cuda().squeeze(0).long()

        def run(rgb, dep):
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            with torch.no_grad():
                return model([rgb, dep])

    # capture decoder input
    _orig = model.decode_head.forward

    def _h(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _h

    def cap(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        out = run(rgb, dep)
        _CAP["mode"] = None
        if isinstance(out, dict) and "pred" in out:
            out = out["pred"]
        return _CAP["store"], out

    names = None
    Z = {}
    KL = {}
    D = {}
    H = {}
    Hclean = torch.zeros(NCLS, NCLS, dtype=torch.long)
    n_done = 0

    for i, b in enumerate(loader):
        if i >= a.n_eval:
            break
        rgb, dep, lab = unpack(b)
        size = rgb.shape[-2:]
        Zc, oc = cap(rgb, dep)
        oc = F.interpolate(oc, size=size, mode="bilinear", align_corners=False)
        Hclean += hist(oc.argmax(1).squeeze(0).cpu(), lab.cpu())
        nz_clean = sum(float((t ** 2).sum()) for t in Zc) ** 0.5

        conds = build_conditions(dep)
        if names is None:
            names = list(conds)
            Z = {c: [] for c in names}
            KL = {c: [] for c in names}
            D = {c: [] for c in names}
            H = {c: torch.zeros(NCLS, NCLS, dtype=torch.long) for c in names}

        ref = None
        logp_ref = None
        for c in names:
            Zk, ok = cap(rgb, conds[c])
            ok = F.interpolate(ok, size=size, mode="bilinear", align_corners=False)
            H[c] += hist(ok.argmax(1).squeeze(0).cpu(), lab.cpu())
            if c == "zero(reference)":
                ref = Zk
                logp_ref = F.log_softmax(ok, 1)
            # distance from the REFERENCE removal, normalised by the removal's own displacement
            dz = sum(float(((k - r) ** 2).sum()) for k, r in zip(Zk, ref)) ** 0.5
            dr = sum(float(((r - z) ** 2).sum()) for r, z in zip(ref, Zc)) ** 0.5
            Z[c].append(dz / max(dr, 1e-12))
            D[c].append(dz / max(nz_clean, 1e-12))
            KL[c].append(float(F.kl_div(F.log_softmax(ok, 1), logp_ref,
                                        log_target=True, reduction="batchmean")))
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done}/{min(a.n_eval, len(loader))}", flush=True)

    clean_m = miou(Hclean)
    lo, hi = GATES[a.arch]
    print(f"\n=== GATE ({a.arch}) clean mIoU {clean_m:.2f}, band [{lo}, {hi}] ===")
    if not (lo <= clean_m <= hi):
        raise SystemExit("REFUSING: gate failed")
    print("  gate: PASS")

    rows = {}
    for c in names:
        rows[c] = {"relZ_vs_removal": sum(Z[c]) / len(Z[c]),
                   "relZ_vs_clean": sum(D[c]) / len(D[c]),
                   "kl_vs_reference": sum(KL[c]) / len(KL[c]),
                   "miou": miou(H[c])}
    ref_miou = rows["zero(reference)"]["miou"]

    canary = rows["self-comparison"]["kl_vs_reference"]
    print(f"\nCANARY self-comparison KL = {canary:.3e}  (must be exactly 0)")
    if canary != 0.0:
        print("REFUSING: the probe cannot resolve equivalence.")
        raise SystemExit(1)

    print(f"\n{'condition':20s} {'||dZ||/||removal||':>19s} {'KL vs ref':>12s} "
          f"{'mIoU':>8s} {'d mIoU':>8s}")
    for c in names:
        r = rows[c]
        print(f"{c:20s} {r['relZ_vs_removal']:19.4f} {r['kl_vs_reference']:12.2f} "
              f"{r['miou']:8.2f} {r['miou']-ref_miou:+8.2f}")

    tested = [c for c in names if c.startswith("const")]
    equivalent = [c for c in tested if rows[c]["kl_vs_reference"] == 0.0]
    print(f"\n=== VERDICT ({a.arch}) ===")
    print(f"  candidate removals tested : {len(tested)}")
    print(f"  removal-EQUIVALENT to zero: {len(equivalent)}  {equivalent}")
    print(f"  NOT equivalent            : {len(tested)-len(equivalent)}")
    worst = max((rows[c]["miou"] - ref_miou) for c in tested)
    best = min((rows[c]["miou"] - ref_miou) for c in tested)
    print(f"  mIoU spread across nominally-identical 'removals': "
          f"[{best:+.2f}, {worst:+.2f}]  = {worst-best:.2f} points")

    out = {"arch": a.arch, "clean_miou": clean_m, "n_eval": n_done, "canary_kl": canary,
           "rows": rows, "n_equivalent": len(equivalent), "n_tested": len(tested),
           "equivalent": equivalent, "miou_spread": worst - best}
    (GAPFILL / f"m37_equivalence_{a.arch}.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GAPFILL / f'm37_equivalence_{a.arch}.json'}")


if __name__ == "__main__":
    main()
