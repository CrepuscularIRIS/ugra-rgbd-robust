"""M11 — does the shuffle collapse live in HOW MANY pixels flip, or in WHICH?

Prereg: plan/PREREG-M11-flips-vs-miou.md. Filed before any number.

THE CONTRADICTION M10 LEFT. First-order predicted flip count does NOT collapse under spatial shuffle
(0.511 / 0.608) while real mIoU damage DOES (0.50 -> -0.03, 2.85 -> 0.34, M3). mIoU damage IS argmax
flips, so exactly one of these holds:

  R-lin-fail    TRUE full-step flips DO collapse; only the linearization failed.
  R-composition TRUE flips do NOT collapse either, and the collapse lives in WHICH pixels flip --
                mIoU is CLASS-AVERAGED over 40 classes, so flips in rare classes cost far more.

R-composition is the uncomfortable branch: it would make the campaign's most load-bearing causal
result partly a property of the METRIC's class weighting rather than of the representation.

THE DESIGN IS PAIRED. Flip count and mIoU come from the SAME decoder forward passes on the SAME
images, so they cannot diverge except through the class weighting the hypothesis names. That pairing
is the whole severity of the test.

    python3 scripts/m11_flips_vs_miou.py --gpu 1 --n-eval 30
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
NCLS = 40
IGNORE = 255
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402

# the two conditions M3 measured the shuffle collapse on, plus two in-support contrasts
CONDS = ["dconst", "invalid_0.10", "depth_saturate", "invalid_0.55"]


def hist(pred, lab):
    """40x40 confusion matrix on valid pixels."""
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def miou(h):
    h = h.double()
    tp = h.diag()
    union = h.sum(0) + h.sum(1) - tp
    iou = tp / union
    valid = union > 0
    return float(iou[valid].mean()) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=30)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(miss.missing_keys)
    print(f"checkpoint: {got}/{len(sd)} tensors loaded")
    if got < 0.9 * len(sd):
        raise SystemExit("REFUSING: <90% loaded")
    model.eval().cuda()
    _orig = model.decode_head.forward

    def _hook(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _hook

    def cap_(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec(feats, size):
        """Decoder logits, bases PINNED, upsampled exactly as builder.encode_decode does."""
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            out = _orig(list(feats))
        if isinstance(out, dict) and "pred" in out:
            out = out["pred"]
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    g = torch.Generator().manual_seed(SEED)
    H = {"clean": torch.zeros(NCLS, NCLS, dtype=torch.long)}
    for c in CONDS:
        for arm in ("real", "shuf", "uni"):
            H[f"{c}|{arm}"] = torch.zeros(NCLS, NCLS, dtype=torch.long)
    flips = {f"{c}|{arm}": [] for c in CONDS for arm in ("real", "shuf", "uni")}
    clsflip = {f"{c}|{arm}": torch.zeros(NCLS, dtype=torch.long)
               for c in CONDS for arm in ("real", "shuf")}

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        canary = False
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            lab = b["label"].cuda().squeeze(0)
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)

            if not canary:
                d_self = float((zl - dec(Zc, size)).abs().max())
                print(f"\nCANARY self-comparison: max|diff| = {d_self:.3e}")
                if d_self > 0:
                    raise SystemExit("REFUSING: decoder nondeterministic; every number would be noise")
                print("  bases pinned. proceeding.\n")
                canary = True

            arg_c = zl.argmax(1).squeeze(0)
            H["clean"] += hist(arg_c.cpu(), lab.cpu())

            for c in CONDS:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                delta = [k - z for k, z in zip(Zk, Zc)]
                sh, un = [], []
                for d in delta:
                    B, C, Hh, Ww = d.shape
                    idx = torch.randperm(Hh * Ww, generator=g).to(d.device)
                    sh.append(d.reshape(B, C, Hh * Ww)[:, :, idx].reshape(B, C, Hh, Ww))
                    un.append(d.mean(dim=(2, 3), keepdim=True).expand_as(d))

                for arm, fs in (("real", Zk),
                                ("shuf", [z + d for z, d in zip(Zc, sh)]),
                                ("uni",  [z + d for z, d in zip(Zc, un)])):
                    zz = dec(fs, size)
                    ar = zz.argmax(1).squeeze(0)
                    H[f"{c}|{arm}"] += hist(ar.cpu(), lab.cpu())
                    fl = (ar != arg_c)
                    flips[f"{c}|{arm}"].append(float(fl.float().mean()))
                    if arm in ("real", "shuf"):
                        v = fl & (lab >= 0) & (lab < NCLS)
                        if v.any():
                            clsflip[f"{c}|{arm}"] += torch.bincount(
                                lab[v].long().cpu(), minlength=NCLS)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.n_eval} images")

    base = miou(H["clean"])
    print(f"\nclean mIoU on this {a.n_eval}-image subset: {base:.2f}\n")
    print(f"{'condition':16s} {'arm':>5s} {'dmIoU':>8s} {'flip%':>8s} "
          f"{'flips/dmIoU':>12s}")
    rows = {}
    for c in CONDS:
        for arm in ("real", "shuf", "uni"):
            k = f"{c}|{arm}"
            dm = base - miou(H[k])
            fp = float(np.mean(flips[k])) * 100
            rows[k] = {"dmiou": dm, "flip_pct": fp,
                       "flips_per_dmiou": fp / dm if abs(dm) > 1e-6 else float("nan")}
            print(f"{c:16s} {arm:>5s} {dm:8.2f} {fp:8.3f} "
                  f"{rows[k]['flips_per_dmiou']:12.2f}")
        print()

    print("--- THE PAIRED COMPARISON (same forward passes) ---")
    print(f"{'condition':16s} {'dmIoU ratio':>12s} {'flip ratio':>11s} {'divergence':>11s}")
    verdicts = {}
    for c in CONDS:
        r_m = rows[f"{c}|shuf"]["dmiou"] / rows[f"{c}|real"]["dmiou"] \
            if abs(rows[f"{c}|real"]["dmiou"]) > 1e-6 else float("nan")
        r_f = rows[f"{c}|shuf"]["flip_pct"] / max(rows[f"{c}|real"]["flip_pct"], 1e-9)
        verdicts[c] = {"dmiou_ratio": r_m, "flip_ratio": r_f,
                       "divergence": r_f / r_m if abs(r_m) > 1e-6 else float("nan")}
        print(f"{c:16s} {r_m:12.3f} {r_f:11.3f} {verdicts[c]['divergence']:11.2f}x")

    ref = [c for c in ("dconst", "invalid_0.10") if c in verdicts]
    p_a = all(verdicts[c]["dmiou_ratio"] <= 0.35 for c in ref)
    p_b = all(verdicts[c]["flip_ratio"] <= 0.35 for c in ref)
    p_c = all(verdicts[c]["divergence"] >= 2.0 for c in ref)

    conc = {}
    for c in CONDS:
        for arm in ("real", "shuf"):
            v = clsflip[f"{c}|{arm}"].double()
            conc[f"{c}|{arm}"] = float(v.sort(descending=True).values[:3].sum() / max(v.sum(), 1))
    p_d = all(conc[f"{c}|real"] > conc[f"{c}|shuf"] for c in ref)

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M11a mIoU collapse replicates (<=0.35) -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M11b TRUE flips collapse (<=0.35)      -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M11c divergence >= 2x                  -> {'HIT' if p_c else 'MISS'}")
    print(f"P-M11d real flips more class-concentrated than shuffled -> "
          f"{'HIT' if p_d else 'MISS'}")
    for c in ref:
        print(f"    {c}: top-3-class share  real {conc[f'{c}|real']:.3f}  "
              f"shuf {conc[f'{c}|shuf']:.3f}")

    print(f"\n--- NEGATIVE CONTROL (uniform arm must be small) ---")
    for c in ref:
        print(f"  {c:16s} uniform dmIoU {rows[f'{c}|uni']['dmiou']:+.2f}  "
              f"flip% {rows[f'{c}|uni']['flip_pct']:.3f}")

    print(f"\n--- VERDICT ---")
    if not p_a:
        print("  P-M11a MISSED: the mIoU collapse does not replicate on this subset. The SUBSET is")
        print("  invalid, not the hypothesis. Nothing else here counts -- rerun with more images.")
    elif p_b:
        print("  R-COMPOSITION IS KILLED. True full-step flips collapse under shuffle; M10 killed")
        print("  only the LINEARIZATION. The shuffle collapse is representational, M3 stands")
        print("  unqualified, and R-assign (non-linear, assignment-dependent) leads.")
    elif p_c:
        print("  THE UNQUALIFIED READING OF M3 IS KILLED. True flips do NOT collapse but mIoU does:")
        print("  the collapse is substantially CLASS COMPOSITION / metric weighting, not purely")
        print("  representational. Levels 1-2 evidence needs a scope edit and a stated")
        print("  construct-validity limit. This is the branch I filed as most likely.")
    else:
        print("  BOTH RIVALS KILLED. Flips and mIoU disagree with no composition explanation ->")
        print("  route to checking-the-instrument, NOT to a new mechanism.")

    o = GAPFILL / "m11_flips_vs_miou.json"
    o.write_text(json.dumps({"clean_miou_subset": base, "rows": rows, "verdicts": verdicts,
                             "class_concentration": conc, "n_eval": a.n_eval,
                             "P_M11a": p_a, "P_M11b": p_b, "P_M11c": p_c,
                             "P_M11d": p_d}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
