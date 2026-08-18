"""M26 — does the UNIVARIATE KL -> damage readout (M25) replicate on a second architecture?

M25 established, on DFormerv2-S: task damage is a single power law in the decoder-input-induced
output KL, damage ~ KL^1.18, R2 = 0.955 across 18 conditions / 6 families / 500x KL range, and
NOTHING else predicts the residual (support band +2.8% resid-SD, R Spearman -0.055, ||Delta||
Spearman +0.191).

THE THREAT THIS ADDRESSES. DFormerv2's decoder is LightHamHead, a rank-r NMF reconstruction. A
univariate readout could be a property of THAT decoder rather than of multimodal representations.
CMNeXt-B4 uses SegFormerHead — per-scale MLP, no NMF, no low-rank reconstruction. If the single
curve reproduces here, the readout law is not an NMF artifact.

DISANALOGY, stated in advance (inherited from M15): CMNeXt's NYU pipeline feeds HHA-encoded depth
(3-channel), not raw depth. Corruption families are ANALOGOUS, not identical, and condition-by-
condition agreement with DFormerv2 is NOT under test. What is under test is whether damage is a
univariate function of KL WITHIN this architecture.

THE GATE. This probe builds its own model, so it reproduces a known reference number before any
internal is read: prior runs on this checkpoint gave clean mIoU 54.2-55.13 at full val; the gate is
[52, 57]. A silent load failure once ran three probes on an untrained model in this campaign.

Prereg + outcome table: .grill/prediction-ledger.md (2026-08-10, M26).

    python3 scripts/m26_cross_arch_readout.py --gpu 1 --n-eval 654
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DELIVER = PROJECT / "repos" / "DELIVER"
sys.path.insert(0, str(DELIVER))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))

CKPT = DELIVER / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
NCLS = 40
GATE_LO, GATE_HI = 52.0, 57.0
_CAP = {"store": None, "mode": None}

# Deliberately WIDE: 20 conditions over 6 families, so the fit is not carried by one family.
# depth_gamma is included because it is M25's single unexplained +2.85 sigma anomaly on DFormerv2 —
# whether the excess harm replicates here is a separate, pre-registered question (P-M26d).
CONDS = {
    "constant":       lambda d, DC: DC.depth_constant(d.clone()),
    "depth_offset":   lambda d, DC: DC.depth_offset(d.clone()),
    "invalid_0.10":   lambda d, DC: DC.depth_invalid(d.clone(), frac=0.10, seed=SEED),
    "invalid_0.25":   lambda d, DC: DC.depth_invalid(d.clone(), frac=0.25, seed=SEED),
    "invalid_0.55":   lambda d, DC: DC.depth_invalid(d.clone(), frac=0.55, seed=SEED),
    "invalid_0.70":   lambda d, DC: DC.depth_invalid(d.clone(), frac=0.70, seed=SEED),
    "shift_1":        lambda d, DC: DC.depth_shift(d.clone(), dx=1, dy=0),
    "shift_4":        lambda d, DC: DC.depth_shift(d.clone(), dx=4, dy=0),
    "shift_8":        lambda d, DC: DC.depth_shift(d.clone(), dx=8, dy=0),
    "shift_16":       lambda d, DC: DC.depth_shift(d.clone(), dx=16, dy=0),
    "shift_32":       lambda d, DC: DC.depth_shift(d.clone(), dx=32, dy=0),
    "gaussian_0.1":   lambda d, DC: DC.gaussian_noise(d.clone(), sigma=0.1, seed=SEED),
    "half_missing":   lambda d, DC: DC.half_missing(d.clone(), seed=SEED),
    "flip_h":         lambda d, DC: DC.depth_flip(d.clone(), axis="h"),
    "depth_gain":     lambda d, DC: DC.depth_gain(d.clone()),
    "depth_gamma":    lambda d, DC: DC.depth_gamma(d.clone()),
    "depth_saturate": lambda d, DC: DC.depth_saturate(d.clone()),
    "salt_pepper":    lambda d, DC: DC.salt_pepper(d.clone(), seed=SEED),
    "smooth_random":  lambda d, DC: DC.depth_smooth_random(d.clone(), seed=SEED),
    "holes":          lambda d, DC: DC.rectangular_holes(d.clone(), seed=SEED),
}


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def miou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    return float((tp / u)[u > 0].mean()) * 100.0


def ols(X, Y):
    n = len(X)
    mx, my = sum(X) / n, sum(Y) / n
    b = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(X, Y)]
    sst = sum((y - my) ** 2 for y in Y)
    return a, b, res, 1 - sum(r * r for r in res) / sst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=654)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DELIVER)

    import depth_corruptions as DC
    from semseg.models.cmnext import CMNeXt
    from semseg.datasets.nyu import NYU
    from semseg.augmentations_mm import get_val_augmentation
    from torch.utils.data import DataLoader

    model = CMNeXt("CMNeXt-B4", NCLS, ["img", "depth"])
    sd = torch.load(str(CKPT), map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    msg = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(msg.missing_keys)
    print(f"checkpoint: {got}/{len(sd)} tensors loaded (missing {len(msg.missing_keys)})")
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

    transform = get_val_augmentation([480, 640])
    ds = NYU(str(DELIVER / "data" / "NYUDepthv2"), "val", transform, ["img", "depth"], None)
    dl = DataLoader(ds, batch_size=1, num_workers=2, pin_memory=False)
    print(f"val set: {len(ds)} images; evaluating {min(a.n_eval, len(ds))}")

    def cap_(sample):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(sample)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            out = _orig(list(feats))
        if isinstance(out, dict) and "pred" in out:
            out = out["pred"]
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)

    Hc = torch.zeros(NCLS, NCLS, dtype=torch.long)
    Hk = {c: torch.zeros(NCLS, NCLS, dtype=torch.long) for c in CONDS}
    acc = {c: {"kl": [], "nrm": []} for c in CONDS}
    canary = False

    for i, batch in enumerate(dl):
        if i >= a.n_eval:
            break
        sample, lbl = batch                      # dataset returns a LIST ordered by self.modals
        mods = [t.cuda() for t in sample]
        lab = lbl.cuda().squeeze(0).long()
        size = mods[0].shape[-2:]
        Zc = cap_(mods)
        zl = dec(Zc, size)
        if not canary:
            d = float((zl - dec(Zc, size)).abs().max())
            print(f"\nCANARY self-comparison: max|diff| = {d:.3e}")
            if d > 0:
                raise SystemExit("REFUSING: nondeterministic decoder")
            print(f"  decoder-input stages: {[tuple(t.shape) for t in Zc]}\n")
            canary = True
        logp_c = F.log_softmax(zl, 1)
        Hc += hist(zl.argmax(1).squeeze(0).cpu(), lab.cpu())

        for c, fn in CONDS.items():
            dep = fn(mods[1], DC)
            Zk = cap_([mods[0], dep])
            zk = dec(Zk, size)
            Hk[c] += hist(zk.argmax(1).squeeze(0).cpu(), lab.cpu())
            acc[c]["kl"].append(float(F.kl_div(F.log_softmax(zk, 1), logp_c,
                                               log_target=True, reduction="batchmean")))
            acc[c]["nrm"].append(float(sum(float(((k - z) ** 2).sum())
                                           for k, z in zip(Zk, Zc)) ** 0.5))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{min(a.n_eval, len(ds))}")

    clean = miou(Hc)
    print("\n=== REFERENCE GATE ===")
    print(f"  clean mIoU = {clean:.2f}   expected band [{GATE_LO}, {GATE_HI}]")
    gate = GATE_LO <= clean <= GATE_HI
    print(f"  gate: {'PASS' if gate else 'FAIL — internals must not be read'}")

    rows = {}
    for c in CONDS:
        mk = miou(Hk[c])
        rows[c] = {"miou": mk, "damage": clean - mk,
                   "kl": sum(acc[c]["kl"]) / len(acc[c]["kl"]),
                   "norm": sum(acc[c]["nrm"]) / len(acc[c]["nrm"])}
        rows[c]["R"] = rows[c]["kl"] / max(rows[c]["norm"] ** 2, 1e-12)

    print(f"\n{'condition':16s} {'mIoU':>7s} {'damage':>8s} {'KL':>11s} {'||D||':>9s} {'R':>8s}")
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        r = rows[c]
        print(f"{c:16s} {r['miou']:7.2f} {r['damage']:8.3f} {r['kl']:11.1f} "
              f"{r['norm']:9.1f} {r['R']:8.4f}")

    use = [c for c in rows if rows[c]["kl"] > 100 and rows[c]["damage"] > 0]
    X = [math.log(rows[c]["kl"]) for c in use]
    Y = [math.log(rows[c]["damage"]) for c in use]
    a0, b0, res, r2 = ols(X, Y)
    sd = (sum(r * r for r in res) / (len(res) - 1)) ** 0.5
    print(f"\n--- READOUT FIT (n={len(use)}) ---")
    print(f"  log damage = {a0:.4f} + {b0:.4f} log KL    R2 = {r2:.4f}    resid SD = {sd:.4f}")
    print(f"  DFormerv2 reference (M25): exponent 1.1817, R2 0.9553, resid SD 0.2808")

    # Does ||Delta|| or R add anything beyond KL? (M25 said no, on the other architecture.)
    def extra(key):
        V = [rows[c][key] for c in use]
        mv = sum(V) / len(V); mr = sum(res) / len(res)
        num = sum((v - mv) * (r - mr) for v, r in zip(V, res))
        den = (sum((v - mv) ** 2 for v in V) * sum((r - mr) ** 2 for r in res)) ** 0.5
        return num / den if den > 0 else float("nan")
    print(f"  Pearson(R, residual)      = {extra('R'):+.3f}")
    print(f"  Pearson(||D||, residual)  = {extra('norm'):+.3f}")

    resid = dict(zip(use, res))
    gam = resid.get("depth_gamma")
    print(f"\n  depth_gamma residual = {gam:+.4f} ({gam/sd:+.2f} sigma)   "
          f"[DFormerv2: +0.8010, +2.85 sigma]" if gam is not None else "  depth_gamma excluded")

    p_a = r2 >= 0.85
    p_b = 0.7 <= b0 <= 1.7
    p_d = (gam is not None) and (gam / sd) >= 1.5
    print(f"\n  P-M26a (R2 >= 0.85):        {'HIT' if p_a else 'MISS'}   ({r2:.4f})")
    print(f"  P-M26b (exponent in [0.7,1.7]): {'HIT' if p_b else 'MISS'}   ({b0:.4f})")
    print(f"  P-M26d (depth_gamma >= +1.5 sigma): {'HIT' if p_d else 'MISS'}")

    out = {"clean_miou": clean, "gate": gate, "n_eval": min(a.n_eval, len(ds)),
           "rows": rows, "fit": {"a": a0, "b": b0, "r2": r2, "resid_sd": sd},
           "resid": resid, "P_M26a": bool(p_a), "P_M26b": bool(p_b), "P_M26d": bool(p_d)}
    GAPFILL.mkdir(parents=True, exist_ok=True)
    (GAPFILL / "m26_cross_arch_readout.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GAPFILL / 'm26_cross_arch_readout.json'}")


if __name__ == "__main__":
    main()
