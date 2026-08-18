"""M30 — THE DECISIVE TEST: is the KL -> damage readout univariate, or does FAMILY matter at matched KL?

WHY THIS EXISTS. The prior-art examiner named arXiv:1810.07014 (KL/log-loss upper-bounds regret for
proper convex losses) as the most dangerous paper, and named the experiment that separates our
EMPIRICAL claim from their generic BOUND:

    matched-KL stratification across perturbation families -- within narrow KL bins, does family have
    zero residual effect on realized damage?

Revision 37 ran it on 4 ad-hoc pairs and it went AGAINST the strong claim: mean 43% damage difference
at matched KL, worst case 1.7x at KL matched to 2.5%. But n = 4, non-independent. This is the proper
version.

THE DESIGN. Instead of hunting for lucky matched pairs, sweep a DOSE parameter within each family.
Every family then traces its own curve in (KL, damage), and matched-KL comparison is done by
interpolation along those curves rather than by coincidence.

    If the readout is univariate, ALL family curves collapse onto ONE curve.
    If family matters, the curves are OFFSET from each other.

9 families x 5 doses = 45 conditions, full val (654), one checkpoint, KL and damage from the SAME
run (Revision 36's lesson: cross-run transport degraded M25's numbers).

Also recorded per condition: damage BREADTH (number of classes whose IoU drops) and top-3
concentration -- Revision 37 identified breadth as the candidate missing coordinate, with a sign
OPPOSITE to the concentration hypothesis that was refuted in M29.

GATE. Clean mIoU in [53, 57] (this checkpoint records 55.18; M29 measured 54.86 at full val).

Prereg + outcome table: .grill/prediction-ledger.md (2026-08-10, M30).

    python3 scripts/m30_matched_kl_stratification.py --gpu 1 --n-eval 654
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
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
sys.path.insert(0, str(PROJECT / "scripts"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
NCLS = 40
GATE_LO, GATE_HI = 53.0, 57.0
_CAP = {"store": None, "mode": None}


def build_conditions(DC):
    """9 families x 5 doses. Dose ranges chosen to span a wide KL range within each family."""
    C = {}
    for f in (0.05, 0.12, 0.25, 0.45, 0.70):
        C[f"invalid|{f}"] = ("invalid", f, lambda d, f=f: DC.depth_invalid(d.clone(), frac=f, seed=SEED))
    for f in (0.05, 0.12, 0.25, 0.40, 0.60):
        C[f"saltpepper|{f}"] = ("saltpepper", f, lambda d, f=f: DC.salt_pepper(d.clone(), frac=f, seed=SEED))
    for s in (0.02, 0.05, 0.10, 0.20, 0.35):
        C[f"gaussian|{s}"] = ("gaussian", s, lambda d, s=s: DC.gaussian_noise(d.clone(), sigma=s, seed=SEED))
    for g in (1.15, 1.35, 1.7, 2.2, 3.0):
        C[f"gain|{g}"] = ("gain", g, lambda d, g=g: DC.depth_gain(d.clone(), gain=g))
    for g in (1.15, 1.35, 1.7, 2.2, 3.0):
        C[f"gamma|{g}"] = ("gamma", g, lambda d, g=g: DC.depth_gamma(d.clone(), gamma=g))
    for q in (0.85, 0.70, 0.55, 0.40, 0.25):
        C[f"saturate|{q}"] = ("saturate", q, lambda d, q=q: DC.depth_saturate(d.clone(), q=q))
    for k in (1, 2, 4, 8, 16):
        C[f"shift|{k}"] = ("shift", k, lambda d, k=k: DC.depth_shift(d.clone(), dx=k, dy=0))
    for n, hf in ((1, 0.10), (2, 0.15), (3, 0.20), (5, 0.22), (8, 0.25)):
        C[f"holes|{n}-{hf}"] = ("holes", n * hf, lambda d, n=n, hf=hf: DC.rectangular_holes(
            d.clone(), num_holes=n, hole_frac=hf, seed=SEED))
    for s in (4, 8, 16, 32, 64):
        C[f"smooth|{s}"] = ("smooth", s, lambda d, s=s: DC.depth_smooth_random(d.clone(), scale=s, seed=SEED))
    return C


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(), minlength=NCLS ** 2).reshape(NCLS, NCLS)


def per_class_iou(h):
    h = h.double(); tp = h.diag(); u = h.sum(0) + h.sum(1) - tp
    return torch.where(u > 0, tp / u, torch.full_like(u, float("nan"))), u


def miou(h):
    iou, u = per_class_iou(h)
    return float(iou[u > 0].mean()) * 100.0


def pixacc(h):
    h = h.double()
    return float(h.diag().sum() / h.sum()) * 100.0


def ols(X, Y):
    n = len(X); mx = sum(X) / n; my = sum(Y) / n
    b = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(X, Y)]
    sst = sum((y - my) ** 2 for y in Y)
    sd = (sum(r * r for r in res) / max(n - 2, 1)) ** 0.5
    return a, b, res, (1 - sum(r * r for r in res) / sst if sst > 0 else float("nan")), sd


def spearman(u, v):
    def rk(z):
        s = sorted(range(len(z)), key=lambda i: z[i]); r = [0] * len(z)
        for i, j in enumerate(s):
            r[j] = i + 1
        return r
    a, b = rk(u), rk(v); n = len(a); ma = sum(a) / n; mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=654)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    import depth_corruptions as DC
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

    def _h(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _h

    def cap_(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    CONDS = build_conditions(DC)
    print(f"conditions: {len(CONDS)} ({len(set(v[0] for v in CONDS.values()))} families x 5 doses)")

    Hc = torch.zeros(NCLS, NCLS, dtype=torch.long)
    Hk = {c: torch.zeros(NCLS, NCLS, dtype=torch.long) for c in CONDS}
    kl = {c: [] for c in CONDS}
    canary = False

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            lab = b["label"].cuda().squeeze(0).long()
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)
            if not canary:
                d0 = float((zl - dec(Zc, size)).abs().max())
                if d0 > 0:
                    raise SystemExit(f"REFUSING: nondeterministic decoder ({d0:.3e})")
                print(f"CANARY max|diff| = {d0:.3e}  OK")
                canary = True
            logp = F.log_softmax(zl, 1)
            Hc += hist(zl.argmax(1).squeeze(0).cpu(), lab.cpu())
            for c, (famname, dose, fn) in CONDS.items():
                zk = dec(cap_(rgb, fn(dep)), size)
                Hk[c] += hist(zk.argmax(1).squeeze(0).cpu(), lab.cpu())
                kl[c].append(float(F.kl_div(F.log_softmax(zk, 1), logp,
                                            log_target=True, reduction="batchmean")))
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{min(a.n_eval, len(vl))}", flush=True)

    clean_m, clean_p = miou(Hc), pixacc(Hc)
    iou_c, u_c = per_class_iou(Hc)
    print(f"\n=== REFERENCE GATE ===\n  clean mIoU {clean_m:.2f} band [{GATE_LO},{GATE_HI}]  pixacc {clean_p:.2f}")
    if not (GATE_LO <= clean_m <= GATE_HI):
        raise SystemExit("REFUSING: gate failed")
    print("  gate: PASS")

    rows = {}
    for c, (famname, dose, _) in CONDS.items():
        iou_k, _ = per_class_iou(Hk[c])
        drops = [max(0.0, float(iou_c[j]) - float(iou_k[j])) for j in range(NCLS)
                 if u_c[j] > 0 and not math.isnan(float(iou_c[j])) and not math.isnan(float(iou_k[j]))]
        drops.sort(reverse=True)
        tot = sum(drops)
        rows[c] = {"family": famname, "dose": dose, "kl": sum(kl[c]) / len(kl[c]),
                   "dmiou": clean_m - miou(Hk[c]), "dpix": clean_p - pixacc(Hk[c]),
                   "breadth": sum(1 for d in drops if d > 1e-6),
                   "top3_share": (sum(drops[:3]) / tot) if tot > 1e-12 else float("nan")}

    print(f"\n{'condition':20s} {'fam':11s} {'KL':>10s} {'dmIoU':>7s} {'dPix':>7s} {'brdth':>6s}")
    for c in sorted(rows, key=lambda x: rows[x]["kl"]):
        r = rows[c]
        print(f"{c:20s} {r['family']:11s} {r['kl']:10.0f} {r['dmiou']:7.3f} {r['dpix']:7.3f} {r['breadth']:6d}")

    out = {"gate": True, "clean_miou": clean_m, "clean_pixacc": clean_p,
           "n_eval": a.n_eval, "rows": rows, "analysis": {}}

    for key in ("dmiou", "dpix"):
        use = [c for c in rows if rows[c]["kl"] > 100 and rows[c][key] > 0]
        fams = sorted(set(rows[c]["family"] for c in use))
        print(f"\n===== {key} =====")

        # (a) per-family fits
        perfam = {}
        for f in fams:
            g = [c for c in use if rows[c]["family"] == f]
            if len(g) < 3:
                continue
            X = [math.log(rows[c]["kl"]) for c in g]; Y = [math.log(rows[c][key]) for c in g]
            a0, b0, _, r2, sd = ols(X, Y)
            perfam[f] = {"a": a0, "b": b0, "r2": r2, "sd": sd, "n": len(g),
                         "lo": min(X), "hi": max(X)}
            print(f"  family {f:11s} n={len(g)}  slope={b0:+.3f}  R2={r2:.4f}")
        p_a = all(v["r2"] >= 0.95 for v in perfam.values())

        # (b) pooled vs pooled+family-intercept
        X = [math.log(rows[c]["kl"]) for c in use]; Y = [math.log(rows[c][key]) for c in use]
        _, _, res, r2p, sdp = ols(X, Y)
        resid = dict(zip(use, res))
        fmean = {f: (sum(resid[c] for c in use if rows[c]["family"] == f)
                     / max(sum(1 for c in use if rows[c]["family"] == f), 1)) for f in fams}
        res2 = [resid[c] - fmean[rows[c]["family"]] for c in use]
        m2 = sum(res2) / len(res2)
        sd2 = (sum((r - m2) ** 2 for r in res2) / max(len(res2) - 1 - len(fams), 1)) ** 0.5
        red = 100 * (1 - sd2 / sdp)
        print(f"  pooled R2={r2p:.4f} sd={sdp:.4f}  ->  +family intercepts sd={sd2:.4f}  "
              f"reduction={red:+.1f}%")
        p_b = red < 20

        # (c) matched-KL max cross-family damage ratio, by interpolation
        ratios = []
        grid = [min(X) + (max(X) - min(X)) * t / 20 for t in range(21)]
        for lk in grid:
            vals = [(f, v["a"] + v["b"] * lk) for f, v in perfam.items() if v["lo"] <= lk <= v["hi"]]
            if len(vals) >= 3:
                lo = min(v for _, v in vals); hi = max(v for _, v in vals)
                ratios.append((math.exp(hi - lo), math.exp(lk), [f for f, _ in vals]))
        if ratios:
            worst = max(ratios, key=lambda z: z[0])
            med = sorted(r[0] for r in ratios)[len(ratios) // 2]
            print(f"  matched-KL cross-family damage ratio: median {med:.2f}x, "
                  f"WORST {worst[0]:.2f}x at KL={worst[1]:.0f} among {len(worst[2])} families")
            p_c = worst[0] < 1.5
        else:
            med = worst = None; p_c = False
            print("  no KL grid point covered by >=3 families")

        # (d) breadth as the missing coordinate
        br = [rows[c]["breadth"] for c in use]
        sp = spearman(br, [resid[c] for c in use])
        print(f"  Spearman(breadth, residual) = {sp:+.3f}")
        p_d = abs(sp) >= 0.5

        print(f"\n  P-M30a (every family R2 >= .95):        {'HIT' if p_a else 'MISS'}")
        print(f"  P-M30b (family intercepts buy < 20%):   {'HIT' if p_b else 'MISS'}  ({red:+.1f}%)")
        print(f"  P-M30c (worst matched-KL ratio < 1.5x): {'HIT' if p_c else 'MISS'}"
              f"  ({worst[0]:.2f}x)" if ratios else "  P-M30c: no data")
        print(f"  P-M30d (|Spearman(breadth,resid)|>=.5): {'HIT' if p_d else 'MISS'}  ({sp:+.3f})")

        out["analysis"][key] = {"perfam": perfam, "pooled_r2": r2p, "pooled_sd": sdp,
                                "fam_sd": sd2, "reduction_pct": red,
                                "fam_means": fmean, "resid": resid,
                                "matched_median": med,
                                "matched_worst": (worst[0] if ratios else None),
                                "matched_worst_kl": (worst[1] if ratios else None),
                                "spearman_breadth": sp,
                                "P_M30a": bool(p_a), "P_M30b": bool(p_b),
                                "P_M30c": bool(p_c), "P_M30d": bool(p_d)}

    (GAPFILL / "m30_matched_kl.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {GAPFILL / 'm30_matched_kl.json'}")


if __name__ == "__main__":
    main()
