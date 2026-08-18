"""M29 — WHY IS depth_gamma A +2.85 SIGMA OUTLIER? The class-averaging hypothesis.

THE ANOMALY. M25 found the KL -> damage readout univariate (R2 0.955) with ONE real exception:
`depth_gamma` sits at +2.85 sigma (damage 0.470 vs predicted 0.211 -- 2.3x excess harm per nat) and
it survives a seed-noise check at 3.95x its own 3-seed spread, so it is not measurement noise. M26
showed it does NOT replicate on CMNeXt (-0.78 sigma), so it is DFormerv2-specific.

THE HYPOTHESIS (H-gamma3). KL is a PER-PIXEL average over the output distribution. mIoU is a
PER-CLASS mean. These weight the image differently: damage concentrated in a few classes -- especially
rare ones occupying few pixels -- moves mIoU a lot while barely moving average KL. So the outlier may
be a property of the METRIC, not of the model or the corruption.

If true this is not a nuisance finding: it states a BOUNDARY CONDITION on the principle. The readout
would be univariate in a per-pixel damage measure, and any class-averaged metric adds a second
coordinate (damage concentration) that the law does not contain.

RIVALS this must beat:
  H-gamma1  small-damage artifact       -- already refuted by the seed-noise check (3.95x)
  H-gamma2  gamma is a monotone remap of depth, so mask_d = |d_i - d_j| is rescaled NON-UNIFORMLY in
            space -- a spatially varying gain on the attention bias, structurally unlike the others.
            NOT tested here; if P-M29c hits, H-gamma2 is not needed.

THE DECISIVE TEST. Re-fit the readout law using PIXEL ACCURACY drop (a per-pixel measure, matched to
KL's weighting) in place of mIoU drop. If the anomaly is class-averaging, depth_gamma's residual
COLLAPSES. If it persists, the anomaly is about the model and H-gamma2 becomes live.

GATE. Reproduces this checkpoint's known clean mIoU (55.18 at full val); band [53, 57].

Prereg + outcome table: .grill/prediction-ledger.md (2026-08-10, M29).

    python3 scripts/m29_gamma_anomaly.py --gpu 1 --n-eval 654
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

from m1_mig_full import COND, corrupt  # noqa: E402


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def per_class_iou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    iou = torch.where(u > 0, tp / u, torch.full_like(u, float("nan")))
    return iou, u


def miou(h):
    iou, u = per_class_iou(h)
    return float(iou[u > 0].mean()) * 100.0


def pixacc(h):
    h = h.double()
    return float(h.diag().sum() / h.sum()) * 100.0


def ols(X, Y):
    n = len(X)
    mx, my = sum(X) / n, sum(Y) / n
    b = sum((x - mx) * (y - my) for x, y in zip(X, Y)) / sum((x - mx) ** 2 for x in X)
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(X, Y)]
    sst = sum((y - my) ** 2 for y in Y)
    sd = (sum(r * r for r in res) / (n - 2)) ** 0.5
    return a, b, res, 1 - sum(r * r for r in res) / sst, sd


def spearman(u, v):
    def rk(z):
        s = sorted(range(len(z)), key=lambda i: z[i])
        r = [0] * len(z)
        for i, j in enumerate(s):
            r[j] = i + 1
        return r
    a, b = rk(u), rk(v)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
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

    m14 = json.loads((GAPFILL / "m14_holdout_rayleigh.json").read_text())["rows"]
    names = [c for c in COND if c in m14]
    print(f"conditions: {len(names)}")

    Hc = torch.zeros(NCLS, NCLS, dtype=torch.long)
    Hk = {c: torch.zeros(NCLS, NCLS, dtype=torch.long) for c in names}
    kl = {c: [] for c in names}
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
                d = float((zl - dec(Zc, size)).abs().max())
                if d > 0:
                    raise SystemExit(f"REFUSING: nondeterministic decoder ({d:.3e})")
                print(f"CANARY self-comparison max|diff| = {d:.3e}  OK")
                canary = True
            logp = F.log_softmax(zl, 1)
            Hc += hist(zl.argmax(1).squeeze(0).cpu(), lab.cpu())
            for c in names:
                zk = dec(cap_(rgb, corrupt(dep, COND[c][1])), size)
                Hk[c] += hist(zk.argmax(1).squeeze(0).cpu(), lab.cpu())
                kl[c].append(float(F.kl_div(F.log_softmax(zk, 1), logp,
                                            log_target=True, reduction="batchmean")))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{min(a.n_eval, len(vl))}")

    clean_m, clean_p = miou(Hc), pixacc(Hc)
    iou_c, u_c = per_class_iou(Hc)
    print(f"\n=== REFERENCE GATE ===\n  clean mIoU {clean_m:.2f} (band [{GATE_LO},{GATE_HI}])  "
          f"pixel acc {clean_p:.2f}")
    gate = GATE_LO <= clean_m <= GATE_HI
    print(f"  gate: {'PASS' if gate else 'FAIL — internals must not be read'}")
    if not gate:
        raise SystemExit("REFUSING: gate failed")

    rows = {}
    for c in names:
        iou_k, _ = per_class_iou(Hk[c])
        drops = []
        for j in range(NCLS):
            if u_c[j] > 0 and not math.isnan(float(iou_c[j])) and not math.isnan(float(iou_k[j])):
                drops.append(max(0.0, float(iou_c[j]) - float(iou_k[j])))
        drops.sort(reverse=True)
        tot = sum(drops)
        rows[c] = {
            "miou": miou(Hk[c]), "dmiou": clean_m - miou(Hk[c]),
            "pixacc": pixacc(Hk[c]), "dpix": clean_p - pixacc(Hk[c]),
            "kl": sum(kl[c]) / len(kl[c]),
            "top3_share": (sum(drops[:3]) / tot) if tot > 1e-12 else float("nan"),
            "n_classes_dropping": sum(1 for d in drops if d > 1e-6),
        }

    print(f"\n{'condition':22s} {'KL':>10s} {'dmIoU':>7s} {'dPixAcc':>8s} {'top3sh':>7s} {'ncls':>5s}")
    for c in sorted(rows, key=lambda x: -rows[x]["kl"]):
        r = rows[c]
        print(f"{c:22s} {r['kl']:10.0f} {r['dmiou']:7.3f} {r['dpix']:8.3f} "
              f"{r['top3_share']:7.3f} {r['n_classes_dropping']:5d}")

    def fit(key):
        use = [c for c in rows if rows[c]["kl"] > 100 and rows[c][key] > 0]
        X = [math.log(rows[c]["kl"]) for c in use]
        Y = [math.log(rows[c][key]) for c in use]
        a0, b0, res, r2, sd = ols(X, Y)
        return use, dict(zip(use, res)), r2, sd, b0

    print("\n--- READOUT FIT, TWO DAMAGE MEASURES ---")
    out = {}
    for key, lab_ in (("dmiou", "mIoU drop (per-CLASS mean)"), ("dpix", "pixel-acc drop (per-PIXEL)")):
        use, res, r2, sd, b0 = fit(key)
        g = res.get("depth_gamma")
        out[key] = {"r2": r2, "resid_sd": sd, "exponent": b0,
                    "gamma_resid": g, "gamma_sigma": (g / sd) if g is not None else None,
                    "resid": res}
        print(f"  {lab_:32s} n={len(use):2d}  R2={r2:.4f}  sd={sd:.4f}  exp={b0:.3f}  "
              f"depth_gamma = {g:+.4f} ({g/sd:+.2f} sigma)")

    conc = [rows[c]["top3_share"] for c in out["dmiou"]["resid"]]
    rsd = [out["dmiou"]["resid"][c] for c in out["dmiou"]["resid"]]
    sp = spearman(conc, rsd)
    print(f"\n  Spearman(top-3 class-drop share, mIoU-readout residual) = {sp:+.3f}")

    gs_m = out["dmiou"]["gamma_sigma"]
    gs_p = out["dpix"]["gamma_sigma"]
    p_a = rows["depth_gamma"]["top3_share"] >= max(
        rows[c]["top3_share"] for c in rows if c != "depth_gamma" and not math.isnan(rows[c]["top3_share"]))
    p_b = sp >= 0.5
    p_c = abs(gs_p) < 1.5 and abs(gs_m) >= 1.5
    print(f"\n  P-M29a (depth_gamma most class-concentrated): {'HIT' if p_a else 'MISS'}")
    print(f"  P-M29b (Spearman concentration~residual >= .5): {'HIT' if p_b else 'MISS'}  ({sp:+.3f})")
    print(f"  P-M29c (anomaly COLLAPSES under pixel-acc):    {'HIT' if p_c else 'MISS'}"
          f"   mIoU {gs_m:+.2f}s -> pix {gs_p:+.2f}s")

    (GAPFILL / "m29_gamma_anomaly.json").write_text(json.dumps(
        {"gate": gate, "clean_miou": clean_m, "clean_pixacc": clean_p, "n_eval": a.n_eval,
         "rows": rows, "fits": out, "spearman_conc_resid": sp,
         "P_M29a": bool(p_a), "P_M29b": bool(p_b), "P_M29c": bool(p_c)}, indent=1))
    print(f"\nwrote {GAPFILL / 'm29_gamma_anomaly.json'}")


if __name__ == "__main__":
    main()
