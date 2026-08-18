"""M24 — the L4 comparative against RIVAL-1: registration vs FEATURE SEPARABILITY.

RIVAL-1, from the primary source (arXiv:2303.15488, NeurIPS 2023, "On the Importance of Feature
Separability in Predicting Out-Of-Distribution Error"):

    "we show that a large domain gap not necessarily leads to a low test accuracy...
     inter-class dispersion is strongly correlated with the model accuracy, while intra-class
     compactness does not reflect the generalization performance on OOD data."

WHY IT IS A REAL RIVAL. It makes the SAME kind of claim we do — displacement magnitude is the wrong
predictor — but names a DIFFERENT quantity: inter-class SEPARABILITY of the features, not
REGISTRATION of the displacement. Both accounts explain "big gap, little damage." They are
distinguishable only by measurement, which is what this does.

CONSTRUCTION. Their dispersion is dataset-level over class centroids under pseudo-labels. Adapted to
dense prediction: per-pixel decoder-input features, pseudo-label = argmax of the model's own
prediction, centroid per present class, dispersion = mean pairwise centroid distance normalised by
mean within-class spread (a Fisher-style ratio; higher = more separable). Measured clean and
corrupted; the PREDICTOR is the DROP.

Prereg + outcome table + severity: .grill/prediction-ledger.md (2026-08-10, M24).

    python3 scripts/m24_rival1_dispersion.py --gpu 1 --n-eval 16
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
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt, spearman  # noqa: E402


def dispersion(feat, pseudo, fixed_classes):
    """Fisher-style separability over a FIXED class set, so it is comparable across conditions.

    RIVAL-1 computes dispersion over class centroids in a FIXED label space. The first version of
    this probe let the class set float with the corrupted pseudo-labels, so a heavily corrupted model
    predicting fewer classes could show HIGHER dispersion — which inverted the statistic. The class
    set is now pinned to the CLEAN prediction's classes for every condition.
    """
    C, H, W = feat.shape
    f = feat.reshape(C, -1).T
    lab = pseudo.reshape(-1)
    cents, spreads = [], []
    for k in fixed_classes:
        m = lab == k
        if int(m.sum()) < 10:
            continue                      # class absent under this condition: skip, but the SET is fixed
        fk = f[m]
        c = fk.mean(0)
        cents.append(c)
        spreads.append((fk - c).norm(dim=1).mean())
    if len(cents) < 2:
        return float("nan"), 0
    Cm = torch.stack(cents)
    d = torch.cdist(Cm.unsqueeze(0), Cm.unsqueeze(0)).squeeze(0)
    n = Cm.shape[0]
    inter = float(d.sum() / (n * (n - 1)))
    intra = float(torch.stack(spreads).mean())
    return inter / max(intra, 1e-9), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=16)
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
    m10 = json.loads((GAPFILL / "m10_margin_crossing.json").read_text())["rows"]
    dmg = {c: v["damage"] for c, v in m10.items()}
    names = [c for c in COND if c in dmg]

    acc = {c: {"disp": [], "kl": [], "nclass": []} for c in names}
    clean_disp = []
    nclass_clean = []
    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)
            if i == 0:
                assert float((zl - dec(Zc, size)).abs().max()) == 0.0, "nondeterministic"
            logp = F.log_softmax(zl, 1)
            f0 = Zc[0][0]
            ph, pw = f0.shape[1], f0.shape[2]
            pseudo = F.interpolate(zl.argmax(1, keepdim=True).float(), size=(ph, pw),
                                   mode="nearest").squeeze().long()
            fixed = [int(k) for k in torch.unique(pseudo) if int((pseudo == k).sum()) >= 10]
            cdv, cn = dispersion(f0, pseudo, fixed)
            clean_disp.append(cdv)
            nclass_clean.append(cn)
            for c in names:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                zk = dec(Zk, size)
                pk = F.interpolate(zk.argmax(1, keepdim=True).float(), size=(ph, pw),
                                   mode="nearest").squeeze().long()
                dv, nn = dispersion(Zk[0][0], pk, fixed)
                acc[c]["disp"].append(dv)
                acc[c]["nclass"].append(nn)
                acc[c]["kl"].append(float(F.kl_div(F.log_softmax(zk, 1), logp,
                                                   log_target=True, reduction="batchmean")))
            if (i + 1) % 8 == 0:
                print(f"  {i+1}/{a.n_eval}")

    cd = float(np.nanmean(clean_disp))
    rows = {}
    for c in names:
        d_ = float(np.nanmean(acc[c]["disp"]))
        rows[c] = {"disp": d_, "disp_drop": cd - d_,
                   "nclass": float(np.mean(acc[c]["nclass"])), "kl": float(np.mean(acc[c]["kl"])),
                   "damage": dmg[c], "band": m14[c]["band"] if c in m14 else "IN",
                   "norm": m14[c]["norm"] if c in m14 else float("nan")}

    print(f"\nclean inter/intra dispersion = {cd:.4f}  over {np.mean(nclass_clean):.1f} classes")
    print(f"CONFOUND CHECK — classes retained per condition (clean = {np.mean(nclass_clean):.1f}):")
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        print(f"   {c:20s} {rows[c]['nclass']:5.1f}")
    print()
    print(f"{'condition':20s} {'band':>4s} {'damage':>7s} {'disp':>8s} {'drop':>8s} {'KL':>11s}")
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        r = rows[c]
        print(f"{c:20s} {r['band']:>4s} {r['damage']:7.2f} {r['disp']:8.4f} "
              f"{r['disp_drop']:8.4f} {r['kl']:11.1f}")

    IN = [c for c in rows if rows[c]["band"] == "IN"]
    OFF = [c for c in rows if rows[c]["band"] == "OFF"]

    def sp(g, key):
        return spearman([rows[c][key] for c in g], [rows[c]["damage"] for c in g])

    print(f"\n--- HEAD-TO-HEAD (Spearman with measured full-val damage) ---")
    res = {}
    for nm, g in (("IN", IN), ("OFF", OFF)):
        if len(g) < 3:
            continue
        d_, k_, n_ = sp(g, "disp_drop"), sp(g, "kl"), sp(g, "norm")
        res[nm] = {"disp_drop": d_, "kl": k_, "norm": n_, "n": len(g)}
        print(f"  {nm:3s} n={len(g):2d}   RIVAL-1 dispersion drop {d_:+.3f}   "
              f"ours KL {k_:+.3f}   [||Delta|| {n_:+.3f}]")

    # partial correlations on ranks, in-band
    def rank(v):
        return np.argsort(np.argsort(v)).astype(float)

    def partial(x, y, z):
        X, Y, Z = rank(x), rank(y), rank(z)
        def r(u, v):
            return float(np.corrcoef(u, v)[0, 1])
        rxy, rxz, ryz = r(X, Y), r(X, Z), r(Y, Z)
        den = np.sqrt(max((1 - rxz ** 2) * (1 - ryz ** 2), 1e-12))
        return (rxy - rxz * ryz) / den

    pk = pd_ = float("nan")
    if len(IN) >= 4:
        dm = [rows[c]["damage"] for c in IN]
        kl = [rows[c]["kl"] for c in IN]
        dd = [rows[c]["disp_drop"] for c in IN]
        pk = partial(kl, dm, dd)
        pd_ = partial(dd, dm, kl)
        print(f"\n--- PARTIAL CORRELATIONS, in-band (n={len(IN)}) ---")
        print(f"  KL with damage, controlling for dispersion drop : {pk:+.3f}")
        print(f"  dispersion drop with damage, controlling for KL : {pd_:+.3f}")

    p_a = "IN" in res and res["IN"]["disp_drop"] is not None and abs(res["IN"]["disp_drop"]) >= 0.7
    p_b = ("IN" in res and res["IN"]["kl"] is not None and res["IN"]["disp_drop"] is not None
           and res["IN"]["kl"] - abs(res["IN"]["disp_drop"]) >= 0.10)
    p_c = max(abs(pk), abs(pd_)) >= 0.3 if np.isfinite(pk) else False

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M24a RIVAL-1 dispersion orders in-band damage (|rho|>=0.7) -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M24b ours beats theirs by >=0.10 in-band -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M24c not redundant (max |partial| >= 0.3) -> {'HIT' if p_c else 'MISS'}")
    print(f"\n--- OUTCOME-TABLE BRANCH ---")
    print(f"  P-M24a {'HIT' if p_a else 'MISS'} | P-M24b {'HIT' if p_b else 'MISS'} | "
          f"P-M24c {'HIT' if p_c else 'MISS'}")
    print("  (interpretation belongs in STANDING-ASSESSMENT, not here)")

    o = GAPFILL / "m24_rival1_dispersion.json"
    o.write_text(json.dumps({"clean_dispersion": cd, "rows": rows, "headtohead": res,
                             "partial_kl_given_disp": float(pk),
                             "partial_disp_given_kl": float(pd_),
                             "P_M24a": bool(p_a), "P_M24b": bool(p_b), "P_M24c": bool(p_c),
                             "n_eval": a.n_eval}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
