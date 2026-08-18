"""M1 — does representation displacement predict damage, at proper n? Pre-reg: plan/MIG-PROGRAM.md.

M0 tested 3 matched-fraction conditions and found pure ||delta|| already reproduced the damage rank
while MIG's top-1 direction did not. M0's n was too small to test the OUT-OF-SAMPLE move, which the
program named as the load-bearing test because it does not degrade as n shrinks.

M1 uses every condition whose damage is ALREADY MEASURED on this checkpoint, so no new evaluation is
needed and no damage number is produced by this script. Damage is read from results/gapfill/*.json
(clean minus corrupted, per seed, then averaged) -- the same files C1 used.

THE TEST. For each condition compute, on the decoder-input representation:
    ||delta||        magnitude of the displacement from clean          (B0)
    maha_excess      Mahalanobis(corrupt) - Mahalanobis(clean)          (B2 / support term)
    coord            projection onto the influence basis                (MIG term)
Then ask which predicts damage, with LEAVE-ONE-FAMILY-OUT, because C1 established that severity
variants of one process are not independent observations.

FILED BEFORE THE NUMBERS (do not edit afterwards):
    P-M1a  Spearman(||delta||, damage) >= 0.70 across families            P_true 0.60
    P-M1b  maha_excess adds NOTHING over ||delta|| -- partial Spearman of
           maha_excess with damage, controlling for ||delta||, is < 0.3   P_true 0.60
    P-M1c  depth_offset displacement stays < 1% of entire_missing's       P_true 0.90

P-M1b is the one that matters: it is the pre-registered statement that MIG's second term is
decoration. Filed at 0.60, deliberately near a coin flip -- the ledger records 8 hits / 11 misses
and every 0.70+ filing that rested on a mechanism story has failed.

    python3 scripts/m1_mig_full.py --gpu 1 --n-fit 60 --n-eval 24
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
sys.path.insert(0, str(DFORMER))
sys.path.insert(0, str(DFORMER / "utils"))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
_FEAT = []

# condition -> (gapfill file suffix, corruption spec, FAMILY)
# FAMILY assignment is by GENERATING PROCESS, declared here before any correlation is computed;
# two conditions share a family iff one is the other with a severity parameter changed.
COND = {
    "invalid_0.10":  ("dinv010",      ("invalid", 0.10),  "sparse-invalid"),
    "invalid_0.25":  ("invalid0.25",  ("invalid", 0.25),  "sparse-invalid"),
    "invalid_0.55":  ("invalid0.55",  ("invalid", 0.55),  "sparse-invalid"),
    "invalid_0.70":  ("invalid0.7",   ("invalid", 0.70),  "sparse-invalid"),
    "holes_1blob":   ("blob1",        ("holes", 1, 0.316), "compact-holes"),
    "holes_2blob":   ("blob2",        ("holes", 2, 0.224), "compact-holes"),
    "depth_offset":  ("ho_depth_offset",  ("offset", 0.2),  "pointwise-monotone"),
    "depth_gain":    ("ho_depth_gain",    ("gain", 1.5),    "pointwise-monotone"),
    "depth_gamma":   ("ho_depth_gamma",   ("gamma", 1.6),   "pointwise-monotone"),
    "depth_saturate":("ho_depth_saturate",("saturate", 0.5),"pointwise-monotone"),
    "depth_shift":   ("ho_depth_shift",   ("shift", 4),     "spatial-displacement"),
    "salt_pepper":   ("ho_salt_pepper",   ("saltpepper", 0.2), "sparse-impulse"),
    "depth_smooth_random": ("ho_depth_smooth_random", ("smooth", 16), "smooth-random-field"),
    "dconst":        ("dconst",       ("constant",),      "global-degeneracy"),
}


def measured_damage():
    """clean - corrupted, per seed, averaged. Read from disk; this script measures no damage."""
    base = {}
    for s in (20260801, 20260802, 20260803):
        f = PROJECT / "results" / "p0" / f"mainB_baseline_s{s}" / "final_eval.json"
        base[s] = json.loads(f.read_text())["results"]["single"]["clean"]["miou"]
    out = {}
    for name, (suf, _, _) in COND.items():
        vals = []
        for f in sorted(glob.glob(str(GAPFILL / f"mainB_baseline_s*_{suf}.json"))):
            m = re.search(r"_s(\d+)_", f)
            if not m:
                continue
            s = int(m.group(1))
            r = json.loads(open(f).read())["results"]["single"]
            vals.append(base[s] - r[next(iter(r))]["miou"])
        if vals:
            out[name] = st.mean(vals)
    return out


def corrupt(d, spec):
    from depth_corruptions import (rectangular_holes, depth_invalid, gaussian_noise,
                                   depth_saturate, depth_shift, salt_pepper,
                                   depth_smooth_random)
    k = spec[0]
    if k == "invalid":   return depth_invalid(d.clone(), frac=spec[1], seed=SEED)
    if k == "holes":     return rectangular_holes(d.clone(), num_holes=spec[1],
                                                  hole_frac=spec[2], seed=SEED)
    if k == "offset":    return d + spec[1]
    if k == "gain":      return d * spec[1]
    if k == "gamma":
        mn = d.min()
        z = (d - mn) / (d.max() - mn + 1e-8)
        return z.clamp(min=0) ** spec[1] * (d.max() - mn) + mn
    if k == "saturate":  return depth_saturate(d.clone(), q=spec[1])
    if k == "shift":     return depth_shift(d.clone(), dx=spec[1], dy=0)
    if k == "saltpepper":return salt_pepper(d.clone(), frac=spec[1], seed=SEED)
    if k == "smooth":    return depth_smooth_random(d.clone(), scale=spec[1], seed=SEED)
    if k == "constant":  return torch.zeros_like(d)
    raise ValueError(k)


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return None
    rx, ry = rank(x), rank(y)
    mx, my = st.mean(rx), st.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return None if dx == 0 or dy == 0 else num / (dx * dy)


def partial_spearman(x, y, z):
    """Spearman of x,y controlling for z — how much x adds beyond z."""
    rxy, rxz, ryz = spearman(x, y), spearman(x, z), spearman(y, z)
    if None in (rxy, rxz, ryz):
        return None
    den = ((1 - rxz ** 2) * (1 - ryz ** 2)) ** 0.5
    return None if den == 0 else (rxy - rxz * ryz) / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-fit", type=int, default=60)
    ap.add_argument("--n-eval", type=int, default=24)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    dmg = measured_damage()
    print(f"conditions with measured damage: {len(dmg)}")
    if len(dmg) < 6:
        raise SystemExit(f"REFUSING: only {len(dmg)} conditions have damage on disk; "
                         f"the out-of-sample move needs more")

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

    def _dec(x, *args, **kw):
        f = x if isinstance(x, (list, tuple)) else [x]
        _FEAT.append(torch.cat([t.mean(dim=(2, 3)) for t in f], dim=1).detach().float().cpu())
        return _orig(x, *args, **kw)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def fwd(rgb, dep):
        _FEAT.clear()
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        return _FEAT[-1].clone()

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        b0 = next(iter(vl))
        r0, d0 = b0["data"].cuda(), b0["modal_x"].cuda()
        noise = (fwd(r0, d0) - fwd(r0, d0)).abs().max().item()
        print(f"CANARY identical-forward delta: {noise:.3e}")
        if noise > 1e-5:
            raise SystemExit("REFUSING: decoder nondeterminism uncontrolled")

        fit = []
        for i, b in enumerate(vl):
            if i >= a.n_fit:
                break
            fit.append(fwd(b["data"].cuda(), b["modal_x"].cuda()))
        Z = torch.cat(fit).numpy()
        mu, S = Z.mean(0), np.cov(Z, rowvar=False)
        S += (1e-3 * np.trace(S) / S.shape[0]) * np.eye(S.shape[0])
        Sinv = np.linalg.inv(S)

        def maha(z):
            d = z - mu
            return float(np.sqrt(np.einsum("ij,jk,ik->i", d, Sinv, d).mean()))

        names = [c for c in COND if c in dmg]
        acc = {c: [] for c in names + ["clean"]}
        vl2, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl2):
            if i < a.n_fit:
                continue
            if i >= a.n_fit + a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            acc["clean"].append(fwd(rgb, dep).numpy())
            for c in names:
                acc[c].append(fwd(rgb, corrupt(dep, COND[c][1])).numpy())
        F = {c: np.concatenate(v) for c, v in acc.items()}

    base = F["clean"]
    m_clean = maha(base)
    rows = []
    D = np.stack([(F[c] - base).mean(0) for c in names])
    U, sv, Vt = np.linalg.svd(D, full_matrices=False)
    k = 3
    for c in names:
        d = (F[c] - base).mean(0)
        rows.append({"cond": c, "family": COND[c][2], "damage": dmg[c],
                     "norm": float(np.linalg.norm(F[c] - base, axis=1).mean()),
                     "maha_excess": maha(F[c]) - m_clean,
                     "coord": float(np.linalg.norm(Vt[:k] @ d))})

    print(f"\n{'condition':22s} {'family':22s} {'damage':>7s} {'|delta|':>9s} "
          f"{'maha_exc':>9s} {'coord_k3':>9s}")
    for r in sorted(rows, key=lambda r: -r["damage"]):
        print(f"{r['cond']:22s} {r['family']:22s} {r['damage']:7.2f} {r['norm']:9.3f} "
              f"{r['maha_excess']:9.3f} {r['coord']:9.3f}")

    dm = [r["damage"] for r in rows]
    nm = [r["norm"] for r in rows]
    mx = [r["maha_excess"] for r in rows]
    cd = [r["coord"] for r in rows]
    print(f"\nSpearman vs damage over {len(rows)} conditions:")
    print(f"  ||delta||     {spearman(nm, dm):+.3f}")
    print(f"  maha_excess   {spearman(mx, dm):+.3f}")
    print(f"  MIG coord(k=3){spearman(cd, dm):+.3f}")
    ps = partial_spearman(mx, dm, nm)
    print(f"  partial(maha_excess, damage | ||delta||) = "
          f"{'n/a' if ps is None else f'{ps:+.3f}'}   <- P-M1b")

    fams = sorted(set(r["family"] for r in rows))
    print(f"\nLEAVE-ONE-FAMILY-OUT ({len(fams)} families) — Spearman(||delta||, damage):")
    for f in fams:
        keep = [r for r in rows if r["family"] != f]
        s = spearman([r["norm"] for r in keep], [r["damage"] for r in keep])
        print(f"  drop {f:22s} n={len(keep):2d}  {'n/a' if s is None else f'{s:+.3f}'}")

    off = next((r for r in rows if r["cond"] == "depth_offset"), None)
    big = max(rows, key=lambda r: r["damage"])
    p_a = spearman(nm, dm) >= 0.70
    p_b = ps is not None and abs(ps) < 0.3
    p_c = off is not None and off["norm"] < 0.01 * big["norm"]
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M1a Spearman(||delta||,damage) >= 0.70 -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M1b maha adds nothing over ||delta|| (|partial| < 0.3) -> "
          f"{'HIT' if p_b else 'MISS'}")
    print(f"P-M1c depth_offset ||delta|| < 1% of {big['cond']}'s -> {'HIT' if p_c else 'MISS'}")

    p = GAPFILL / "m1_mig_full.json"
    p.write_text(json.dumps({"rows": rows, "singular_values": [float(x) for x in sv],
                             "spearman_norm": spearman(nm, dm),
                             "spearman_maha": spearman(mx, dm),
                             "spearman_coord": spearman(cd, dm),
                             "partial_maha_given_norm": ps,
                             "P_M1a": bool(p_a), "P_M1b": bool(p_b),
                             "P_M1c": bool(p_c)}, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
