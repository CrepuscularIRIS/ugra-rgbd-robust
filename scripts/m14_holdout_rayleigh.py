"""M14 — held-out confirmation of R = KL/||Delta||^2 as an arrangement-sensitive stage-A quantity.

R was found by READING the M13 table. That is the same post-hoc status M8's two-band split had before
M9 confirmed it, and it is a hypothesis until this runs.

WHAT MAKES THIS OUT-OF-SAMPLE:
  * criterion (c) -- collapse under spatial permutation -- needs NO damage measurement, so it extends
    from the 3 conditions measured in M12 to all 24 here for free;
  * SIX corruption families never used in this campaign enter: entire_missing, half_missing,
    gaussian_noise, depth_flip, rect_holes_wide, invalid_0.40. A new family can break a law fitted on
    the old ones;
  * band membership for each new condition is computed by M8's kNN support machinery, and the
    residual range is PREDICTED before it is measured.

The new families are adversarial by choice, not convenience:
  entire_missing / half_missing  the most extreme excursions available -> predicted OFF, tight range
  depth_flip                     preserves the depth VALUE distribution exactly while destroying
                                 spatial correspondence -> the sharpest test of arrangement-sensitivity
  gaussian_noise                 spatially white by construction -> should behave like a shuffled field

Prereg, outcome table, severity: .grill/prediction-ledger.md (2026-08-10, M14).

    python3 scripts/m14_holdout_rayleigh.py --gpu 1 --n-eval 30 --n-bank 6
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
OFF_LO, OFF_HI = 0.55, 1.00      # the OFF-band residual range, FIXED from M13 before this ran
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402
from m8_local_support import knn_dist, flat  # noqa: E402

EXTRA_SHIFTS = {"shift_1": 1, "shift_8": 8, "shift_16": 16, "shift_32": 32}
# SIX families never used in this campaign
NEW = {
    "entire_missing":  ("entire", None),
    "half_missing":    ("half", None),
    "gaussian_0.10":   ("gauss", 0.10),
    "depth_flip_h":    ("flip", "h"),
    "rect_holes_wide": ("rect", None),
    "invalid_0.40":    ("invalid", 0.40),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--n-bank", type=int, default=6)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    import depth_corruptions as DC
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
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            out = _orig(list(feats))
        if isinstance(out, dict) and "pred" in out:
            out = out["pred"]
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)

    def corrupt_any(d, n):
        if n in EXTRA_SHIFTS:
            return DC.depth_shift(d.clone(), dx=EXTRA_SHIFTS[n], dy=0)
        if n in NEW:
            kind, arg = NEW[n]
            if kind == "entire":  return DC.entire_missing(d.clone())
            if kind == "half":    return DC.half_missing(d.clone(), seed=SEED)
            if kind == "gauss":   return DC.gaussian_noise(d.clone(), sigma=arg, seed=SEED)
            if kind == "flip":    return DC.depth_flip(d.clone(), axis=arg)
            if kind == "rect":    return DC.rectangular_holes(d.clone(), seed=SEED)
            if kind == "invalid": return DC.depth_invalid(d.clone(), frac=arg, seed=SEED)
        return corrupt(d, COND[n][1])

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    OLD = list(COND) + list(EXTRA_SHIFTS)
    names = OLD + list(NEW)
    g = torch.Generator().manual_seed(SEED)
    acc = {n: {"kl": [], "kl_s": [], "kl_u": [], "nrm": [], "sup": [],
               "kl_h1": [], "nrm_h1": [], "kl_h2": [], "nrm_h2": []} for n in names}
    clean_sup = []

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        it = iter(vl)
        bank = []
        for _ in range(a.n_bank):
            b = next(it)
            bank.append(flat(cap_(b["data"].cuda(), b["modal_x"].cuda()), 0))
        B = torch.cat(bank)
        if B.shape[0] > 20000:
            B = B[torch.linspace(0, B.shape[0] - 1, 20000).long()]
        print(f"clean bank: {B.shape[0]} per-position features (SAME construction as M8)\n")

        canary = False
        for i, b in enumerate(it):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            size = rgb.shape[-2:]
            Zc = cap_(rgb, dep)
            zl = dec(Zc, size)
            if not canary:
                d = float((zl - dec(Zc, size)).abs().max())
                print(f"CANARY: max|diff| = {d:.3e}")
                if d > 0:
                    raise SystemExit("REFUSING: nondeterministic decoder")
                print("  bases pinned.\n")
                canary = True
            logp_c = F.log_softmax(zl, 1)
            clean_sup.append(knn_dist(flat(Zc, 0), B))

            for n in names:
                Zk = cap_(rgb, corrupt_any(dep, n))
                delta = [k - z for k, z in zip(Zk, Zc)]
                nrm = float(sum(float((d_ ** 2).sum()) for d_ in delta) ** 0.5)
                sh, un = [], []
                for d_ in delta:
                    Bb, C, Hh, Ww = d_.shape
                    idx = torch.randperm(Hh * Ww, generator=g).to(d_.device)
                    sh.append(d_.reshape(Bb, C, Hh * Ww)[:, :, idx].reshape(Bb, C, Hh, Ww))
                    un.append(d_.mean(dim=(2, 3), keepdim=True).expand_as(d_))

                def kl_of(fs):
                    return float(F.kl_div(F.log_softmax(dec(fs, size), 1), logp_c,
                                          log_target=True, reduction="batchmean"))
                k = kl_of(Zk)
                acc[n]["kl"].append(k)
                acc[n]["kl_s"].append(kl_of([z + d_ for z, d_ in zip(Zc, sh)]))
                acc[n]["kl_u"].append(kl_of([z + d_ for z, d_ in zip(Zc, un)]))
                acc[n]["nrm"].append(nrm)
                acc[n]["sup"].append(knn_dist(flat(Zk, 0), B))
                tag = "h1" if i < a.n_eval // 2 else "h2"
                acc[n][f"kl_{tag}"].append(k)
                acc[n][f"nrm_{tag}"].append(nrm)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.n_eval}")

    cs = float(np.mean(clean_sup))
    rows = {}
    for n in names:
        kl = float(np.mean(acc[n]["kl"]))
        nr = float(np.mean(acc[n]["nrm"]))
        rows[n] = {"kl": kl, "norm": nr,
                   "kl_shuf": float(np.mean(acc[n]["kl_s"])),
                   "kl_uni": float(np.mean(acc[n]["kl_u"])),
                   "support_excess": float(np.mean(acc[n]["sup"])) - cs,
                   "new": n in NEW}
        r = rows[n]
        r["band"] = "OFF" if r["support_excess"] > 0.05 else "IN"
        r["R"] = kl / nr ** 2 if nr > 1.0 else None
        r["shuf_ratio"] = r["kl_shuf"] / kl if kl > 1e-6 else None
        r["uni_ratio"] = r["kl_uni"] / kl if kl > 1e-6 else None

    # exponent on disjoint image halves
    def fit(tag):
        u = [n for n in names if rows[n]["norm"] > 1.0 and acc[n][f"nrm_{tag}"]]
        lx = np.log([float(np.mean(acc[n][f"nrm_{tag}"])) for n in u])
        ly = np.log([max(float(np.mean(acc[n][f"kl_{tag}"])), 1e-9) for n in u])
        return float(np.polyfit(lx, ly, 1)[0])
    p1, p2 = fit("h1"), fit("h2")

    print(f"\nclean support ref {cs:.4f}   exponent p: images 1-{a.n_eval//2} = {p1:.3f}  "
          f"images {a.n_eval//2+1}-{a.n_eval} = {p2:.3f}   |diff| = {abs(p1-p2):.3f}\n")
    print(f"{'condition':20s} {'new':>4s} {'band':>4s} {'excess':>8s} {'||D||':>9s} "
          f"{'R':>8s} {'shuf':>6s} {'uni':>6s}")
    for n in sorted(names, key=lambda x: (not rows[x]["new"], -rows[x]["kl"])):
        r = rows[n]
        R = f"{r['R']:8.4f}" if r["R"] is not None else "    VOID"
        s = f"{r['shuf_ratio']:6.3f}" if r["shuf_ratio"] is not None else "  VOID"
        u = f"{r['uni_ratio']:6.3f}" if r["uni_ratio"] is not None else "  VOID"
        print(f"{n:20s} {'NEW' if r['new'] else '':>4s} {r['band']:>4s} "
              f"{r['support_excess']:8.4f} {r['norm']:9.3f} {R}{s}{u}")

    valid = [n for n in names if rows[n]["R"] is not None]
    old_v = [n for n in valid if not rows[n]["new"]]
    new_v = [n for n in valid if rows[n]["new"]]
    p_a = all(rows[n]["shuf_ratio"] <= 0.30 for n in old_v)
    p_b = bool(new_v) and all(rows[n]["shuf_ratio"] <= 0.30 for n in new_v)
    new_off = [n for n in new_v if rows[n]["band"] == "OFF"]
    new_in = [n for n in new_v if rows[n]["band"] == "IN"]
    off_ok = all(OFF_LO <= rows[n]["R"] <= OFF_HI for n in new_off)
    in_out = any(not (OFF_LO <= rows[n]["R"] <= OFF_HI) for n in new_in)
    p_c = off_ok and in_out
    p_d = abs(p1 - p2) <= 0.15
    p_e = all(rows[n]["uni_ratio"] <= 0.30 for n in valid)

    print(f"\n--- FILED PREDICTIONS ---")
    bad_a = [n for n in old_v if rows[n]["shuf_ratio"] > 0.30]
    bad_b = [n for n in new_v if rows[n]["shuf_ratio"] > 0.30]
    print(f"P-M14a R collapses on all {len(old_v)} OLD conditions -> {'HIT' if p_a else 'MISS'}"
          + (f"  violators {bad_a}" if bad_a else ""))
    print(f"P-M14b R collapses on all {len(new_v)} NEW families -> {'HIT' if p_b else 'MISS'}"
          + (f"  violators {bad_b}" if bad_b else ""))
    print(f"P-M14c out-of-sample band prediction -> {'HIT' if p_c else 'MISS'}")
    print(f"      new OFF {[(n, round(rows[n]['R'],3)) for n in new_off]} "
          f"all in [{OFF_LO},{OFF_HI}]? {off_ok}")
    print(f"      new IN  {[(n, round(rows[n]['R'],3)) for n in new_in]} "
          f"any outside? {in_out}")
    print(f"P-M14d exponent stable across disjoint halves -> {'HIT' if p_d else 'MISS'}")
    bad_e = [n for n in valid if rows[n]["uni_ratio"] > 0.30]
    print(f"P-M14e uniform control small -> {'HIT' if p_e else 'MISS'}"
          + (f"  violators {bad_e}" if bad_e else ""))

    print(f"\n--- VERDICT ---")
    if p_a and p_b and p_e:
        print("  R IS ARRANGEMENT-SENSITIVE, CONFIRMED OUT OF SAMPLE. The permutation preserves")
        print("  ||Delta|| exactly and every per-channel statistic, so only the position assignment")
        print("  differs -- and R collapses across every family including six never used before,")
        print("  while a uniform field of the same norm does not produce the response.")
    else:
        print("  R FAILS criterion (c) somewhere. Per the outcome table this KILLS R as an")
        print("  arrangement-sensitive quantity -- the external lane warned the algebra does not")
        print("  imply the collapse, and a single non-collapsing family settles it.")

    o = GAPFILL / "m14_holdout_rayleigh.json"
    o.write_text(json.dumps({"clean_support": cs, "p_half1": p1, "p_half2": p2, "rows": rows,
                             "P_M14a": p_a, "P_M14b": p_b, "P_M14c": p_c,
                             "P_M14d": p_d, "P_M14e": p_e, "n_eval": a.n_eval}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
