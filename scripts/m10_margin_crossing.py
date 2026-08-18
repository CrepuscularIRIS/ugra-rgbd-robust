"""M10 — margin-crossing linear response. Prereg: plan/PREREG-M10-margin-crossing.md

THE OBJECT. Not ||J.Delta|| -- that factorises as ||Delta|| * gain and inherits the norm's
correlation mechanically. Damage is mIoU, i.e. an ARGMAX FLIP, so a large response in a high-margin
region contributes exactly zero. The statistic is therefore the first-order predicted flip fraction:

    dz   = J.Delta          directional derivative of decoder logits along the true displacement
    FLIP = mean[ argmax(z_clean + dz) != argmax(z_clean) ]

J is never formed (480x640x40 outputs makes it absurd). The directional derivative is taken by finite
difference on the DECODER ONLY: dz = (dec(Z + eps*Delta) - dec(Z)) / eps. Two decoder calls.

THE BARS, from E-1 (computed on cached artifacts today, and they retracted the standing framing):
    in-support  Spearman(||Delta||, damage) = +0.833   (n=8; +0.750 without depth_offset)
    off-support                             = +0.943   (n=6)
A pooled score is NOT acceptable. Pooling is the defect this campaign has paid for three times.

FILED BEFORE ANY NUMBER (see prereg for P_true and the outcome table):
    P-M10a  FLIP_shuffled / FLIP_real <= 0.3 on dconst and invalid_0.10          0.70
    P-M10b  in-band Spearman(FLIP, damage) >= 0.85, beating the 0.833 bar        0.40
    P-M10c  gain factor ||J.Delta||/||Delta|| varies >= 1.5x across conditions   0.55
    P-M10d  linearization validity higher in-band than off-band by >= 0.10       0.60
    P-M10e  dconst ranked top-1 within the in-support band by FLIP               0.65

P-M10a is THE test. ||Delta|| is exactly shuffle-invariant by construction, so no disguised-magnitude
statistic can collapse under the shuffle. If P-M10a fails, the object is dead -- no reformulation.

    python3 scripts/m10_margin_crossing.py --gpu 1 --n-eval 8
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
sys.path.insert(0, str(PROJECT / "scripts"))

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
EPS = 1e-3
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, measured_damage, spearman, corrupt  # noqa: E402

# held-out shifts, damage already on disk from M9 -- all predicted IN-support.
EXTRA_SHIFTS = {"shift_1": 1, "shift_8": 8, "shift_16": 16, "shift_32": 32}


def shift_damage():
    """M9's held-out shift damages, read from disk. Same construction as m9_holdout_bands."""
    base = {}
    for s in (20260801, 20260802, 20260803):
        f = PROJECT / "results" / "p0" / f"mainB_baseline_s{s}" / "final_eval.json"
        base[s] = json.loads(f.read_text())["results"]["single"]["clean"]["miou"]
    out = {}
    for name, n in EXTRA_SHIFTS.items():
        vals = []
        for f in sorted(glob.glob(str(GAPFILL / f"mainB_baseline_s*_shift{n}.json"))):
            m = re.search(r"_s(\d+)_", f)
            if not m:
                continue
            r = json.loads(open(f).read())["results"]["single"]
            vals.append(base[int(m.group(1))] - r[next(iter(r))]["miou"])
        if vals:
            out[name] = st.mean(vals)
    return out


def flat_corr(a, b):
    """Pearson between two flattened tensors, on a subsample."""
    x = a.reshape(-1).float()
    y = b.reshape(-1).float()
    n = min(x.numel(), 200000)
    idx = torch.linspace(0, x.numel() - 1, n).long().to(x.device)
    x, y = x[idx], y[idx]
    x = x - x.mean()
    y = y - y.mean()
    d = (x.norm() * y.norm()).item()
    return float((x * y).sum().item() / d) if d > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=8)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    dmg = measured_damage()
    dmg.update(shift_damage())
    print(f"conditions with measured damage: {len(dmg)}")

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    from depth_corruptions import depth_shift
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

    def _dec_hook(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _dec_hook

    def cap_(rgb, dep):
        """Capture the decoder-input feature list for one input."""
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec(feats):
        """Decoder logits for a feature list. Ham/NMF bases PINNED per forward."""
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            return _orig(list(feats))

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    names = [c for c in COND if c in dmg] + [c for c in EXTRA_SHIFTS if c in dmg]
    g = torch.Generator().manual_seed(SEED)

    def corrupt_any(d, name):
        if name in EXTRA_SHIFTS:
            return depth_shift(d.clone(), dx=EXTRA_SHIFTS[name], dy=0)
        return corrupt(d, COND[name][1])

    acc = {c: {"flip": [], "flip_shuf": [], "flip_uni": [], "gain": [],
               "lin": [], "nrm": []} for c in names}

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        canary_done = False
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            Zc = cap_(rgb, dep)
            zl = dec(Zc)

            if not canary_done:
                zl2 = dec(Zc)
                d_self = float((zl - zl2).abs().max())
                print(f"\nCANARY self-comparison (same input twice): max|diff| = {d_self:.3e}")
                if d_self > 0:
                    raise SystemExit("REFUSING: decoder is nondeterministic; reseeding failed. "
                                     "Every number from this probe would be noise.")
                print("  bases are pinned. proceeding.\n")
                canary_done = True

            base_arg = zl.argmax(1)
            for c in names:
                Zk = cap_(rgb, corrupt_any(dep, c))
                delta = [k - z for k, z in zip(Zk, Zc)]
                nrm = float(sum(float((d ** 2).sum()) for d in delta) ** 0.5)

                # --- first-order directional derivative, decoder only
                zl_e = dec([z + EPS * d for z, d in zip(Zc, delta)])
                dz = (zl_e - zl) / EPS
                flip = float((( zl + dz).argmax(1) != base_arg).float().mean())

                # --- linearization validity: predicted vs TRUE logit change
                true_dz = dec(Zk) - zl
                lin = flat_corr(dz, true_dz)

                # --- SHUFFLE control: same magnitude, same per-channel stats, permuted positions
                sh = []
                for d in delta:
                    B, C, H, W = d.shape
                    idx = torch.randperm(H * W, generator=g).to(d.device)
                    sh.append(d.reshape(B, C, H * W)[:, :, idx].reshape(B, C, H, W))
                dz_s = (dec([z + EPS * d for z, d in zip(Zc, sh)]) - zl) / EPS
                flip_s = float(((zl + dz_s).argmax(1) != base_arg).float().mean())

                # --- UNIFORM control: spatial mean broadcast (M2's arm)
                un = [d.mean(dim=(2, 3), keepdim=True).expand_as(d) for d in delta]
                dz_u = (dec([z + EPS * d for z, d in zip(Zc, un)]) - zl) / EPS
                flip_u = float(((zl + dz_u).argmax(1) != base_arg).float().mean())

                acc[c]["flip"].append(flip)
                acc[c]["flip_shuf"].append(flip_s)
                acc[c]["flip_uni"].append(flip_u)
                acc[c]["gain"].append(float(dz.norm()) / max(nrm, 1e-9))
                acc[c]["lin"].append(lin)
                acc[c]["nrm"].append(nrm)
            print(f"  image {i+1}/{a.n_eval} done")

    # ---- assemble
    m8 = json.loads((GAPFILL / "m8_local_support.json").read_text())
    cl8 = m8["clean_ref"]
    band = {}
    for c in names:
        if c in m8["rows"]:
            band[c] = "OFF" if (m8["rows"][c]["knn"] - cl8) > 0.05 else "IN"
        else:
            band[c] = "IN"   # M9 confirmed all held-out shifts in-band, 4/4

    rows = {c: {"flip": float(np.mean(acc[c]["flip"])),
                "flip_shuf": float(np.mean(acc[c]["flip_shuf"])),
                "flip_uni": float(np.mean(acc[c]["flip_uni"])),
                "gain": float(np.mean(acc[c]["gain"])),
                "lin": float(np.nanmean(acc[c]["lin"])),
                "norm": float(np.mean(acc[c]["nrm"])),
                "damage": dmg[c], "band": band[c]} for c in names}

    print(f"\n{'condition':22s} {'band':>4s} {'damage':>7s} {'|D|':>8s} {'FLIP':>8s} "
          f"{'shuf':>8s} {'unif':>8s} {'gain':>7s} {'lin':>6s}")
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        r = rows[c]
        print(f"{c:22s} {r['band']:>4s} {r['damage']:7.2f} {r['norm']:8.3f} {r['flip']:8.4f} "
              f"{r['flip_shuf']:8.4f} {r['flip_uni']:8.4f} {r['gain']:7.3f} {r['lin']:6.3f}")

    IN = [c for c in rows if rows[c]["band"] == "IN"]
    OFF = [c for c in rows if rows[c]["band"] == "OFF"]

    def sp(g_, key):
        return spearman([rows[c][key] for c in g_], [rows[c]["damage"] for c in g_])

    print(f"\n--- WITHIN-BAND ORDERING (pooled scores are NOT acceptable) ---")
    out = {}
    for nm, g_ in (("IN", IN), ("OFF", OFF)):
        if len(g_) < 3:
            continue
        f_, n_ = sp(g_, "flip"), sp(g_, "norm")
        out[nm] = {"flip": f_, "norm": n_, "n": len(g_)}
        print(f"  {nm:3s} n={len(g_):2d}   Spearman(FLIP,damage) = {f_:+.3f}   "
              f"[bar: ||Delta|| = {n_:+.3f}]")

    # ---- predictions
    ratios = {c: rows[c]["flip_shuf"] / max(rows[c]["flip"], 1e-9)
              for c in ("dconst", "invalid_0.10") if c in rows}
    p_a = bool(ratios) and all(v <= 0.3 for v in ratios.values())
    p_b = "IN" in out and out["IN"]["flip"] is not None and out["IN"]["flip"] >= 0.85
    gains = [rows[c]["gain"] for c in rows if rows[c]["norm"] > 1e-6]
    gspread = (max(gains) / max(min(gains), 1e-9)) if gains else float("nan")
    p_c = gspread >= 1.5
    lin_in = float(np.mean([rows[c]["lin"] for c in IN]))
    lin_off = float(np.mean([rows[c]["lin"] for c in OFF])) if OFF else float("nan")
    p_d = (lin_in - lin_off) >= 0.10
    in_rank = sorted(IN, key=lambda c: -rows[c]["flip"])
    p_e = bool(in_rank) and in_rank[0] == "dconst"

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M10a shuffle collapse <=0.3: {ratios} -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M10b in-band Spearman(FLIP) >= 0.85 -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M10c gain spread {gspread:.2f}x >= 1.5x -> {'HIT' if p_c else 'MISS'}")
    print(f"P-M10d linearization in {lin_in:.3f} vs off {lin_off:.3f} "
          f"(diff {lin_in-lin_off:+.3f}) >= 0.10 -> {'HIT' if p_d else 'MISS'}")
    print(f"P-M10e dconst top-1 in-band by FLIP (got '{in_rank[0] if in_rank else None}') "
          f"-> {'HIT' if p_e else 'MISS'}")

    print(f"\n--- NEGATIVE CONTROLS (these MUST be small) ---")
    for c in ("dconst", "invalid_0.10"):
        if c in rows:
            print(f"  {c:16s} uniform-arm FLIP {rows[c]['flip_uni']:.4f} vs real {rows[c]['flip']:.4f}")
    if "depth_offset" in rows:
        print(f"  depth_offset FLIP {rows['depth_offset']['flip']:.5f}  "
              f"(CANARY ONLY -- Delta~0, every functional predicts zero; carries NO evidential weight)")

    print(f"\n--- VERDICT ---")
    if not p_a:
        print("  P-M10a MISSED. FLIP does not collapse under spatial shuffle, which is the one")
        print("  constraint that defines candidacy. H7 is DEAD as stated. Do not reformulate it in")
        print("  a new basis -- that is what the outcome table forbids.")
    elif p_a and p_b:
        print("  P-M10a AND P-M10b HIT. R-mag is KILLED: ||Delta|| is shuffle-invariant by")
        print("  construction, so a disguised-magnitude statistic could not have collapsed.")
        print("  H7 survives to cross-architecture replication.")
    else:
        print("  P-M10a HIT, P-M10b MISSED. First-order sufficiency in-band is KILLED -- the")
        print("  influence-function failure mode reproducing at an internal interface.")
        print("  Survivor: R-assign, a non-linear assignment-dependent account.")

    res = {"rows": rows, "within_band": out, "shuffle_ratios": ratios,
           "gain_spread": gspread, "lin_in": lin_in, "lin_off": lin_off,
           "P_M10a": p_a, "P_M10b": bool(p_b), "P_M10c": bool(p_c),
           "P_M10d": bool(p_d), "P_M10e": p_e, "eps": EPS, "n_eval": a.n_eval}
    o = GAPFILL / "m10_margin_crossing.json"
    o.write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
