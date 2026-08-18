"""M35 — RE-AUDIT UNDER R_eff: does the stage-0 defect overturn anything already concluded?

Revision 41 found `decode_head.in_index = [1,2,3]`: stage 0 is causally inert (zeroing its
displacement leaves KL bitwise unchanged) yet carries 5.5-20% of ||Delta||^2, a 3.63x spread across
conditions. Two consequences were logged as OWED rather than resolved:

  (1) cross-condition R orderings might flip under R_eff = KL / ||Delta_{1:3}||^2;
  (2) M27/M28's training characterisation used ||Delta|| over ALL stages. If a trained arm shifts
      displacement into or out of the inert stage, the measured "global pathway attenuation"
      (implied gain 0.525, CV 0.109) could be partly an artifact of counting inert displacement.

(2) is the sharper of the two. M28's headline — three interventions are three settings of one knob,
with relative ||Delta|| predicting clean cost at +1.000 on arm means — rests on a norm that includes
a causally dead component.

THIS SCRIPT RE-DERIVES BOTH, on the same checkpoints, with per-stage norms recorded so the inert
component can be separated rather than assumed away.

GATE: baseline per-condition R (all-stage convention) must reproduce M14's recorded values, median
ratio within 10% — the same gate M27/M28 passed at 0.9236.

Prereg: .grill/prediction-ledger.md (2026-08-10, M35).

    python3 scripts/m35_reff_reaudit.py --gpu 1 --n-eval 30
"""
import argparse
import json
import os
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT = Path(__file__).resolve().parents[1]
DFORMER = PROJECT / "repos" / "DFormer"
for _p in (DFORMER, DFORMER / "utils", PROJECT / "diagnostics" / "corruptions", PROJECT / "scripts"):
    sys.path.insert(0, str(_p))

GAPFILL = PROJECT / "results" / "gapfill"
P0 = PROJECT / "results" / "p0"
SEED = 20260721
USED = [1, 2, 3]
_CAP = {"store": None, "mode": None}

ARMS = ["baseline", "attenuation", "constant", "dropout"]
PREFIX = {"baseline": "mainB_baseline", "attenuation": "mainC_attenuation",
          "constant": "mainC_constant", "dropout": "mainC_dropout"}
SEEDS = [20260801, 20260802]

from m1_mig_full import COND, corrupt  # noqa: E402


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


def pearson(u, v):
    n = len(u); mu = sum(u) / n; mv = sum(v) / n
    num = sum((x - mu) * (y - mv) for x, y in zip(u, v))
    den = (sum((x - mu) ** 2 for x in u) * sum((y - mv) ** 2 for y in v)) ** 0.5
    return num / den if den else float("nan")


def measure(ckpt, cfg, names, loader, n_eval):
    from models.builder import EncoderDecoder
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    if len(sd) - len(miss.missing_keys) < 0.9 * len(sd):
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

    def dec(f_, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _orig(list(f_))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    acc = {c: {"kl": [], "n_all": [], "n_eff": [], "n0": []} for c in names}
    zc_all, zc_eff = [], []
    for i, b in enumerate(loader()):
        if i >= n_eval:
            break
        rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
        size = rgb.shape[-2:]
        Zc = cap_(rgb, dep)
        logp = F.log_softmax(dec(Zc, size), 1)
        zc_all.append(sum(float((t ** 2).sum()) for t in Zc) ** 0.5)
        zc_eff.append(sum(float((Zc[j] ** 2).sum()) for j in USED) ** 0.5)
        for c in names:
            Zk = cap_(rgb, corrupt(dep, COND[c][1]))
            D = [k - z for k, z in zip(Zk, Zc)]
            n2 = [float((d ** 2).sum()) for d in D]
            acc[c]["n_all"].append(sum(n2) ** 0.5)
            acc[c]["n_eff"].append(sum(n2[j] for j in USED) ** 0.5)
            acc[c]["n0"].append(n2[0] ** 0.5)
            acc[c]["kl"].append(float(F.kl_div(F.log_softmax(dec(Zk, size), 1), logp,
                                               log_target=True, reduction="batchmean")))
    out = {}
    za = sum(zc_all) / len(zc_all); ze = sum(zc_eff) / len(zc_eff)
    for c in names:
        kl = sum(acc[c]["kl"]) / len(acc[c]["kl"])
        na = sum(acc[c]["n_all"]) / len(acc[c]["n_all"])
        ne = sum(acc[c]["n_eff"]) / len(acc[c]["n_eff"])
        n0 = sum(acc[c]["n0"]) / len(acc[c]["n0"])
        out[c] = {"kl": kl, "n_all": na, "n_eff": ne, "n0": n0,
                  "R_all": kl / max(na ** 2, 1e-12), "R_eff": kl / max(ne ** 2, 1e-12),
                  "rel_all": na / za, "rel_eff": ne / ze,
                  "inert_share": (n0 ** 2) / max(na ** 2, 1e-12)}
    return out


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
    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    m14 = json.loads((GAPFILL / "m14_holdout_rayleigh.json").read_text())["rows"]
    names = [c for c in COND if c in m14]
    engine = Engine(custom_parser=None)
    vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

    res = {}
    for s in SEEDS:
        for arm in ARMS:
            ck = P0 / f"{PREFIX[arm]}_s{s}" / "ckpt" / "latest.pth"
            if not ck.exists():
                continue
            print(f"=== {arm} s{s} ===", flush=True)
            res[(arm, s)] = measure(ck, cfg, names, lambda: vl, a.n_eval)

    base = res[("baseline", SEEDS[0])]
    gc = [c for c in names if m14[c].get("R") is not None]
    ratios = sorted(base[c]["R_all"] / m14[c]["R"] for c in gc)
    med = ratios[len(ratios) // 2]
    gate = abs(med - 1) < 0.10
    print(f"\n=== GATE (baseline R_all vs M14) median {med:.4f} -> {'PASS' if gate else 'FAIL'}")
    if not gate:
        raise SystemExit("REFUSING: gate failed")

    # ---------- (1) cross-condition ordering under R_eff ----------
    live = [c for c in names if base[c]["n_all"] > 1e-6]
    Ra = [base[c]["R_all"] for c in live]
    Re = [base[c]["R_eff"] for c in live]
    sp = spearman(Ra, Re)
    oa = sorted(live, key=lambda c: -base[c]["R_all"])
    oe = sorted(live, key=lambda c: -base[c]["R_eff"])
    flips = sum(1 for k in range(len(live)) if oa[k] != oe[k])
    print(f"\n=== (1) CROSS-CONDITION ORDERING ===")
    print(f"  Spearman(R_all, R_eff) = {sp:.4f}   positions changed: {flips}/{len(live)}")
    print(f"  inert share: min {min(base[c]['inert_share'] for c in live):.4f} "
          f"max {max(base[c]['inert_share'] for c in live):.4f}")
    print(f"\n  {'rank':>4s} {'by R_all':22s} {'by R_eff':22s}")
    for k in range(len(live)):
        mark = "  <-- flip" if oa[k] != oe[k] else ""
        print(f"  {k+1:4d} {oa[k]:22s} {oe[k]:22s}{mark}")

    # ---------- (2) does M27/M28 survive? ----------
    print(f"\n=== (2) TRAINING CHARACTERISATION UNDER ||Delta_eff|| ===")
    fe = {}
    for s in SEEDS:
        b = res.get(("baseline", s))
        if b is None:
            continue
        for arm in ARMS:
            r = res.get((arm, s))
            if r is None or arm == "baseline":
                continue
            L = [c for c in live if c != "depth_offset"]
            dN_all = [100 * (r[c]["n_all"] / b[c]["n_all"] - 1) for c in L]
            dN_eff = [100 * (r[c]["n_eff"] / b[c]["n_eff"] - 1) for c in L]
            d0 = [100 * (r[c]["n0"] / b[c]["n0"] - 1) for c in L]
            fe[f"{arm}_s{s}"] = {"dN_all": st.mean(dN_all), "dN_eff": st.mean(dN_eff),
                                 "dN0": st.mean(d0),
                                 "cv_eff": st.stdev([1 + x / 100 for x in dN_eff])
                                 / st.mean([1 + x / 100 for x in dN_eff])}
    fev = json.loads((GAPFILL / "m28_all_arms_lever.json").read_text())["summary"]
    print(f"  {'arm-seed':20s} {'d|D|_all%':>10s} {'d|D|_eff%':>10s} {'d|D|_st0%':>10s} "
          f"{'CV_eff':>7s} {'dclean':>7s}")
    xs, ys = [], []
    for k, v in fe.items():
        dc = fev[k]["dclean"]
        xs.append(v["dN_eff"]); ys.append(dc)
        print(f"  {k:20s} {v['dN_all']:+10.1f} {v['dN_eff']:+10.1f} {v['dN0']:+10.1f} "
              f"{v['cv_eff']:7.3f} {dc:+7.2f}")
    pe = pearson(xs, ys)
    pa = pearson([v["dN_all"] for v in fe.values()], ys)
    print(f"\n  Pearson(d||Delta||_ALL, dclean) = {pa:+.3f}   [M28 reported +0.715]")
    print(f"  Pearson(d||Delta||_EFF, dclean) = {pe:+.3f}")

    order_eff = sorted(fe, key=lambda k: fe[k]["dN_eff"])
    print(f"\n  arm ordering by attenuation (most first), under EFF:")
    for s in SEEDS:
        o = [k for k in order_eff if k.endswith(str(s))]
        print(f"    seed {s}: " + " -> ".join(k.split('_')[0] for k in o))

    p_a = sp >= 0.95
    p_b = pe >= 0.6
    p_c = max(abs(v["dN_eff"] - v["dN0"]) for v in fe.values()) < 15
    print(f"\n  P-M35a (Spearman(R_all,R_eff) >= 0.95):        {'HIT' if p_a else 'MISS'} ({sp:.4f})")
    print(f"  P-M35b (Pearson(d|D|_eff, dclean) >= 0.6):     {'HIT' if p_b else 'MISS'} ({pe:+.3f})")
    print(f"  P-M35c (stage-0 attenuation tracks stages1-3, gap < 15pp): "
          f"{'HIT' if p_c else 'MISS'}")

    (GAPFILL / "m35_reff_reaudit.json").write_text(json.dumps(
        {"gate_median": med, "spearman_Rall_Reff": sp, "flips": flips,
         "conditions": {c: base[c] for c in live}, "arms": fe,
         "pearson_eff": pe, "pearson_all": pa,
         "P_M35a": bool(p_a), "P_M35b": bool(p_b), "P_M35c": bool(p_c)}, indent=1))
    print(f"\nwrote {GAPFILL / 'm35_reff_reaudit.json'}")


if __name__ == "__main__":
    main()
