"""M39 (E1) — THE MATCHED POSITIONAL CONTROL for the reliance result.

THE OBJECTION THIS ANSWERS, and it is the first thing a reviewer will say about C3:

    "You scaled a term during training and the model's depth reliance fell. Would scaling ANY term
     have done that? Is this about depth, or about attenuating something?"

`mainD_posatten` is exactly that control and it was trained but never evaluated. It attenuates the
POSITIONAL term of the same attention bias, at hyperparameters matched to the depth-attenuation arm
(POS_ATTEN=1, POS_ALPHAMIN=0.0508, POS_PCLEAN=0.5), 500 epochs, 2 seeds. Same schedule, same budget,
same architecture, same everything — a different term scaled.

    depth-attenuation arm : scales the DEPTH term   -> d||Delta|| -29.0%, d missing +2.76
    positional control    : scales the POSITIONAL term -> ?

If the positional control shows the same reliance drop and the same missing-modality gain, C3 is
about attenuation in general and the depth-specific reading is wrong. If it does not, the effect is
specific to the depth pathway and C3 stands with its matched control discharged.

Measures both sides on the same checkpoints:
  (a) OUTCOME  — clean and depth-absent mIoU, full validation set
  (b) RELIANCE — relative ||Delta|| over the decoder-read stages, the C3 statistic

GATE: baseline per-condition sensitivity reproduces M14, median ratio within 10%.

Prereg: .grill/prediction-ledger.md (2026-08-10, M39).

    python3 scripts/m39_posatten_control.py --gpu 1 --n-eval 30 --n-miou 654
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
NCLS = 40
_CAP = {"store": None, "mode": None}

# the control, and the reference arms it must be read against
TARGETS = {
    "baseline":    ("mainB_baseline", [20260801, 20260802]),
    "posatten":    ("mainD_posatten", [20260801, 20260802]),
    "attenuation": ("mainC_attenuation", [20260801, 20260802]),
}

from m1_mig_full import COND, corrupt  # noqa: E402
from m35_reff_reaudit import measure  # noqa: E402


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def miou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    return float((tp / u)[u > 0].mean()) * 100.0


def eval_outcome(ckpt, cfg, loader, n_miou):
    """clean and depth-absent mIoU on the full validation set."""
    from models.builder import EncoderDecoder
    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    if len(sd) - len(miss.missing_keys) < 0.9 * len(sd):
        raise SystemExit("REFUSING: <90% loaded")
    model.eval().cuda()
    Hc = torch.zeros(NCLS, NCLS, dtype=torch.long)
    Hm = torch.zeros(NCLS, NCLS, dtype=torch.long)
    for i, b in enumerate(loader()):
        if i >= n_miou:
            break
        rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
        lab = b["label"].cuda().squeeze(0).long()
        size = rgb.shape[-2:]
        for H, d in ((Hc, dep), (Hm, torch.zeros_like(dep))):
            torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
            with torch.no_grad():
                o = model(rgb, d)
            if isinstance(o, dict) and "pred" in o:
                o = o["pred"]
            o = F.interpolate(o, size=size, mode="bilinear", align_corners=False)
            H += hist(o.argmax(1).squeeze(0).cpu(), lab.cpu())
    return miou(Hc), miou(Hm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=30)
    ap.add_argument("--n-miou", type=int, default=654)
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

    out = {}
    for arm, (pref, seeds) in TARGETS.items():
        for s in seeds:
            ck = P0 / f"{pref}_s{s}" / "ckpt" / "latest.pth"
            if not ck.exists():
                print(f"  SKIP {arm} s{s}")
                continue
            print(f"=== {arm} s{s} ===", flush=True)
            cl, ms = eval_outcome(ck, cfg, lambda: vl, a.n_miou)
            rel = measure(ck, cfg, names, lambda: vl, a.n_eval)
            out[(arm, s)] = {"clean": cl, "missing": ms, "rel": rel}
            print(f"    clean {cl:.2f}   depth-absent {ms:.2f}", flush=True)

    b0 = out[("baseline", 20260801)]["rel"]
    gc = [c for c in names if m14[c].get("R") is not None]
    ratios = sorted(b0[c]["R_all"] / m14[c]["R"] for c in gc)
    med = ratios[len(ratios) // 2]
    print(f"\n=== GATE median {med:.4f} -> {'PASS' if abs(med-1)<0.10 else 'FAIL'}")
    if abs(med - 1) >= 0.10:
        raise SystemExit("REFUSING: gate failed")

    live = [c for c in names if b0[c]["n_all"] > 1e-6 and c != "depth_offset"]

    rows = {}
    for (arm, s), v in out.items():
        if arm == "baseline":
            continue
        b = out[("baseline", s)]
        dN = [100 * (v["rel"][c]["n_eff"] / b["rel"][c]["n_eff"] - 1) for c in live]
        rows[f"{arm}_s{s}"] = {
            "arm": arm, "seed": s,
            "dN": st.mean(dN), "dN_sd": st.stdev(dN),
            "dclean": v["clean"] - b["clean"],
            "dmiss": v["missing"] - b["missing"],
        }

    print(f"\n{'arm-seed':24s} {'d|D|%':>9s} {'sd':>6s} {'d clean':>9s} {'d absent':>9s}")
    for k in sorted(rows):
        r = rows[k]
        print(f"{k:24s} {r['dN']:+9.1f} {r['dN_sd']:6.1f} {r['dclean']:+9.2f} {r['dmiss']:+9.2f}")

    summ = {}
    for arm in ("posatten", "attenuation"):
        v = [rows[k] for k in rows if rows[k]["arm"] == arm]
        if not v:
            continue
        summ[arm] = {k: st.mean([x[k] for x in v]) for k in ("dN", "dclean", "dmiss")}
        summ[arm]["n"] = len(v)
    print(f"\n{'arm':14s} {'n':>2s} {'d|D|%':>9s} {'d clean':>9s} {'d absent':>9s}")
    for arm in ("posatten", "attenuation"):
        if arm in summ:
            v = summ[arm]
            print(f"{arm:14s} {v['n']:2d} {v['dN']:+9.1f} {v['dclean']:+9.2f} {v['dmiss']:+9.2f}")

    if "posatten" in summ and "attenuation" in summ:
        p, d = summ["posatten"], summ["attenuation"]
        p_a = abs(p["dN"]) < 0.5 * abs(d["dN"])
        p_b = p["dmiss"] < 0.5 * d["dmiss"]
        print(f"\n  P-M39a (positional control moves reliance < half as much): "
              f"{'HIT' if p_a else 'MISS'}   ({p['dN']:+.1f} vs {d['dN']:+.1f})")
        print(f"  P-M39b (positional control gains < half the depth-absent effect): "
              f"{'HIT' if p_b else 'MISS'}   ({p['dmiss']:+.2f} vs {d['dmiss']:+.2f})")
        summ["P_M39a"], summ["P_M39b"] = bool(p_a), bool(p_b)

    (GAPFILL / "m39_posatten_control.json").write_text(json.dumps(
        {"gate_median": med, "rows": rows, "summary": summ,
         "outcome": {f"{k[0]}_s{k[1]}": {"clean": v["clean"], "missing": v["missing"]}
                     for k, v in out.items()},
         "n_eval": a.n_eval, "n_miou": a.n_miou}, indent=1))
    print(f"\nwrote {GAPFILL / 'm39_posatten_control.json'}")


if __name__ == "__main__":
    main()
