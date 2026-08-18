"""M12 — anomaly triage RUNG 1: is the shuffle collapse a METRIC ARTIFACT?

`research-judgment.md` §4 rung 1: "does it survive a *continuous* metric, a different aggregation, a
different split? (Two-regime findings die here more often than anywhere else.)"

THE ANOMALY. Shuffling removes ~100% of mIoU damage but only ~65% of argmax flips (M11), both from
the same forward passes. mIoU is BOTH argmax-thresholded AND class-averaged; those are two separate
suspects and this campaign has never separated them.

FIVE MEASURES ON THE SAME LOGITS, removing the suspects one at a time:

    mIoU        argmax + class-averaged      <- the campaign's metric, both suspects present
    pixel_acc   argmax, NOT class-averaged   <- removes class-averaging only
    fwIoU       argmax, frequency-weighted   <- removes class-averaging a second way
    CE          continuous, needs GT         <- removes argmax
    KL          continuous, NO GT, NO argmax <- removes BOTH; cannot be rescued by aggregation

If the collapse is representational it must appear in KL, which sees every logit change with no
threshold and no class weighting. If it appears ONLY in mIoU, the campaign's most load-bearing causal
result is a property of the metric.

Prereg + outcome table + severity: .grill/prediction-ledger.md (2026-08-10, M12).
RATIO DISCIPLINE: raw values always; a ratio only where the real arm clears a floor, else VOID.

    python3 scripts/m12_metric_artifact.py --gpu 1 --n-eval 30
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
FLOOR = 0.5          # a ratio is VOID unless the real arm exceeds this
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, corrupt  # noqa: E402

CONDS = ["dconst", "invalid_0.10", "invalid_0.55"]
ARMS = ("real", "shuf", "uni")


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def metrics_from_hist(h):
    h = h.double()
    tp = h.diag()
    union = h.sum(0) + h.sum(1) - tp
    iou = tp / union
    v = union > 0
    miou = float(iou[v].mean()) * 100.0
    pacc = float(tp.sum() / h.sum()) * 100.0
    freq = h.sum(1) / h.sum()
    fw = float((freq[v] * iou[v]).sum()) * 100.0
    return miou, pacc, fw


def ratio(real, shuf, floor=FLOOR):
    """Guarded ratio — VOID when the denominator cannot support one (lesson D-1)."""
    return (shuf / real) if abs(real) > floor else None


def fmt(x):
    return "  VOID" if x is None else f"{x:6.3f}"


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
    acc = {}
    for c in CONDS:
        for arm in ARMS:
            H[f"{c}|{arm}"] = torch.zeros(NCLS, NCLS, dtype=torch.long)
            acc[f"{c}|{arm}"] = {"kl": [], "ce": [], "c2w": [], "w2c": [], "w2w": []}
    ce_clean = []

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        canary = False
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
                print(f"\nCANARY self-comparison: max|diff| = {d:.3e}")
                if d > 0:
                    raise SystemExit("REFUSING: nondeterministic decoder")
                print("  bases pinned.\n")
                canary = True

            arg_c = zl.argmax(1).squeeze(0)
            valid = (lab >= 0) & (lab < NCLS)
            corr_c = (arg_c == lab) & valid
            H["clean"] += hist(arg_c.cpu(), lab.cpu())
            logp_c = F.log_softmax(zl, 1)
            ce_clean.append(float(F.nll_loss(logp_c, lab.unsqueeze(0), ignore_index=255)))

            for c in CONDS:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                delta = [k - z for k, z in zip(Zk, Zc)]
                sh, un = [], []
                for d_ in delta:
                    B, C, Hh, Ww = d_.shape
                    idx = torch.randperm(Hh * Ww, generator=g).to(d_.device)
                    sh.append(d_.reshape(B, C, Hh * Ww)[:, :, idx].reshape(B, C, Hh, Ww))
                    un.append(d_.mean(dim=(2, 3), keepdim=True).expand_as(d_))
                for arm, fs in (("real", Zk),
                                ("shuf", [z + d_ for z, d_ in zip(Zc, sh)]),
                                ("uni",  [z + d_ for z, d_ in zip(Zc, un)])):
                    zz = dec(fs, size)
                    ar = zz.argmax(1).squeeze(0)
                    H[f"{c}|{arm}"] += hist(ar.cpu(), lab.cpu())
                    k = acc[f"{c}|{arm}"]
                    # KL(clean || perturbed): continuous, NO ground truth, NO argmax
                    k["kl"].append(float(F.kl_div(F.log_softmax(zz, 1), logp_c,
                                                  log_target=True, reduction="batchmean")))
                    k["ce"].append(float(F.nll_loss(F.log_softmax(zz, 1),
                                                    lab.unsqueeze(0), ignore_index=255)))
                    corr_k = (ar == lab) & valid
                    ch = (ar != arg_c) & valid
                    n = float(valid.sum())
                    k["c2w"].append(float((ch & corr_c & ~corr_k).sum()) / n * 100)
                    k["w2c"].append(float((ch & ~corr_c & corr_k).sum()) / n * 100)
                    k["w2w"].append(float((ch & ~corr_c & ~corr_k).sum()) / n * 100)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{a.n_eval}")

    bm, bp, bf = metrics_from_hist(H["clean"])
    bce = float(np.mean(ce_clean))
    print(f"\nclean subset: mIoU {bm:.2f}  pixAcc {bp:.2f}  fwIoU {bf:.2f}  CE {bce:.4f}\n")

    R = {}
    print(f"{'cond':14s} {'arm':>5s} | {'dmIoU':>7s} {'dPixAcc':>8s} {'dfwIoU':>7s} "
          f"{'dCE':>7s} {'KL':>7s} | {'C->W':>6s} {'W->C':>6s} {'W->W':>6s}")
    for c in CONDS:
        for arm in ARMS:
            m, p, f_ = metrics_from_hist(H[f"{c}|{arm}"])
            k = acc[f"{c}|{arm}"]
            R[f"{c}|{arm}"] = {
                "dmiou": bm - m, "dpixacc": bp - p, "dfwiou": bf - f_,
                "dce": float(np.mean(k["ce"])) - bce, "kl": float(np.mean(k["kl"])),
                "c2w": float(np.mean(k["c2w"])), "w2c": float(np.mean(k["w2c"])),
                "w2w": float(np.mean(k["w2w"]))}
            r = R[f"{c}|{arm}"]
            print(f"{c:14s} {arm:>5s} | {r['dmiou']:7.2f} {r['dpixacc']:8.2f} "
                  f"{r['dfwiou']:7.2f} {r['dce']:7.3f} {r['kl']:7.4f} | "
                  f"{r['c2w']:6.2f} {r['w2c']:6.2f} {r['w2w']:6.2f}")
        print()

    print("--- SHUF/REAL RATIO PER MEASURE (VOID where the real arm is below the floor) ---")
    print(f"{'cond':14s} {'mIoU':>7s} {'pixAcc':>7s} {'fwIoU':>7s} {'CE':>7s} {'KL':>7s} {'C->W':>7s}")
    ratios = {}
    for c in CONDS:
        rr, ss = R[f"{c}|real"], R[f"{c}|shuf"]
        ratios[c] = {
            "miou": ratio(rr["dmiou"], ss["dmiou"]),
            "pixacc": ratio(rr["dpixacc"], ss["dpixacc"]),
            "fwiou": ratio(rr["dfwiou"], ss["dfwiou"]),
            "ce": ratio(rr["dce"], ss["dce"], 0.05),
            "kl": ratio(rr["kl"], ss["kl"], 0.01),
            "c2w": ratio(rr["c2w"], ss["c2w"], 0.5)}
        q = ratios[c]
        print(f"{c:14s} {fmt(q['miou'])} {fmt(q['pixacc'])} {fmt(q['fwiou'])} "
              f"{fmt(q['ce'])} {fmt(q['kl'])} {fmt(q['c2w'])}")

    def allv(key, op, thr):
        vals = [ratios[c][key] for c in CONDS if ratios[c][key] is not None]
        return bool(vals) and all(op(v, thr) for v in vals), vals

    p_a, kls = allv("kl", lambda v, t: v >= t, 0.30)
    p_b, pas = allv("pixacc", lambda v, t: v <= t, 0.35)
    p_c, ces = allv("ce", lambda v, t: v >= t, 0.30)
    p_d, c2s = allv("c2w", lambda v, t: v <= t, 0.35)

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M12a KL does NOT collapse (>=0.30): {[round(v,3) for v in kls]} -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M12b pixAcc DOES collapse (<=0.35): {[round(v,3) for v in pas]} -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M12c CE does NOT collapse (>=0.30): {[round(v,3) for v in ces]} -> {'HIT' if p_c else 'MISS'}")
    print(f"P-M12d C->W flips DO collapse (<=0.35): {[round(v,3) for v in c2s]} -> {'HIT' if p_d else 'MISS'}")

    print(f"\n--- VERDICT (rung 1 of the anomaly triage ladder) ---")
    if not p_a and not p_c:
        print("  CONTINUOUS METRICS COLLAPSE TOO. The metric-artifact hypothesis is KILLED: the")
        print("  shuffle collapse is visible without argmax and without class weighting, so it is")
        print("  REPRESENTATIONAL. The anomaly CLEARS rung 1 and the odd measure was the flip COUNT")
        print("  -> promote to rung 2 (implementation bug in the flip counter).")
    elif p_a and p_b:
        print("  LOCALISED TO ARGMAX CROSSING. Continuous divergence survives the shuffle while")
        print("  every argmax-based measure collapses -> the shuffle does not remove the")
        print("  representational change, it removes the DECISION change. 'Damage is representation")
        print("  displacement' is KILLED in its unqualified form.")
    elif p_a and not p_b:
        print("  CLASS-AVERAGING IS IMPLICATED. Only mIoU collapses; pixel accuracy does not.")
        print("  M3 is not representational evidence -> Level-1/2 evidence needs a scope edit.")
    else:
        print("  MIXED. Report the raw table; do not promote past rung 1 on a mixed result.")

    o = GAPFILL / "m12_metric_artifact.json"
    o.write_text(json.dumps({"clean": {"miou": bm, "pixacc": bp, "fwiou": bf, "ce": bce},
                             "rows": R, "ratios": ratios, "n_eval": a.n_eval,
                             "P_M12a": p_a, "P_M12b": p_b, "P_M12c": p_c,
                             "P_M12d": p_d}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
