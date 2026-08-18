"""M8 — PER-POSITION support. H4 revived in the only form M7 leaves open.

M7 established that the causal variable is a property of the displacement-to-POSITION ASSIGNMENT:
shuffling changes only which position receives which displacement, and damage collapses, while every
scalar field statistic tried so far (magnitude, channel direction, magnitude co-location, local
orthogonal fraction) is unchanged or uninformative.

H4 (representation support) is on the register as CONTESTED -- but it was only ever tested GLOBALLY,
as Mahalanobis distance on spatially-POOLED features. Pooling is exactly the operation that discards
assignment. So the per-position form of H4 has never been measured, and it is the one candidate that
lives in the space M7 identified.

THE IDEA, stated so it can be wrong:
    Z(p) + delta(p)   is a feature the network actually produces for some input -> ON the realisable
                      per-position feature manifold
    Z(p) + delta(q)   (shuffled) is a combination no input produces           -> OFF it

If damage requires the perturbed feature to remain PLAUSIBLE -- wrong but realisable, so the decoder
acts on it rather than being unmoved by it -- then the true assignment stays in-support and the
shuffle leaves it, and that difference is what shuffling destroys.

SUPPORT STATISTIC: kNN distance (k=8) from a per-position feature to a bank of CLEAN per-position
features, built from held-out images. Chosen over Mahalanobis because a per-position feature
distribution is multi-modal (sky, wall, edge...) and a single Gaussian would blur exactly the
structure at issue. Bank is clean-only and never sees damage.

FILED BEFORE ANY NUMBER:
    P-M8a  kNN(Z+delta_true) < kNN(Z+delta_shuffled) -- the true assignment stays closer to the
           realisable manifold than the shuffled one                              P_true 0.70
    P-M8b  across the 14 conditions, damage correlates with the perturbed features REMAINING
           in-support (|Spearman| >= 0.50 of damage against mean kNN distance)     P_true 0.40
    P-M8c  depth_offset's perturbed features are indistinguishable from clean
           (kNN ratio within 1%) -- positive control, it must hold                 P_true 0.90

P-M8b at 0.40, below a coin flip: five candidates have died here and four of them died by being
uninformative rather than wrong. The honest prior is that this one joins them. P-M8a is the
mechanism test and is filed higher because it is close to a construction fact -- but it is exactly
what M7 assumed about the shuffle and DISPROVED for local geometry, so it is not free.

    python3 scripts/m8_local_support.py --gpu 1 --n-eval 10 --n-bank 8
"""
import argparse
import json
import os
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
SEED = 20260721
K = 8
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, measured_damage, spearman, corrupt  # noqa: E402


def knn_dist(Q, bank, k=K, cap=4096):
    """mean distance from each query per-position feature to its k-th nearest clean neighbour."""
    if Q.shape[0] > cap:
        idx = torch.linspace(0, Q.shape[0] - 1, cap).long()
        Q = Q[idx]
    d = torch.cdist(Q.unsqueeze(0), bank.unsqueeze(0)).squeeze(0)
    kk = min(k, d.shape[1])
    return float(d.topk(kk, largest=False).values[:, -1].mean())


def flat(feats, stage=0):
    f = feats[stage][0]
    return f.reshape(f.shape[0], -1).T.contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--n-bank", type=int, default=8)
    ap.add_argument("--stage", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    dmg = measured_damage()
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

    def _dec(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def cap_(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    names = [c for c in COND if c in dmg]
    g = torch.Generator().manual_seed(SEED)

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        it = iter(vl)
        # ---- CLEAN BANK, built from held-out images, never sees a corruption
        bank = []
        for _ in range(a.n_bank):
            b = next(it)
            bank.append(flat(cap_(b["data"].cuda(), b["modal_x"].cuda()), a.stage))
        B = torch.cat(bank)
        if B.shape[0] > 20000:
            B = B[torch.linspace(0, B.shape[0] - 1, 20000).long()]
        print(f"clean bank: {B.shape[0]} per-position features, dim {B.shape[1]}")

        acc = {c: [] for c in names}
        clean_ref, true_ref, shuf_ref = [], [], []
        for i, b in enumerate(it):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            Zc = cap_(rgb, dep)
            zc = flat(Zc, a.stage)
            clean_ref.append(knn_dist(zc, B))
            for c in names:
                Zk = cap_(rgb, corrupt(dep, COND[c][1]))
                acc[c].append(knn_dist(flat(Zk, a.stage), B))
            # P-M8a: true vs shuffled assignment, on the M3 reference condition
            Zk = cap_(rgb, corrupt(dep, COND["invalid_0.10"][1]))
            d0 = Zk[a.stage] - Zc[a.stage]
            C, H, W = d0.shape[1], d0.shape[2], d0.shape[3]
            idx = torch.randperm(H * W, generator=g).to(d0.device)
            sh = d0.reshape(1, C, H * W)[:, :, idx].reshape(1, C, H, W)
            true_ref.append(knn_dist(flat([Zc[a.stage] + d0], 0), B))
            shuf_ref.append(knn_dist(flat([Zc[a.stage] + sh], 0), B))

    cl = float(np.mean(clean_ref))
    rows = {c: {"knn": float(np.mean(acc[c])), "damage": dmg[c]} for c in names}
    print(f"\nclean reference kNN distance: {cl:.4f}\n")
    print(f"{'condition':22s} {'damage':>7s} {'kNN':>10s} {'excess':>10s}")
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        r = rows[c]
        print(f"{c:22s} {r['damage']:7.2f} {r['knn']:10.4f} {r['knn']-cl:10.4f}")

    ns = list(rows)
    d = [rows[c]["damage"] for c in ns]
    r_k = spearman([rows[c]["knn"] for c in ns], d)
    print(f"\nSpearman(kNN excess, damage) = {r_k:+.3f}")

    tr, sr = float(np.mean(true_ref)), float(np.mean(shuf_ref))
    print(f"\nP-M8a — true vs shuffled assignment (invalid_0.10):")
    print(f"  true assignment    kNN {tr:.4f}")
    print(f"  shuffled assignment kNN {sr:.4f}   ratio {sr/max(tr,1e-9):.3f}")

    off = rows.get("depth_offset", {}).get("knn", float("nan"))
    p_a = tr < sr
    p_b = r_k is not None and abs(r_k) >= 0.50
    p_c = abs(off - cl) / max(cl, 1e-9) < 0.01
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M8a true stays closer to the manifold than shuffled -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M8b |Spearman(kNN, damage)| >= 0.50: {r_k:+.3f} -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M8c depth_offset within 1% of clean: {abs(off-cl)/max(cl,1e-9)*100:.2f}% -> "
          f"{'HIT' if p_c else 'MISS'}")

    out = PROJECT / "results" / "gapfill" / "m8_local_support.json"
    out.write_text(json.dumps({"clean_ref": cl, "rows": rows, "true_knn": tr, "shuf_knn": sr,
                               "spearman_knn_damage": r_k, "P_M8a": bool(p_a),
                               "P_M8b": bool(p_b), "P_M8c": bool(p_c)}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
