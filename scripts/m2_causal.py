"""M2 — Level 2. Does DIRECTION matter at matched magnitude, or only magnitude?

This is MIG's fairest and last test. M0/M1 refuted MIG CORRELATIONALLY: ||delta|| at the decoder
input predicts damage at rho=0.952 across 7 families, better than the low-rank coordinate (0.921).
But correlation cannot distinguish "magnitude is what matters" from "the corruptions we happen to
have all move along the influence directions, so magnitude and direction are confounded".

M2 breaks that confound by INTERVENING: inject a displacement directly at the decoder input, at a
MATCHED magnitude, along different directions.

    influence   the top-1 direction of the measured corruption-displacement basis
    orthogonal  a direction in the orthogonal complement of the top-k influence basis
    random      an isotropic random direction

All three at identical ||delta||, applied to the SAME images, measured with the SAME metric.

OUTCOME TABLE — every branch kills something:
    damage(influence) >> damage(orthogonal) at matched magnitude
        -> DIRECTION MATTERS. MIG revives at Level 2: there IS a modality-influence subspace, and
           M1's correlational result was confounded. The low-rank object is real after all.
    damage(influence) ~= damage(orthogonal) ~= damage(random)
        -> ONLY MAGNITUDE MATTERS. MIG is dead causally as well as correlationally, and
           Displacement-Bounded training is the correct principle.
    damage(orthogonal) >> damage(influence)
        -> the basis is anti-informative; the measurement is wrong and both prior probes are void.

FILED BEFORE THE NUMBERS:
    P-M2a  damage(influence) > damage(orthogonal) by more than the clean-vs-clean noise   P_true 0.45
    P-M2b  damage(random) lies between them                                               P_true 0.50

P-M2a at 0.45 -- BELOW a coin flip. M1 already showed the low-rank projection predicts damage WORSE
than the raw norm, so the honest prior is that direction does not matter. Filing above 0.5 here
would be filing against my own most recent measurement.

Damage is measured as mIoU drop against the SAME subset's clean mIoU, using the repo's own
hist_info/compute_score, so it is comparable within this script. It is NOT comparable to the frozen
654-image table and is never reported as if it were.

    python3 scripts/m2_causal.py --gpu 1 --n-eval 100
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

CKPT = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
SEED = 20260721
_FEAT, _INJECT = [], {"vec": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--n-basis", type=int, default=24)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    from utils.metric import hist_info, compute_score
    from depth_corruptions import depth_invalid, rectangular_holes

    model = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(miss.missing_keys)
    print(f"checkpoint: {got}/{len(sd)} tensors loaded")
    if got < 0.9 * len(sd):
        raise SystemExit("REFUSING: <90% loaded")
    model.eval().cuda()

    NC = cfg.num_classes
    _orig = model.decode_head.forward

    def _dec(x, *args, **kw):
        feats = x if isinstance(x, (list, tuple)) else [x]
        _FEAT.append(torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1).detach().float().cpu())
        v = _INJECT["vec"]
        if v is not None:
            # add the displacement to every spatial position of every stage, split by channel block
            out, o = [], 0
            for f in feats:
                c = f.shape[1]
                out.append(f + v[o:o + c].to(f.device).view(1, c, 1, 1))
                o += c
            feats = out
            x = feats if isinstance(x, (list, tuple)) else feats[0]
        return _orig(x, *args, **kw)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def run(rgb, dep):
        _FEAT.clear()
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = model(rgb, dep)
        return o, _FEAT[-1].clone()

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

        # ---- 1. build the influence basis from real corruptions (no damage used)
        disp = []
        for i, b in enumerate(vl):
            if i >= a.n_basis:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            _, zc = run(rgb, dep)
            for f in (lambda d: depth_invalid(d.clone(), frac=0.10, seed=SEED),
                      lambda d: depth_invalid(d.clone(), frac=0.55, seed=SEED),
                      lambda d: rectangular_holes(d.clone(), num_holes=2,
                                                  hole_frac=0.224, seed=SEED)):
                _, zk = run(rgb, f(dep))
                disp.append((zk - zc).numpy()[0])
        D = np.stack(disp)
        U, sv, Vt = np.linalg.svd(D, full_matrices=False)
        dim = D.shape[1]
        k = 3
        v_inf = Vt[0]
        # orthogonal complement of the top-k influence basis
        g = np.random.default_rng(SEED).normal(size=dim)
        g -= Vt[:k].T @ (Vt[:k] @ g)
        v_orth = g / (np.linalg.norm(g) + 1e-12)
        rr = np.random.default_rng(SEED + 1).normal(size=dim)
        v_rand = rr / (np.linalg.norm(rr) + 1e-12)
        leak = float(abs(Vt[:k] @ v_orth).max())
        print(f"basis dim {dim}, top-3 singular {[round(float(x),2) for x in sv[:3]]}, "
              f"orthogonality leak {leak:.2e}")
        if leak > 1e-6:
            raise SystemExit("REFUSING: orthogonal direction is not orthogonal")

        # magnitude matched to the MEAN real-corruption displacement
        mag = float(np.linalg.norm(D, axis=1).mean())
        print(f"matched injection magnitude ||delta|| = {mag:.3f}\n")

        arms = {"clean": None, "influence": v_inf, "orthogonal": v_orth, "random": v_rand}
        hists = {a_: np.zeros((NC, NC)) for a_ in arms}
        corr = {a_: [0, 0] for a_ in arms}
        vl2, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl2):
            if i >= a.n_eval:
                break
            rgb = b["data"].cuda()
            dep = b["modal_x"].cuda()
            gt = b["label"].numpy()[0]
            for name, v in arms.items():
                _INJECT["vec"] = None if v is None else torch.from_numpy(v * mag).float()
                out, _ = run(rgb, dep)
                pred = out.argmax(1).cpu().numpy()[0]
                if pred.shape != gt.shape:
                    import torch.nn.functional as F
                    pr = F.interpolate(out.float(), size=gt.shape, mode="bilinear",
                                       align_corners=False)
                    pred = pr.argmax(1).cpu().numpy()[0]
                h, lab, cor = hist_info(NC, pred, gt)
                hists[name] += h
                corr[name][0] += cor
                corr[name][1] += lab
        _INJECT["vec"] = None

    res = {}
    for name in arms:
        iu, mean_IU, _, mean_pixel_acc = compute_score(hists[name], corr[name][0], corr[name][1])[:4]
        res[name] = float(mean_IU) * 100
    clean = res["clean"]
    print(f"{'arm':12s} {'mIoU':>8s} {'damage':>8s}")
    for name in arms:
        print(f"{name:12s} {res[name]:8.2f} {clean - res[name]:8.2f}")

    d_inf, d_orth, d_rand = (clean - res[k_] for k_ in ("influence", "orthogonal", "random"))
    p_a = d_inf > d_orth + 0.5
    p_b = min(d_inf, d_orth) <= d_rand <= max(d_inf, d_orth)
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M2a damage(influence) > damage(orthogonal) + 0.5 -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M2b damage(random) between them -> {'HIT' if p_b else 'MISS'}")
    print("\n--- VERDICT ---")
    if d_inf > d_orth + 0.5:
        print("  DIRECTION MATTERS. MIG revives at Level 2: at matched magnitude, displacement")
        print("  along the influence basis does more damage than orthogonal displacement.")
    elif abs(d_inf - d_orth) <= 0.5:
        print("  ONLY MAGNITUDE MATTERS. At matched ||delta||, direction is irrelevant.")
        print("  MIG is dead causally as well as correlationally; the scalar is the object.")
    else:
        print("  ANTI-INFORMATIVE BASIS — orthogonal hurts more than influence. Both prior")
        print("  probes are void and the measurement must be re-derived.")

    p = PROJECT / "results" / "gapfill" / "m2_causal.json"
    p.write_text(json.dumps({"n_eval": a.n_eval, "magnitude": mag, "miou": res,
                             "damage": {"influence": d_inf, "orthogonal": d_orth,
                                        "random": d_rand},
                             "P_M2a": bool(p_a), "P_M2b": bool(p_b)}, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
