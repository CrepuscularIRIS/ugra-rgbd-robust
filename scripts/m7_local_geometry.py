"""M7 — the standing puzzle. What does spatial shuffling destroy that varies with damage?

THE PUZZLE. Shuffling the displacement field collapses damage (-0.03 vs 0.50; 0.34 vs 2.85) while
preserving its magnitude and every per-channel statistic. Matched-magnitude injection does 0.12-0.19
where the real corruption does 1.84. Both stand. Nothing measured so far explains either:
  ||delta||   rho 0.952 but NOT causal (M2)
  direction   causally irrelevant at matched magnitude (M2)
  low rank    channel basis worse than a scalar (M1); spatial rank-1 dominant but non-monotone (M4)
  A_magnitude M6's alignment: spatial corr between ||delta(h,w)|| and ||Z(h,w)|| -- rho 0.240, and
              alpha=0 is NOT its argmax, so dropout does not fall out. REFUTED.

THE UNTESTED CANDIDATE. M6 compared MAGNITUDES. Shuffling destroys something magnitudes cannot see:
the PER-POSITION GEOMETRIC RELATION between the displacement and the local feature it lands on.

    at each position (h,w), decompose  delta(h,w)  relative to  Z_clean(h,w):
        parallel     component along Z(h,w)   -- rescales the local feature (confidence-like)
        orthogonal   component across Z(h,w)  -- changes the local feature's IDENTITY

Shuffling moves delta from position p to position q, where the local feature direction differs, so
it RANDOMISES the parallel/orthogonal split while leaving ||delta||, the per-channel statistics and
the multiset of local vectors exactly intact. That is precisely the invariance structure of the
shuffle control, and it is why this quantity -- unlike M6's -- is a candidate for what shuffling
destroys.

    R = mean over positions of  ||delta_perp(h,w)|| / ||delta(h,w)||     (orthogonal fraction)

FILED BEFORE ANY NUMBER (do not edit afterwards):
    P-M7a  R predicts damage across the 14 conditions at Spearman >= 0.70          P_true 0.45
    P-M7b  ||delta|| * R beats ||delta|| alone (currently +0.930)                  P_true 0.45
    P-M7c  SHUFFLING RAISES R toward the isotropic baseline -- i.e. R is a quantity
           the shuffle control actually destroys, measured directly on the shuffled
           field rather than inferred                                              P_true 0.70

P-M7a/b at 0.45, BELOW a coin flip: three quantities have already died here, and M6's near-identical
construction reached only rho 0.240. Filing above 0.5 would ignore my own most recent measurement.
P-M7c is the one I expect to hold, because it is close to a geometric identity -- and if it FAILS,
the shuffle control is not doing what I have claimed it does for four probes, which would be the
most important negative in the program.

    python3 scripts/m7_local_geometry.py --gpu 1 --n-eval 12
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
_CAP = {"store": None, "mode": None}

from m1_mig_full import COND, measured_damage, spearman, corrupt  # noqa: E402


def orth_fraction(delta, Zc):
    """R = mean over positions of ||delta_perp|| / ||delta||, relative to the LOCAL clean feature."""
    num, den = 0.0, 0
    for d, z in zip(delta, Zc):
        d0, z0 = d[0], z[0]                       # (C,H,W)
        C = d0.shape[0]
        dv = d0.reshape(C, -1).T                  # (P,C)
        zv = z0.reshape(C, -1).T
        zn = zv / (zv.norm(dim=1, keepdim=True) + 1e-12)
        par = (dv * zn).sum(1, keepdim=True) * zn
        perp = dv - par
        dn = dv.norm(dim=1)
        m = dn > 1e-8
        if m.sum() == 0:
            continue
        num += float((perp.norm(dim=1)[m] / dn[m]).sum())
        den += int(m.sum())
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=12)
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

    def cap(rgb, dep):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(rgb, dep)
        _CAP["mode"] = None
        return _CAP["store"]

    names = [c for c in COND if c in dmg]
    acc = {c: {"R": [], "nrm": []} for c in names}
    shuf_R, real_R = [], []
    g = torch.Generator().manual_seed(SEED)

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            Zc = cap(rgb, dep)
            for c in names:
                Zk = cap(rgb, corrupt(dep, COND[c][1]))
                delta = [k - z for k, z in zip(Zk, Zc)]
                acc[c]["R"].append(orth_fraction(delta, Zc))
                acc[c]["nrm"].append(float(sum(float((d ** 2).sum()) for d in delta) ** 0.5))
            # P-M7c: does SHUFFLING raise R? measured on invalid_0.10, the M3 reference
            Zk = cap(rgb, corrupt(dep, COND["invalid_0.10"][1]))
            delta = [k - z for k, z in zip(Zk, Zc)]
            sh = []
            for d in delta:
                B, C, H, W = d.shape
                idx = torch.randperm(H * W, generator=g).to(d.device)
                sh.append(d.reshape(B, C, H * W)[:, :, idx].reshape(B, C, H, W))
            real_R.append(orth_fraction(delta, Zc))
            shuf_R.append(orth_fraction(sh, Zc))

    rows = {c: {"R": float(np.nanmean(acc[c]["R"])), "nrm": float(np.mean(acc[c]["nrm"])),
                "damage": dmg[c]} for c in names}
    print(f"\n{'condition':22s} {'damage':>7s} {'|delta|':>10s} {'R(orth frac)':>13s}")
    for c in sorted(rows, key=lambda x: -rows[x]["damage"]):
        r = rows[c]
        print(f"{c:22s} {r['damage']:7.2f} {r['nrm']:10.1f} {r['R']:13.4f}")

    ns = list(rows)
    d = [rows[c]["damage"] for c in ns]
    r_n = spearman([rows[c]["nrm"] for c in ns], d)
    r_R = spearman([rows[c]["R"] for c in ns], d)
    r_p = spearman([rows[c]["nrm"] * rows[c]["R"] for c in ns], d)
    print(f"\nSpearman vs damage ({len(ns)} conditions):")
    print(f"  ||delta||        {r_n:+.3f}")
    print(f"  R                {r_R:+.3f}")
    print(f"  ||delta|| * R    {r_p:+.3f}")

    mr, ms = float(np.nanmean(real_R)), float(np.nanmean(shuf_R))
    print(f"\nP-M7c — does shuffling change R?  real {mr:.4f}  shuffled {ms:.4f}  "
          f"delta {ms-mr:+.4f}")

    p_a = r_R is not None and abs(r_R) >= 0.70
    p_b = r_p is not None and r_n is not None and r_p > r_n
    p_c = ms > mr + 0.005
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M7a |Spearman(R, damage)| >= 0.70 -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M7b ||delta||*R beats ||delta|| -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M7c shuffling RAISES R -> {'HIT' if p_c else 'MISS'}")
    if not p_c:
        print("  *** If P-M7c MISSED, the shuffle control does NOT randomise local geometry.")
        print("  *** Four probes have leaned on an interpretation of that control. Report it.")

    out = PROJECT / "results" / "gapfill" / "m7_local_geometry.json"
    out.write_text(json.dumps({"rows": rows, "real_R": mr, "shuffled_R": ms,
                               "spearman": {"norm": r_n, "R": r_R, "prod": r_p},
                               "P_M7a": bool(p_a), "P_M7b": bool(p_b),
                               "P_M7c": bool(p_c)}, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
