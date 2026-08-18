"""M0 — the cheapest test that could falsify MIG. Pre-registered at plan/MIG-PROGRAM.md.

THE FALSIFIER. Three conditions zero ~10% of depth pixels and differ ONLY in spatial arrangement:

    invalid_0.10  scattered     damage 1.84   (1.73/2.05/1.75)
    holes_1blob   one blob      damage 0.54   (0.53/0.63/0.47)
    holes_2blob   two blobs     damage 0.24   (0.07/0.41/0.24)

Same modality, fraction, fill value and corruption kind. If MIG's two terms are equal across them
while damage differs by 1.60, the pre-registered forbid is violated and the hypothesis is dead.

FILED BEFORE ANY NUMBER (do not edit after seeing output):
    P-M0a  Mahalanobis rank matches damage rank: invalid > blob1 > blob2      P_true 0.65
    P-M0b  D_M(invalid)/D_M(blob2) > 2.0                                      P_true 0.55
    P-M0c  positive control: depth_offset influence-coord norm < 5% of invalid P_true 0.85

BASELINES MIG MUST BEAT (added after the corpus sweep surfaced two rivals):
    B0  ||delta||        pure magnitude
    B1  single global direction   -- the CVPR-2026 modality-gap precedent: ONE vector monotonically
                                     governs perturbation robustness, so one dimension may suffice
    B2  support alone    -- RIVAL-1: NeurIPS 2023 arXiv:2303.15488 shows a large domain gap does
                            NOT imply low accuracy, i.e. support may not predict damage at all

CANARIES, all of which can fail:
  * checkpoint must load >=90% of tensors (a silent strict=False once ran 3 probes on an untrained
    model and produced a whole false finding)
  * Ham/NMF decoder redraws random bases EVERY forward -> reseed per forward, and prove it by
    comparing two identical forwards; without this every delta below is noise
  * matched-norm random perturbation must NOT reproduce the ordering

    python3 scripts/m0_mig.py --gpu 1 --n-fit 120 --n-eval 48
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

# measured damage, per-seed mean, from results/gapfill (verified 2026-08-09)
DAMAGE = {"invalid_0.10": 1.84, "holes_1blob": 0.54, "holes_2blob": 0.24, "depth_offset": 0.00}

_FEAT = []


def corrupt(d, name, rng):
    from depth_corruptions import rectangular_holes, depth_invalid
    if name == "clean":
        return d
    if name == "invalid_0.10":
        return depth_invalid(d.clone(), frac=0.10, seed=SEED)
    if name == "holes_1blob":
        return rectangular_holes(d.clone(), num_holes=1, hole_frac=0.316, seed=SEED)
    if name == "holes_2blob":
        return rectangular_holes(d.clone(), num_holes=2, hole_frac=0.224, seed=SEED)
    if name == "depth_offset":
        return d + 0.5                      # analytically zero-damage: mask_d is offset-invariant
    if name == "random_matched":
        # NEGATIVE CONTROL: gaussian noise, norm matched per-image to invalid_0.10's displacement
        ref = depth_invalid(d.clone(), frac=0.10, seed=SEED)
        target = (ref - d).norm()
        g = torch.randn(d.shape, generator=rng).to(d.device)
        return d + g * (target / (g.norm() + 1e-12))
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-fit", type=int, default=120)
    ap.add_argument("--n-eval", type=int, default=48)
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

    # Capture the DECODER INPUT — the representation the support statistic is defined on.
    # builder.py:267 calls `self.decode_head.forward(x)` DIRECTLY, which bypasses PyTorch hooks
    # (they fire on `module(...)`, not `module.forward(...)`). A registered hook here would
    # silently never fire and every feature would be missing — so wrap the method instead.
    _orig_dec = model.decode_head.forward

    def _dec(x, *args, **kw):
        feats = x if isinstance(x, (list, tuple)) else [x]
        _FEAT.append(torch.cat([f.mean(dim=(2, 3)) for f in feats], dim=1).detach().float().cpu())
        return _orig_dec(x, *args, **kw)

    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader, get_train_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def fwd(rgb, dep):
        _FEAT.clear()
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)   # Ham/NMF reseed
        with torch.no_grad():
            model(rgb, dep)
        return _FEAT[-1].clone()

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

        # ---- CANARY 0: identical forwards must reproduce, or every delta is decoder noise
        it = iter(vl)
        b0 = next(it)
        r0, d0 = b0["data"].cuda(), b0["modal_x"].cuda()
        f1, f2 = fwd(r0, d0), fwd(r0, d0)
        noise = (f1 - f2).abs().max().item()
        print(f"CANARY identical-forward feature delta: {noise:.3e}")
        if noise > 1e-5:
            raise SystemExit("REFUSING: decoder nondeterminism not controlled; deltas are noise")

        # ---- fit mu, Sigma on CLEAN features (fit set), never on corrupted, never on damage
        fit = []
        for i, b in enumerate(vl):
            if i >= a.n_fit:
                break
            fit.append(fwd(b["data"].cuda(), b["modal_x"].cuda()))
        Z = torch.cat(fit).numpy()
        mu = Z.mean(0)
        S = np.cov(Z, rowvar=False)
        S += (1e-3 * np.trace(S) / S.shape[0]) * np.eye(S.shape[0])   # shrinkage, pinned
        Sinv = np.linalg.inv(S)
        print(f"support model fitted on {Z.shape[0]} clean images, dim {Z.shape[1]}")

        def maha(z):
            d = z - mu
            return float(np.sqrt(np.einsum("ij,jk,ik->i", d, Sinv, d).mean()))

        # ---- evaluate every condition on a disjoint eval set
        conds = ["clean", "invalid_0.10", "holes_1blob", "holes_2blob",
                 "depth_offset", "random_matched"]
        rng = torch.Generator().manual_seed(SEED)
        acc = {c: [] for c in conds}
        vl2, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl2):
            if i < a.n_fit:
                continue
            if i >= a.n_fit + a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            for c in conds:
                acc[c].append(fwd(rgb, corrupt(dep, c, rng)).numpy())
        F = {c: np.concatenate(v) for c, v in acc.items()}

    base = F["clean"]
    out = {}
    for c in conds:
        if c == "clean":
            continue
        D = F[c] - base
        out[c] = {"maha": maha(F[c]), "maha_clean_ref": maha(base),
                  "norm": float(np.linalg.norm(D, axis=1).mean()),
                  "mean_disp": D.mean(0).tolist()}

    # influence directions: SVD over the mean displacements of the three matched conditions
    trio = ["invalid_0.10", "holes_1blob", "holes_2blob"]
    M = np.stack([np.array(out[c]["mean_disp"]) for c in trio])
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    for c in conds:
        if c == "clean":
            continue
        v = np.array(out[c]["mean_disp"])
        out[c]["coord"] = float(np.linalg.norm(Vt @ v))
        out[c]["coord_top1"] = float(abs(Vt[0] @ v))

    print(f"\n{'condition':16s} {'damage':>7s} {'maha':>9s} {'|delta|':>9s} "
          f"{'coord':>9s} {'top1':>9s}")
    print(f"{'clean (ref)':16s} {'0.00':>7s} {maha(base):9.3f} {'-':>9s} {'-':>9s} {'-':>9s}")
    for c in conds[1:]:
        dm = DAMAGE.get(c)
        print(f"{c:16s} {('%.2f'%dm) if dm is not None else '?':>7s} "
              f"{out[c]['maha']:9.3f} {out[c]['norm']:9.3f} "
              f"{out[c]['coord']:9.3f} {out[c]['coord_top1']:9.3f}")
    print(f"\nsingular values of the 3-condition displacement matrix: "
          f"{[round(float(x),3) for x in sv]}")

    # ---- score the filed predictions
    order = sorted(trio, key=lambda c: -out[c]["maha"])
    p_a = order == ["invalid_0.10", "holes_1blob", "holes_2blob"]
    ratio = out["invalid_0.10"]["maha"] / max(1e-9, out["holes_2blob"]["maha"])
    p_b = ratio > 2.0
    off_rel = out["depth_offset"]["coord"] / max(1e-9, out["invalid_0.10"]["coord"])
    p_c = off_rel < 0.05

    print("\n--- FILED PREDICTIONS ---")
    print(f"P-M0a rank {order}  -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M0b maha ratio invalid/blob2 = {ratio:.2f} (>2.0) -> {'HIT' if p_b else 'MISS'}")
    print(f"P-M0c depth_offset coord / invalid coord = {off_rel:.4f} (<0.05) "
          f"-> {'HIT' if p_c else 'MISS'}")

    print("\n--- CONTROLS ---")
    rn = out["random_matched"]
    print(f"negative control (matched-norm random): maha {rn['maha']:.3f}, "
          f"|delta| {rn['norm']:.3f} vs invalid |delta| {out['invalid_0.10']['norm']:.3f}")
    print("  If random matches invalid on BOTH maha and coord, the geometry measures norm only.")

    print("\n--- BASELINES MIG MUST BEAT (rank over the matched trio) ---")
    for name, key in [("B0 |delta|", "norm"), ("B1 top-1 direction", "coord_top1"),
                      ("B2 support alone", "maha")]:
        o = sorted(trio, key=lambda c: -out[c][key])
        ok = o == ["invalid_0.10", "holes_1blob", "holes_2blob"]
        print(f"  {name:22s} {o}  {'matches damage rank' if ok else 'does NOT match'}")

    p = PROJECT / "results" / "gapfill" / "m0_mig.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"n_fit": int(Z.shape[0]), "dim": int(Z.shape[1]),
                             "singular_values": [float(x) for x in sv],
                             "conditions": {c: {k: v for k, v in out[c].items()
                                                if k != "mean_disp"} for c in out},
                             "P_M0a": bool(p_a), "P_M0b": bool(p_b), "P_M0c": bool(p_c),
                             "maha_ratio": float(ratio)}, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
