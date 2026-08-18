"""M15 — CROSS-ARCHITECTURE generality of R. Does the shuffle collapse survive a non-NMF decoder?

THE CONFOUND. DFormerv2's decoder is LightHamHead — a rank-r NMF reconstruction, which annihilates
spatially incoherent input BY CONSTRUCTION. So the 7-13x shuffle collapse of R = KL/||Delta||^2 may
be a property of THAT DECODER rather than of multimodal representations. It is the single largest
threat to the account.

THE TEST. CMNeXt-B4 uses SegFormerHead — a per-scale MLP head with no NMF and no low-rank
reconstruction. There is no mechanism in it that preferentially annihilates spatially incoherent
input, so if the collapse reproduces here it is not an NMF artifact.

DISANALOGY (stated, not discovered later): CMNeXt's NYU pipeline feeds HHA-encoded depth (3-channel),
not raw depth. Corruption families are ANALOGOUS, not identical. What is under test is the
architecture-generality of the shuffle collapse and the quadratic law — not condition-by-condition
agreement, which the disanalogy forbids.

THE GATE. This probe builds its own model, so it must reproduce a known reference number before any
internal is read: prior R1 runs on this checkpoint gave clean mIoU 54.2-55.13, so the gate is
[52, 57]. A silent load failure once ran three probes on an untrained model in this campaign.

Prereg: .grill/prediction-ledger.md (2026-08-10, M15).

    python3 scripts/m15_cross_arch.py --gpu 1 --n-eval 30
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
DELIVER = PROJECT / "repos" / "DELIVER"
sys.path.insert(0, str(DELIVER))
sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))

CKPT = DELIVER / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
NCLS = 40
GATE_LO, GATE_HI = 52.0, 57.0
_CAP = {"store": None, "mode": None}

CONDS = {
    "constant":     lambda d, DC: DC.depth_constant(d.clone()),
    "invalid_0.10": lambda d, DC: DC.depth_invalid(d.clone(), frac=0.10, seed=SEED),
    "invalid_0.55": lambda d, DC: DC.depth_invalid(d.clone(), frac=0.55, seed=SEED),
    "shift_4":      lambda d, DC: DC.depth_shift(d.clone(), dx=4, dy=0),
    "shift_32":     lambda d, DC: DC.depth_shift(d.clone(), dx=32, dy=0),
    "gaussian_0.1": lambda d, DC: DC.gaussian_noise(d.clone(), sigma=0.1, seed=SEED),
    "half_missing": lambda d, DC: DC.half_missing(d.clone(), seed=SEED),
    "flip_h":       lambda d, DC: DC.depth_flip(d.clone(), axis="h"),
}


def hist(pred, lab):
    m = (lab >= 0) & (lab < NCLS)
    return torch.bincount(NCLS * lab[m].long() + pred[m].long(),
                          minlength=NCLS ** 2).reshape(NCLS, NCLS)


def miou(h):
    h = h.double()
    tp = h.diag()
    u = h.sum(0) + h.sum(1) - tp
    return float((tp / u)[u > 0].mean()) * 100.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=30)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DELIVER)

    import depth_corruptions as DC
    from semseg.models.cmnext import CMNeXt
    from semseg.datasets.nyu import NYU
    from semseg.augmentations_mm import get_val_augmentation
    from torch.utils.data import DataLoader

    model = CMNeXt("CMNeXt-B4", NCLS, ["img", "depth"])
    sd = torch.load(str(CKPT), map_location="cpu")
    sd = sd.get("state_dict", sd.get("model", sd))
    msg = model.load_state_dict(sd, strict=False)
    got = len(sd) - len(msg.missing_keys)
    print(f"checkpoint: {got}/{len(sd)} tensors loaded  (missing {len(msg.missing_keys)}, "
          f"unexpected {len(msg.unexpected_keys)})")
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

    transform = get_val_augmentation([480, 640])
    ds = NYU(str(DELIVER / "data" / "NYUDepthv2"), "val", transform, ["img", "depth"], None)
    dl = DataLoader(ds, batch_size=1, num_workers=2, pin_memory=False)

    def cap_(sample):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            model(sample)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            out = _orig(list(feats))
        if isinstance(out, dict) and "pred" in out:
            out = out["pred"]
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)

    g = torch.Generator().manual_seed(SEED)
    Hc = torch.zeros(NCLS, NCLS, dtype=torch.long)
    acc = {c: {"kl": [], "kl_s": [], "kl_u": [], "nrm": []} for c in CONDS}
    canary = False

    for i, batch in enumerate(dl):
        if i >= a.n_eval:
            break
        sample, lbl = batch          # dataset returns a LIST ordered by self.modals
        mods = [t.cuda() for t in sample]
        lab = lbl.cuda().squeeze(0).long()
        size = mods[0].shape[-2:]
        Zc = cap_(mods)
        zl = dec(Zc, size)
        if not canary:
            d = float((zl - dec(Zc, size)).abs().max())
            print(f"\nCANARY self-comparison: max|diff| = {d:.3e}")
            if d > 0:
                raise SystemExit("REFUSING: nondeterministic decoder")
            print(f"  decoder-input feature stages: {[tuple(t.shape) for t in Zc]}\n")
            canary = True
        logp_c = F.log_softmax(zl, 1)
        Hc += hist(zl.argmax(1).squeeze(0).cpu(), lab.cpu())

        for c, fn in CONDS.items():
            dep = fn(mods[1], DC)
            Zk = cap_([mods[0], dep])
            delta = [k - z for k, z in zip(Zk, Zc)]
            nrm = float(sum(float((d_ ** 2).sum()) for d_ in delta) ** 0.5)
            sh, un = [], []
            for d_ in delta:
                B, C, Hh, Ww = d_.shape
                idx = torch.randperm(Hh * Ww, generator=g).to(d_.device)
                sh.append(d_.reshape(B, C, Hh * Ww)[:, :, idx].reshape(B, C, Hh, Ww))
                un.append(d_.mean(dim=(2, 3), keepdim=True).expand_as(d_))

            def kl_of(fs):
                return float(F.kl_div(F.log_softmax(dec(fs, size), 1), logp_c,
                                      log_target=True, reduction="batchmean"))
            acc[c]["kl"].append(kl_of(Zk))
            acc[c]["kl_s"].append(kl_of([z + d_ for z, d_ in zip(Zc, sh)]))
            acc[c]["kl_u"].append(kl_of([z + d_ for z, d_ in zip(Zc, un)]))
            acc[c]["nrm"].append(nrm)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{a.n_eval}")

    clean = miou(Hc)
    print(f"\n=== REFERENCE GATE ===")
    print(f"  clean mIoU on {a.n_eval} images = {clean:.2f}   expected band [{GATE_LO}, {GATE_HI}] "
          f"(prior R1 runs: 54.2-55.13)")
    gate = GATE_LO <= clean <= GATE_HI
    if not gate:
        print("  *** GATE FAILED — the harness does not reproduce the known number. ***")
        print("  Per the prereg NOTHING below counts. Reporting the numbers for diagnosis only.")
    else:
        print("  GATE PASSED. Internals may be read.\n")

    rows = {}
    for c in CONDS:
        kl = float(np.mean(acc[c]["kl"]))
        nr = float(np.mean(acc[c]["nrm"]))
        rows[c] = {"kl": kl, "norm": nr,
                   "R": kl / nr ** 2 if nr > 1.0 else None,
                   "shuf_ratio": float(np.mean(acc[c]["kl_s"])) / kl if kl > 1e-6 else None,
                   "uni_ratio": float(np.mean(acc[c]["kl_u"])) / kl if kl > 1e-6 else None}

    print(f"{'condition':16s} {'||D||':>10s} {'KL':>12s} {'R':>9s} {'shuf':>7s} {'uni':>7s}")
    for c in sorted(rows, key=lambda x: -rows[x]["kl"]):
        r = rows[c]
        R = f"{r['R']:9.4f}" if r["R"] is not None else "     VOID"
        s = f"{r['shuf_ratio']:7.3f}" if r["shuf_ratio"] is not None else "   VOID"
        u = f"{r['uni_ratio']:7.3f}" if r["uni_ratio"] is not None else "   VOID"
        print(f"{c:16s} {r['norm']:10.3f} {r['kl']:12.1f} {R}{s}{u}")

    use = [c for c in rows if rows[c]["R"] is not None]
    lx = np.log([rows[c]["norm"] for c in use])
    ly = np.log([max(rows[c]["kl"], 1e-9) for c in use])
    p = float(np.polyfit(lx, ly, 1)[0])
    sh = [rows[c]["shuf_ratio"] for c in use]
    un = [rows[c]["uni_ratio"] for c in use]
    p_a = 1.7 <= p <= 2.4
    p_b = all(v <= 0.30 for v in sh)
    p_c = all(u <= s for u, s in zip(un, sh))

    print(f"\nfitted exponent p = {p:.3f}   (DFormerv2 gave 2.055)")
    print(f"shuffle ratio: {min(sh):.3f} - {max(sh):.3f}  (mean {np.mean(sh):.3f})   "
          f"[DFormerv2: 0.076-0.139]")
    print(f"uniform ratio: {min(un):.3f} - {max(un):.3f}  (mean {np.mean(un):.3f})   "
          f"[DFormerv2: 0.019-0.072]")

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M15gate clean mIoU in [{GATE_LO},{GATE_HI}] -> {'HIT' if gate else 'MISS'}")
    print(f"P-M15a  p in [1.7,2.4] -> {'HIT' if p_a else 'MISS'}")
    bad = [c for c in use if rows[c]["shuf_ratio"] > 0.30]
    print(f"P-M15b  R collapses on all conditions -> {'HIT' if p_b else 'MISS'}"
          + (f"   violators {bad}" if bad else ""))
    print(f"P-M15c  uniform <= shuffle -> {'HIT' if p_c else 'MISS'}")

    print(f"\n--- VERDICT ---")
    if not gate:
        print("  GATE FAILED — no conclusion is licensed. Fix the harness or abandon this lane.")
    elif p_b:
        print("  THE Ham/NMF EXPLANATION IS KILLED. R's arrangement-sensitivity reproduces on a")
        print("  SegFormer MLP decoder with no low-rank reconstruction, so the shuffle collapse is")
        print("  not an artifact of DFormerv2's decoder. The account is about representations.")
    else:
        print("  THE GENERALITY CLAIM IS KILLED. The collapse does not reproduce on an MLP decoder,")
        print("  so it is a property of DFormerv2's low-rank head. M3/M4/M12/M13/M14 are all")
        print("  retro-scoped to one architecture and must be written that way.")

    o = GAPFILL / "m15_cross_arch.json"
    o.write_text(json.dumps({"clean_miou": clean, "gate": gate, "exponent_p": p, "rows": rows,
                             "P_M15gate": gate, "P_M15a": p_a, "P_M15b": p_b, "P_M15c": p_c,
                             "n_eval": a.n_eval}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
