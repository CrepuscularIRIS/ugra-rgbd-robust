"""M21 — the PHASE arm, done correctly. M16's version was voided by my own bug.

WHAT WENT WRONG IN M16. `phase_randomize` drew a uniform random phase directly on the rfft2 output
WITHOUT enforcing Hermitian symmetry, so `irfft2` returned a field whose power spectrum deviated
0.938 from the original. The arm did not do what its name claimed and its result proved nothing.
The instrument check I had added caught it — which is the only reason it is not a false claim.

THE FIX. Take the phase of a random REAL field. Because that field is real, its rfft2 phase carries
the correct Hermitian symmetry by construction, so combining it with the true magnitude and inverting
yields a real field whose power spectrum is preserved.

    F_new = |F_true| * exp(i * angle(rfft2(random_real_field)))

WHY THE ARM MATTERS. Phase randomization is the SECOND spectrum-preserving control, independent of
translation. Translation preserves the spectrum and shifts registration rigidly; phase randomization
preserves the spectrum and destroys registration entirely. If both collapse R, the alignment
conclusion rests on two transforms with different mechanisms rather than one.

PRIOR ART, read before this ran (arXiv:2606.17037, "The Importance of Phase in Neural
Representations: An Internal Oppenheim-Lim Test"): transplanting the Fourier PHASE of one image onto
the MAGNITUDE of another AT A HIDDEN LAYER makes the prediction follow the PHASE donor, across
PRISM2D, GFNet and ViT-B/16. Identity rides on phase; image-specific magnitude is "largely
dispensable to the readout." **So a phase result here CONFIRMS a known asymmetry in a new setting —
it does not discover it, and must not be written as if it did.**

FILED BEFORE RUNNING:
    P-M21a  the fixed transform preserves the power spectrum: deviation < 1e-3        0.85
    P-M21b  phase randomization collapses R by >= 3x on DFormerv2                     0.70
    P-M21c  it collapses R on CMNeXt too                                              0.65
    P-M21d  phase collapse is DEEPER than translate(8) on both -- destroying
            registration everywhere beats displacing it by a fixed offset             0.60

    python3 scripts/m21_phase_fixed.py --gpu 1 --n-eval 10
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
GAPFILL = PROJECT / "results" / "gapfill"
SEED = 20260721
_CAP = {"store": None, "mode": None}


def phase_randomize(d, gen):
    """Randomise phase, PRESERVE the power spectrum, via the phase of a random REAL field."""
    H, W = d.shape[-2], d.shape[-1]
    mag = torch.fft.rfft2(d.float()).abs()
    noise = torch.randn(d.shape, generator=gen, device="cpu").to(d.device)
    ph = torch.angle(torch.fft.rfft2(noise.float()))
    return torch.fft.irfft2(mag * torch.exp(1j * ph), s=(H, W)).to(d.dtype)


def translate(d, k):
    return torch.roll(d, shifts=(k, k), dims=(2, 3))


def spectrum_dev(a, b):
    P = lambda t: torch.fft.rfft2(t.float()).abs()
    p0 = P(a)
    return float((P(b) - p0).abs().max() / p0.max())


def hook(model):
    _orig = model.decode_head.forward

    def _h(x, *g, **k):
        f = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [t.detach().clone() for t in f]
        return _orig(x, *g, **k)
    model.decode_head.forward = _h
    return _orig


def measure(cap_, dec, iterator, conds, corrupt_fn, n_eval, gen):
    acc = {c: {"real": [], "phase": [], "trans8": [], "shuffle": []} for c in conds}
    dev = []
    for i, item in enumerate(iterator):
        if i >= n_eval:
            break
        mods, size = item
        Zc = cap_(mods)
        zl = dec(Zc, size)
        if i == 0:
            assert float((zl - dec(Zc, size)).abs().max()) == 0.0, "nondeterministic decoder"
        logp = F.log_softmax(zl, 1)
        for c in conds:
            Zk = cap_(corrupt_fn(mods, c))
            delta = [k - z for k, z in zip(Zk, Zc)]

            def kl_of(dl):
                return float(F.kl_div(F.log_softmax(dec([z + d for z, d in zip(Zc, dl)], size), 1),
                                      logp, log_target=True, reduction="batchmean"))
            ph = [phase_randomize(d, gen) for d in delta]
            if i == 0:
                dev.append(spectrum_dev(delta[0], ph[0]))
            sh = []
            for d in delta:
                B, C, H, W = d.shape
                idx = torch.randperm(H * W, generator=gen).to(d.device)
                sh.append(d.reshape(B, C, H * W)[:, :, idx].reshape(B, C, H, W))
            acc[c]["real"].append(kl_of(delta))
            acc[c]["phase"].append(kl_of(ph))
            acc[c]["trans8"].append(kl_of([translate(d, 8) for d in delta]))
            acc[c]["shuffle"].append(kl_of(sh))
    return acc, float(np.mean(dev))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=10)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    gen = torch.Generator().manual_seed(SEED)
    out = {}

    # ---------- DFormerv2
    DF = PROJECT / "repos" / "DFormer"
    sys.path.insert(0, str(DF)); sys.path.insert(0, str(DF / "utils"))
    sys.path.insert(0, str(PROJECT / "diagnostics" / "corruptions"))
    sys.path.insert(0, str(PROJECT / "scripts"))
    os.chdir(DF)
    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    from m1_mig_full import COND, corrupt
    ck = PROJECT / "results" / "p0" / "mainB_baseline_s20260801" / "ckpt" / "latest.pth"
    m = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d)
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd.get("state_dict", sd))
    miss = m.load_state_dict(sd, strict=False)
    assert len(sd) - len(miss.missing_keys) >= 0.9 * len(sd)
    m = m.eval().cuda()
    _orig = hook(m)

    def cap_df(mods):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            m(mods[0], mods[1])
        _CAP["mode"] = None
        return _CAP["store"]

    def dec_df(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _orig(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")
    conds = ["dconst", "invalid_0.10", "invalid_0.55"]
    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)

        def it():
            for b in vl:
                yield ([b["data"].cuda(), b["modal_x"].cuda()], b["data"].shape[-2:])
        acc_df, dev_df = measure(cap_df, dec_df, it(), conds,
                                 lambda mods, c: [mods[0], corrupt(mods[1], COND[c][1])],
                                 a.n_eval, gen)
    out["DFormerv2"] = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in acc_df.items()}

    # ---------- CMNeXt
    DL = PROJECT / "repos" / "DELIVER"
    sys.path.insert(0, str(DL))
    os.chdir(DL)
    import depth_corruptions as DC
    from semseg.models.cmnext import CMNeXt
    from semseg.datasets.nyu import NYU
    from semseg.augmentations_mm import get_val_augmentation
    from torch.utils.data import DataLoader
    ck2 = DL / "checkpoints" / "cmnext_nyu" / "NYU_Depth_V2" / "cmnext_b4_nyu_rgbd.pth"
    cm = CMNeXt("CMNeXt-B4", 40, ["img", "depth"])
    sd2 = torch.load(str(ck2), map_location="cpu")
    sd2 = sd2.get("state_dict", sd2.get("model", sd2))
    msg = cm.load_state_dict(sd2, strict=False)
    assert len(sd2) - len(msg.missing_keys) >= 0.9 * len(sd2)
    cm = cm.eval().cuda()
    _o2 = hook(cm)

    def cap_cm(mods):
        _CAP["mode"], _CAP["store"] = "store", None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            cm(mods)
        _CAP["mode"] = None
        return _CAP["store"]

    def dec_cm(feats, size):
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = _o2(list(feats))
        if isinstance(o, dict) and "pred" in o:
            o = o["pred"]
        return F.interpolate(o, size=size, mode="bilinear", align_corners=False)

    ds = NYU(str(DL / "data" / "NYUDepthv2"), "val",
             get_val_augmentation([480, 640]), ["img", "depth"], None)
    dl = DataLoader(ds, batch_size=1, num_workers=2)
    c2 = {"constant": lambda d: DC.depth_constant(d.clone()),
          "invalid_0.10": lambda d: DC.depth_invalid(d.clone(), frac=0.10, seed=SEED),
          "invalid_0.55": lambda d: DC.depth_invalid(d.clone(), frac=0.55, seed=SEED)}

    def it2():
        for sample, _ in dl:
            mods = [t.cuda() for t in sample]
            yield (mods, mods[0].shape[-2:])
    acc_cm, dev_cm = measure(cap_cm, dec_cm, it2(), list(c2),
                             lambda mods, c: [mods[0], c2[c](mods[1])], a.n_eval, gen)
    out["CMNeXt"] = {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in acc_cm.items()}

    print(f"\n=== INSTRUMENT CHECK — power-spectrum deviation of the FIXED phase transform ===")
    print(f"  DFormerv2 {dev_df:.6f}   CMNeXt {dev_cm:.6f}   (M16's broken version: 0.937801)")

    print(f"\n{'arch|condition':28s} {'phase':>8s} {'trans8':>8s} {'shuffle':>8s}")
    rows = {}
    for arch, data in out.items():
        for c, d in data.items():
            r = {k: d[k] / max(d["real"], 1e-12) for k in ("phase", "trans8", "shuffle")}
            rows[f"{arch}|{c}"] = r
            print(f"{arch+'|'+c:28s} {r['phase']:8.3f} {r['trans8']:8.3f} {r['shuffle']:8.3f}")

    p_a = max(dev_df, dev_cm) < 1e-3
    df_ph = [rows[k]["phase"] for k in rows if k.startswith("DFormerv2")]
    cm_ph = [rows[k]["phase"] for k in rows if k.startswith("CMNeXt")]
    p_b = all(v <= 1 / 3 for v in df_ph)
    p_c = all(v <= 1 / 3 for v in cm_ph)
    p_d = all(rows[k]["phase"] <= rows[k]["trans8"] for k in rows)

    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M21a fixed transform preserves spectrum (<1e-3) -> {'HIT' if p_a else 'MISS'}")
    print(f"P-M21b phase collapses >=3x on DFormerv2: {[round(v,3) for v in df_ph]} -> "
          f"{'HIT' if p_b else 'MISS'}")
    print(f"P-M21c phase collapses >=3x on CMNeXt: {[round(v,3) for v in cm_ph]} -> "
          f"{'HIT' if p_c else 'MISS'}")
    print(f"P-M21d phase deeper than translate(8) everywhere -> {'HIT' if p_d else 'MISS'}")
    print(f"\n--- OUTCOME-TABLE BRANCH ---")
    print(f"  P-M21a {'HIT' if p_a else 'MISS'} | P-M21b {'HIT' if p_b else 'MISS'} | "
          f"P-M21c {'HIT' if p_c else 'MISS'} | P-M21d {'HIT' if p_d else 'MISS'}")
    print("  (interpretation belongs in STANDING-ASSESSMENT, not here)")

    o = GAPFILL / "m21_phase_fixed.json"
    o.write_text(json.dumps({"raw": out, "ratios": rows, "spec_dev_dformer": dev_df,
                             "spec_dev_cmnext": dev_cm, "P_M21a": bool(p_a),
                             "P_M21b": bool(p_b), "P_M21c": bool(p_c), "P_M21d": bool(p_d),
                             "n_eval": a.n_eval}, indent=2) + "\n")
    print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
