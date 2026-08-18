"""M3 — is the decoder input where damage is DETERMINED, and is spatial structure the causal part?

M2 showed a spatially UNIFORM displacement of matched magnitude does 0.15 damage where the real
corruption does 1.84. Two rival explanations, and they have opposite consequences:

  R-A  The decoder input IS the bottleneck, but the causal object is the SPATIALLY RESOLVED
       displacement field, which mean-pooling discarded. Then a principle exists: bound the
       spatially-resolved displacement.
  R-B  The damage is determined UPSTREAM, inside the encoder. Then NO statistic of the decoder
       input can be the principle, and the whole decoder-interface framing is the wrong place
       to look.

THE TEST. Feed CLEAN depth, then inject at the decoder input the exact per-position displacement
field a real corruption produces. If damage returns to the real corruption's level, R-A. If it
stays near zero, R-B.

    real         run the actual corruption                                   (reference)
    field        clean depth + the corruption's own spatially-resolved delta  (R-A test)
    shuffled     same field, spatially permuted -- same magnitude, same per-channel statistics,
                 spatial correspondence destroyed                             (is position load-bearing?)
    uniform      the M2 constant-vector arm, re-measured here for comparability

FILED BEFORE THE NUMBERS:
    P-M3a  damage(field) >= 0.7 * damage(real)   -- the decoder input is the bottleneck   P_true 0.60
    P-M3b  damage(shuffled) < 0.5 * damage(field) -- spatial correspondence is load-bearing P_true 0.65

P-M3a at 0.60: the encoder feeds the decoder only through these tensors, so injecting the exact
difference SHOULD reproduce the output difference -- unless the decode head's internal state (Ham/NMF
bases fitted per forward) depends on the input in a way the injection cannot reproduce. That caveat
is why it is 0.60 and not higher.

    python3 scripts/m3_spatial.py --gpu 1 --n-eval 60
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
_CAP = {"store": None, "inject": None, "mode": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--cond", default="invalid_0.10",
                    choices=["invalid_0.10", "dconst"],
                    help="dconst is the EXACT removal for DFormerv2 (mask_d==0)")
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    os.chdir(DFORMER)

    from importlib import import_module
    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    cfg.pad = False
    from models.builder import EncoderDecoder
    from utils.metric import hist_info, compute_score
    from depth_corruptions import depth_invalid

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
        feats = list(x) if isinstance(x, (list, tuple)) else [x]
        if _CAP["mode"] == "store":
            _CAP["store"] = [f.detach().clone() for f in feats]
        elif _CAP["mode"] == "inject" and _CAP["inject"] is not None:
            feats = [f + d.to(f.device) for f, d in zip(feats, _CAP["inject"])]
        y = feats if isinstance(x, (list, tuple)) else feats[0]
        return _orig(y, *args, **kw)
    model.decode_head.forward = _dec

    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset
    from utils.engine.engine import Engine
    sys.argv = [sys.argv[0]]
    os.environ.setdefault("LOCAL_RANK", "0")

    def run(rgb, dep, mode=None, inject=None):
        _CAP["mode"], _CAP["inject"], _CAP["store"] = mode, inject, None
        torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        with torch.no_grad():
            o = model(rgb, dep)
        _CAP["mode"] = None
        return o

    arms = ["clean", "real", "field", "shuffled", "uniform"]
    hists = {k: np.zeros((NC, NC)) for k in arms}
    corr = {k: [0, 0] for k in arms}
    g = torch.Generator().manual_seed(SEED)

    with Engine(custom_parser=None) as engine:
        vl, _ = get_val_loader(engine, RGBXDataset, cfg, val_batch_size=1)
        for i, b in enumerate(vl):
            if i >= a.n_eval:
                break
            rgb, dep = b["data"].cuda(), b["modal_x"].cuda()
            gt = b["label"].numpy()[0]
            cdep = (torch.zeros_like(dep) if a.cond == 'dconst'
                    else depth_invalid(dep.clone(), frac=0.10, seed=SEED))

            run(rgb, dep, mode="store");   Zc = _CAP["store"]
            run(rgb, cdep, mode="store");  Zk = _CAP["store"]
            delta = [(k - c) for k, c in zip(Zk, Zc)]

            shuf = []
            for d in delta:
                B, C, H, W = d.shape
                flat = d.reshape(B, C, H * W)
                idx = torch.randperm(H * W, generator=g).to(d.device)
                shuf.append(flat[:, :, idx].reshape(B, C, H, W))
            uni = [d.mean(dim=(2, 3), keepdim=True).expand_as(d) *
                   (d.norm() / (d.mean(dim=(2, 3), keepdim=True).expand_as(d).norm() + 1e-12))
                   for d in delta]

            outs = {
                "clean":    run(rgb, dep),
                "real":     run(rgb, cdep),
                "field":    run(rgb, dep, mode="inject", inject=delta),
                "shuffled": run(rgb, dep, mode="inject", inject=shuf),
                "uniform":  run(rgb, dep, mode="inject", inject=uni),
            }
            for k, o in outs.items():
                pred = o.argmax(1).cpu().numpy()[0]
                if pred.shape != gt.shape:
                    import torch.nn.functional as F
                    pred = F.interpolate(o.float(), size=gt.shape, mode="bilinear",
                                         align_corners=False).argmax(1).cpu().numpy()[0]
                h, lab, cor = hist_info(NC, pred, gt)
                hists[k] += h; corr[k][0] += cor; corr[k][1] += lab

    res = {k: float(compute_score(hists[k], corr[k][0], corr[k][1])[1]) * 100 for k in arms}
    clean = res["clean"]
    print(f"\n{'arm':12s} {'mIoU':>8s} {'damage':>8s}")
    for k in arms:
        print(f"{k:12s} {res[k]:8.2f} {clean - res[k]:8.2f}")

    d_real = clean - res["real"]
    d_field = clean - res["field"]
    d_shuf = clean - res["shuffled"]
    d_uni = clean - res["uniform"]
    p_a = d_field >= 0.7 * d_real
    p_b = d_shuf < 0.5 * d_field if d_field > 0 else False
    print(f"\n--- FILED PREDICTIONS ---")
    print(f"P-M3a damage(field) >= 0.7*damage(real): {d_field:.2f} vs {0.7*d_real:.2f} "
          f"-> {'HIT' if p_a else 'MISS'}")
    print(f"P-M3b damage(shuffled) < 0.5*damage(field): {d_shuf:.2f} vs {0.5*d_field:.2f} "
          f"-> {'HIT' if p_b else 'MISS'}")

    print("\n--- VERDICT ---")
    if p_a:
        print("  R-A: the DECODER INPUT is the bottleneck. Injecting the spatially-resolved field")
        print("  reproduces the damage, so the causal object is the displacement FIELD, not its")
        print("  pooled magnitude. A principle over the field is derivable.")
        if p_b:
            print("  And spatial correspondence is LOAD-BEARING: shuffling position destroys it.")
        else:
            print("  But shuffling position does NOT destroy it — the field's per-channel")
            print("  magnitude distribution, not its spatial arrangement, carries the damage.")
    else:
        print("  R-B: the damage is NOT determined at the decoder input. Even the exact")
        print("  spatially-resolved field fails to reproduce it, so no statistic of this")
        print("  interface can be the principle — the decoder-interface framing is wrong.")

    p = PROJECT / "results" / "gapfill" / f"m3_spatial_{a.cond}.json"
    p.write_text(json.dumps({"n_eval": a.n_eval, "cond": a.cond, "miou": res,
                             "damage": {"real": d_real, "field": d_field,
                                        "shuffled": d_shuf, "uniform": d_uni},
                             "P_M3a": bool(p_a), "P_M3b": bool(p_b)}, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
