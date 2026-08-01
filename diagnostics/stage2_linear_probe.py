#!/usr/bin/env python
"""THE DISCRIMINATING TEST for the paper's central word: "under-development" vs "co-adaptation".

The thesis says the baseline's missing-depth failure is UNDER-DEVELOPED primary-pathway
REPRESENTATION. The rival says the representation is fine and the failure is DOWNSTREAM
calibration / co-adaptation with the depth-conditioned operator.

A frozen linear probe on stage-2 features (depth removed) separates them:
  · baseline features probe POORLY, and method/RGB-only features probe BETTER
        => the representation really is under-developed. "under-development" is the right word.
  · baseline features probe just as WELL as the others, yet the full model still fails
        => the representation was fine all along; the failure is downstream.
        "under-development" is WRONG and the paper must say "co-adaptation".

Linear probe = a single 1x1 conv (256 -> n_class) on frozen stage-2 features, trained on cached
train features and scored on cached val features. Everything else in the network is discarded, so
the probe measures ONLY what stage 2 linearly encodes about the labels.

VALIDATION GATE (standing rule: a probe proves itself before its internals are read): the checkpoint
load must match >=90% of tensors, and the probe must beat the majority-class baseline by a wide
margin -- otherwise the features or the labels are misaligned and no comparison is meaningful.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJECT, "repos", "DFormer"))
sys.path.insert(0, os.path.join(_PROJECT, "repos"))

CAP = {}


def build(ckpt, cfg):
    from models.builder import EncoderDecoder
    m = EncoderDecoder(cfg=cfg, criterion=None, norm_layer=torch.nn.BatchNorm2d, syncbn=False)
    w = torch.load(ckpt, map_location="cpu")
    sd = w.get("model", w.get("state_dict", w))
    r = m.load_state_dict(sd, strict=False)
    matched = len(m.state_dict()) - len(r.missing_keys)
    if matched < 0.9 * len(m.state_dict()):
        raise SystemExit(f"FATAL: {ckpt} matched only {matched}/{len(m.state_dict())}")
    for nm, mod in m.named_modules():
        if nm.endswith("backbone.layers.2"):
            mod.register_forward_hook(lambda _m, _i, o: CAP.__setitem__("f", (o[0] if isinstance(o, (tuple, list)) else o).detach()))
    return m.cuda().eval()


def collect(model, loader, n):
    """Cache stage-2 features and label maps, depth REMOVED."""
    X, Y = [], []
    seen = 0
    for mb in loader:
        if seen >= n:
            break
        img = mb["data"].cuda().float()
        modal = torch.zeros_like(mb["modal_x"].cuda().float())     # DEPTH REMOVED
        gt = mb["label"].cuda()
        if img.dim() == 3:
            img, modal = img[None], modal[None]
        if gt.dim() == 2:
            gt = gt[None]
        torch.manual_seed(20260721)
        CAP.clear()
        with torch.no_grad():
            model(img, modal)
        f = CAP["f"]                                   # (B,H,W,C) or (B,C,H,W)
        if f.shape[-1] in (256,) and f.dim() == 4 and f.shape[1] not in (256,):
            f = f.permute(0, 3, 1, 2).contiguous()     # -> (B,C,H,W)
        lab = F.interpolate(gt[:, None].float(), size=f.shape[-2:], mode="nearest")[:, 0].long()
        X.append(f.flatten(2).permute(0, 2, 1).reshape(-1, f.shape[1]).half().cpu())
        Y.append(lab.reshape(-1).cpu())
        seen += 1
    return torch.cat(X), torch.cat(Y)


def probe(Xtr, Ytr, Xva, Yva, nclass=40, iters=300, lr=0.05):
    keep = (Ytr != 255) & (Ytr < nclass)
    Xtr, Ytr = Xtr[keep].float().cuda(), Ytr[keep].cuda()
    keepv = (Yva != 255) & (Yva < nclass)
    Xva, Yva = Xva[keepv].float().cuda(), Yva[keepv].cuda()
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True) + 1e-5
    Xtr, Xva = (Xtr - mu) / sd, (Xva - mu) / sd
    cnt = torch.bincount(Ytr, minlength=nclass).float()
    wts = torch.where(cnt > 0, 1.0 / cnt.clamp(min=1), torch.zeros_like(cnt))
    wts = wts / wts[wts > 0].mean()            # inverse-frequency, mean-normalised
    W = torch.zeros(Xtr.shape[1], nclass, device="cuda", requires_grad=True)
    b = torch.zeros(nclass, device="cuda", requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    n = Xtr.shape[0]
    for i in range(iters):
        idx = torch.randint(0, n, (16384,), device="cuda")
        loss = F.cross_entropy(Xtr[idx] @ W + b, Ytr[idx], weight=wts)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = (Xva @ W + b).argmax(1)
        acc = (pred == Yva).float().mean().item()
        # BALANCED accuracy = mean per-class recall over classes PRESENT in val: the probe analogue
        # of mIoU's equal weighting, and the metric that can see rare-class gains.
        recs = []
        for c in torch.unique(Yva).tolist():
            m = Yva == c
            if m.sum() > 0:
                recs.append((pred[m] == c).float().mean().item())
        bal = float(np.mean(recs))
        major = torch.bincount(Yva, minlength=nclass).float().max().item() / Yva.numel()
    return acc, bal, major


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntr", type=int, default=150)
    ap.add_argument("--nva", type=int, default=80)
    a = ap.parse_args()
    import types
    from importlib import import_module
    from utils.dataloader.dataloader import get_train_loader, get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset

    cfg = import_module("local_configs.NYUDepthv2.DFormerv2_S").C
    for k, v in (("pad", False), ("pad_SUNRGBD", False), ("num_workers", 2)):
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    eng = types.SimpleNamespace(distributed=False, world_size=1)

    P = _PROJECT
    ck = {
        "baseline":    f"{P}/repos/DFormer/checkpoints/trained/DFormerv2_Small_NYU.pth",
        "method":      f"{P}/repos/DFormer/checkpoints/NYUDepthv2_DFormerv2_S_20260725-155828/latest.pth",
        "rgb_only_ft": f"{P}/results/score_push/runs/l2_nof/screens/rgb_only_finetune/ckpt_final_c2.pth",
    }
    print(f"{'model':<14}{'pixel acc':>11}{'BALANCED':>11}{'majority':>11}")
    res = {}
    for name, path in ck.items():
        if not os.path.exists(path):
            print(f"{name:<14}  MISSING"); continue
        tr, _ = get_train_loader(eng, RGBXDataset, cfg)
        va, _ = get_val_loader(eng, RGBXDataset, cfg, val_batch_size=1)
        model = build(path, cfg)
        Xtr, Ytr = collect(model, tr, a.ntr)
        Xva, Yva = collect(model, va, a.nva)
        del model; torch.cuda.empty_cache()
        acc, bal, major = probe(Xtr, Ytr, Xva, Yva)
        res[name] = (acc, bal)
        print(f"{name:<14}{acc:>11.4f}{bal:>11.4f}{major:>11.4f}", flush=True)
        del Xtr, Ytr, Xva, Yva; torch.cuda.empty_cache()
    if "baseline" in res and "method" in res:
        for i, lbl in ((0, "pixel"), (1, "BALANCED")):
            dm = res["method"][i] - res["baseline"][i]
            dr = res.get("rgb_only_ft", (float("nan"),) * 2)[i] - res["baseline"][i]
            print(f"\n{lbl}: method-baseline={dm:+.4f}  rgb_only-baseline={dr:+.4f}")
        print("\nREADING: if the BALANCED gaps are LARGE (>~+0.05) and rank-order with task gain,")
        print("         the rare-class caveat overturns the co-adaptation verdict.")
        print("         If balanced gaps stay ~1pp like pixel acc, co-adaptation STANDS.")


if __name__ == "__main__":
    main()
