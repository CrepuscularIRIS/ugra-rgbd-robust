"""Post-training repair routes of the paper's Fig. 2, for the DFormer family.

    features : frozen-model feature repair per encoder stage -- oracle (uses real depth), blind (estimated
               from the missing-depth features only), random-subspace control   [dat.analysis.featrepair]
    weights  : plain interpolation C -> R, and Geodesic Unlearning (GU) of every eligible layer toward the
               reference R                                                       [dat.analysis.gu / apply]

R is the matched RGB-only reference: same architecture, trained with constant depth
(`python -m dat.train --preset RGB_ONLY --init <C checkpoint>` for a quick warm-started reference, or from the
pretrained checkpoint for the strict one). Each route reports the share of the complete-missing gap it closes.
The HAM decoder is stochastic (+-0.1-0.2 mIoU between identical runs).
"""
from __future__ import annotations

import torch

from . import apply, featrepair as fr
from ..evaluate import evaluate


def _both(host, model, loader, limit, device):
    r = evaluate(host, model, loader, ("complete", "missing"), limit=limit, device=device)
    return r["complete"]["mIoU"], r["missing"]["mIoU"]


def weight_routes(host, model, c_ckpt: str, r_ckpt: str, loader, k: int = 16, limit: int | None = None,
                  device: str = "cuda") -> list[dict]:
    """Interpolation and GU(all layers) between C and R, scored as (complete, missing) mIoU."""
    host.load_weights(model, c_ckpt)
    c_sd = {n: p.detach().clone() for n, p in model.state_dict().items()}
    r_state = torch.load(r_ckpt, map_location="cpu", weights_only=False)
    r_sd = r_state.get("state_dict", r_state)
    layers = apply.candidate_layers(c_sd, k)
    every = [n for n, p in model.named_parameters() if n.startswith("backbone.") and p.ndim >= 2]
    rows = []
    s_c, s_m = _both(host, model, loader, limit, device)
    rows.append({"route": "C", "complete": s_c, "missing": s_m})
    for label, names, eta, mode in (("interp eta=0.5", every, 0.5, "interp"), ("interp eta=1.0 (=R)", every, 1.0, "interp"),
                                    ("GU all layers eta=0.5", layers, 0.5, "gu"), ("GU all layers eta=1.0", layers, 1.0, "gu")):
        with apply.edited(model, c_sd, r_sd, names, k, eta, mode):
            c_, m_ = _both(host, model, loader, limit, device)
        rows.append({"route": label, "complete": c_, "missing": m_})
    return rows


@torch.no_grad()
def feature_routes(host, model, calib_loader, eval_loader, ks=(2, 8, 32), n_calib: int = 200,
                   limit: int | None = None, device: str = "cuda") -> dict:
    """Oracle / blind / random subspace repair of the missing-depth encoder features, all four stages at once."""
    from utils.metrics_new import Metrics

    state = {"fn": None}
    model.backbone.register_forward_hook(lambda m, i, out: state["fn"](out) if state["fn"] else None)
    model.eval()

    def feats(img, depth):
        return model.backbone(img, depth)

    stats = None
    for i, b in enumerate(calib_loader):
        if i >= n_calib:
            break
        img, d = b["data"].to(device), b["modal_x"].to(device)
        zc, zm = feats(img, d), feats(img, torch.full_like(d, host.empty_value))
        stats = stats or [fr.StageStats(z.shape[1], device) for z in zc]
        for s, (a, m) in zip(stats, zip(zc, zm)):
            s.update(a, m)
    spectrum = [{"stage": i + 1, "channels": s.c, "rel_shift": s.rel_shift(),
                 "eff_rank": fr.effective_rank(s.spectrum()),
                 "explained": {k: fr.explained(s.spectrum(), k) for k in ks}} for i, s in enumerate(stats)]

    ops = {}
    for k in ks:
        us = [s.basis(min(k, s.c)) for s in stats]
        ms = [s.blind_map(u) for s, u in zip(stats, us)]
        rs = [fr.random_basis(s.c, min(k, s.c), seed=si, device=device) for si, s in enumerate(stats)]
        ops[f"oracle@{k}"] = lambda zm, zc, us=us: tuple(fr.repair_oracle(m, c, u) for m, c, u in zip(zm, zc, us))
        ops[f"blind@{k}"] = lambda zm, zc, us=us, ms=ms: tuple(fr.repair_blind(m, u, mm) for m, u, mm in zip(zm, us, ms))
        ops[f"random@{k}"] = lambda zm, zc, rs=rs: tuple(fr.repair_oracle(m, c, u) for m, c, u in zip(zm, zc, rs))

    names = ["complete", "missing", *ops]
    mets = {n: Metrics(host.num_classes, host.cfg.background, device) for n in names}
    for i, b in enumerate(eval_loader):
        if limit is not None and i >= limit:
            break
        img, d, y = b["data"].to(device), b["modal_x"].to(device), b["label"].to(device)
        z0, zc = torch.full_like(d, host.empty_value), feats(img, d)
        mets["complete"].update(model(img, d).softmax(1), y)
        mets["missing"].update(model(img, z0).softmax(1), y)
        for name, fn in ops.items():
            state["fn"] = lambda zm, fn=fn: fn(zm, zc)
            mets[name].update(model(img, z0).softmax(1), y)
            state["fn"] = None
    miou = {n: mets[n].compute_iou()[1] for n in names}
    gap = miou["complete"] - miou["missing"]
    share = {n: 100 * (miou[n] - miou["missing"]) / gap for n in ops} if gap > 0 else {}
    return {"spectrum": spectrum, "miou": miou, "gap": gap, "gap_closed_pct": share}
