"""n7 cycle-4 (M3) PCT precondition probes — run on an early checkpoint (~epoch 5).

Loads a checkpoint into DFormerv2WithBoundaryGate, runs forward on clean val images (model.eval()),
reads model.last_B ([B,1,H2,W2]) + model.last_B_logits, and computes the three kill-gates:

  PCT-1 (boundary learnability): boundary-F1 of sigmoid(last_B_logits)>0.5 vs
         label_boundary_target(label) downsampled to (H2,W2). Compare ep5 vs an ep0 (random-M3) probe.
         PASS if F1_ep5 - F1_ep0 >= 0.10.  KILL otherwise.
  PCT-2 (planar-gate suppression): mean last_B on INTERIOR pixels of the planar cluster
         {bathtub, tv, toilet, whiteboard} (mask by GT, exclude a boundary band).
         PASS if mean_B_planar < 0.10.  KILL if >= 0.10.
  PCT-3 (missing-depth fallback): re-run with depth zeroed (depth_available=0); Shannon entropy of
         last_B over spatial positions.  PASS if entropy > 0.30.  KILL if <= 0.30.

Every number comes from a real forward. No fabrication. Run:
  python pct_probe.py --config=local_configs.NYUDepthv2.DFormerv2_S \
      --ckpt <ep5.pth> [--ckpt0 <ep0/none>] --num_batches 6
from inside repos/DFormer (so dataset relative paths resolve).
"""
import argparse
import os
import sys
from importlib import import_module

import torch
import torch.nn.functional as F

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
DFORMER_ROOT = os.path.join(EXP_DIR, "..", "..", "repos", "DFormer")
sys.path.insert(0, EXP_DIR)
sys.path.insert(0, DFORMER_ROOT)
sys.path.insert(0, os.path.join(DFORMER_ROOT, "utils"))

import torch.nn as nn
from models.builder import EncoderDecoder as segmodel
from model_wrapper_m3 import DFormerv2WithBoundaryGate
from boundary_gate_adapter import label_boundary_target
from utils.dataloader.dataloader import get_val_loader
from utils.dataloader.RGBXDataset import RGBXDataset


# NYUDepthv2 40-class trainIDs (0-indexed) for the planar cluster used in PCT-2.
# NYU class order (1..40): 1 wall,2 floor,3 cabinet,4 bed,5 chair,6 sofa,7 table,8 door,9 window,
# 10 bookshelf,11 picture,12 counter,13 blinds,14 desk,15 shelves,16 curtain,17 dresser,18 pillow,
# 19 mirror,20 floormat,21 clothes,22 ceiling,23 books,24 refridgerator,25 television,26 paper,
# 27 towel,28 shower_curtain,29 box,30 whiteboard,31 person,32 nightstand,33 toilet,34 sink,
# 35 lamp,36 bathtub,37 bag,38 otherstructure,39 otherfurniture,40 otherprop.
# labels arrive 0-indexed (trainID = raw-1). planar cluster = {bathtub, tv, toilet, whiteboard}:
PLANAR_TRAINIDS = [35, 24, 32, 29]  # bathtub=36, television=25, toilet=33, whiteboard=30 -> minus 1


class _FakeEngine:
    distributed = False
    local_rank = 0
    world_size = 1


def build_model(config):
    criterion = nn.CrossEntropyLoss(reduction="none", ignore_index=config.background)
    base = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d, syncbn=False)
    model = DFormerv2WithBoundaryGate(base, num_classes=config.num_classes)
    return model


def load_ckpt(model, ckpt_path):
    if ckpt_path in (None, "", "none", "random"):
        return "random-init (no ckpt loaded)"
    w = torch.load(ckpt_path, map_location="cpu")
    w = w.get("state_dict", w.get("model", w)) if isinstance(w, dict) else w
    w = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in w.items()}
    rep = model.load_state_dict(w, strict=False)
    return f"loaded {ckpt_path}: missing={len(rep.missing_keys)} unexpected={len(rep.unexpected_keys)}"


def boundary_f1(logits, label, bg, thresh=0.5):
    """F1 of (sigmoid(logits)>thresh) vs label_boundary_target downsampled to logits res."""
    H2, W2 = logits.shape[-2:]
    tgt = label_boundary_target(label.long(), ignore_index=bg, width=2)  # [B,1,H,W]
    tgt_ds = (F.interpolate(tgt.float(), size=(H2, W2), mode="area") > 0.10).float()
    pred = (torch.sigmoid(logits) > thresh).float()
    tp = (pred * tgt_ds).sum().item()
    fp = (pred * (1 - tgt_ds)).sum().item()
    fn = ((1 - pred) * tgt_ds).sum().item()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


def planar_interior_meanB(B, label, bg, band=2):
    """mean last_B over INTERIOR pixels of planar-cluster classes (exclude a boundary band)."""
    H2, W2 = B.shape[-2:]
    lab = label.long()
    planar = torch.zeros_like(lab, dtype=torch.float32)
    for c in PLANAR_TRAINIDS:
        planar = planar + (lab == c).float()
    planar = (planar > 0).float().unsqueeze(1)  # [B,1,H,W]
    # exclude boundary band: erode planar mask so only interiors remain
    k = 2 * band + 1
    interior = -F.max_pool2d(-planar, k, 1, band)
    interior_ds = (F.interpolate(interior, size=(H2, W2), mode="area") > 0.5).float()
    n = interior_ds.sum().item()
    if n < 1:
        return None, 0
    return (B * interior_ds).sum().item() / n, int(n)


def spatial_entropy(B):
    """Shannon entropy (nats, normalized to [0,1] by log(N)) of last_B over spatial positions."""
    ent_list = []
    Bf = B.flatten(2)  # [B,1,H*W]
    for b in range(Bf.shape[0]):
        v = Bf[b, 0]
        s = v.sum()
        if s <= 1e-8:
            ent_list.append(0.0)
            continue
        p = v / s
        p = p[p > 0]
        ent = -(p * p.log()).sum().item()
        ent_list.append(ent / (torch.log(torch.tensor(float(v.numel()))).item()))
    return sum(ent_list) / max(len(ent_list), 1)


def run_probe(model, val_loader, config, num_batches, tag):
    model.eval()
    dev = torch.device("cuda")
    model.to(dev)
    f1s, planarBs, planarNs, ents = [], [], [], []
    it = iter(val_loader)
    for i in range(num_batches):
        try:
            mb = next(it)
        except StopIteration:
            break
        rgb = mb["data"].to(dev)
        label = mb["label"].to(dev)
        modal_x = mb["modal_x"].to(dev)
        with torch.no_grad():
            # clean forward (depth present)
            _ = model(rgb, modal_x, label=None, depth_available=1.0)
            B_logits = model.last_B_logits.clone()
            B = model.last_B.clone()
            f1s.append(boundary_f1(B_logits, label, config.background))
            mb_planar, n = planar_interior_meanB(B, label, config.background)
            if mb_planar is not None:
                planarBs.append(mb_planar); planarNs.append(n)
            # missing-depth forward
            _ = model(rgb, torch.zeros_like(modal_x), label=None, depth_available=0.0)
            ents.append(spatial_entropy(model.last_B.clone()))
    f1 = sum(f1s) / max(len(f1s), 1)
    planarB = (sum(w * b for w, b in zip(planarNs, planarBs)) / max(sum(planarNs), 1)
               if planarBs else float("nan"))
    ent = sum(ents) / max(len(ents), 1)
    print(f"[{tag}] batches={len(f1s)} boundary_F1={f1:.4f} "
          f"planar_interior_meanB={planarB:.4f} (px={sum(planarNs)}) missing_depth_entropy={ent:.4f}")
    return f1, planarB, ent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="local_configs.NYUDepthv2.DFormerv2_S")
    ap.add_argument("--ckpt", required=True, help="epoch-~5 checkpoint")
    ap.add_argument("--ckpt0", default="random", help="epoch-0 / random baseline for PCT-1 (default random-init)")
    ap.add_argument("--num_batches", type=int, default=6)
    args = ap.parse_args()

    config = getattr(import_module(args.config), "C")
    if not hasattr(config, "pad"):
        config.pad = False
    engine = _FakeEngine()
    val_loader, _ = get_val_loader(engine, RGBXDataset, config, val_batch_size=int(config.batch_size))

    print("=== PCT PROBE (ep0 baseline) ===")
    m0 = build_model(config)
    print(load_ckpt(m0, args.ckpt0))
    f1_0, _, _ = run_probe(m0, val_loader, config, args.num_batches, "ep0")
    del m0; torch.cuda.empty_cache()

    print("=== PCT PROBE (ep5 checkpoint) ===")
    m5 = build_model(config)
    print(load_ckpt(m5, args.ckpt))
    f1_5, planarB_5, ent_5 = run_probe(m5, val_loader, config, args.num_batches, "ep5")

    print("\n=== PCT VERDICTS ===")
    d_f1 = f1_5 - f1_0
    pct1 = d_f1 >= 0.10
    pct2 = (planarB_5 < 0.10)
    pct3 = ent_5 > 0.30
    print(f"PCT-1 boundary_learnability: F1 ep0={f1_0:.4f} ep5={f1_5:.4f} d={d_f1:+.4f} "
          f"(need >=+0.10) -> {'PASS' if pct1 else 'KILL'}")
    print(f"PCT-2 planar_suppression:    meanB_planar={planarB_5:.4f} "
          f"(need <0.10) -> {'PASS' if pct2 else 'KILL'}")
    print(f"PCT-3 missing_depth_fallback: entropy={ent_5:.4f} "
          f"(need >0.30) -> {'PASS' if pct3 else 'KILL'}")
    print(f"OVERALL PCT: {'PASS' if (pct1 and pct2 and pct3) else 'KILL'}")
