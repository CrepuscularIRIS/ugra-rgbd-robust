"""Load frozen DFormerv2-S (NYU) for diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

DFORMER_ROOT = Path(__file__).resolve().parents[2] / "repos" / "DFormer"
if str(DFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DFORMER_ROOT))


def load_dformerv2_s_nyu(
    ckpt_path: Optional[str] = None,
    device: str = "cuda:0",
) -> Tuple[nn.Module, Any]:
    """Build EncoderDecoder with DFormerv2_S config and load NYU weights."""
    from importlib import import_module

    from models.builder import EncoderDecoder as segmodel

    config = getattr(import_module("local_configs.NYUDepthv2.DFormerv2_S"), "C")
    # ValPre / SUN padding flags used at runtime by dataloader
    if not hasattr(config, "pad"):
        config.pad = False
    if ckpt_path is None:
        ckpt_path = str(DFORMER_ROOT / "checkpoints" / "trained" / "DFormerv2_Small_NYU.pth")

    model = segmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d, syncbn=False)
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, config


@torch.no_grad()
def forward_with_stage_feats(
    model: nn.Module,
    rgb: torch.Tensor,
    depth: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Return (logits [B,K,H,W], list of encoder stage maps [B,C,h,w])."""
    orisize = rgb.shape
    feats = model.backbone(rgb, depth)
    if isinstance(feats, tuple):
        feats = list(feats)
    # ham decoder uses stages 1..3
    out = model.decode_head.forward(feats)
    if isinstance(out, dict) and "pred" in out:
        logits = out["pred"]
    else:
        logits = out
    logits = F.interpolate(logits, size=orisize[-2:], mode="bilinear", align_corners=False)
    return logits, feats


@torch.no_grad()
def logits_to_miou(
    logits: torch.Tensor,
    label: torch.Tensor,
    num_classes: int = 40,
    ignore_index: int = 255,
) -> float:
    """Single-batch or multi-batch mean IoU (macro over classes present)."""
    pred = logits.argmax(dim=1)
    label = label.long()
    ious = []
    for c in range(num_classes):
        pred_c = pred == c
        label_c = label == c
        valid = label != ignore_index
        inter = (pred_c & label_c & valid).sum().float()
        union = ((pred_c | label_c) & valid).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    if not ious:
        return 0.0
    return float(sum(ious) / len(ious))
