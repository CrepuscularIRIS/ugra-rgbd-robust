"""MP-A oracle gate (EVAL-ONLY, env-gated) for DFormerv2-S GeoPriorGen injection.

Falsification / sizing probe for the "gradient-discontinuity conflation"
missing-point (MP-A): DFormerv2-S injects the GeoPriorGen depth signal into the
GSA attention as a pairwise depth-decay bias  weight[1] * |depth_i - depth_j| *
decay  (added to the attention logits). MP-A hypothesizes that this bias is
applied indiscriminately over WITHIN-object SMOOTH depth gradients (a receding
wall) as well as BETWEEN-object depth DISCONTINUITIES (a table edge), so that
smooth-gradient injection manufactures phantom semantic boundaries on planar
surfaces (damaging clean + gaussian mIoU).

This module installs an EVAL-TIME gate on the depth-prior term ONLY -- the
positional term  weight[0] * pos_decay  and every other tensor path are left
byte-identical. It monkeypatches GeoPriorGen.forward; NO parameter is changed and
NO training occurs. The gate is controlled entirely by env vars, so importing
this module with the gate disabled is a no-op.

Env contract:
  MPA_ORACLE   "off" | "edge" | "gradient"          [default: off]
      edge     -> inject the depth prior ONLY where the depth map has a true
                  discontinuity (|grad depth| > tau); zero it on smooth pixels.
      gradient -> inject ONLY on smooth pixels (the complement); should
                  REPLICATE the conflation damage if MP-A is true.
  MPA_TAU      edge threshold on normalized-depth Sobel gradient magnitude
                                                     [default: 0.15]
  MPA_STAGES   comma list of stage indices to gate  [default: 2]  (stage-2 GSA)

The edge mask M is computed per forward from the SAME interpolated depth map the
prior already uses (GeoPriorGen interpolates x_e to (H,W) at the top of forward),
via a median-prefiltered Sobel gradient magnitude, thresholded at tau. The
pairwise depth term mask_d[i,j] is gated by an endpoint factor
  edge:      g[i,j] = max(M_i, M_j)
  gradient:  g[i,j] = 1 - max(M_i, M_j)
so a pair keeps its depth-difference bias iff (edge) at least one endpoint sits on
a discontinuity, or (gradient) neither does. gate ONLY multiplies the
weight[1]*mask_d contribution.

A module-level dict `LAST_GATE_STATS` records the mean gated-pixel fraction of the
most recent gated forward, so the runner can report %pixels gated for the tau
sweep without touching the metric path.
"""

from __future__ import annotations

import os
from typing import Dict

import torch
import torch.nn.functional as F

# Read once at import; the runner sets these before importing the eval module.
_MODE = os.environ.get("MPA_ORACLE", "off").strip().lower()
_TAU = float(os.environ.get("MPA_TAU", "0.15"))
_STAGES = {
    int(s) for s in os.environ.get("MPA_STAGES", "2").split(",") if s.strip() != ""
}

LAST_GATE_STATS: Dict[str, float] = {"pixel_gated_frac": float("nan"), "n_calls": 0.0}

# Sobel kernels (registered lazily per device/dtype).
_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])


def _edge_mask(depth_hw: torch.Tensor, tau: float) -> torch.Tensor:
    """Robust depth-discontinuity mask at stage resolution.

    depth_hw: (B, 1, H, W) normalized depth already interpolated to (H, W).
    Returns M: (B, H, W) float in {0,1}, 1 where |grad depth| (normalized) > tau.
    """
    b, _, h, w = depth_hw.shape
    # median prefilter (3x3) to suppress single-pixel spikes -> robust gradient.
    if h >= 3 and w >= 3:
        patches = F.unfold(depth_hw, kernel_size=3, padding=1)  # (B, 9, H*W)
        med = patches.median(dim=1).values.view(b, 1, h, w)
    else:
        med = depth_hw
    kx = _SOBEL_X.to(med).view(1, 1, 3, 3)
    ky = _SOBEL_Y.to(med).view(1, 1, 3, 3)
    gx = F.conv2d(med, kx, padding=1)
    gy = F.conv2d(med, ky, padding=1)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-12)  # (B,1,H,W)
    # normalize gradient magnitude per-image to [0,1] so tau is scale-stable.
    gmax = grad.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    grad_n = grad / gmax
    m = (grad_n > tau).float()  # (B,1,H,W)
    return m[:, 0]  # (B,H,W)


def _pair_gate_1d(mask_line: torch.Tensor) -> torch.Tensor:
    """Given per-position edge membership along a 1-D axis, build the pairwise
    endpoint gate g[i,j] = max(M_i, M_j).

    mask_line: (..., L) float in {0,1}. Returns (..., L, L).
    """
    mi = mask_line.unsqueeze(-1)  # (...,L,1)
    mj = mask_line.unsqueeze(-2)  # (...,1,L)
    return torch.maximum(mi, mj)


def install(GeoPriorGen_cls) -> None:
    """Monkeypatch GeoPriorGen.forward with the env-gated oracle branch.

    When MPA_ORACLE == 'off' this is a no-op (original forward retained). The
    patched forward reproduces the original math exactly, then -- only for the
    split (decomposed) stage-2 path and only for stages in _STAGES -- multiplies
    the weight[1]*mask_d contribution by the edge (or 1-edge) endpoint gate.
    """
    if _MODE not in ("edge", "gradient"):
        return  # gate disabled -> leave the frozen forward untouched.

    orig_forward = GeoPriorGen_cls.forward
    # tag each GeoPriorGen with its stage index so we can gate only _STAGES.
    # We infer stage from embed_dim via the decay buffer length is not enough;
    # instead we gate ALL split-stage GeoPriorGen and record which fired. In
    # DFormerv2-S only stage-2 uses split_or_not=True among the mid stages, and
    # stage-0/1 are also split but at higher res. To honor MPA_STAGES we compare
    # against a per-instance stage attribute set by a light patch on the parent.
    def patched_forward(self, HW_tuple, depth_map, split_or_not=False):
        # exact original interpolation + math, re-derived so we can intercept
        # mask_d before it is folded into mask_h/mask_w.
        depth_i = F.interpolate(
            depth_map, size=HW_tuple, mode="bilinear", align_corners=False
        )
        H, W = HW_tuple

        gate_this = split_or_not and getattr(self, "_mpa_stage", None) in _STAGES

        if split_or_not:
            index = torch.arange(H * W).to(self.decay)
            sin = torch.sin(index[:, None] * self.angle[None, :]).reshape(H, W, -1)
            cos = torch.cos(index[:, None] * self.angle[None, :]).reshape(H, W, -1)

            mask_d_h = self.generate_1d_depth_decay(H, W, depth_i.transpose(-2, -1))
            mask_d_w = self.generate_1d_depth_decay(W, H, depth_i)

            mask_h_pos = self.generate_1d_decay(H)
            mask_w_pos = self.generate_1d_decay(W)

            if gate_this:
                # edge mask at stage resolution: (B, H, W)
                M = _edge_mask(depth_i, _TAU)  # (B,H,W)
                if _MODE == "gradient":
                    M = 1.0 - M
                # record gated-pixel fraction (per-pixel, before pairing).
                frac = M.mean().item()
                n = LAST_GATE_STATS["n_calls"]
                prev = LAST_GATE_STATS["pixel_gated_frac"]
                LAST_GATE_STATS["pixel_gated_frac"] = (
                    frac if n == 0 or prev != prev else (prev * n + frac) / (n + 1)
                )
                LAST_GATE_STATS["n_calls"] = n + 1

                # mask_d_h: (B, nheads, W, H, H) pairwise over rows (H) per col w.
                # endpoint edge membership along H for each column w: M[:, :, w]
                # -> shape (B, W, H). build pairwise (B,W,H,H) then broadcast heads.
                M_hcol = M.permute(0, 2, 1)  # (B, W, H)
                g_h = _pair_gate_1d(M_hcol)  # (B, W, H, H)
                g_h = g_h.unsqueeze(1)  # (B,1,W,H,H) -> broadcast over heads
                # mask_d_w: (B, nheads, H, W, W) pairwise over cols (W) per row h.
                M_wrow = M  # (B, H, W)
                g_w = _pair_gate_1d(M_wrow)  # (B, H, W, W)
                g_w = g_w.unsqueeze(1)  # (B,1,H,W,W)

                mask_d_h = mask_d_h * g_h
                mask_d_w = mask_d_w * g_w

            mask_h = (
                self.weight[0] * mask_h_pos.unsqueeze(0).unsqueeze(2)
                + self.weight[1] * mask_d_h
            )
            mask_w = (
                self.weight[0] * mask_w_pos.unsqueeze(0).unsqueeze(2)
                + self.weight[1] * mask_d_w
            )
            return ((sin, cos), (mask_h, mask_w))

        # non-split path (stage-3): unchanged, never gated.
        return orig_forward(self, HW_tuple, depth_map, split_or_not=split_or_not)

    GeoPriorGen_cls.forward = patched_forward


def tag_stages(model_root) -> None:
    """Walk the encoder and stamp each GeoPriorGen with its stage index so the
    gate can restrict itself to MPA_STAGES. Stage index = position in .layers."""
    # locate the dformerv2 encoder (segmodel.backbone or .encoder)
    enc = None
    for attr in ("backbone", "encoder"):
        cand = getattr(model_root, attr, None)
        if cand is not None and hasattr(cand, "layers"):
            enc = cand
            break
    if enc is None:
        # fallback: search modules for one carrying `.layers` of BasicLayer
        for m in model_root.modules():
            if hasattr(m, "layers") and hasattr(m, "num_layers"):
                enc = m
                break
    if enc is None:
        return
    for stage_idx, layer in enumerate(enc.layers):
        for blk in getattr(layer, "blocks", []):
            geo = getattr(blk, "Geo", None)
            if geo is not None:
                geo._mpa_stage = stage_idx
