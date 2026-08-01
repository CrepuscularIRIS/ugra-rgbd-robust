"""n7 cycle-4 (M3) model wrapper — Detached Semantic-Boundary Gate + Zero-Init Parallel Edge Adapter.

Same forward contract as the base segmodel and the cycle-3 wrapper:
  - label is not None  -> returns scalar loss (base CE + aux + boundary BCE)
  - label is None      -> returns logits

Anti-conflation invariants (must survive review):
  * B (boundary map) is supervised ONLY by the SEMANTIC label boundary (BCE), never depth edges.
  * B is STOP-GRADIENTED before it gates the adapter: gate = sigmoid(B_logits).detach().
  * The parallel adapter is zero-init => epoch-0 output == baseline for ANY B (identity at init).
  * Under missing depth (depth_available=0) the depth response and availability mask are 0, so B
    falls back to RGB-stem evidence (the entire_missing lever).
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
if EXP_DIR not in sys.path:
    sys.path.insert(0, EXP_DIR)

from boundary_gate_adapter import (
    BoundaryHead, ZeroInitParallelAdapter,
    second_order_depth_response, label_boundary_target, depth_edge_target,
)


class DFormerv2WithBoundaryGate(nn.Module):

    def __init__(self, base_model, num_classes=40, adapter_stage_idx=2, rgb_stem_idx=0,
                 boundary_weight=0.10, boundary_pos_weight=30.0, boundary_width=2,
                 supervision="label", stop_grad=True):
        super().__init__()
        self.base = base_model
        self.cfg = base_model.cfg
        self.adapter_stage_idx = adapter_stage_idx
        self.rgb_stem_idx = rgb_stem_idx
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width
        assert supervision in ("label", "depth")
        self.supervision = supervision      # "label" = canonical M3; "depth" = ablation (should regress)
        self.stop_grad = stop_grad          # True = canonical M3; False = ablation (task-grad leak)
        self.register_buffer("boundary_pos_weight", torch.tensor(float(boundary_pos_weight)))

        dim = base_model.channels[adapter_stage_idx]        # 256
        rgb_in = base_model.channels[rgb_stem_idx]          # 64
        self.boundary_head = BoundaryHead(rgb_in_dim=rgb_in, rgb_proj_dim=32, hidden=32)
        self.adapter = ZeroInitParallelAdapter(dim=dim, bottleneck=64)

        # diagnostics filled each forward (read by PCT probes); not part of the graph
        self.last_B = None
        self.last_B_logits = None

    def _depth_1ch(self, modal_x):
        if modal_x is None:
            return None
        return modal_x[:, 0:1] if modal_x.dim() == 4 else modal_x.unsqueeze(1)

    @staticmethod
    def _per_sample_depth_present(d1: torch.Tensor) -> torch.Tensor:
        """[B,1,1,1] float mask: 1 if sample has non-structureless depth.

        Eval entire_missing = zeros_like(normalized modal_x) with default depth_available=1.
        Train modality-dropout also zeros modal_x. Detect structureless zeros so avail_mask
        matches train (0) under eval missing, not avail=1 + empty depth_resp (distribution shift).
        """
        # abs-max over C,H,W per sample; zeros_like -> 0
        return (d1.detach().abs().flatten(1).amax(dim=1) > 1e-8).to(dtype=d1.dtype).view(-1, 1, 1, 1)

    def forward(self, rgb, modal_x=None, label=None, depth_available: float = 1.0):
        feats = list(self.base.backbone(rgb, modal_x))
        stage2 = feats[self.adapter_stage_idx]
        rgb_stem = feats[self.rgb_stem_idx]
        H2, W2 = stage2.shape[-2:]
        B_, _, _, _ = stage2.shape
        dev, dt = stage2.device, stage2.dtype

        # --- second-order depth response @ stage-2 res (0 under missing depth) ---
        d1 = self._depth_1ch(modal_x)
        if d1 is not None and depth_available > 0:
            present = self._per_sample_depth_present(d1.to(dt))          # [B,1,1,1]
            resp_full = second_order_depth_response(d1.to(dt))
            depth_resp = F.interpolate(resp_full, size=(H2, W2), mode="area") * present
            avail = present.expand(B_, 1, H2, W2).to(dtype=dt)
        else:
            depth_resp = torch.zeros(B_, 1, H2, W2, device=dev, dtype=dt)
            avail = torch.zeros(B_, 1, H2, W2, device=dev, dtype=dt)
            present = torch.zeros(B_, 1, 1, 1, device=dev, dtype=dt)

        rgb_stem_ds = F.interpolate(rgb_stem, size=(H2, W2), mode="bilinear", align_corners=False)

        # --- boundary head + stop-grad gate ---
        B_logits = self.boundary_head(depth_resp, rgb_stem_ds, avail)   # [B,1,H2,W2]
        B_prob = torch.sigmoid(B_logits)
        B_gate = B_prob.detach() if self.stop_grad else B_prob          # canonical M3: STOP-GRADIENT
        self.last_B_logits = B_logits.detach()
        self.last_B = B_prob.detach()

        # --- zero-init parallel adapter, gated ---
        # B_gate [B,1,H2,W2] broadcasts over adapter channels [B,C,H2,W2]
        feats[self.adapter_stage_idx] = stage2 + B_gate * self.adapter(stage2)

        # --- decode ---
        feats_tuple = tuple(feats)
        if getattr(self.cfg, "decoder", "") == "CMaGAE":
            depth_1ch = self._depth_1ch(modal_x)
            out = self.base.decode_head.forward(feats_tuple, depth_1ch)
        else:
            out = self.base.decode_head.forward(feats_tuple)
        logits = out["pred"] if isinstance(out, dict) and "pred" in out else out
        logits = F.interpolate(logits, size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        if label is None:
            return logits

        bg = self.cfg.background
        lab = label.long()
        mask = lab != bg
        # empty-mask guard: mean() of empty tensor is NaN and would poison the run
        if mask.any():
            loss = self.base.criterion(logits, lab)[mask].mean()
        else:
            loss = logits.sum() * 0.0

        if self.base.aux_head:
            aux_idx = self.base.aux_index if hasattr(self.base, "aux_index") else 0
            aux_fm = self.base.aux_head(feats[aux_idx])
            aux_fm = F.interpolate(aux_fm, size=rgb.shape[-2:], mode="bilinear", align_corners=False)
            if mask.any():
                loss = loss + self.base.aux_rate * self.base.criterion(aux_fm, lab)[mask].mean()

        # --- boundary aux (trains B ONLY; gate is detached in canonical M3) ---
        if self.training:
            if self.supervision == "label":
                tgt = label_boundary_target(lab, ignore_index=bg, width=self.boundary_width)  # [B,1,H,W]
            else:
                # depth-edge ablation: skip when depth absent / structureless this batch
                depth_ok = (d1 is not None and depth_available > 0 and float(present.amax().item()) > 0.0)
                tgt = depth_edge_target(d1.to(dt)) if depth_ok else None
            if tgt is not None:
                tgt_ds = F.interpolate(tgt, size=(H2, W2), mode="area")
                tgt_ds = (tgt_ds > 0.10).to(dt)            # grok-review: soften 0.25->0.10 to avoid boundary mass-erasure at 1/16 res
                bce = F.binary_cross_entropy_with_logits(
                    B_logits, tgt_ds, pos_weight=self.boundary_pos_weight.to(dt))
                loss = loss + self.boundary_weight * bce

        return loss
