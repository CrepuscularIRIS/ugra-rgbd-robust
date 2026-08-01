"""Modified EncoderDecoder wrapper for n7: RGB Decomposition Adapter + Depth Witness.

Drop-in replacement for the base segmodel. Same forward(rgb, modal_x, label) contract:
  - label is not None -> returns scalar loss (base CE + aux losses)
  - label is None -> returns logits

The adapter modifies the stage-2 (index 2, dim=256) backbone output.
Aux heads are training-only — skipped when self.training is False.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

from rgb_decomp_adapter import RGBDecompAdapter, DepthWitnessHead, RGBSegHead
from depth_witness_targets import compute_witness_targets


class DFormerv2WithAdapter(nn.Module):

    def __init__(self, base_model, num_classes=40, adapter_stage_idx=2,
                 witness_weight=0.05, seg_aux_weight=0.2, n_bins=8):
        super().__init__()
        self.base = base_model
        self.cfg = base_model.cfg
        self.adapter_stage_idx = adapter_stage_idx
        self.witness_weight = witness_weight
        self.seg_aux_weight = seg_aux_weight
        self.n_bins = n_bins

        in_dim = base_model.channels[adapter_stage_idx]
        self.adapter = RGBDecompAdapter(in_dim=in_dim, sub_dim=64)
        self.witness_head = DepthWitnessHead(in_dim=in_dim, hidden_dim=128, n_bins=n_bins)
        self.seg_aux_head = RGBSegHead(in_dim=64 * 3, num_classes=num_classes)

    def forward(self, rgb, modal_x=None, label=None):
        backbone_out = self.base.backbone(rgb, modal_x)
        feats = list(backbone_out)

        if self.training:
            stage2_feat = feats[self.adapter_stage_idx]
            adapted_feat, decomp_concat = self.adapter(stage2_feat)
            feats[self.adapter_stage_idx] = adapted_feat
        else:
            stage2_feat = feats[self.adapter_stage_idx]
            adapted_feat, _ = self.adapter(stage2_feat)
            feats[self.adapter_stage_idx] = adapted_feat

        feats_tuple = tuple(feats)
        if getattr(self.cfg, "decoder", "") == "CMaGAE":
            depth_1ch = modal_x[:, 0:1] if modal_x is not None and modal_x.dim() == 4 else modal_x
            out = self.base.decode_head.forward(feats_tuple, depth_1ch)
        else:
            out = self.base.decode_head.forward(feats_tuple)

        if isinstance(out, dict) and "pred" in out:
            logits = out["pred"]
        else:
            logits = out

        orisize = rgb.shape
        logits = F.interpolate(logits, size=orisize[-2:], mode="bilinear", align_corners=False)

        if label is None:
            return logits

        bg = self.cfg.background
        mask = label.long() != bg
        loss = self.base.criterion(logits, label.long())[mask].mean()

        if self.base.aux_head:
            aux_fm = self.base.aux_head(feats[0] if not hasattr(self.base, 'aux_index') else feats[self.base.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode="bilinear", align_corners=False)
            loss += self.base.aux_rate * self.base.criterion(aux_fm, label.long())[mask].mean()

        if self.training:
            seg_aux_logits = self.seg_aux_head(decomp_concat)
            seg_aux_logits = F.interpolate(seg_aux_logits, size=orisize[-2:], mode="bilinear", align_corners=False)
            seg_aux_loss = self.base.criterion(seg_aux_logits, label.long())[mask].mean()
            loss += self.seg_aux_weight * seg_aux_loss

            if modal_x is not None:
                rgb_feat_detached = adapted_feat.detach()
                bin_logits, normal_pred = self.witness_head(rgb_feat_detached)

                bin_targets, normal_targets, valid_mask = compute_witness_targets(modal_x, self.n_bins)

                h_feat, w_feat = bin_logits.shape[2:]
                bin_targets_down = F.interpolate(
                    bin_targets.unsqueeze(1).float(), size=(h_feat, w_feat), mode="nearest"
                ).squeeze(1).long()
                normal_targets_down = F.interpolate(
                    normal_targets, size=(h_feat, w_feat), mode="bilinear", align_corners=False
                )
                valid_mask_down = F.interpolate(
                    valid_mask.unsqueeze(1).float(), size=(h_feat, w_feat), mode="nearest"
                ).squeeze(1)

                bin_loss = F.cross_entropy(bin_logits, bin_targets_down, reduction="none")
                bin_loss = (bin_loss * valid_mask_down).sum() / (valid_mask_down.sum() + 1e-6)

                normal_pred_norm = F.normalize(normal_pred, dim=1, eps=1e-6)
                normal_targets_norm = F.normalize(normal_targets_down, dim=1, eps=1e-6)
                cos_loss = 1 - (normal_pred_norm * normal_targets_norm).sum(dim=1)
                cos_loss = (cos_loss * valid_mask_down).sum() / (valid_mask_down.sum() + 1e-6)

                witness_loss = bin_loss + cos_loss
                loss += self.witness_weight * witness_loss

        return loss
