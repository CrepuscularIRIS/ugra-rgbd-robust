#!/usr/bin/env python
"""Eval wrapper for HHD+GRP — hooks into the eval_robust_suite model loading
to apply per-head depth coupling BEFORE evaluation.

The HHD patch must be applied AFTER load_state_dict so that the per-head
_weight_depth_perhead parameters are loaded from the checkpoint.

Usage: same env vars as eval_robust_suite.py + GEO_RATIO(0.5).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(os.path.dirname(_HERE))
_ROOT = os.path.join(_PROJECT, "repos", "DFormer")
for _p in (os.path.join(_ROOT, "utils"), _ROOT, _PROJECT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_ROOT)
os.environ.setdefault("LOCAL_RANK", "0")

import torch
from hhd_geo_prior import patch_backbone


def main():
    import scripts.eval_robust_suite as suite

    _orig_load_state_dict = torch.nn.Module.load_state_dict

    def _hooked_load_state_dict(self, state_dict, strict=True, **kwargs):
        geo_ratio = float(os.environ.get("GEO_RATIO", "0.5"))
        if hasattr(self, "backbone"):
            info = patch_backbone(self.backbone, geo_ratio=geo_ratio)
            print(f"[hhd-eval] Patched {info['n_patched']} GeoPriorGen before loading weights")
        result = _orig_load_state_dict(self, state_dict, strict=strict, **kwargs)
        return result

    torch.nn.Module.load_state_dict = _hooked_load_state_dict
    try:
        suite.main()
    finally:
        torch.nn.Module.load_state_dict = _orig_load_state_dict


if __name__ == "__main__":
    main()
