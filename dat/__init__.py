"""Depth-availability training (DAT): code for the paper's training strategy and its evaluation protocol.

    dat.availability   availability sampling: complete / missing / partial depth (Sec. 3.4, Algorithm 1)
    dat.presets        the training arms of Table 1 and Table 3 as availability mixtures
    dat.metrics        S_c, S_m, S_RGB, T/B/G (Eq. 1) and classwise changes (Eq. 5)
    dat.hosts          architecture adapters: DFormerv1/v2, CMNeXt, GeminiFusion
    dat.train/.evaluate  single-GPU training and endpoint evaluation for the DFormer family
    dat.analysis       post-training repair routes of Fig. 2 (reference fine-tuning, interpolation, GU, feature repair)
"""
from .availability import (AvailabilitySampler, Mixture, PartialMaskSpec, apply_availability, missing_depth,
                           partial_depth)
from .metrics import Endpoints

__all__ = ["AvailabilitySampler", "Mixture", "PartialMaskSpec", "apply_availability", "missing_depth",
           "partial_depth", "Endpoints"]
