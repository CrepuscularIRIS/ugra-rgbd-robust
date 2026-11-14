"""Host interface: what DAT needs from an RGB-D architecture.

DAT changes only the inputs, so a host adapter is small: build the network, load a checkpoint, run
`(rgb, depth) -> logits`, and compute the segmentation loss. The empty-input constant `c` is the value
the architecture's own default training uses for "no depth" after its own preprocessing/standardization.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class Host(ABC):
    name: str  # e.g. "DFormerv2-S"
    family: str  # "dformerv1" | "dformerv2" | "cmnext" | "geminifusion"
    empty_value: float  # c in D~ = M*D + (1-M)*c
    supports_rgb_only: bool = True  # the constant-depth RGB-only baseline of Sec. 3.1 can be trained (same network,
    #                                 depth replaced by the empty-input constant in training and testing)
    single_stream_mode: bool = False  # the architecture can also run with no depth stream at all (a different network)

    @abstractmethod
    def build(self, num_classes: int = 40, device: str = "cuda") -> nn.Module:
        """Instantiate the network (no pretrained download) and move it to `device`."""

    @abstractmethod
    def load_weights(self, model: nn.Module, path: str) -> None:
        """Load a released or self-trained checkpoint; must fail loudly on missing/unexpected keys."""

    @abstractmethod
    def forward(self, model: nn.Module, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Logits of shape (B, K, H, W) at the input resolution."""

    def loss(self, model: nn.Module, rgb: torch.Tensor, depth: torch.Tensor, label: torch.Tensor,
             ignore_index: int = 255) -> torch.Tensor:
        """Pixelwise cross-entropy averaged over valid labelled pixels (Eq. 4)."""
        logits = self.forward(model, rgb, depth)
        return F.cross_entropy(logits, label.long(), ignore_index=ignore_index)


_REGISTRY = {
    "dformer": "dat.hosts.dformer",
    "cmnext": "dat.hosts.cmnext",
    "geminifusion": "dat.hosts.geminifusion",
}


def get_host(name: str) -> Host:
    """`get_host("DFormerv2-S")`, `get_host("DFormerv1-T")`, `get_host("CMNeXt-B4")`, `get_host("GeminiFusion-B3")`."""
    key = name.lower()
    fam = "dformer" if key.startswith("dformer") else "cmnext" if key.startswith("cmnext") else \
        "geminifusion" if key.startswith("geminifusion") else None
    if fam is None:
        raise KeyError(f"unknown host {name!r}")
    return importlib.import_module(_REGISTRY[fam]).make_host(name)
