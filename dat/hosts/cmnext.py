"""CMNeXt host (jamycheung/DELIVER, `semseg/models/cmnext.py`), RGB + depth on NYU Depth V2.

Facts about upstream's depth input that fix the empty-input constant `c` (see also patches/cmnext-dat.patch):

* NYU depth is fed as a 3-channel HHA jpeg (`semseg/datasets/nyu.py`, `HHA/<name>.jpg`), so depth is (B, 3, H, W).
* Its only preprocessing is `/= 255` (`semseg/augmentations_mm.py::Normalize`); there is no mean/std, so depth is in
  [0, 1] and "standardized" depth is the same tensor.
* Upstream's own "no data" value is 0: `RandomResizedCrop` pads every non-image modality with `fill=0` after cropping
  and `val_mm.pad_image` zero-pads. A zero tensor is therefore the all-black HHA image, and c = 0.0.

RGB-only: `CMNeXt(modals=['img'])` has `num_modals == 0` (the backbone drops the whole depth branch, FRM and FFM) and
is a different, smaller network than the RGB-D model fed a constant depth map. The paper's RGB-only reference
(Sec. 3.1) is the latter (same network, `RGB_ONLY` preset); `build(rgb_only=True)` gives the former for comparison.

Upstream's `cmnext_settings` only defines B2, B4 and B5 (B0/B1/B3 are commented out), so those are the variants here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import paths
from .base import Host

VARIANTS = ("B2", "B4", "B5")


def _unwrap(ckpt: dict) -> dict:
    """Released files are raw state_dicts; train_mm.py's *_checkpoint.pth wraps one in 'model_state_dict'."""
    for key in ("model_state_dict", "state_dict", "model"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


class CMNeXtHost(Host):
    family = "cmnext"
    empty_value = 0.0  # HHA / 255, zero-padded by upstream; no mean/std (module docstring)
    single_stream_mode = True  # modals=['img'] builds the depth-free network

    def __init__(self, variant: str):
        if variant not in VARIANTS:
            raise KeyError(f"CMNeXt variant {variant!r} unsupported; upstream defines {VARIANTS}")
        self.variant = variant
        self.name = f"CMNeXt-{variant}"

    def build(self, num_classes: int = 40, device: str = "cuda", rgb_only: bool = False) -> nn.Module:
        paths.activate(paths.CMNEXT_ROOT)
        from semseg.models import CMNeXt  # needs fvcore, tabulate (pure python)
        modals = ["img"] if rgb_only else ["img", "depth"]
        return CMNeXt(self.name, num_classes, modals).to(device)

    def load_weights(self, model: nn.Module, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = {k.removeprefix("module."): v for k, v in _unwrap(ckpt).items()}  # DDP-saved files
        # strict=True raises RuntimeError listing missing and unexpected keys
        model.load_state_dict(state, strict=True)

    def forward(self, model: nn.Module, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if model.backbone.num_modals == 0:  # modals=['img'] model: no depth branch
            return model([rgb])
        if depth.shape[1] == 1:  # upstream feeds 3-channel HHA; accept a single channel by replication
            depth = depth.expand(-1, 3, -1, -1)
        return model([rgb, depth])  # interpolated to rgb's H x W inside CMNeXt.forward


def make_host(name: str) -> CMNeXtHost:
    """`make_host("CMNeXt-B4")`; the bare variant ("B4") is accepted too."""
    return CMNeXtHost(name.split("-")[-1].upper())
