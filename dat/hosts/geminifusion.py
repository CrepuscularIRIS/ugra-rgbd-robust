"""GeminiFusion host (MiT-B3 / MiT-B5, NYU Depth V2, RGB + depth).

Upstream: https://github.com/JiaDingCN/GeminiFusion (pinned by scripts/setup_geminifusion.sh).

Facts about the architecture that shape this adapter (all verified against the pinned commit):

* No single-stream mode. `models/modules.py` hard-wires `num_parallel = 2` and `WeTr.forward`
  (models/segformer.py) always decodes both streams, so a network without the depth stream would be a different
  architecture (`single_stream_mode = False`). The RGB-only baseline of Sec. 3.1 does not need one: it is the same
  network trained and tested with the constant depth map (`--dat-preset RGB_ONLY`), exactly as for DFormerv1,
  so `supports_rgb_only = True`.
* Input is a list `[rgb, depth]`, both (B, 3, H, W). Depth is a grey PNG, replicated to 3 channels
  (utils/datasets.py `read_image`) and scaled by `/255` only (utils/augmentations_mm.py `Normalize`:
  only the "rgb" key gets mean/std). So depth lives in [0, 1] with NO standardization.
* Empty depth c = 0.0: the only "no depth" upstream ever feeds is zero padding, and it pads depth with
  `fill=0` before `/255` (`RandomResizedCrop`: `TF.pad(v, padding, fill=0)`), which stays 0 afterwards.
  There is no depth standardization that could move it, so c = 0 in the network's input space.
* `WeTr.forward` returns `([out_rgb, out_depth, ens], None)`; `ens = sum_l softmax(alpha)_l * out_l.detach()`.
  The headline numbers (validate(): the last entry, "ens") use `ens`, so `forward` returns `ens`. Because
  `ens` is detached, training needs all three outputs: upstream's loss is the SUM of the three
  cross-entropies, which `loss` reproduces (main.py `train`). DAT leaves that loss untouched.
* Outputs are at 1/4 resolution; upstream upsamples bilinearly for the loss and bicubically (cv2) for
  validation. `forward` upsamples bilinearly.
* Importing `models.segformer` also imports Swin code that needs `mmdet` and `mmcv.fileio` (old mmcv 1.x).
  MiT does not use them, so inert stubs are installed if the real packages are unavailable.
* Upstream's top-level package names (`models`, `utils`) collide with DFormer's; use one family per process.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from dat import paths
from dat.hosts.base import Host

logger = logging.getLogger(__name__)

EMPTY_VALUE = 0.0  # c: zero depth, see module docstring
_DPR = {"b3": 0.4, "b5": 0.35}  # upstream README training commands (--dpr)
# The released NYU MiT-B3 checkpoint carries 80 `encoder.score_predictor.*` tensors (1.26M params), left over from
# TokenFusion. No code in the pinned upstream defines or reads them (upstream loads with strict=False), so they are
# the only unexpected keys tolerated; anything else, and every missing key, is an error.
KNOWN_UNUSED_PREFIXES = ("encoder.score_predictor.",)


def _install_stubs() -> None:
    """Make `models.segformer` importable without mmdet / mmcv 1.x (only Swin needs them)."""
    try:
        import mmdet.utils  # noqa: F401
        import mmcv.fileio  # noqa: F401
        return
    except ImportError:
        pass
    mmdet = types.ModuleType("mmdet")
    mmdet_utils = types.ModuleType("mmdet.utils")
    mmdet_utils.get_root_logger = lambda *a, **k: logging.getLogger("mmdet")
    mmdet.utils = mmdet_utils
    custom = types.ModuleType("mmcv_custom")
    custom.load_checkpoint = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("Swin checkpoints are not supported by this stubbed import"))
    sys.modules.setdefault("mmdet", mmdet)
    sys.modules.setdefault("mmdet.utils", mmdet_utils)
    sys.modules.setdefault("mmcv_custom", custom)


def _segformer():
    paths.activate(paths.GEMINIFUSION_ROOT)
    _install_stubs()
    from models import segformer  # noqa: WPS433 (upstream package)
    return segformer


@contextlib.contextmanager
def _skip_pretrained(segformer):
    """`WeTr.__init__` loads `pretrained/<backbone>.pth` (relative path) strictly; neutralize that."""
    real_load, real_expand = torch.load, segformer.expand_state_dict
    torch.load = lambda *a, **k: {"head.weight": None, "head.bias": None}
    segformer.expand_state_dict = lambda model_dict, state_dict, n: model_dict
    try:
        yield
    finally:
        torch.load, segformer.expand_state_dict = real_load, real_expand


def _strip_ddp_prefix(sd: dict) -> dict:
    """Checkpoints are saved from DDP: every key starts with 'module.'. The model's own inner
    `ModuleParallel.module` keys never start with it, so stripping one leading prefix is exact."""
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}


class GeminiFusionHost(Host):
    family = "geminifusion"
    empty_value = EMPTY_VALUE
    supports_rgb_only = True  # constant-depth RGB-only, same network (see docstring)
    single_stream_mode = False  # num_parallel=2 and both streams are always decoded

    def __init__(self, name: str, backbone: str):
        self.name, self.backbone = name, backbone
        self.last_load_report: dict | None = None

    def build(self, num_classes: int = 40, device: str = "cuda") -> nn.Module:
        seg = _segformer()
        key = self.backbone.split("_")[1]
        with _skip_pretrained(seg):
            model = seg.WeTr(self.backbone, num_classes=num_classes, n_heads=8, dpr=_DPR[key], drop_rate=0.0)
        return model.to(device)

    def load_weights(self, model: nn.Module, path: str) -> None:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except Exception:  # pickled non-tensor entries in some self-trained checkpoints
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = _strip_ddp_prefix(ckpt["segmenter"] if "segmenter" in ckpt else ckpt)
        res = model.load_state_dict(sd, strict=False)
        unexpected = [k for k in res.unexpected_keys if not k.startswith(KNOWN_UNUSED_PREFIXES)]
        if res.missing_keys or unexpected:
            raise RuntimeError(
                f"{path}: {len(res.missing_keys)} missing, {len(unexpected)} unexpected keys; "
                f"missing[:5]={res.missing_keys[:5]} unexpected[:5]={unexpected[:5]}")
        self.last_load_report = {"loaded": len(sd) - len(res.unexpected_keys), "missing": 0,
                                 "unexpected": 0, "ignored_known_unused": len(res.unexpected_keys)}

    @staticmethod
    def _depth3(depth: torch.Tensor) -> torch.Tensor:
        return depth.expand(-1, 3, -1, -1) if depth.shape[1] == 1 else depth

    def forward_all(self, model: nn.Module, rgb: torch.Tensor, depth: torch.Tensor) -> list[torch.Tensor]:
        """[rgb-stream, depth-stream, ensemble] logits, all upsampled to the input resolution."""
        outs, _ = model([rgb, self._depth3(depth)])
        size = rgb.shape[-2:]
        return [F.interpolate(o, size=size, mode="bilinear", align_corners=False) for o in outs]

    def forward(self, model: nn.Module, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Ensemble logits (what upstream reports as the model's prediction), (B, K, H, W)."""
        return self.forward_all(model, rgb, depth)[2]

    def loss(self, model, rgb, depth, label, ignore_index: int = 255) -> torch.Tensor:
        """Upstream's training loss unchanged: sum of CE over the two streams and the (detached) ensemble.
        The ensemble term has gradient only w.r.t. `alpha`; the base-class loss on `forward` alone would
        train nothing else."""
        label = label.long()
        return sum(F.cross_entropy(o, label, ignore_index=ignore_index)
                   for o in self.forward_all(model, rgb, depth))


def make_host(name: str) -> GeminiFusionHost:
    key = name.lower().replace("_", "-")
    if key not in ("geminifusion-b3", "geminifusion-b5"):
        raise KeyError(f"unknown GeminiFusion host {name!r}; use 'GeminiFusion-B3' or 'GeminiFusion-B5'")
    return GeminiFusionHost(name, "mit_" + key[-2:])
