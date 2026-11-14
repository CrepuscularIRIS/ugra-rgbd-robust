"""DFormerv1 (T/S/B/L) and DFormerv2 (S/B/L) on NYU Depth V2 and SUN RGB-D, via the unpatched upstream code.

DAT needs no change to DFormer: it only changes the depth tensor passed to `model(rgb, depth)`.
Upstream is expected at $DFORMER_ROOT (default repos/DFormer, see scripts/setup_dformer.sh; any checkout that has
`models/builder.py` and `local_configs/` works, patched or not). The empty-input constant is c = 0 in the
standardized depth space used by upstream (mean 0.48, std 0.28), as stated in Sec. 3.1 of the paper.
"""
from __future__ import annotations

import copy
import importlib
import os
import re
import tempfile

import torch
import torch.nn as nn

from .. import paths
from .base import Host

#           name          config file       family
_MODELS = {
    "DFormerv1-T": ("DFormer_Tiny", "dformerv1"),
    "DFormerv1-S": ("DFormer_Small", "dformerv1"),
    "DFormerv1-B": ("DFormer_Base", "dformerv1"),
    "DFormerv1-L": ("DFormer_Large", "dformerv1"),
    "DFormerv2-S": ("DFormerv2_S", "dformerv2"),
    "DFormerv2-B": ("DFormerv2_B", "dformerv2"),
    "DFormerv2-L": ("DFormerv2_L", "dformerv2"),
}
DATASETS = ("NYUDepthv2", "SUNRGBD")


class DFormerHost(Host):
    empty_value = 0.0
    supports_rgb_only = True  # constant depth in training and testing (Sec. 3.1)

    def __init__(self, name: str, dataset: str = "NYUDepthv2", pretrained: str | None = None,
                 data_root: str | None = None):
        if name not in _MODELS:
            raise KeyError(f"{name!r}: choose from {sorted(_MODELS)}")
        if dataset not in DATASETS:
            raise KeyError(f"{dataset!r}: choose from {DATASETS}")
        self.name, self.dataset, self.pretrained = name, dataset, pretrained
        self.config_name, self.family = _MODELS[name]
        self.data_root = data_root or paths.DATA_ROOT
        paths.activate(paths.DFORMER_ROOT)
        self._cfg = None

    # ---- config ------------------------------------------------------------------------------------------
    @property
    def cfg(self):
        """A private copy of the upstream config with dataset paths pointed at $RGBD_DATA_ROOT."""
        if self._cfg is None:
            cfg = copy.deepcopy(self._fresh_config())
            root = os.path.join(self.data_root, self.dataset)
            cfg.dataset_path = root
            cfg.rgb_root_folder = os.path.join(root, "RGB")
            cfg.gt_root_folder = os.path.join(root, "Label")
            cfg.x_root_folder = os.path.join(root, "Depth")
            cfg.train_source = os.path.join(root, "train.txt")
            cfg.eval_source = os.path.join(root, "test.txt")
            cfg.pad = False
            cfg.pretrained_model = self.pretrained  # None: random init (a full checkpoint is loaded afterwards)
            self._cfg = cfg
        return self._cfg

    def _fresh_config(self):
        """Upstream configs all mutate one module-level `C`; reload the chain so another host's values never leak in."""
        # The configs also create `checkpoints/<run>/` log folders relative to the cwd at import time; do the
        # import from a throw-away directory so nothing is left behind in the caller's tree.
        with tempfile.TemporaryDirectory() as scratch:
            cwd = os.getcwd()
            os.chdir(scratch)
            try:
                base = importlib.reload(importlib.import_module("local_configs._base_"))  # a new, empty C
                for name in (f"local_configs._base_.datasets.{self.dataset}",
                             f"local_configs.{self.dataset}.{self.config_name}"):
                    mod = importlib.reload(importlib.import_module(name))
            finally:
                os.chdir(cwd)
        assert mod.C is base.C
        return mod.C

    @property
    def num_classes(self) -> int:
        return int(self.cfg.num_classes)

    # ---- Host interface ----------------------------------------------------------------------------------
    def build(self, num_classes: int | None = None, device: str = "cuda") -> nn.Module:
        from models.builder import EncoderDecoder

        if num_classes is not None and num_classes != self.num_classes:
            raise ValueError(f"{self.dataset} has {self.num_classes} classes, got {num_classes}")
        model = EncoderDecoder(cfg=self.cfg, norm_layer=nn.BatchNorm2d)
        return model.to(device)

    def load_weights(self, model: nn.Module, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("state_dict", "model"):  # released checkpoints use one of these wrappers
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        res = model.load_state_dict(state, strict=False)
        unexpected = list(res.unexpected_keys)
        if self.family == "dformerv1":  # released v1 checkpoints also store four norm layers the network never uses
            unexpected = [k for k in unexpected if not re.fullmatch(r"backbone\.norm[0-3]\.(weight|bias)", k)]
        if res.missing_keys or unexpected:
            raise RuntimeError(f"{path}: {len(res.missing_keys)} missing {res.missing_keys[:3]}, "
                               f"{len(unexpected)} unexpected {unexpected[:3]}")

    def forward(self, model: nn.Module, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return model(rgb, depth)

    # ---- helpers used by dat.train / dat.evaluate --------------------------------------------------------
    def schedule(self) -> dict:
        """The original per-model training schedule from the upstream config."""
        c = self.cfg
        return {"lr": c.lr, "batch_size": c.batch_size, "epochs": c.nepochs, "warmup_epochs": c.warm_up_epoch,
                "weight_decay": c.weight_decay, "num_train": c.num_train_imgs,
                "iters_per_epoch": c.num_train_imgs // c.batch_size + 1}


def make_host(name: str, dataset: str = "NYUDepthv2", **kw) -> DFormerHost:
    return DFormerHost(name, dataset, **kw)
