"""NYU val/test loader for diagnostics (DFormer preprocessing)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import torch

DFORMER_ROOT = Path(__file__).resolve().parents[2] / "repos" / "DFormer"
if str(DFORMER_ROOT) not in sys.path:
    sys.path.insert(0, str(DFORMER_ROOT))


class _DummyEngine:
    """Minimal stand-in for Engine used by get_val_loader."""

    def __init__(self):
        self.distributed = False
        self.local_rank = 0
        self.world_size = 1


def get_nyu_val_loader(config: Any, batch_size: int = 4, num_workers: int = 2):
    from utils.dataloader.dataloader import get_val_loader
    from utils.dataloader.RGBXDataset import RGBXDataset

    engine = _DummyEngine()
    # get_val_loader signature from DFormer
    val_loader, val_sampler = get_val_loader(
        engine,
        RGBXDataset,
        config,
        val_batch_size=batch_size,
    )
    return val_loader


def _to_tensor(x):
    import numpy as np

    if torch.is_tensor(x):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.as_tensor(x)


def iter_limited(loader, max_images: int) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Yield (rgb, depth, label) until max_images samples."""
    n = 0
    for batch in loader:
        # default collate of dict: {data: Tensor[B,...], modal_x, label, ...}
        if isinstance(batch, dict):
            rgb = _to_tensor(batch["data"])
            depth = _to_tensor(batch["modal_x"])
            label = _to_tensor(batch["label"])
        elif isinstance(batch, (list, tuple)):
            if len(batch) >= 3 and isinstance(batch[0], dict):
                # list of sample dicts (if collate disabled)
                rgb = torch.stack([_to_tensor(s["data"]) for s in batch])
                depth = torch.stack([_to_tensor(s["modal_x"]) for s in batch])
                label = torch.stack([_to_tensor(s["label"]) for s in batch])
            else:
                rgb, depth, label = _to_tensor(batch[0]), _to_tensor(batch[1]), _to_tensor(batch[2])
        else:
            raise RuntimeError(f"Unexpected batch type {type(batch)}")
        b = rgb.shape[0]
        take = min(b, max_images - n)
        yield rgb[:take], depth[:take], label[:take]
        n += take
        if n >= max_images:
            break
