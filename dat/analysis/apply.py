"""Temporarily edit a model's weights (GU or plain interpolation) and restore them afterwards."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from . import gu


def candidate_layers(sd: dict, k: int, prefix: str = "backbone.") -> list[str]:
    """Backbone weight tensors (linear or conv) large enough for a top-k input subspace."""
    return [n for n, w in sd.items()
            if n.startswith(prefix) and n.endswith(".weight") and w.ndim in (2, 4)
            and min(w.shape[0], w[0].numel()) >= 2 * k]


@contextmanager
def edited(model, c_sd: dict, r_sd: dict, layers: list[str], k: int, eta: float, mode: str = "gu"):
    """mode "gu": rotate each layer's top-k input subspace toward R by `eta`.
    mode "interp": W = (1-eta) W_C + eta W_R (the layer-interpolation control, no subspace structure)."""
    params = dict(model.named_parameters())
    try:
        for n in layers:
            if mode == "gu":
                new = gu.edit_weight(c_sd[n].float().cpu(), gu.top_k_svd(r_sd[n].float(), k)[2], k, eta)
            else:
                new = (1 - eta) * c_sd[n].float().cpu() + eta * r_sd[n].float()
            params[n].data.copy_(new.to(params[n].device, params[n].dtype))
        yield model
    finally:
        for n in layers:
            params[n].data.copy_(c_sd[n])
