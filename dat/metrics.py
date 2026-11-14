"""Endpoint scores and the retained-advantage decomposition of Sec. 3.1, plus classwise changes (Eq. 5).

    T = S_m - S_RGB      retained advantage: what an RGB-D model still holds over RGB-only without depth
    B = S_c - S_RGB      complete-input benefit of depth
    G = S_c - S_m        depth dependence: accuracy lost when depth is withdrawn
    B = T + G

A deployed RGB-D segmenter is preferable to the RGB-only baseline under missing depth only if T > 0;
a small G alone does not establish this.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Endpoints:
    s_c: float  # complete-input mIoU
    s_m: float  # missing-input mIoU
    s_rgb: float | None = None  # RGB-only baseline of the same architecture (None if it cannot be trained)

    @property
    def g(self) -> float:
        return self.s_c - self.s_m

    @property
    def t(self) -> float | None:
        return None if self.s_rgb is None else self.s_m - self.s_rgb

    @property
    def b(self) -> float | None:
        return None if self.s_rgb is None else self.s_c - self.s_rgb

    def check_identity(self, tol: float = 1e-9) -> None:
        if self.s_rgb is not None and abs(self.b - (self.t + self.g)) > tol:
            raise AssertionError("B = T + G violated")

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(T=self.t, B=self.b, G=self.g)
        return d


def iou_from_hist(hist: np.ndarray) -> np.ndarray:
    """Per-class IoU in percent from a (K, K) confusion matrix (rows = ground truth, cols = prediction).

    Counts are aggregated over the whole dataset before dividing, as in the paper; a class with no
    ground-truth and no prediction contributes 0, matching the official DFormer evaluation.
    """
    hist = np.asarray(hist, dtype=np.float64)
    tp = np.diag(hist)
    denom = hist.sum(0) + hist.sum(1) - tp
    iou = np.where(denom > 0, tp / np.maximum(denom, 1e-12), 0.0)
    return 100.0 * iou


def miou_from_hist(hist: np.ndarray) -> float:
    return float(iou_from_hist(hist).mean())


def classwise_delta(hist_a: np.ndarray, hist_c: np.ndarray) -> tuple[np.ndarray, float]:
    """Eq. (5): delta_j = IoU_j(A) - IoU_j(C) under the same input condition, and its mean over classes."""
    delta = iou_from_hist(hist_a) - iou_from_hist(hist_c)
    return delta, float(delta.mean())


def format_row(name: str, e: Endpoints) -> str:
    t = "  n/a" if e.t is None else f"{e.t:+5.1f}"
    rgb = "  n/a" if e.s_rgb is None else f"{e.s_rgb:5.1f}"
    return f"{name:<18} RGB-only {rgb}  S_c {e.s_c:5.1f}  S_m {e.s_m:5.1f}  T {t}  G {e.g:5.1f}"
