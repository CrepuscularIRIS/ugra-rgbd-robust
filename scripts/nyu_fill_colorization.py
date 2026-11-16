"""Approximate RGB-guided Laplacian depth fill (NYU-official-style, NOT a verified
port — Sol review P2-10: this is a symmetric graph-Laplacian colorization with hard
known-value constraints, structurally close to but not identical to the toolbox
`fill_depth_colorization`; F3 therefore replays an official-STYLE fill, and any claim
of exact official-error replay must be dropped).

System: minimize sum_ij w_ij (z_i - z_j)^2 over unknown pixels with known depths as
hard constraints; w_ij = exp(-||I_i - I_j||^2 / (2 sigma^2)) over 8-neighbourhood.
Solved as a reduced sparse SPD system (unknowns only) via CG with splu fallback.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, splu


def fill_depth_colorization(rgb: np.ndarray, depth_sparse: np.ndarray,
                            sigma: float = 0.1) -> np.ndarray:
    """Fill missing depth (depth_sparse == 0) guided by rgb (H, W, 3 in [0, 1]).

    Solves the REDUCED system on unknown pixels only, with known depths as hard
    constraints (Sol review P1-1: the previous soft-constraint CG let global
    tolerance be dominated by the 1e4-weighted known rows, so large holes were
    left unfilled while info==0 reported success)."""
    h, w = depth_sparse.shape
    n = h * w
    known = (depth_sparse > 0).ravel()
    if known.all():
        return depth_sparse.copy()

    # 8-neighbourhood edge list
    offsets = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]
    rows, cols, vals = [], [], []
    idx = np.arange(n).reshape(h, w)
    for dy, dx in offsets:
        ys0, xs0 = slice(max(0, -dy), h - max(0, dy)), slice(max(0, -dx), w - max(0, dx))
        ys1, xs1 = slice(max(0, dy), h - max(0, -dy)), slice(max(0, dx), w - max(0, -dx))
        a = idx[ys0, xs0].ravel()
        b = idx[ys1, xs1].ravel()
        diff = (rgb[ys0, xs0] - rgb[ys1, xs1]).reshape(-1, 3)
        wgt = np.exp(-(diff ** 2).sum(axis=1) / (2 * sigma ** 2))
        rows.append(a); cols.append(b); vals.append(wgt)
    r = np.concatenate(rows); c = np.concatenate(cols); v = np.concatenate(vals)
    W = sp.csr_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n))
    deg = np.asarray(W.sum(axis=1)).ravel()
    L = (sp.diags(deg) - W).tocsc()

    unk = np.flatnonzero(~known)
    zk = np.zeros(n)
    zk[known] = depth_sparse.ravel()[known]
    A = L[unk][:, unk]
    rhs = -L[unk][:, known] @ zk[known]
    # ponytail: CG on the reduced SPD system; splu fallback if it fails to converge
    M = sp.diags(1.0 / np.maximum(A.diagonal(), 1e-12))
    z_u, info = cg(A, rhs, rtol=1e-8, maxiter=5000, M=M)
    if info != 0:
        z_u = splu(A).solve(rhs)
    out = zk
    out[unk] = z_u
    return out.reshape(h, w).astype(np.float32)


def _self_check() -> None:
    h, w = 24, 32
    rgb = np.zeros((h, w, 3))
    rgb[:, w // 2:] = 0.9  # strong color edge at the middle
    d = np.zeros((h, w), np.float32)
    d[:, 0] = 2.0          # known: left wall at 2 m
    d[:, -1] = 5.0         # known: right wall at 5 m
    z = fill_depth_colorization(rgb, d)
    assert np.isfinite(z).all()
    assert abs(z[:, 0].mean() - 2.0) < 0.05 and abs(z[:, -1].mean() - 5.0) < 0.05
    # color edge must keep the two sides apart
    assert z[:, w // 2 - 2].mean() < 3.2 < z[:, w // 2 + 2].mean(), (z[:, w // 2 - 2].mean(), z[:, w // 2 + 2].mean())
    # all-known input is untouched
    d2 = np.ones((h, w), np.float32) * 3.0
    assert np.allclose(fill_depth_colorization(rgb, d2), d2)
    # P1-1 regression: a LARGE hole in a constant scene must be fully filled
    rgb3 = np.zeros((120, 160, 3))
    d3 = np.ones((120, 160), np.float32) * 2.0
    d3[30:90, 40:120] = 0.0
    z3 = fill_depth_colorization(rgb3, d3)
    assert np.abs(z3[30:90, 40:120] - 2.0).max() < 0.05, np.abs(z3[30:90, 40:120] - 2.0).max()


if __name__ == "__main__":
    _self_check()
    print("nyu_fill_colorization self-check OK")
