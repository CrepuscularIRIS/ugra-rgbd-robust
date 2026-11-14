"""Geodesic Unlearning (GU): Grassmannian subspace edit of one weight matrix.

Reference: arXiv:2610.04792, Sec. 5 and App. D.1 (Eqs. 2-4, 6-9).

Given a multimodal weight W_TI and the weight W_T of a matched unimodal reference
(same shapes, trained with the missing modality replaced by a null input), GU rotates the
top-k *input* singular subspace of W_TI toward that of W_T along the Grassmann geodesic,
keeping the output factor, the principal coefficients and the residual of W_TI:

    W_TI = U_k S_k V_k^T + R        (Eq. 2)
    W'   = R + U_k S_k V(eta)^T     (Eq. 4)

eta = 0 leaves the layer unchanged, eta = 1 reaches the reference subspace.
There is no loss and no gradient step: the reference only supplies the target subspace.
"""
from __future__ import annotations

import torch

_EPS = 1e-6


def as_matrix(w: torch.Tensor) -> torch.Tensor:
    """Linear (out, in) stays; conv (out, in, kh, kw) -> (out, in*kh*kw)."""
    return w.reshape(w.shape[0], -1)


def top_k_svd(w: torch.Tensor, k: int):
    """Return U_k (out,k), S_k (k,), V_k (in,k) of the top-k SVD of w."""
    u, s, vh = torch.linalg.svd(as_matrix(w).double(), full_matrices=False)
    return u[:, :k], s[:k], vh[:k].T


def principal_angles(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Principal angles (radians, ascending) between span(v1) and span(v2); bases are (d, k)."""
    overlap = v1.T @ v2
    cos = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)  # descending cos = ascending angle
    sin = torch.linalg.svdvals(v2 - v1 @ overlap).flip(0)  # sines of the same angles, ascending
    return torch.atan2(sin, cos)  # arccos alone loses ~1e-8 accuracy for nearly equal subspaces


def grassmann_distance(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """Canonical Grassmann geodesic distance: l2 norm of the principal-angle vector (Eq. 10)."""
    return principal_angles(v1, v2).norm()


def geodesic_basis(v_ti: torch.Tensor, v_t: torch.Tensor, eta: float) -> torch.Tensor:
    """Orthonormal basis V(eta) of the point at fraction `eta` of the geodesic span(v_ti) -> span(v_t).

    Implements Eqs. (6)-(9): V_ti^T V_t = P cos(Theta) Q^T, A = V_ti P, B = V_t Q,
    Y = (B - A cos Theta)(sin Theta)^+, V~(eta) = A cos(eta Theta) + Y sin(eta Theta), V(eta) = V~ P^T.
    V(0) = v_ti and V(1) = v_t Q P^T (same span as v_t).
    """
    p, c, qh = torch.linalg.svd(v_ti.T @ v_t)
    c = c.clamp(0.0, 1.0)
    theta = torch.arccos(c)
    a = v_ti @ p
    b = v_t @ qh.T
    sin = torch.sin(theta)
    inv_sin = torch.where(sin > _EPS, 1.0 / sin.clamp_min(_EPS), torch.zeros_like(sin))  # pseudo-inverse
    y = (b - a * c) * inv_sin  # column-wise; zero where theta ~ 0
    v_tilde = a * torch.cos(eta * theta) + y * torch.sin(eta * theta)
    return v_tilde @ p.T


def edit_weight(w_ti: torch.Tensor, v_ref: torch.Tensor, k: int, eta: float) -> torch.Tensor:
    """Apply GU to one weight tensor.

    `v_ref` is the top-k input singular basis (in, k) of the matched reference layer
    (use `top_k_svd(w_ref, k)[2]`). Returns a tensor with the shape and dtype of `w_ti`.
    """
    mat = as_matrix(w_ti).double()
    u_k, s_k, v_k = top_k_svd(w_ti, k)
    residual = mat - (u_k * s_k) @ v_k.T  # Eq. (2)
    v_eta = geodesic_basis(v_k, v_ref.double(), eta)
    edited = residual + (u_k * s_k) @ v_eta.T  # Eq. (4)
    return edited.reshape(w_ti.shape).to(w_ti.dtype)


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(_EPS)


def layer_scores(mean_angles: torch.Tensor, harm_gaps: torch.Tensor) -> torch.Tensor:
    """Target-layer score s_l = z(theta_l) + z(delta_harm_l) across candidate layers (Eq. 1)."""
    return zscore(mean_angles.double()) + zscore(harm_gaps.double())
