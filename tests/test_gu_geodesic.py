"""Property tests of the Grassmannian geodesic edit (GU): geometric claims of arXiv:2610.04792 App. D, not just shapes."""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dat.analysis import gu  # noqa: E402

torch.manual_seed(0)
D, K = 64, 8


def _pair(overlap_noise=0.7):
    """Two orthonormal bases in R^D whose subspaces overlap partially."""
    a = torch.linalg.qr(torch.randn(D, K, dtype=torch.double))[0]
    b = torch.linalg.qr(a + overlap_noise * torch.randn(D, K, dtype=torch.double))[0]
    return a, b


def _same_span(a, b, tol=1e-8):
    return gu.principal_angles(a, b).max().item() < tol


def test_endpoints():
    a, b = _pair()
    assert _same_span(gu.geodesic_basis(a, b, 0.0), a)
    assert _same_span(gu.geodesic_basis(a, b, 1.0), b)
    assert torch.allclose(gu.geodesic_basis(a, b, 0.0), a, atol=1e-9)  # V(0) = V_TI exactly


def test_orthonormal_along_path():
    a, b = _pair()
    for eta in (0.1, 0.25, 0.5, 0.9):
        v = gu.geodesic_basis(a, b, eta)
        assert torch.allclose(v.T @ v, torch.eye(K, dtype=torch.double), atol=1e-9)


def test_each_principal_angle_advances_by_eta():
    """d(S_eta, S_T) = (1-eta) D and every angle to the reference shrinks to (1-eta)*theta_i (Eq. 5, 11)."""
    a, b = _pair()
    theta = gu.principal_angles(a, b)
    big = gu.grassmann_distance(a, b)
    for eta in (0.25, 0.5, 0.75):
        v = gu.geodesic_basis(a, b, eta)
        assert math.isclose(gu.grassmann_distance(a, v).item(), eta * big.item(), rel_tol=1e-6)
        assert math.isclose(gu.grassmann_distance(v, b).item(), (1 - eta) * big.item(), rel_tol=1e-6)
        assert torch.allclose(gu.principal_angles(v, b), (1 - eta) * theta, atol=1e-6)


def test_identical_subspaces_are_fixed_points():
    a, _ = _pair()
    rot = torch.linalg.qr(torch.randn(K, K, dtype=torch.double))[0]  # same span, different basis
    v = gu.geodesic_basis(a, a @ rot, 0.6)
    assert _same_span(v, a)


def test_edit_weight_keeps_output_factor_coefficients_and_residual():
    w = torch.randn(48, D, dtype=torch.double)
    w_ref = w + 0.8 * torch.randn_like(w)
    u, s, v = gu.top_k_svd(w, K)
    v_ref = gu.top_k_svd(w_ref, K)[2]
    assert torch.allclose(gu.edit_weight(w, v_ref, K, 0.0), w, atol=1e-9)  # eta = 0: no-op
    w1 = gu.edit_weight(w, v_ref, K, 1.0)
    # the part of W outside the principal term is untouched: R = W - U S V^T
    r = w - (u * s) @ v.T
    r1 = w1 - (u * s) @ gu.geodesic_basis(v, v_ref, 1.0).T
    assert torch.allclose(r, r1, atol=1e-9)
    # at eta = 1 the edited principal input directions span the reference subspace
    assert _same_span(gu.geodesic_basis(v, v_ref, 1.0), v_ref)


def test_conv_weight_shape_and_dtype_preserved():
    w = torch.randn(32, 16, 3, 3)
    w_ref = torch.randn(32, 16, 3, 3)
    v_ref = gu.top_k_svd(w_ref, 4)[2]
    out = gu.edit_weight(w, v_ref, 4, 0.5)
    assert out.shape == w.shape and out.dtype == w.dtype


def test_layer_scores_rank_layers_by_both_signals():
    ang = torch.tensor([1.0, 5.0, 2.0])
    harm = torch.tensor([0.1, 0.9, 0.2])
    assert gu.layer_scores(ang, harm).argmax().item() == 1


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
