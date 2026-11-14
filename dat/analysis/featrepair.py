"""Feature-space subspace repair for missing depth (frozen model, no training).

For each encoder stage, with Z_c = features under real depth and Z_m = features under missing depth,
the displacement is dZ = Z_m - Z_c. We ask how much of dZ lives in a low-dimensional subspace and how much
of the clean-depth behaviour is recovered by removing the estimated displacement inside it:

    oracle : Z' = Z_m - U U^T (Z_m - Z_c)        upper bound; needs the real depth, so not deployable
    blind  : Z' = Z_m - U (M [Z_m; 1])           coefficients regressed from Z_m only (calibrated, ridge)
    random : same as oracle with a random basis  control for "any k-dim subspace works"

U (C x k) holds the top-k eigenvectors of the *centred* covariance of dZ (fit on calibration images);
the projection acts on the actual uncentred displacement.
"""
from __future__ import annotations

import torch


class StageStats:
    """Streaming second-moment statistics of (Z_c, Z_m) token pairs for one stage."""

    def __init__(self, channels: int, device="cuda"):
        c = channels
        self.c, self.n = c, 0
        self.sum_d = torch.zeros(c, dtype=torch.float64, device=device)
        self.s_dd = torch.zeros(c, c, dtype=torch.float64, device=device)
        self.s_dm = torch.zeros(c, c + 1, dtype=torch.float64, device=device)
        self.s_mm = torch.zeros(c + 1, c + 1, dtype=torch.float64, device=device)
        self.sq_c = 0.0
        self.sq_d = 0.0

    @torch.no_grad()
    def update(self, zc: torch.Tensor, zm: torch.Tensor) -> None:
        tok = lambda z: z.permute(0, 2, 3, 1).reshape(-1, z.shape[1]).float()  # noqa: E731  (N, C)
        zc, zm = tok(zc), tok(zm)
        d = zm - zc
        zt = torch.cat([zm, torch.ones(zm.shape[0], 1, device=zm.device)], dim=1)
        self.n += zm.shape[0]
        self.sum_d += d.sum(0).double()
        self.s_dd += (d.T @ d).double()
        self.s_dm += (d.T @ zt).double()
        self.s_mm += (zt.T @ zt).double()
        self.sq_c += float((zc * zc).sum())
        self.sq_d += float((d * d).sum())

    def rel_shift(self) -> float:
        return (self.sq_d / max(self.sq_c, 1e-12)) ** 0.5

    def _centred_cov(self) -> torch.Tensor:
        mean = self.sum_d / self.n
        return self.s_dd / self.n - torch.outer(mean, mean)

    def spectrum(self) -> torch.Tensor:
        """Eigenvalues of the centred dZ covariance, descending."""
        return torch.linalg.eigvalsh(self._centred_cov()).flip(0).clamp_min(0)

    def basis(self, k: int) -> torch.Tensor:
        _, vecs = torch.linalg.eigh(self._centred_cov())
        return vecs[:, -k:].flip(1).float()  # (C, k), top-k first

    def blind_map(self, u: torch.Tensor, ridge: float = 1e-3) -> torch.Tensor:
        """Ridge regression of the coefficients U^T dZ on [Z_m; 1]: returns M (k, C+1)."""
        smm = self.s_mm + ridge * torch.diag(self.s_mm).mean() * torch.eye(self.c + 1, device=self.s_mm.device,
                                                                          dtype=self.s_mm.dtype)
        return (u.double().T @ self.s_dm @ torch.linalg.inv(smm)).float()


def effective_rank(eigs: torch.Tensor) -> float:
    """Participation ratio (sum lam)^2 / sum lam^2: how many directions share the displacement energy."""
    e = eigs.double()
    return float(e.sum() ** 2 / (e * e).sum().clamp_min(1e-30))


def explained(eigs: torch.Tensor, k: int) -> float:
    return float(eigs[:k].sum() / eigs.sum().clamp_min(1e-30))


def repair_oracle(zm, zc, u):
    a = torch.einsum("ck,bchw->bkhw", u, zm - zc)
    return zm - torch.einsum("ck,bkhw->bchw", u, a)


def repair_blind(zm, u, m):
    c = zm.shape[1]
    a = torch.einsum("kc,bchw->bkhw", m[:, :c], zm) + m[:, c].view(1, -1, 1, 1)
    return zm - torch.einsum("ck,bkhw->bchw", u, a)


def random_basis(c: int, k: int, seed: int, device="cuda") -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.linalg.qr(torch.randn(c, k, generator=g))[0].to(device)
