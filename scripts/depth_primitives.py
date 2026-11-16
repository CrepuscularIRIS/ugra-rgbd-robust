"""Depth corruption primitives — small composable array ops, dataset-agnostic.

Families are compositions of these; primitives carry no dataset or label logic
(regions and labels are passed in). Unified 2026-11-15 per review: one op set,
audited once; every family is a readable composition. nyu_corrupt_families.py
now only composes these.

    p_zero(z, reg)             zeros on reg                       (announced erasure)
    p_fill_ns(z, reg)          harmonic NS inpaint on reg          (silent erasure)
    p_fill_color(z, reg, rgb)  official colorization fill          (replay native fill)
    p_offset(z, reg, s, fill)  z + s*(fill-z) on reg               (wrong distance)
    p_swap(z, reg, donor)      donor values on reg                 (real-but-elsewhere)
    p_warp(z, t)               rigid shift + moved-in mask         (misregistration)
    p_noise(z, reg, a, bank, rng)  disparity alpha-mix on reg&z>0  (noise transfer; a=1 == NTR)

    select_objects(z, lab, exclude_ids, k, eps)   canonical F4/F5 candidate selector
"""
from __future__ import annotations

import numpy as np
import cv2


def p_zero(z: np.ndarray, reg: np.ndarray) -> np.ndarray:
    out = z.copy()
    out[reg] = 0.0
    return out


def p_fill_ns(z: np.ndarray, reg: np.ndarray, radius: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Harmonic (Navier-Stokes) fill; returns (filled map, fill values everywhere)."""
    fill = cv2.inpaint(z, reg.astype(np.uint8), radius, cv2.INPAINT_NS)
    out = z.copy()
    out[reg] = fill[reg]
    return out, fill


def p_fill_color(z: np.ndarray, reg: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Official NYU colorization fill: zero reg, then Levin-colorize every missing pixel.

    rgb: (H, W, 3) float in [0, 1]. Algorithm is dataset-agnostic; whether replaying
    the native fill pipeline is a meaningful story is a per-dataset question (it is
    on NYU, it is not on SUN).
    """
    from nyu_fill_colorization import fill_depth_colorization
    z_in = z.copy()
    z_in[reg] = 0.0
    return fill_depth_colorization(rgb, z_in)


def p_offset(z: np.ndarray, reg: np.ndarray, s: float, fill: np.ndarray) -> np.ndarray:
    """Move region toward the fill by fraction s (s<1 stays in front of background)."""
    out = z.copy()
    out[reg] = z[reg] + s * (fill[reg] - z[reg])
    return out


def p_swap(z: np.ndarray, reg: np.ndarray, donor: np.ndarray) -> np.ndarray:
    out = z.copy()
    out[reg] = donor[reg]
    return out


def p_warp(z: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Rigid pixel shift (t in x, t//2 in y); returns (warped, moved-in-undef mask)."""
    h, w = z.shape
    M = np.float32([[1, 0, t], [0, 1, t // 2]])
    out = cv2.warpAffine(z, M, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.float32)
    inside = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h),
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out, inside == 0


def p_noise(z: np.ndarray, reg: np.ndarray, alpha: float, bank: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """Real-noise transfer in the disparity domain (stationary there): d~ = d + a*N.

    Guarded: only pixels in reg AND z > 0 are touched — holes stay exactly 0
    (unguarded, a hole would be clip'd to 0.3 m and fabricated into a reading).
    alpha=1 with a residual bank is exactly NTR texture re-synthesis.
    """
    if not reg.any() or bank is None or not len(bank):
        return z
    field = _tiled(bank, rng, *z.shape)
    d = 1.0 / np.clip(z, 0.3, 12.0)
    active = reg & (z > 0)
    d2 = np.where(active, d + alpha * field, d)
    out = np.where(d2 > 1e-4, 1.0 / np.clip(d2, 1e-4, None), z).astype(np.float32)
    out[~active] = z[~active]
    return out


def _tiled(bank: np.ndarray, rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    ph, pw = bank[0].shape
    return np.tile(bank[int(rng.integers(len(bank)))], (h // ph + 1, w // pw + 1))[:h, :w]


def select_objects(z: np.ndarray, lab: np.ndarray, cand_ids: list,
                   k: int, eps: float, n_cand: int = 20, min_area: int = 200,
                   ) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Canonical F4/F5 candidate selector (single source of truth for both).

    cand_ids: label values eligible for corruption (callers pass the non-structural
    classes). Returns (keep cc ids, CC label map, NS fill of all candidates); keeps
    the k CCs whose fill differs most from the original (median |fill - z| per CC
    must exceed eps)."""
    cls = np.isin(lab, cand_ids)
    if not cls.any():
        return [], np.zeros_like(lab), z.copy()
    _, lcc, st, _ = cv2.connectedComponentsWithStats(cls.astype(np.uint8), connectivity=8)
    cand = [c for c in np.argsort(-st[1:, cv2.CC_STAT_AREA])[:n_cand] + 1
            if st[c, cv2.CC_STAT_AREA] >= min_area]
    if not cand:
        return [], lcc, z.copy()
    _, fill = p_fill_ns(z, np.isin(lcc, cand))
    err = {c: float(np.median(np.abs(fill - z)[lcc == c])) for c in cand}
    keep = [int(c) for c in sorted(cand, key=lambda c: -err[c])[:k] if err[c] > eps]
    return keep, lcc, fill


def _self_check() -> None:
    rng = np.random.default_rng(0)
    z = np.full((40, 60), 3.0, np.float32)
    reg = np.zeros((40, 60), bool)
    reg[10:25, 20:40] = True
    z[reg] = 2.0
    assert p_zero(z, reg)[15, 30] == 0 and p_zero(z, reg)[0, 0] == 3
    zf, fill = p_fill_ns(z, reg)
    assert np.abs(zf[15, 30] - 3.0) < 0.2 and fill[0, 0] == 3.0
    zo = p_offset(z, reg, 0.9, fill)
    assert 2.0 < zo[15, 30] < 3.0
    donor = np.full_like(z, 1.5)
    assert p_swap(z, reg, donor)[15, 30] == 1.5 and p_swap(z, reg, donor)[0, 0] == 3
    m = np.zeros((40, 60), np.float32)
    m[20:22, 20:22] = 1.0
    zw, undef = p_warp(m, 4)
    assert zw[22:24, 24:26].min() == 1.0 and undef[:, :4].any()
    bank = rng.normal(0, 0.003, (8, 8, 8)).astype(np.float32)
    # noise leaves holes untouched and only moves reg
    zh = z.copy()
    zh[35:38, 0:5] = 0.0  # a hole
    zn = p_noise(zh, np.ones_like(zh, bool), 4.0, bank, rng)
    assert (zh[35:38, 0:5] == 0).all()
    valid = zh > 0
    assert np.array_equal(zn[~valid], zh[~valid]) and np.abs(zn[valid] - zh[valid]).max() < 0.7  # ~5-sigma tail at alpha=4
    zr = p_noise(z, reg, 4.0, bank, rng)  # region-limited call
    assert np.array_equal(zr[~reg], z[~reg]) and np.abs(zr[reg] - 2.0).max() < 0.7
    # select_objects: protruding block kept, flat patch skipped
    lab = np.zeros((40, 60), np.uint8)
    lab[reg] = 5
    lab[30:33, 45:50] = 7  # flat against background -> below eps
    keep, lcc, fill = select_objects(z, lab, [5, 7], k=2, eps=0.1)
    assert len(keep) == 1 and (lcc == keep[0])[10:25, 20:40].all()


if __name__ == "__main__":
    _self_check()
    print("depth_primitives self-check OK")
