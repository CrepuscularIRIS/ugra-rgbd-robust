"""NYU corruption families — thin compositions of depth_primitives.

Signatures are unchanged from the pre-unification version so build_nyu_corrupt.py
needs no edits; internals now compose the single primitive set (review 2026-11-15).
Regression gate: F1/F2/F6/F8 rebuilt frames must be byte-identical to the
pre-refactor data/NYU_corrupt outputs (see check in scripts/regress_primitives.py).
"""
from __future__ import annotations

import numpy as np

from depth_primitives import (
    p_zero, p_fill_color, p_offset, p_swap, p_warp, p_noise, select_objects,
)


def f0_missing(depth_filled: np.ndarray, c: float = 0.0) -> tuple[np.ndarray, None, dict]:
    """ESWA endpoint: the whole map replaced by a constant (design F0)."""
    return np.full_like(depth_filled, c), None, {"c": c}


def f1_hole_zero(depth_filled: np.ndarray, donor_inner_mask: np.ndarray) -> tuple[np.ndarray, None, dict]:
    """Donor inner holes stamped as zeros (announced twin of F3)."""
    holes = donor_inner_mask == 0
    return p_zero(depth_filled, holes), None, {"stamped_frac": float(holes.mean())}


def f2_noise(depth_filled: np.ndarray, bank: np.ndarray, alpha: float,
             rng: np.random.Generator) -> tuple[np.ndarray, None, dict]:
    """SIRST-5K-style real-noise transfer, disparity domain."""
    return p_noise(depth_filled, np.ones(depth_filled.shape, bool), alpha, bank, rng), None, {"alpha": alpha}


def f3_hole_reinpain(rgb: np.ndarray, raw: np.ndarray, donor_inner_mask: np.ndarray,
                     bank: np.ndarray | None = None,
                     rng: np.random.Generator | None = None) -> tuple[np.ndarray, None, dict]:
    """Stamp donor holes on raw, re-fill with the RGB-guided Laplacian fill
    (official-STYLE, approximate — see nyu_fill_colorization header); NTR texture
    (p_noise alpha=1) on the stamped holes kills the smooth-fill tell."""
    holes = donor_inner_mask == 0
    z_t = p_fill_color(raw, holes, rgb.astype(np.float32) / 255.0)
    meta = {"stamped_frac": float(holes.mean())}
    if bank is not None and len(bank) and rng is not None:
        z_t = p_noise(z_t, holes, 1.0, bank, rng)
        meta["ntr"] = True
    return z_t, None, meta


def f4_erase(depth_filled: np.ndarray, label: np.ndarray, obj_ids: list[int],
             rng: np.random.Generator, k: int = 3, eps_abs: float = 0.0,
             bank: np.ndarray | None = None,
             ) -> tuple[np.ndarray, None, dict]:
    """See-through: erase the k instances whose harmonic fill differs most from the
    original; NTR re-synthesizes sensor texture on the erased regions."""
    keep, lcc, fill = select_objects(depth_filled, label, obj_ids, k, eps_abs)
    reg = np.isin(lcc, keep)
    z_t = depth_filled.copy()
    if keep:
        z_t = np.where(reg, fill, depth_filled).astype(np.float32)
        if bank is not None and len(bank):
            z_t = p_noise(z_t, reg, 1.0, bank, rng)
    return z_t, None, {"erased_frac": float(reg.mean()), "n_regions": len(keep)}


def f5_instance_offset(depth_filled: np.ndarray, label: np.ndarray, obj_ids: list[int],
                       rng: np.random.Generator, s: float = 0.9, k: int = 3,
                       eps_abs: float = 0.0,
                       ) -> tuple[np.ndarray, None, dict]:
    """Object at a plausible wrong distance: anchored to the F4 harmonic fill,
    z + s*(fill - z), s < 1 keeps the object in front of the background. The only
    family whose depth EDGES stay RGB-aligned while VALUES are wrong."""
    keep, lcc, fill = select_objects(depth_filled, label, obj_ids, k, eps_abs)
    z_t = depth_filled.copy()
    for c in keep:
        z_t = p_offset(z_t, lcc == c, s, fill)
    return z_t, None, {"s": s, "n_objects": len(keep)}


def f6_misreg(depth_filled: np.ndarray, t: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rigid pixel misregistration of the whole depth map."""
    z_t, undef = p_warp(depth_filled, t)
    return z_t, undef, {"t": t, "moved_in_frac": float(undef.mean())}


def f8_swap(depth_donor: np.ndarray) -> tuple[np.ndarray, None, dict]:
    """Full-frame swap with another real depth map (encoding handled by the caller)."""
    return p_swap(depth_donor, np.ones(depth_donor.shape, bool), depth_donor), None, {}


def _self_check() -> None:
    rng = np.random.default_rng(1)
    d = np.full((40, 60), 3.0, np.float32)
    m = np.ones((40, 60), np.uint8) * 255
    m[5:10, 5:15] = 0
    z, u, meta = f1_hole_zero(d, m)
    assert z[6, 6] == 0 and z[0, 0] == 3 and abs(meta["stamped_frac"] - 50 / 2400) < 1e-9
    # F2 identity at alpha=0 (disparity unchanged -> z unchanged up to float32 roundtrip)
    bank = rng.normal(0, 0.003, (8, 8, 8)).astype(np.float32)
    z2, _, m2 = f2_noise(d, bank, 0.0, rng)
    assert np.allclose(z2, d) and m2["alpha"] == 0.0
    # F4: protruding object erased to background level
    yy, xx = np.mgrid[0:40, 0:60].astype(np.float32)
    wall = 3.0 + 0.002 * xx + 0.001 * yy
    scene = wall.copy()
    scene[10:25, 20:40] = 2.0
    lab = np.zeros((40, 60), np.uint16)
    lab[10:25, 20:40] = 7
    z, _, meta = f4_erase(scene, lab, [7], rng, k=1, eps_abs=0.1)
    assert meta["n_regions"] == 1
    assert np.abs(z[10:25, 20:40] - wall[10:25, 20:40]).max() < 0.3
    assert np.array_equal(z[0:10, :], scene[0:10, :])
    # F5: offset toward the fill, stays short of the wall
    z, _, m5 = f5_instance_offset(scene, lab, [7], rng, s=0.9, k=1, eps_abs=0.1)
    assert 2.0 < z[17, 30] < wall[17, 30] and m5["n_objects"] == 1
    # F6: marker block lands at (+t, +t//2)
    src = np.zeros((40, 60), np.float32)
    src[20:22, 20:22] = 1.0
    z, undef, meta = f6_misreg(src, 5)
    assert undef[0, :5].all() and not undef[25, 30] and meta["t"] == 5
    assert z[22:24, 25:27].min() == 1.0 and z[20:22, 20:22].max() == 0.0
    # F8
    z, u, _ = f8_swap(np.ones((40, 60), np.float32) * 1.5)
    assert z[0, 0] == 1.5
    # F0
    z, _, m0 = f0_missing(np.ones((5, 5), np.float32) * 3, 0.0)
    assert (z == 0).all() and m0["c"] == 0.0
    # F3: wrapped colorization fills the stamped hole in-range
    rgb = np.zeros((24, 32, 3), np.float32)
    raw = np.full((24, 32), 2.0, np.float32)
    dm = np.ones((24, 32), np.uint8) * 255
    dm[8:16, 8:20] = 0
    z, _, mf = f3_hole_reinpain(rgb, raw, dm)
    assert mf["stamped_frac"] == 96 / 768
    assert (z[8:16, 8:20] > 1.5).all()


if __name__ == "__main__":
    _self_check()
    print("nyu_corrupt_families self-check OK")
