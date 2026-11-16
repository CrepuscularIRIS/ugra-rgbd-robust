"""Core utilities for the NYU corrupted-depth construction.

Design source: docs/benchmark_design_opus55_20261115.md (S3.0, 4.1-4.3, Step 0b).
Key verified facts baked in (2026-11-15, reference/verify_20261115/):
  - label values are 1-based -> names[v-1]; wall/floor/ceiling ids from name search
  - pseudo-clean z_pc = rawDepths on valid pixels (NOT `depths`, which differs at p99 by 5.3 cm)
  - InnerMask (data/NYU_realmask/InnerMask) = holes minus registration border
  - DFormer depth PNG is a per-frame monotone nonlinear stretch -> frozen per-frame encoder

All spatial arrays are (H=480, W=640); the mat stores (W, H) and RGB (3, W, H).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import cv2

# Kinect v1 intrinsics for the official 640x480 crop (NYU toolbox camera_params).
# ponytail: constant intrinsics; verify_20261115 flagged them as toolbox-memory. The
# plane-fit inlier self-check below fails loudly if they are badly wrong.
FX, FY, CX, CY = 518.8579, 519.4611, 325.5808, 253.7691

WALL_FLOOR_CEIL = {"wall", "floor", "ceiling"}
SPECULAR_SUBSTR = ("mirror", "window", "glass")

E_CLEAN, E_ERR, E_UNDEF = 0, 1, 255

# toolbox crop of the valid region (verified 2026-11-15); ~4% of out-of-crop pixels
# carry raw readings but sit where RGB/depth/labels are not trustworthy
NYU_CROP = (slice(45, 471), slice(41, 601))


def _decode_str(ds: h5py.Dataset) -> str:
    return "".join(chr(int(c[0])) for c in ds[()])


def load_names(f: h5py.File) -> list[str]:
    """894 raw class names, index i is the name of label value i+1."""
    return [_decode_str(f[r]) for r in f["names"][0]]


def label_ids_by_name(names: list[str], wanted: set[str]) -> list[int]:
    """1-based label ids whose name is exactly in `wanted`."""
    return [i + 1 for i, n in enumerate(names) if n in wanted]


def label_ids_by_substring(names: list[str], subs: tuple[str, ...]) -> list[int]:
    return [i + 1 for i, n in enumerate(names) if any(s in n for s in subs)]


def read_frame(f: h5py.File, i: int) -> dict[str, np.ndarray]:
    """One frame, spatial axes transposed to (480, 640)."""
    return {
        "rgb": np.transpose(np.asarray(f["images"][i]), (2, 1, 0)).copy(),  # (C,W,H)->(H,W,C)
        "depth": np.transpose(np.asarray(f["depths"][i]), (1, 0)).astype(np.float32),  # official filled
        "raw": np.transpose(np.asarray(f["rawDepths"][i]), (1, 0)).astype(np.float32),  # sensor, 0 = hole
        "label": np.transpose(np.asarray(f["labels"][i]), (1, 0)),
        "scene": _decode_str(f[f["scenes"][0][i]]),
    }


def load_split(dformer_root: Path) -> tuple[list[int], list[int]]:
    """DFormer train/test ids; lines look like 'RGB/2.jpg', mat index == file id (proven 2026-09-19)."""
    def _ids(p: Path) -> list[int]:
        return [int(line.strip().rsplit("/", 1)[-1].split(".")[0])
                for line in p.read_text().splitlines() if line.strip()]
    return _ids(dformer_root / "train.txt"), _ids(dformer_root / "test.txt")


# ---------------------------------------------------------------- planes / epsilon

def _backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
    x = (u - CX) * z / FX
    y = (v - CY) * z / FY
    return np.stack([x, y, z], axis=1)


def _plane_z_at(n: np.ndarray, d: float, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Depth along the ray of pixel (u, v) where plane n.P = d (n unit, n_z > 0)."""
    denom = n[2] + n[0] * (u - CX) / FX + n[1] * (v - CY) / FY
    return d / denom


def fit_plane_ransac(
    z: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
    n_iter: int = 200, thr: float = 0.02, min_inlier: float = 0.5,
) -> tuple[np.ndarray, float, float] | None:
    """RANSAC plane on masked pixels. Returns (n, d, inlier_ratio) or None if degenerate."""
    v_idx, u_idx = np.nonzero(mask)
    if v_idx.size < 200:
        return None
    pts = _backproject(u_idx.astype(np.float64), v_idx.astype(np.float64), z[v_idx, u_idx].astype(np.float64))
    if v_idx.size > 4000:  # subsample, keep plane fit cheap
        pts = pts[rng.choice(v_idx.size, 4000, replace=False)]
    best = None
    for _ in range(n_iter):
        try:
            p = pts[rng.choice(pts.shape[0], 3, replace=False)]
        except ValueError:
            return None
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9 or n[2] <= 0:
            continue
        n /= norm
        d = float(n @ p[0])
        resid = np.abs(pts @ n - d)
        inl = resid < thr
        ratio = inl.mean()
        if best is None or ratio > best[2]:
            best = (n, d, ratio)
    if best is None or best[2] < min_inlier:
        return None
    n, d, _ = best  # refit least squares on inliers
    resid = np.abs(pts @ n - d)
    inl = resid < thr
    if inl.sum() >= 3:
        c = pts[inl].mean(axis=0)
        _, _, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
        n2 = vt[-1]
        if n2[2] > 0:
            n, d = n2, float(n2 @ c)
    return n, d, 1.0


def calibrate_eps(
    f: h5py.File, frame_ids: list[int], names: list[str], step: int = 3, seed: int = 0,
) -> dict:
    """eps(z) from q99 of |z_raw - z_plane| on wall/floor/ceiling pixels (design S3.0).

    Writes {'bin_centers': [...], 'q99': [...], 'eps_abs': float, 'n_frames': int, 'n_pts': int}.
    """
    rng = np.random.default_rng(seed)
    wfc_ids = label_ids_by_name(names, WALL_FLOOR_CEIL)
    specular = label_ids_by_substring(names, SPECULAR_SUBSTR)
    residuals, depths = [], []
    used = 0
    for i in frame_ids[::step]:
        fr = read_frame(f, i)
        m = np.isin(fr["label"], wfc_ids) & (fr["raw"] > 0) & ~np.isin(fr["label"], specular)
        fit = fit_plane_ransac(fr["raw"], m, rng)
        if fit is None:
            continue
        n, d, _ = fit
        v_idx, u_idx = np.nonzero(m)
        u64, v64 = u_idx.astype(np.float64), v_idx.astype(np.float64)
        denom = n[2] + n[0] * (u64 - CX) / FX + n[1] * (v64 - CY) / FY
        near = denom > 0.2  # near-grazing rays: z_plane explodes, exclude
        z_raw = fr["raw"][v_idx, u_idx].astype(np.float64)
        z_plane = _plane_z_at(n, d, u64, v64)
        r = np.abs(z_raw - z_plane)
        keep = near & (r < 0.1)  # calibration measures sensor noise, not cross-plane distance
        residuals.append(r[keep])
        depths.append(z_plane[keep])
        used += 1
    r = np.concatenate(residuals)
    z = np.concatenate(depths)
    edges = np.arange(0.5, 10.01, 0.5)
    centers, q99 = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (z >= lo) & (z < hi)
        if sel.sum() > 500:
            centers.append(float((lo + hi) / 2))
            q99.append(float(np.quantile(r[sel], 0.99)))
    return {
        "bin_centers": centers,
        "q99": q99,
        "eps_abs": float(np.quantile(r, 0.99)),
        "n_frames": used,
        "n_pts": int(r.size),
    }


def eps_of(z: np.ndarray, calib: dict) -> np.ndarray:
    """Tolerance map: max(eps_abs, binned q99) (design Eq. 1)."""
    binned = np.interp(z, calib["bin_centers"], calib["q99"])
    return np.maximum(binned, calib["eps_abs"])


# ---------------------------------------------------------------- V set / oracle

def v_mask(fr: dict[str, np.ndarray], names: list[str], calib: dict, band_dilate: int = 3) -> np.ndarray:
    """Verifiable pixels V (design S3.0): raw valid, inside the toolbox crop, labelled,
    non-specular, off the z_pc discontinuity band. (Sol review P2-4: crop + label!=0
    exclusions were missing; out-of-crop and unlabelled pixels must be U.)"""
    specular = label_ids_by_substring(names, SPECULAR_SUBSTR)
    v = (fr["raw"] > 0) & ~np.isin(fr["label"], specular) & (fr["label"] != 0)
    crop = np.zeros(fr["raw"].shape, bool)
    crop[NYU_CROP] = True
    v &= crop
    # discontinuity band on z_pc: 3x3 max-min jump > eps, dilated
    z = np.where(v, fr["raw"], 0).astype(np.float32)
    thr = eps_of(np.maximum(z, 0.5), calib)
    jump = cv2.dilate(z, np.ones((3, 3), np.uint8)) - cv2.erode(z, np.ones((3, 3), np.uint8))
    band = cv2.dilate((jump > thr).astype(np.uint8), np.ones((band_dilate, band_dilate), np.uint8)) > 0
    v &= ~band
    return v & (fr["raw"] > 0)


def oracle(z_tilde: np.ndarray, fr: dict[str, np.ndarray], v: np.ndarray, calib: dict) -> np.ndarray:
    """E in {0 clean, 1 err, 255 undefined} (design Eq. 1, construction-known truth)."""
    z_pc = fr["raw"]
    err = np.abs(z_tilde - z_pc) > eps_of(z_pc, calib)
    e = np.full(z_pc.shape, E_UNDEF, np.uint8)
    e[v & ~err] = E_CLEAN
    e[v & err] = E_ERR
    return e


# ---------------------------------------------------------------- frozen encoder

class FrozenEncoder:
    """Per-frame monotone map from filled metric depth to the DFormer 8-bit PNG scale.

    Fit on the clean pair (depths, DFormer PNG), then FROZEN and applied to corrupted
    depth, so corruption cannot trigger a whole-image rescale (design 4.2).
    """

    def __init__(self, n_bins: int = 64):
        self.n_bins = n_bins
        self.knots: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def fit(self, fid: int, depth: np.ndarray, png8: np.ndarray) -> float:
        """Fit quantile knots; returns median abs reproduction error in PNG levels."""
        m = depth > 0
        z, lv = depth[m].astype(np.float64), png8[m].astype(np.float64)
        qs = np.quantile(z, np.linspace(0, 1, self.n_bins + 1))
        zk, lk = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            sel = (z >= lo) & (z <= hi if hi == qs[-1] else z < hi)
            if sel.sum() > 50:
                zk.append(float(np.median(z[sel])))
                lk.append(float(np.median(lv[sel])))
        if len(zk) < 2:
            raise ValueError(f"frame {fid}: degenerate encoder fit")
        zk_a = np.maximum.accumulate(np.array(zk))
        lk_a = np.maximum.accumulate(np.array(lk))
        uk, idx = np.unique(zk_a, return_index=True)
        self.knots[fid] = (uk, lk_a[idx])
        recon = self.apply(fid, depth)[m]
        return float(np.median(np.abs(recon - lv)))

    def apply(self, fid: int, z: np.ndarray) -> np.ndarray:
        zk, lk = self.knots[fid]
        out = np.clip(np.interp(z, zk, lk), 0, 255)
        out[z == 0] = 0  # metric zero must stay zero (Sol P1-3): holes read as holes
        return out


def write_depth_png(path: Path, png8: np.ndarray) -> None:
    """Replicate DFormer's 4-channel depth PNG (R=G=B=level, A=255)."""
    v = np.clip(np.round(png8), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.merge([v, v, v, v * 0 + 255]))


def write_e_png(path: Path, e: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), e)


def load_png8(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img.astype(np.float32)


def _self_check() -> None:
    rng = np.random.default_rng(0)
    # plane fit on a synthetic tilted floor
    h, w = 60, 80
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    z = 2.0 + 0.001 * xx + 0.002 * yy + rng.normal(0, 0.003, (h, w)).astype(np.float32)
    fit = fit_plane_ransac(z, np.ones((h, w), bool), rng)
    assert fit is not None and fit[2] > 0.9, fit
    n, d, _ = fit
    zp = _plane_z_at(n, d, xx.astype(float), yy.astype(float))
    assert np.abs(zp - z).max() < 0.03
    # eps_of
    calib = {"bin_centers": [2.0, 4.0], "q99": [0.02, 0.06], "eps_abs": 0.03}
    e = eps_of(np.array([[1.0, 3.0], [9.0, 9.9]]), calib)
    assert np.allclose(e, [[0.03, 0.04], [0.06, 0.06]]), e
    # encoder monotone + reproduction
    enc = FrozenEncoder()
    z = np.sort(rng.uniform(0.5, 8.0, 5000))
    lv = np.clip(10 + 30 * z, 0, 255)
    err = enc.fit(0, z, lv)
    assert err < 1.0 and np.all(np.diff(enc.knots[0][1]) >= -1e-9)
    # oracle polarity
    fr = {"raw": np.array([[2.0, 2.0], [2.0, 2.0]], np.float32)}
    v = np.ones((2, 2), bool)
    o = oracle(np.array([[2.0, 2.0], [2.0, 5.0]], np.float32), fr, v, calib)
    assert o.tolist() == [[0, 0], [0, 1]], o


if __name__ == "__main__":
    _self_check()
    print("nyu_corrupt_core self-check OK")
