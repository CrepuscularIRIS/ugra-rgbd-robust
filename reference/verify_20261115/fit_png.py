"""Fit DFormer 8-bit depth PNG against mat depths: affine in z, affine in 1/z, and explicit min-max hypotheses."""
import json, sys
import h5py, numpy as np, cv2

MAT, DDIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
f = h5py.File(MAT, "r"); dep = f["depths"]; raw = f["rawDepths"]
crop = np.zeros((480, 640), bool); crop[44:471, 40:601] = True
rng = np.random.default_rng(0); ids = sorted(rng.choice(1449, 60, replace=False).tolist())

def lsq(x, y):
    A = np.stack([x, np.ones_like(x)], 1); c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c; return c, float(np.sqrt((r ** 2).mean())), float(np.abs(r).max())

rows, chan_same = [], 0
for i in ids:
    png = cv2.imread(f"{DDIR}/{i}.png", cv2.IMREAD_UNCHANGED)
    chan_same += int(all((png[..., k] == png[..., 0]).all() for k in range(1, 3)))
    alpha = png[..., 3] if png.shape[2] == 4 else None
    p = png[..., 0].astype(np.float64)
    z = np.asarray(dep[i]).T.astype(np.float64); zr = np.asarray(raw[i]).T.astype(np.float64)
    row = {"id": i, "alpha_unique": np.unique(alpha).tolist()[:5] if alpha is not None else None}
    for nm, m in {"full": np.ones_like(crop), "crop": crop}.items():
        row[f"lin_{nm}"] = lsq(z[m], p[m])[1:]; row[f"inv_{nm}"] = lsq(1 / z[m], p[m])[1:]
        row[f"coef_lin_{nm}"] = lsq(z[m], p[m])[0].tolist()
    for nm, m in {"full": np.ones_like(crop), "crop": crop}.items():  # explicit per-image min-max of filled depths
        lo, hi = z[m].min(), z[m].max()
        pred = np.clip(np.round(255 * (z - lo) / (hi - lo)), 0, 255)
        row[f"minmax_{nm}_mad"] = float(np.abs(pred - p).mean())
    fixed = np.clip(np.round(255 * z / 10.0), 0, 255)  # fixed metric z/10m hypothesis
    row["fixed10m_mad"] = float(np.abs(fixed - p).mean())
    v = zr > 0
    row["lin_raw_valid"] = lsq(zr[v], p[v])[1:]
    row["lin_filled_on_same_pixels"] = lsq(z[v], p[v])[1:]
    rows.append(row)

def med(k, j=0): return float(np.median([r[k][j] if isinstance(r[k], (list, tuple)) else r[k] for r in rows]))
summ = {"n": len(rows), "rgb_channels_identical": chan_same, "alpha_unique_example": rows[0]["alpha_unique"]}
for k in ["lin_full", "lin_crop", "inv_full", "inv_crop", "lin_raw_valid", "lin_filled_on_same_pixels"]:
    summ[k + "_rmse_median"] = med(k, 0); summ[k + "_maxabs_median"] = med(k, 1)
    summ[k + "_rmse_max"] = float(max(r[k][0] for r in rows))
for k in ["minmax_full_mad", "minmax_crop_mad", "fixed10m_mad"]:
    summ[k + "_median"] = med(k); summ[k + "_max"] = float(max(r[k] for r in rows))
slopes = np.array([r["coef_lin_full"][0] for r in rows]); icpt = np.array([r["coef_lin_full"][1] for r in rows])
summ["slope_cv"] = float(slopes.std() / slopes.mean()); summ["slope_range"] = [float(slopes.min()), float(slopes.max())]
summ["icpt_range"] = [float(icpt.min()), float(icpt.max())]
json.dump({"summary": summ, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summ, indent=1))
