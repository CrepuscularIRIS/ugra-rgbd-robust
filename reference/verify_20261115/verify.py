"""CPU stats for benchmark_design §8: border vs inner holes, scenes, accelData, depths-vs-raw."""
import json, sys
import h5py, numpy as np, cv2

MAT, TESTJ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
f = h5py.File(MAT, "r")
raw = f["rawDepths"]; dep = f["depths"]
N = raw.shape[0]
cnt = np.zeros((480, 640), np.int32)
for i in range(N):
    cnt += (np.asarray(raw[i]).T != 0)
res = {"N": N}
for thr_name, m in {"any": cnt > 0, "ge50pct": cnt >= N * 0.5, "ge95pct": cnt >= N * 0.95}.items():
    ys, xs = np.where(m)
    res[f"bbox_{thr_name}"] = [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())]
    res[f"area_{thr_name}"] = float(m.mean())
res["never_valid_frac"] = float((cnt == 0).mean())
# row/col profile of valid rate (to see sharpness of the border)
rate = cnt / N
res["row_rate_first60"] = [round(float(v), 3) for v in rate.mean(1)[:60]]
res["col_rate_first60"] = [round(float(v), 3) for v in rate.mean(0)[:60]]
res["row_rate_last20"] = [round(float(v), 3) for v in rate.mean(1)[-20:]]
res["col_rate_last50"] = [round(float(v), 3) for v in rate.mean(0)[-50:]]
np.save(OUT + "_cnt.npy", cnt)

# toolbox crop (1-indexed rows 45:471, cols 41:601) -> 0-indexed inclusive
crop = np.zeros((480, 640), bool); crop[44:471, 40:601] = True
res["toolbox_crop_area"] = float(crop.mean())
inner = crop
test_ids = set(json.load(open(TESTJ))["ids"])
full, inn, out_, inn_comp = [], [], [], []
comp_sizes = {"train": [], "test": []}
diff_abs, diff_rel = [], []
for i in range(N):
    r = np.asarray(raw[i]).T
    hole = r == 0
    full.append(hole.mean()); inn.append(hole[inner].mean()); out_.append(hole[~inner].mean())
    hm = (hole & inner).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(hm, connectivity=8)
    sz = st[1:, cv2.CC_STAT_AREA]
    comp_sizes["test" if i in test_ids else "train"].extend(sz.tolist())
    if i % 10 == 0:  # depths vs raw on raw-valid inner pixels, subsampled frames
        d = np.asarray(dep[i]).T
        v = (~hole) & inner
        e = d[v] - r[v]
        diff_abs.append(np.abs(e)); diff_rel.append(np.abs(e) / r[v])
full, inn, out_ = map(np.array, (full, inn, out_))
ids = np.arange(N); te = np.array([i in test_ids for i in ids])

def q(a):
    return {k: float(v) for k, v in zip(["min", "p25", "median", "p75", "max", "mean"],
            [a.min(), *np.percentile(a, [25, 50, 75]), a.max(), a.mean()])}
for name, sel in {"all": np.ones(N, bool), "test": te, "train": ~te}.items():
    res[f"hole_full_{name}"] = q(full[sel]); res[f"hole_inner_{name}"] = q(inn[sel])
    res[f"hole_outside_{name}"] = q(out_[sel])
res["hole_inner_bins_test"] = np.histogram(inn[te], [0, .01, .02, .05, .1, .2, .3, .5, 1])[0].tolist()
res["share_of_holes_outside_crop_test_mean"] = float(
    np.mean((full[te] - inn[te] * crop.mean()) / full[te]))
for k, sz in comp_sizes.items():
    sz = np.array(sz); tot = sz.sum()
    res[f"cc_{k}"] = {
        "n_components": int(len(sz)), "per_frame": float(len(sz) / max(1, (te if k == "test" else ~te).sum())),
        "size_pct": {p: float(np.percentile(sz, p)) for p in [50, 90, 99]}, "max": int(sz.max()),
        "area_share_lt10px": float(sz[sz < 10].sum() / tot), "area_share_lt100px": float(sz[sz < 100].sum() / tot),
        "area_share_ge1000px": float(sz[sz >= 1000].sum() / tot),
        "area_share_ge10000px": float(sz[sz >= 10000].sum() / tot)}
da = np.concatenate(diff_abs); dr = np.concatenate(diff_rel)
res["depths_minus_raw_on_valid"] = {"exact_equal_frac": float((da == 0).mean()),
    "abs_median_m": float(np.median(da)), "abs_p99_m": float(np.percentile(da, 99)),
    "rel_p99": float(np.percentile(dr, 99)), "abs_max_m": float(da.max())}
# scenes / sceneTypes / accelData
def cellstr(name):
    return ["".join(chr(c) for c in np.asarray(f[ref]).ravel()) for ref in f[name][0]]
scenes = cellstr("scenes"); stypes = cellstr("sceneTypes")
from collections import Counter
sc = Counter(scenes)
res["n_scenes"] = len(sc); res["n_sceneTypes"] = len(set(stypes))
res["frames_per_scene_hist"] = dict(sorted(Counter(sc.values()).items()))
tr_sc = {scenes[i] for i in range(N) if not te[i]}; te_sc = {scenes[i] for i in range(N) if te[i]}
res["n_train_frames_by_json"] = int((~te).sum()); res["n_test_frames"] = int(te.sum())
res["scene_overlap_train_test"] = sorted(tr_sc & te_sc)
res["n_scenes_train"] = len(tr_sc); res["n_scenes_test"] = len(te_sc)
acc = np.asarray(f["accelData"])
res["accelData"] = {"shape": list(acc.shape), "first3_frames": acc[:, :3].T.tolist(),
                    "norm_xyz_median": float(np.median(np.linalg.norm(acc[:3], axis=0))),
                    "n_unique_cols": int(len(np.unique(acc.T, axis=0)))}
json.dump({"scenes": scenes, "sceneTypes": stypes}, open(OUT + "_scenes.json", "w"))
json.dump(res, open(OUT + ".json", "w"), indent=1)
print(json.dumps(res, indent=1))
