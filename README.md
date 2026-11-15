# rgbd-realdepth

Real sensor missing-depth masks for RGB-D segmentation benchmarks, in the DFormer data layout.
The standard NYU Depth V2 / SUN RGB-D packages ship hole-filled depth; this rebuilds where the sensor actually failed.

| Dataset | Official source | Real holes | Status |
|---|---|---|---|
| NYU Depth V2 | `nyu_depth_v2_labeled.mat` (2.97 GB) | `rawDepths == 0` | `scripts/build_nyu_masks.py` |
| SUN RGB-D | the DFormer package already holds the official raw `depth/` | `Depth == 0` | `scripts/build_sun_masks.py` |

```bash
python scripts/build_nyu_masks.py --mat data/nyu_depth_v2_labeled.mat --out data/NYU_realmask
python scripts/build_sun_masks.py --sun /data/datasets/rgbd/SUNRGBD --out data/SUN_realmask
```

Needs `numpy`, `h5py`, `opencv-python`. Datasets and generated masks go under `data/` and are never committed.

`reference/nyu_rawdepths_20260919/` holds the first version of this study (2026-09-19): the mat-index-to-file-id
mapping proof, the real-mask evaluation and its results. Its imports point at a deleted repository, so it is kept
for reading only.
