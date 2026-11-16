# RGB-D depth robustness: real sensor holes, a corrupted-depth benchmark, A2S training

Code for three things built on NYU Depth V2 and SUN RGB-D:

1. **Dataset**: where the depth sensor actually failed. The standard benchmark packages ship hole-filled depth;
   these scripts rebuild per-image real sensor-hole masks from the official raw depth.
2. **Benchmark**: corrupted-depth variants of both datasets, from *announced* corruption (depth visibly missing)
   to *silent* corruption (depth present but wrong), each with an exact per-pixel error oracle.
3. **Training protocol (A2S)**: an announced-to-silent curriculum that extends depth-availability training
   (complete / missing / partial depth) with gated corrupted-depth levels.

Nothing here changes a network. Every variant keeps RGB, labels and splits and only replaces `Depth/`, in the
DFormer data layout, so any DFormer-style loader can train or evaluate on it unchanged.

## 1. Dataset: real sensor-hole masks

| | NYU Depth V2 | SUN RGB-D |
|---|---|---|
| Source of real holes | `rawDepths == 0` in the official `nyu_depth_v2_labeled.mat` (2.97 GB) | the DFormer SUN package already holds the official **raw** `depth/` (holes = 0), verified against the official `SUNRGBD.zip` |
| Script | `scripts/build_nyu_masks.py` | `scripts/build_sun_masks.py` |
| Output | `Mask/<id>.png`, `InnerMask/<id>.png`, `holes.json`, `inner_holes.json` | `Mask/<name>.png`, `holes.json` (with sensor) |

Mask polarity: 255 = depth valid, 0 = sensor hole. NYU frame index `i` in the mat equals DFormer file id `i`
(checked by depth correlation, `reference/nyu_rawdepths_20260919/verify_nyu_mapping.py`).

What the data shows (measured with these scripts):

- **NYU has a registration border.** About 19% of every frame has no raw reading in all frames: it is the
  depth-to-RGB registration border, not sensor failure. `InnerMask` keeps only holes inside the valid region
  (rows 45–471, cols 41–601). Whole-frame hole median 0.29; inside the valid region 0.11.
- **SUN has no such border**; hole rates depend on the sensor:

  | sensor | frames | hole median | mean |
  |---|---|---|---|
  | Kinect v1 | 2003 | 0.061 | 0.086 |
  | Kinect v2 | 3784 | 0.213 | 0.238 |
  | RealSense | 1159 | 0.308 | 0.313 |
  | Xtion | 3389 | 0.018 | 0.049 |

- **Holes sit on the surfaces the literature names.** Hole rate by class, relative to the mean: window ×4.6
  (NYU) / ×3.4 (SUN), mirror ×3.5 / ×2.6, TV ×3.8 / ×2.4. NYU pixels farther than 4 m are holes twice as often
  as nearer ones.
- **Real holes are small and fragmented.** Median 9 (NYU) / 5 (SUN) separate holes per frame; the largest
  single hole covers a median 4–5% of the frame. No frame in either dataset has ≥ 90% holes.
- **The two "complete" inputs differ.** NYU depth is officially inpainted (the network never sees a hole);
  SUN depth in the DFormer package is raw (≈ 15% zero pixels on average).

## 2. Benchmark: corrupted-depth families

`scripts/build_nyu_corrupt.py` (all 1449 frames) and `scripts/build_sun_corrupt.py` (per sensor, `--sensor`)
build the variants from the official data plus the DFormer package:

| family | mechanism | kind | severities |
|---|---|---|---|
| F0 | constant depth map (the missing-depth endpoint) | announced | zero |
| F1 | real inner holes transplanted from another frame of the same split, set to zero | announced | a05–a40 |
| F2 | real sensor noise mixed in the disparity domain | silent | a1–a16 |
| F3 | the F1 holes re-filled by RGB-guided colorization (silent twin of F1) | silent | a05–a40 |
| F4 | see-through: object instances erased and harmonically in-filled | silent | k1/k3/k5 |
| F5 | object moved to a plausible wrong distance | silent | s50/s90 |
| F6 | rigid depth-to-RGB misregistration | silent | t04–t32 |
| F8 | full-frame swap with another real depth map (cross- or same-scene donor) | silent | cross/same |

How it stays measurable:

- **Pseudo-clean depth** `z_pc` = raw depth on verifiable pixels V; tolerance `eps(z)` is self-calibrated from
  wall/floor/ceiling plane residuals (≈ 8 mm near, ≈ 10 cm at 6 m).
- **Exact oracle** `E/<id>.png`: 0 = depth honest, 1 = depth wrong, 255 = not scoreable, known by construction.
  The oracle floor is 0.22% (official inpainted depth vs raw on V); variants below ~0.5% count as uncorrupted.
- **Frozen encoder.** DFormer's 8-bit NYU depth PNG is a per-frame monotone stretch; corrupted depth is encoded
  with the clean frame's frozen map, so a local corruption cannot rescale the whole image. SUN uses the official
  16-bit shifted encoding, re-encoded exactly.
- **Reproducible.** Per-task RNG (family / severity / frame); `scripts/regress_primitives.py` checks Depth and E
  byte for byte. Donors always come from the target's own split.

Families are compositions of seven primitives in `scripts/depth_primitives.py` (zero, harmonic fill,
colorization fill, offset, swap, warp, noise). `scripts/make_variants.py` packages each family/severity as a
standalone root `variants/<FAM>_<SEV>/` (RGB/Label symlinked, corrupted Depth, E, split files) plus `manifest.json`.

### Scoring

```bash
python scripts/eval_variants.py --dataset NYUDepthv2 --host DFormerv2-S --device cuda:0   # mIoU per variant
python scripts/eval_strata.py --dataset NYUDepthv2 --host DFormerv2-S --variant F8/cross  # per-stratum
```

`eval_variants.py` scores a host on every variant (single scale; corruption lives in the files, not in a tensor
overwrite). `eval_strata.py` splits pixels by the oracle into E = 0 (depth honest) and E = 1 (depth lying) and
compares the same model under corrupted and clean depth, giving the per-stratum shortfall
`F_X = mIoU_X(clean) − mIoU_X(corrupted)`.

## 3. Training protocol: A2S (announced-to-silent)

`scripts/train_a2s.py` starts from depth-availability training and adds corrupted depth in gated levels:

| phase | depth shown in training (complete : missing : partial : corrupted) |
|---|---|
| start (no level enabled) | 50 : 25 : 25 : 0 — plain depth-availability training |
| a level enabled | 40 : 20 : 20 : 20 — the corrupted slot draws uniformly over enabled levels, then over families in the level |

- Levels: 1 = `F8/cross` (full swap), 2 = `F5/s50`, `F5/s90` (wrong distance).
- Gates at T/3 and 2T/3 probe 200 fixed training images (single scale). Level j is enabled when the lower 95%
  bound of `H_j = S_m − S_j` is positive (scene-clustered paired bootstrap), i.e. when the model currently does
  *worse* with that corruption than with no depth at all. Skipped levels are rechecked at the next gate.
- Clean and corrupted copies of a sample get identical augmentation (random-state replay, checked per sample).

```bash
python scripts/train_a2s.py --steps 120 --batch-size 8 --out runs/a2s_smoke              # smoke run
python scripts/train_a2s.py --preset-only DAT --steps 60 --batch-size 8 --out runs/dat    # plain baseline
```

Status: the mechanism (gates, pairing, sampling) passes a 120-step smoke run; no full training run and no
trained weights yet.

## Setup

```bash
# host code: DFormer adapters and the depth-availability harness (branch `dat` of this repository)
git clone -b dat https://github.com/CrepuscularIRIS/ugra-rgbd-robust ~/workspace/ugra-rgbd-robust
bash ~/workspace/ugra-rgbd-robust/scripts/setup_dformer.sh
pip install numpy h5py opencv-python scipy

python scripts/build_nyu_masks.py --mat data/nyu_depth_v2_labeled.mat --out data/NYU_realmask
python scripts/build_sun_masks.py --sun /data/datasets/rgbd/SUNRGBD --out data/SUN_realmask
python scripts/build_nyu_corrupt.py --mat data/nyu_depth_v2_labeled.mat --dformer /data/datasets/rgbd/NYUDepthv2 \
    --realmask data/NYU_realmask --out data/NYU_corrupt
python scripts/make_variants.py --corrupt data/NYU_corrupt --base /data/datasets/rgbd/NYUDepthv2
```

Inputs: the official `nyu_depth_v2_labeled.mat`, and the DFormer NYU Depth V2 / SUN RGB-D packages
(Hugging Face `bbynku/DFormerv2`, `datasets/*.zip`). `$DFORMER_ROOT` points at upstream DFormer;
`$DFORMERV1_WEIGHTS` at the released DFormerv1 checkpoints. Several defaults are paths on the authors' machine
and can be overridden by flags.

## Repository layout

```
scripts/     masks, corruption primitives and families, builders, packaging, regression guard, evaluation, A2S
reference/   nyu_rawdepths_20260919/  first real-hole study (mapping proof, results; read-only, imports a removed repo)
             verify_20261115/         checks behind the encoder and pseudo-clean choices
data/        datasets, masks, variants, runs — never committed
```

Datasets, generated masks, corrupted variants and checkpoints are not distributed; NYU Depth V2 and SUN RGB-D
are subject to their own terms.
