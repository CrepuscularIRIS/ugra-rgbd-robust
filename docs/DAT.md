# Depth-availability training (DAT)

Code for the paper *Toward Deployable RGB-D Semantic Segmentation: Outperforming Color Alone under Missing Depth via
Depth-Availability Training*. DAT trains one segmenter on complete, missing and partially missing depth with the
standard segmentation loss. It adds no parameters, changes no architecture, and needs no second stage: **it only
changes the depth tensor that enters the network during training.**

## Paper -> code

| paper | code |
|---|---|
| Sec. 3.1 empty-input constant `c`, T / B / G (Eq. 1) | `dat/hosts/*.py` (`empty_value`), `dat/metrics.py` (`Endpoints`) |
| Sec. 3.4 availability sampling, Eq. (2)-(3), `Q_p` | `dat/availability.py` (`AvailabilitySampler`, `apply_availability`, `sample_partial_mask`) |
| Sec. 3.5 / Algorithm 1: one CE loss, one step | `dat/train.py` |
| Table 1 / 3 arms (C, RGB-only, DAT, ablations) | `dat/presets.py` |
| complete / missing / partial evaluation, S_c, S_m, T | `dat/evaluate.py` |
| Eq. (5) classwise change | `dat/metrics.py` (`classwise_delta`) |
| Fig. 2 post-training repair routes | `dat/analysis/` and `scripts/dat_repair_routes.py` |
| Table 2 other architectures | `dat/hosts/cmnext.py`, `dat/hosts/geminifusion.py`, `patches/*-dat.patch` |

## Quick start (DFormer family)

```bash
bash scripts/setup_dformer.sh            # stock upstream at a pinned commit; or point DFORMER_ROOT at your own checkout
export RGBD_DATA_ROOT=/path/to/data      # NYUDepthv2/{RGB,Depth,Label,train.txt,test.txt}, SUNRGBD/ likewise

# DAT, conventional (C) and RGB-only arms: same schedule, only the depth input differs
python -m dat.train --host DFormerv2-S --preset DAT      --pretrained DFormerv2_Small_pretrained.pth --out runs/dat
python -m dat.train --host DFormerv2-S --preset C        --pretrained DFormerv2_Small_pretrained.pth --out runs/c
python -m dat.train --host DFormerv2-S --preset RGB_ONLY --pretrained DFormerv2_Small_pretrained.pth --out runs/rgb

# endpoints and the retained advantage T = S_m - S_RGB
python -m dat.evaluate --host DFormerv2-S --ckpt runs/dat/final.pth --rgb-only-ckpt runs/rgb/final.pth --partial 0.25
```

Hosts: `DFormerv1-{T,S,B,L}`, `DFormerv2-{S,B,L}`; datasets `NYUDepthv2` (40 classes), `SUNRGBD` (37 classes).
Schedule (learning rate, batch size, epochs, warm-up, weight-decay groups) is read from each model's upstream config.
DFormerv2-S on NYU at batch size 16 runs at about 1.3 it/s on one RTX 4090 in fp32, i.e. about 5-6 h for the 25k
steps of the paper; fp32 is the default because reduced precision is unstable for this architecture.

## Choices the paper does not fix

* **`Q_p` (partial masks).** The paper specifies one rectangle or several separated rectangles, not their sizes.
  `PartialMaskSpec` draws the removed share uniformly from 10-50 % of the image, aspect ratio log-uniform in
  [0.5, 2], 2-4 regions for the multi-rectangle shape. Change it in one place if the original setting differs.
* **Two-condition ablation ratios** (Table 3 rows "w/o fully missing", "w/o partial missing") are assumed to be
  50:50 between the two kept conditions (`dat/presets.py`); only 75 : 12.5 : 12.5 and 50 : 25 : 25 are printed.
* **Evaluation protocol.** Two protocols are implemented in `dat.evaluate`. `single` (default) is one scale, no flip.
  `official` (`--protocol official`) is what upstream's `infer.py` runs: scales 0.5-1.5 with horizontal flip, i.e.
  `evaluate_msf`; training-time validation upstream is single-scale. Published DFormer numbers come from the multi-scale
  protocol, so compare a re-measured checkpoint against a published score under `official`, and against a
  single-scale table only under `single`. `scripts/retest_official.py` measures both for the released checkpoints.
* **Run-to-run noise.** The upstream HAM decoder draws a random initialisation on every forward pass, so one
  checkpoint evaluated twice differs by about 0.1-0.2 mIoU.

## Other architectures (Table 2)

| host | RGB-only baseline | how DAT is applied |
|---|---|---|
| DFormerv1 / v2 | yes: constant depth in training and testing | `dat.train` (this repo) |
| CMNeXt-B2/B4/B5 | yes, as the paper defines it: the `RGB_ONLY` preset on the full RGB-D network (constant depth). `modals=['img']` also builds, but it is a different network without the depth branch | `patches/cmnext-dat.patch` (`tools/train_mm.py`), `scripts/setup_cmnext.sh` |
| GeminiFusion-B3/B5 | **no**: `num_parallel = 2` is hard-wired and both streams are decoded unconditionally, so only S_c and S_m are reported and T is undefined | `patches/geminifusion-dat.patch` (`main.py`), `scripts/setup_geminifusion.sh` |

Both patches add one flag (`--dat-preset`, default off, so conventional training is unchanged) and replace only the
depth tensor after it is moved to the GPU; neither touches the loss. `bash scripts/setup_cmnext.sh` /
`setup_geminifusion.sh` clone upstream at a pinned commit and apply the patch (`--check` reports the state).

Facts established from the upstream code (the host adapters encode them):

* **Empty-input constant.** `c = 0` for both. CMNeXt feeds 3-channel HHA divided by 255 (no mean/std); GeminiFusion
  feeds 3-channel grey depth divided by 255. Zero is also what each upstream pads with, so it is the upstream
  convention for "no depth" rather than a measured sensor-failure encoding.
* **GeminiFusion inference.** The released model returns `[rgb stream, depth stream, ens]`; the headline score is
  `ens` (softmax-weighted sum of the two streams). `ens` is detached, so the host overrides `loss` to reproduce
  upstream's sum of the three cross-entropies.
* **Checkpoints load strictly:** CMNeXt-B4 NYU (`InSAI-Lab/CMNeXt`, 1971 keys) and GeminiFusion-B3/B5 NYU, with 0
  missing keys (GeminiFusion-B3 carries 80 leftover `encoder.score_predictor.*` keys, allow-listed).
* **One host family per process.** Upstream `models/` and `utils/` packages collide with DFormer's.
* **Not run end to end.** The two patches are applied and syntax-checked, but a full training run needs each upstream's
  own data layout, DDP and (old) mmcv; evaluation under missing depth should go through the host adapters.

**Reproducibility warning for GeminiFusion.** On the DFormer-layout NYU copy, the released B3 checkpoint scores 49.9
mIoU with real depth and 43.6 with constant depth, versus 56.8 published. The missing-depth score matches Table 2 of
the paper (43.6); the complete-depth score does not. The released weights were trained on the TokenFusion data layout,
whose depth is normalised differently, so complete-input numbers for this host should be re-measured on that layout
before being compared with a missing-input number.

## Tests

```bash
python -m pytest tests/test_dat_core.py tests/test_gu_geodesic.py tests/test_host_dformer.py -q
```

`test_dat_core.py` checks the sampling frequencies, Eq. (2), the partial-mask shapes and the T/B/G identity against a
Table 1 row. `test_host_dformer.py` builds all seven DFormer models, runs a DAT step, verifies the Sec. 3.1 claim that
a constant depth map gives identical DFormerv2 features whatever the constant, and loads the released checkpoint with
no missing or unexpected keys (skipped when upstream code or weights are absent).
