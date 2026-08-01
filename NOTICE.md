# Third-party notices

This repository is [MIT licensed](LICENSE). It does **not** redistribute any of the
following. Each is fetched at setup time or obtained separately, and remains under its
own license.

---

## DFormer / DFormerv2

- **Source:** https://github.com/VCIP-RGBD/DFormer
- **License:** MIT, Copyright (c) 2023 VCIP-RGBD
- **How it is used:** [`patches/dformer-ugra.patch`](patches/dformer-ugra.patch) is a
  **derivative work** of this codebase, distributed under the same MIT terms. The upstream
  source is cloned and patched by `bash scripts/setup_dformer.sh`, pinned to commit
  `23cd8f833df12c74af45bc671e75891e48f8ff1b`.

The patch modifies five files: `models/builder.py`, `models/encoders/DFormerv2.py`,
`utils/train.py`, `utils/dataloader/dataloader.py`, `utils/engine/engine.py`. See
[`patches/README.md`](patches/README.md).

Please cite DFormerv2 (CVPR 2025) and DFormer (ICLR 2024) if you use this work.

---

## GeminiFusion

- **Source:** https://github.com/JiaDingCN/GeminiFusion
- **How it is used:** by `diagnostics/gf_fill_sensitivity.py` only, as the second
  (dual-stream) architecture in the fill-value sensitivity measurement. Optional — every
  other part of this repository runs without it. Neither its code nor its weights are
  redistributed here.

---

## Datasets

Neither dataset is redistributed. Both are governed by their own terms, and users must
obtain them from the original sources. See [`docs/DATASETS.md`](docs/DATASETS.md).

| Dataset | Source |
|---|---|
| NYU Depth v2 | https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html |
| SUN RGB-D | https://rgbd.cs.princeton.edu/ |

---

## Pretrained weights

The ImageNet-pretrained backbone and the released NYU Depth v2 checkpoint are
distributed by the DFormer authors from their release page and are not mirrored here.
