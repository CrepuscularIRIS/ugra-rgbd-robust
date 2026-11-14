# Third-party notices

This repository is MIT licensed (see `LICENSE`). It does **not** redistribute the following; each is fetched at
setup time or obtained separately and remains under its own license. Check each upstream before reuse.

| component | source | how it is used |
|---|---|---|
| DFormer / DFormerv2 | https://github.com/VCIP-RGBD/DFormer (MIT, Copyright (c) 2023 VCIP-RGBD) | cloned by `scripts/setup_dformer.sh` at a pinned commit, used unpatched |
| CMNeXt (DELIVER) | https://github.com/jamycheung/DELIVER | `scripts/setup_cmnext.sh` clones it and applies `patches/cmnext-dat.patch` |
| GeminiFusion | https://github.com/JiaDingCN/GeminiFusion | `scripts/setup_geminifusion.sh` clones it and applies `patches/geminifusion-dat.patch` |
| NYU Depth V2, SUN RGB-D | the original dataset pages; packaged copies by the DFormer authors on Hugging Face (`bbynku/DFormerv2`) | data are never committed |
| Released checkpoints | the respective authors' pages / Hugging Face | never committed |

The patches under `patches/` are derivative works of the respective upstream files and are distributed under the
upstream license terms. `dat/analysis/gu.py` implements the geodesic edit described in arXiv:2610.04792 from the paper;
no code of that work is included.

Please cite DFormerv2 (CVPR 2025), DFormer (ICLR 2024), CMNeXt (CVPR 2023) and GeminiFusion (ICML 2024) if you use them.
