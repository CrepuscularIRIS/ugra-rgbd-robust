# Depth-Availability Training (DAT)

Code for **depth-availability training**, a single-stage strategy for RGB-D semantic segmentation that keeps a model
useful when depth is missing.

RGB-D segmenters trained on complete inputs can fall *below* a model of the same architecture trained on color alone
once depth is removed. DAT trains one segmenter from the start on a mixture of **complete**, **missing** and
**partially missing** depth with the ordinary segmentation loss. It adds no parameters, changes no architecture,
needs no second training stage, and leaves inference untouched: it only changes the depth tensor that enters the
network during training.

```text
S_c  complete-input mIoU          S_m  missing-input mIoU (depth replaced by a constant map)
S_RGB  RGB-only baseline: the same architecture trained and tested on color alone
T = S_m - S_RGB   retained advantage      G = S_c - S_m   depth dependence      B = S_c - S_RGB = T + G
```

A deployed RGB-D model beats the RGB-only baseline under missing depth only if `T > 0`.

## Quick start

```bash
pip install -r requirements.txt                 # install torch for your CUDA build first
bash scripts/setup_dformer.sh                   # upstream DFormer at a pinned commit (stock, no patch needed)
export RGBD_DATA_ROOT=/path/to/data             # NYUDepthv2/ and SUNRGBD/ in the DFormer layout

# the three arms differ only in the depth the network sees during training
python -m dat.train --host DFormerv2-S --preset DAT      --pretrained DFormerv2_Small_pretrained.pth --out runs/dat
python -m dat.train --host DFormerv2-S --preset C        --pretrained DFormerv2_Small_pretrained.pth --out runs/c
python -m dat.train --host DFormerv2-S --preset RGB_ONLY --pretrained DFormerv2_Small_pretrained.pth --out runs/rgb

# S_c, S_m, T  (add --protocol official for the upstream multi-scale + flip protocol)
python -m dat.evaluate --host DFormerv2-S --ckpt runs/dat/final.pth --rgb-only-ckpt runs/rgb/final.pth --partial 0.25
```

Datasets and released checkpoints: Hugging Face `bbynku/DFormerv2` (`datasets/*.zip`, `DFormerv2/*`) and the DFormer
GitHub page. Nothing of this kind is stored in the repository.

## What is in the box

| | |
|---|---|
| `dat/availability.py` | availability sampling: complete 0.50 / missing 0.25 / partial 0.25, `D~ = M*D + (1-M)*c` |
| `dat/presets.py` | the arms of the paper: `C`, `RGB_ONLY`, `DAT` and the ablation mixtures |
| `dat/metrics.py` | S_c, S_m, T/B/G with the `B = T + G` identity, classwise IoU changes |
| `dat/hosts/` | architecture adapters: **DFormerv1-T/S/B/L, DFormerv2-S/B/L** (NYU Depth V2 and SUN RGB-D), **CMNeXt-B2/B4/B5**, **GeminiFusion-B3/B5** |
| `dat/train.py`, `dat/evaluate.py` | single-GPU training with the upstream schedule; endpoint and partial-depth evaluation |
| `dat/analysis/` | post-training repair routes for comparison: weight interpolation, geodesic subspace edit, feature-space repair |
| `patches/` | one-flag DAT patches for the CMNeXt and GeminiFusion training loops (`--dat-preset`, off by default) |
| `scripts/retest_official.py` | re-measure the released checkpoints under the single-scale and the official protocol |
| `tests/` | 34+ unit and integration tests |

GeminiFusion hard-wires two parallel streams and decodes both unconditionally, so it has no single-modality mode: no
RGB-only baseline can be trained for it and only S_c and S_m are meaningful there. CMNeXt trains an RGB-only baseline as
the `RGB_ONLY` preset on the full network. Details, the choices the paper does not fix, and the paper-to-code map:
**[`docs/DAT.md`](docs/DAT.md)**.

## Tests

```bash
python -m pytest tests -q          # host tests skip cleanly when upstream code or weights are absent
```

## Not included

Datasets, checkpoints, training runs and the third-party code under `repos/` are never committed.
See [`NOTICE.md`](NOTICE.md) for upstream licenses.

## License

MIT, see [`LICENSE`](LICENSE).
