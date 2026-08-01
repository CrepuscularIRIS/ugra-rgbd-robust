# Datasets and pretrained weights

Neither datasets nor weights are redistributed here. Both are obtained from their
original sources and stay **by path** — nothing is copied into the repository.

---

## Layout

Upstream DFormer configs resolve every path from a single `root_dir`, so the layout is
fixed:

```text
$DFORMER_ROOT/                       # default: repos/DFormer
├── datasets/
│   ├── NYUDepthv2/
│   │   ├── RGB/        1449 .jpg
│   │   ├── Depth/      1449 .png     raw single-channel sensor depth
│   │   ├── Label/      1449 .png     40-class labels, 255 = ignore
│   │   ├── train.txt   795 lines
│   │   └── test.txt    654 lines
│   └── SUNRGBD/
│       ├── RGB/  Depth/  Label/
│       ├── train.txt   5285 lines
│       └── test.txt    5050 lines
└── checkpoints/
    ├── pretrained/DFormerv2_Small_pretrained.pth    # ImageNet backbone — needed to TRAIN
    └── trained/DFormerv2_Small_NYU.pth              # released NYU model — needed to EVAL
```

A symlink is fine and is what we used:

```bash
ln -s /path/to/your/datasets "$DFORMER_ROOT/datasets"
```

---

## Getting the data

Both datasets, in exactly this layout, are distributed by the DFormer authors — use their
links, since the folder structure and split files already match:

- **[DFormer README → Datasets](https://github.com/VCIP-RGBD/DFormer#datasets)**

Original sources, if you prefer to build the layout yourself:

| Dataset | Source | Split used |
|---|---|---|
| NYU Depth v2 | [cs.nyu.edu](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) | standard 795 train / 654 val, 40 classes |
| SUN RGB-D | [rgbd.cs.princeton.edu](https://rgbd.cs.princeton.edu/) | 5285 train / 5050 val, 37 classes |

### How depth is consumed — this matters

Depth is the **raw single-channel sensor map replicated across three channels**. It is
**not** an HHA encoding and **not** a disparity transform. The depth pathway receives the
sensor signal directly.

This is load-bearing for the whole study. The neutral-element argument is about
`mask_depth = |d_i − d_j|` vanishing for a *constant depth field*; substituting an HHA
encoding changes what "constant" means and the criterion would have to be re-derived. If
you swap the depth representation, you are running a variant.

Normalization: depth is normalized with mean `0.48` and std `0.28` per channel.

---

## Pretrained weights

From the [DFormer release page](https://github.com/VCIP-RGBD/DFormer#pretrained-models):

| File | Purpose | Needed for |
|---|---|---|
| `DFormerv2_Small_pretrained.pth` | ImageNet-pretrained backbone | **training** any arm |
| `DFormerv2_Small_NYU.pth` | released NYU Depth v2 model | evaluation, reproduction reference |

Place them under `$DFORMER_ROOT/checkpoints/{pretrained,trained}/` as above.

### Reproduction reference

The released checkpoint appears in the paper only as a reproduction reference, never as
an arm. Our harness scores it at:

| | mIoU |
|---|---:|
| Published (DFormerv2 paper, NYU Depth v2) | 56.0 |
| Our harness, multi-scale | 56.19 |
| Our harness, **single-scale** (the protocol used throughout) | **55.70** |
| Our own retrain, single-scale | 56.39 |

If your harness reproduces `55.70` single-scale on the released checkpoint, your pipeline
matches ours. If it does not, fix that before reading any other number.

---

## Our checkpoints are not shipped

The trained arms are roughly **7 GB** of `.pth` files and are excluded. To obtain them:

- **Retrain** — [`docs/REPRODUCTION.md`](REPRODUCTION.md). ~1–2 days per 500-epoch run on
  2 GPUs.
- Or work from the **frozen result artifacts**, which carry every number the paper
  reports without needing the weights at all: `python3 scripts/verify_results.py`.

Most questions about the *results* can be answered from `results/` alone. The weights are
only needed to run new *diagnostics*.

---

## Evaluation conditions

Four named conditions, all implemented in
[`diagnostics/corruptions/depth_corruptions.py`](../diagnostics/corruptions/depth_corruptions.py)
and injected by the frozen harness through a single `forward_pre_hook`:

| Condition | What it does |
|---|---|
| `clean` | modality-complete — real depth |
| `entire_missing` | complete depth absence, via the **verified neutral element** — not an unrelated input sentinel |
| `holes` | structured invalid regions |
| `gaussian` | additive Gaussian depth corruption |

Further families (`depth_shift`, `depth_saturate`, `salt_pepper`) and an end-to-end
invalid-pixel-fraction sweep are used in the analysis; see
[`docs/EXPERIMENTS.md`](EXPERIMENTS.md).

> `entire_missing` is **not** implemented as "set the input tensor to zero and hope". It
> goes through the removal that `diagnostics/single_path_invariance.py` proves is exact
> for this architecture. That distinction is the paper.
