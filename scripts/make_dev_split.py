#!/usr/bin/env python3
"""Carve a fixed dev split out of NYUv2 *train* — the R3 (validation-leakage) fix.

The 654-image NYUv2 val set was doing ladder screening, hyperparameter choice,
checkpoint selection AND final reporting. This produces a held-out split drawn
from train so that everything except final reporting can move off val.

Protocol this enables (`plan/paper-requirements.md`, "Independent final evaluation"):

    1. screen / tune / select checkpoints on  train_dev.txt  ->  dev.txt
    2. freeze the config
    3. RETRAIN on the full train.txt
    4. evaluate on val (test.txt) ONCE, for final reporting

Step 3 is why the dev split costs no training data in the main table.

The split is fixed once and shared by every arm. It is written with a recorded
SHA256; regenerating it with the same seed reproduces it byte-for-byte.

    python3 scripts/make_dev_split.py --root repos/DFormer/datasets/NYUDepthv2
"""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

DEV_SPLIT_SEED = 20260805  # fixed once; changing it invalidates every screen run


def sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding train.txt / test.txt")
    ap.add_argument("--n-dev", type=int, default=100)
    ap.add_argument("--seed", type=int, default=DEV_SPLIT_SEED)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()

    root = Path(args.root)
    train_txt = root / "train.txt"
    out_train = root / "train_dev.txt"
    out_dev = root / "dev.txt"
    manifest = root / "dev_split_manifest.json"

    if (out_train.exists() or out_dev.exists()) and not args.force:
        raise SystemExit(
            f"refusing to overwrite {out_train.name}/{out_dev.name} without --force: "
            "the split is frozen once and shared by every arm; regenerating it "
            "silently would invalidate screens already run against it"
        )

    lines = [ln for ln in train_txt.read_text().splitlines() if ln.strip()]
    # Sort first: the input file order must not affect the split.
    lines = sorted(lines)
    if args.n_dev >= len(lines):
        raise SystemExit(f"--n-dev {args.n_dev} >= train size {len(lines)}")

    rng = random.Random(args.seed)
    dev = sorted(rng.sample(lines, args.n_dev))
    dev_set = set(dev)
    keep = [ln for ln in lines if ln not in dev_set]

    assert len(keep) + len(dev) == len(lines), "split is not a partition"
    assert not (set(keep) & dev_set), "train_dev and dev overlap"

    out_train.write_text("\n".join(keep) + "\n")
    out_dev.write_text("\n".join(dev) + "\n")

    # Class coverage: a dev split that never shows a class cannot select on it.
    coverage = {}
    try:
        import numpy as np
        from PIL import Image
        for tag, subset in (("dev", dev), ("train_dev", keep)):
            seen = Counter()
            for ln in subset:
                gt = root / ln.split()[-1]
                if gt.exists():
                    seen.update(np.unique(np.array(Image.open(gt))).tolist())
            coverage[tag] = sorted(int(c) for c in seen)
    except Exception as e:  # pragma: no cover - reporting only
        coverage["error"] = str(e)

    meta = {
        "seed": args.seed,
        "n_train_original": len(lines),
        "n_train_dev": len(keep),
        "n_dev": len(dev),
        "sha256": {
            "train.txt": sha256(train_txt),
            "train_dev.txt": sha256(out_train),
            "dev.txt": sha256(out_dev),
        },
        "n_classes_present": {k: len(v) for k, v in coverage.items() if k != "error"},
        "purpose": "R3: screening/tuning/checkpoint-selection move here; val is touched once.",
    }
    manifest.write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
