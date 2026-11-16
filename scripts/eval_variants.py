"""Step 0 first shot: evaluate DFormerv2-S on corrupted-depth variants (both datasets).

    python scripts/eval_variants.py --dataset NYUDepthv2 --device cuda:0
    python scripts/eval_variants.py --dataset SUNRGBD   --device cuda:1

Per condition, builds a shim data root (RGB/Label(s)/txt symlinked from the base
package, Depth symlinked to the variant) and runs the ugra harness single-scale
eval with condition 'complete' — the corruption lives in the files, no tensor
overwrite. The model is built and loaded once. SUN runs on the 12-frame demo
subset (train-split names, base txt lines kept verbatim).
Writes data/eval_variants/<dataset>_DFormerv2-S.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

UGRA = Path.home() / "workspace/ugra-rgbd-robust"
sys.path.insert(0, str(UGRA))
os.environ.setdefault("DFORMER_ROOT", "/data/projects/DFormer")

REPO = Path(__file__).resolve().parent.parent
NYU_BASE = Path("/data/datasets/rgbd/NYUDepthv2")
SUN_BASE = Path("/data/datasets/rgbd/SUNRGBD")
NYU_C = REPO / "data/NYU_corrupt"
SUN_C = REPO / "data/SUN_corrupt_test_full"
CKPT = {"NYUDepthv2": Path("/data/projects/DFormer/checkpoints/trained/NYUDepthv2/DFormerv2_Small_NYU.pth"),
        "SUNRGBD": Path("/data/projects/DFormer/checkpoints/trained/SUNRGBD/DFormerv2_Small_SUNRGBD.pth")}
OLD_DFORMER = Path(os.environ.get("DFORMERV1_WEIGHTS", "/data/weights/DFormer"))  # DFormer/{NYUDepthv2,SUNRGBD}/*.pth
V1_SIZE = {"DFormerv1-T": "Tiny", "DFormerv1-S": "Small", "DFormerv1-B": "Base", "DFormerv1-L": "Large"}


V2_ROOT = Path("/data/projects/DFormer")


def ckpt_for(host: str, dataset: str) -> Path:
    if host.startswith("DFormerv1"):
        pre = "NYUv2" if dataset == "NYUDepthv2" else "SUNRGBD"
        return OLD_DFORMER / dataset / f"{pre}_DFormer_{V1_SIZE[host]}.pth"
    if host == "DFormerv2-S":
        return CKPT[dataset]
    tag = "NYU" if dataset == "NYUDepthv2" else "SUNRGBD"
    return V2_ROOT / f"DFormerv2_Large_{tag}.pth"  # v2-L released weights at the repo root

NYU_CONDITIONS = {
    "clean": NYU_BASE / "Depth",
    "F0_zero": NYU_C / "F0/zero/Depth",
    "F1_a20": NYU_C / "F1/a20/Depth",
    "F2_a8": NYU_C / "F2/a8/Depth",
    "F6_t16": NYU_C / "F6/t16/Depth",
    "F8_cross": NYU_C / "F8/cross/Depth",
    "F8_same": NYU_C / "F8/same/Depth",
}
SUN_CONDITIONS = {  # SUN demo writes depth directly under <fam>/<sev>/ (no Depth/ subdir)
    "clean": SUN_BASE / "Depth",
    "F0_zero": SUN_C / "F0/zero",
    "F1_holes": SUN_C / "F1/holes",
    "F2_a1": SUN_C / "F2/a1",
    "F2_a8": SUN_C / "F2/a8",
    "F4_k3": SUN_C / "F4/k3",
    "F5_s90": SUN_C / "F5/s90",
    "F6_t16": SUN_C / "F6/t16",
    "F8_swap": SUN_C / "F8/swap",
}


def shim_root(dataset: str, base: Path, depth_dir: Path, subset_names: set[str] | None) -> Path:
    """<tmp>/<dataset>/ with everything from the base except Depth; returns the parent."""
    root = Path(tempfile.mkdtemp(prefix="v2s_")) / dataset
    root.mkdir(parents=True)
    for sub in ("RGB", "Label", "labels", "intrinsics"):
        src = base / sub
        if src.exists():
            (root / sub).symlink_to(src.resolve())
    if subset_names is None:
        for txt in ("train.txt", "test.txt"):
            (root / txt).symlink_to((base / txt).resolve())
    else:
        # subset frames may live in either split; filter whichever has matches and
        # write those lines into BOTH txts (eval reads test.txt)
        lines = []
        for src in ("train.txt", "test.txt"):
            lines = [l for l in (base / src).read_text().splitlines()
                     if l.strip() and Path(l.split()[0]).stem in subset_names]
            if lines:
                break
        for txt in ("train.txt", "test.txt"):
            (root / txt).write_text("\n".join(lines) + "\n")
    (root / "Depth").symlink_to(depth_dir.resolve())
    return root.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(CKPT))
    ap.add_argument("--host", default="DFormerv2-S",
                    choices=["DFormerv2-S", "DFormerv2-L", "DFormerv1-T",
                             "DFormerv1-S", "DFormerv1-B", "DFormerv1-L"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sun-dir", default=str(SUN_C), help="SUN variant root (default: train-split demo)")
    a = ap.parse_args()
    sun_c = Path(a.sun_dir)

    from dat.data import eval_loader
    from dat.evaluate import evaluate
    from dat.hosts.dformer import make_host

    base = NYU_BASE if a.dataset == "NYUDepthv2" else SUN_BASE
    if a.dataset == "NYUDepthv2":
        conds = NYU_CONDITIONS
    else:  # build at runtime so --sun-dir takes effect
        conds = {"clean": SUN_BASE / "Depth",
                 **{k: sun_c / f"{k.split('_')[0]}/{v}" for k, v in
                    [("F0_zero", "zero"), ("F1_holes", "holes"), ("F2_a1", "a1"), ("F2_a8", "a8"),
                     ("F4_k3", "k3"), ("F5_s90", "s90"), ("F6_t16", "t16"), ("F8_swap", "swap")]}}
    subset = None
    if a.dataset == "SUNRGBD":
        subset = {p.stem for p in (sun_c / "F0/zero").glob("*.png")}

    results = {}
    model = host = None
    ck = ckpt_for(a.host, a.dataset)
    for name, depth_dir in conds.items():
        data_root = shim_root(a.dataset, base, depth_dir, subset)
        h = make_host(a.host, a.dataset, data_root=str(data_root))
        if model is None:
            model = h.build(device=a.device)
            h.load_weights(model, str(ck))
            host = h
        loader = eval_loader(h)
        t0 = time.time()
        r = evaluate(host, model, loader, ("complete",), device=a.device)
        results[name] = {"mIoU": round(r["complete"]["mIoU"], 2), "n": r["complete"]["n"],
                         "seconds": round(time.time() - t0)}
        print(f"{a.dataset} {a.host} {name:10s} mIoU {results[name]['mIoU']:6.2f}  "
              f"(n={results[name]['n']}, {results[name]['seconds']}s)", flush=True)

    out = REPO / "data/eval_variants"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.dataset}_{a.host}.json").write_text(json.dumps(
        {"dataset": a.dataset, "host": a.host, "ckpt": str(ck),
         "protocol": "single-scale", "results": results}, indent=1))


if __name__ == "__main__":
    main()
