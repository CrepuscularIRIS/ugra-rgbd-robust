"""Per-stratum scoring: mIoU on E=0 (depth honest) vs E=1 (depth lying) pixels,
for the SAME model under corrupted vs clean depth (design report S4, F_S/G decomposition).

    python scripts/eval_strata.py --dataset NYUDepthv2 --host DFormerv2-S --variant F8/cross

Two synchronized single-scale passes (corrupted variant + clean package) over the same
frames; strata come from the variant's E oracle. Reports mIoU per (input, stratum) and
the shortfall F_X = mIoU_X(clean) - mIoU_X(corrupted) per stratum. mIoU convention
matches dat.metrics (empty classes count 0, as in the official DFormer evaluation).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_variants import NYU_BASE, SUN_BASE, ckpt_for, shim_root  # noqa: E402

UGRA = Path.home() / "workspace/ugra-rgbd-robust"
sys.path.insert(0, str(UGRA))
os.environ.setdefault("DFORMER_ROOT", "/data/projects/DFormer")

REPO = Path(__file__).resolve().parent.parent
NYU_E = REPO / "data/NYU_corrupt"


def e_dir(dataset: str, variant: str) -> Path:
    if dataset == "NYUDepthv2":
        return NYU_E / variant / "E"
    return REPO / "data/SUN_corrupt_test_full" / variant / "E"


def frame_ids(shim: Path, dataset: str) -> list[str]:
    txt = (shim / dataset / "test.txt").resolve() if (shim / dataset / "test.txt").is_symlink() \
        else (shim / dataset / "test.txt")
    ids = []
    for l in txt.read_text().splitlines():
        if l.strip():
            ids.append(l.split()[0].rsplit("/", 1)[-1].split(".")[0])
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["NYUDepthv2", "SUNRGBD"])
    ap.add_argument("--host", default="DFormerv2-S")
    ap.add_argument("--variant", required=True, help="e.g. F8/cross")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from dat.data import eval_loader
    from dat.evaluate import predict
    from dat.hosts.dformer import make_host
    from dat import metrics

    base = NYU_BASE if a.dataset == "NYUDepthv2" else SUN_BASE
    corrupt_depth = (NYU_E / a.variant / "Depth" if a.dataset == "NYUDepthv2"
                     else REPO / "data/SUN_corrupt_test_full" / a.variant)
    subset = None
    if a.dataset == "SUNRGBD":
        subset = {p.stem for p in corrupt_depth.glob("*.png")}

    shim_c = shim_root(a.dataset, base, corrupt_depth, subset)
    shim_k = shim_root(a.dataset, base, base / "Depth", subset)
    host = make_host(a.host, a.dataset, data_root=str(shim_c))
    model = host.build(device=a.device)
    host.load_weights(model, str(ckpt_for(a.host, a.dataset)))
    loader_c = eval_loader(host)  # corrupted files under this shim's cfg
    ids = frame_ids(shim_c, a.dataset)
    host_k = make_host(a.host, a.dataset, data_root=str(shim_k))
    loader_k = eval_loader(host_k)

    k = host.num_classes
    hist = {(inp, s): torch.zeros(k, k, dtype=torch.long, device=a.device)
            for inp in ("corrupt", "clean") for s in (0, 1)}
    model.eval()
    e_root = e_dir(a.dataset, a.variant)
    n = 0
    with torch.no_grad():
        for bc, bk in zip(loader_c, loader_k):
            fid = ids[n]
            em = cv2.imread(str(e_root / f"{fid}.png"), cv2.IMREAD_UNCHANGED)
            if em is None:
                raise FileNotFoundError(e_root / f"{fid}.png")
            e = torch.from_numpy(em.astype(np.int64)).to(a.device)
            for inp, b in (("corrupt", bc), ("clean", bk)):
                rgb, depth, label = b["data"].to(a.device), b["modal_x"].to(a.device), b["label"].to(a.device)
                pred = predict(host, model, rgb, depth).argmax(1)
                keep = label != 255
                for s in (0, 1):
                    m = keep & (e == s)
                    hist[(inp, s)] += torch.bincount(label[m] * k + pred[m], minlength=k * k).view(k, k)
            n += 1

    out = {"dataset": a.dataset, "host": a.host, "variant": a.variant, "n": n, "strata": {}}
    for s in (0, 1):
        mc = metrics.miou_from_hist(hist[("corrupt", s)].cpu().numpy())
        mk = metrics.miou_from_hist(hist[("clean", s)].cpu().numpy())
        out["strata"][f"E{s}"] = {"corrupt": round(float(mc), 2), "clean": round(float(mk), 2),
                                  "F_X": round(float(mk - mc), 2)}
    print(json.dumps(out, indent=1))
    out_dir = REPO / "data/eval_strata"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{a.dataset}_{a.host}_{a.variant.replace('/', '_')}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
