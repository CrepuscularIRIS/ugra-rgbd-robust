"""Re-evaluate the released DFormer checkpoints (the "C" column of Table 1) under both inference protocols.

    python scripts/retest_official.py --out audit/retest [--datasets NYUDepthv2 SUNRGBD] [--only DFormerv2-S ...]

For every (dataset, host) with a released checkpoint it measures the complete-input score S_c and the
missing-input score S_m (constant depth) with
    single   one scale, no flip
    official the upstream infer.py protocol: scales 0.5-1.5 + horizontal flip
and writes one JSON per (dataset, host, protocol); finished runs are skipped, so the script can be resumed.
$OFFICIAL_WEIGHTS must point at the folder holding DFormer/{NYUDepthv2,SUNRGBD}/*.pth and DFormerv2/{NYU,SUNRGBD}/*.pth.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SIZE = {"T": "Tiny", "S": "Small", "B": "Base", "L": "Large"}
HOSTS = ["DFormerv2-S", "DFormerv2-B", "DFormerv2-L", "DFormerv1-T", "DFormerv1-S", "DFormerv1-B", "DFormerv1-L"]


def checkpoint(root: str, dataset: str, host: str) -> str:
    fam, size = host.split("-")
    name = SIZE[size]
    if fam == "DFormerv2":
        sub, tag = ("NYU", "NYU") if dataset == "NYUDepthv2" else ("SUNRGBD", "SUNRGBD")
        return os.path.join(root, "DFormerv2", sub, f"DFormerv2_{name}_{tag}.pth")
    pre = "NYUv2" if dataset == "NYUDepthv2" else "SUNRGBD"
    return os.path.join(root, "DFormer", dataset, f"{pre}_DFormer_{name}.pth")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=os.environ.get("OFFICIAL_WEIGHTS"))
    ap.add_argument("--datasets", nargs="+", default=["NYUDepthv2"], choices=["NYUDepthv2", "SUNRGBD"])
    ap.add_argument("--only", nargs="+", default=HOSTS, choices=HOSTS)
    ap.add_argument("--protocols", nargs="+", default=["single", "official"], choices=["single", "official"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not a.weights:
        ap.error("set --weights or $OFFICIAL_WEIGHTS")
    os.makedirs(a.out, exist_ok=True)

    from dat.data import eval_loader
    from dat.evaluate import OFFICIAL_SCALES, evaluate
    from dat.hosts.dformer import make_host

    for dataset in a.datasets:
        for name in a.only:
            host = make_host(name, dataset)
            ck = checkpoint(a.weights, dataset, name)
            todo = [p for p in a.protocols if not os.path.exists(os.path.join(a.out, f"{dataset}_{name}_{p}.json"))]
            if not todo:
                continue
            model = host.build(device=a.device)
            host.load_weights(model, ck)  # strict
            loader = eval_loader(host)
            for proto in todo:
                kw = dict(scales=OFFICIAL_SCALES, flip=True) if proto == "official" else {}
                t0 = time.time()
                r = evaluate(host, model, loader, ("complete", "missing"), limit=a.limit, device=a.device, **kw)
                out = {"dataset": dataset, "host": name, "protocol": proto, "ckpt": ck, "n": r["complete"]["n"],
                       "S_c": r["complete"]["mIoU"], "S_m": r["missing"]["mIoU"],
                       "G": round(r["complete"]["mIoU"] - r["missing"]["mIoU"], 2), "seconds": round(time.time() - t0)}
                json.dump(out, open(os.path.join(a.out, f"{dataset}_{name}_{proto}.json"), "w"))
                print(f"{dataset:11} {name:12} {proto:8} S_c {out['S_c']:6.2f}  S_m {out['S_m']:6.2f}  "
                      f"G {out['G']:5.2f}  ({out['seconds']}s)", flush=True)
            del model


if __name__ == "__main__":
    main()
