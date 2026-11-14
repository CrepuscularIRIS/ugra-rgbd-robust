"""Post-training repair routes (Fig. 2 of the paper) for a DFormer model.

    python scripts/dat_repair_routes.py features --host DFormerv2-S --ckpt C.pth
    python scripts/dat_repair_routes.py weights  --host DFormerv2-S --ckpt C.pth --ref R.pth

See dat/analysis/routes.py. `R` is the RGB-only reference (python -m dat.train --preset RGB_ONLY ...).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("route", choices=("features", "weights"))
    ap.add_argument("--host", required=True)
    ap.add_argument("--dataset", default="NYUDepthv2", choices=("NYUDepthv2", "SUNRGBD"))
    ap.add_argument("--ckpt", required=True, help="conventional RGB-D checkpoint C")
    ap.add_argument("--ref", help="reference R checkpoint (weights route)")
    ap.add_argument("--k", type=int, default=16, help="GU subspace rank")
    ap.add_argument("--ks", default="2,8,32", help="feature-repair subspace ranks")
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out")
    a = ap.parse_args()

    from dat.analysis import routes
    from dat.data import eval_loader
    from dat.hosts.dformer import make_host

    host = make_host(a.host, a.dataset)
    model = host.build(device=a.device)
    host.load_weights(model, a.ckpt)
    if a.route == "features":
        res = routes.feature_routes(host, model, eval_loader(host, "train"), eval_loader(host),
                                    tuple(int(k) for k in a.ks.split(",")), a.calib, a.limit, a.device)
        for s in res["spectrum"]:
            print("stage {stage}  C={channels:<4} rel.shift {rel_shift:.3f}  eff.rank {eff_rank:.1f}  "
                  "explained {explained}".format(**s))
        print(f"complete {res['miou']['complete']:.2f}  missing {res['miou']['missing']:.2f}")
        for n, v in res["gap_closed_pct"].items():
            print(f"  {n:<12} mIoU {res['miou'][n]:.2f}  closes {v:5.1f}% of the gap")
    else:
        if not a.ref:
            ap.error("--ref is required for the weights route")
        res = routes.weight_routes(host, model, a.ckpt, a.ref, eval_loader(host), a.k, a.limit, a.device)
        for r in res:
            print(f"  {r['route']:<24} missing {r['missing']:6.2f}  complete {r['complete']:6.2f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
