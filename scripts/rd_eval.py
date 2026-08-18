"""Evaluate the CMNeXt arms under the three PRE-REGISTERED conditions.

Protocol fixed in plan/PREREG-removal-dropout.md BEFORE any arm finished:

    clean       real depth, normal forward           — the non-regression floor
    zero_in     depth tensor -> zeros at the input   — the standard benchmark condition, and what
                                                       an unsignalled sensor failure delivers
    removal_R   the named removal invoked at         — the deployment case where absence is KNOWN
                inference (FRM bypassed, x_cam)        and the depth-free path can be selected

BOTH arms are evaluated under BOTH missing conditions. Scoring only each arm in the mode it was
trained for would decide the result by choice of yardstick; the off-diagonal cells are the transfer
check and may not be dropped.

CHECKPOINT POLICY. DELIVER saves best-on-eval, not `latest.pth` as TARGET.md fixes for the DFormer
arms. Both CMNeXt arms share this policy, so the contrast between them is fair, but the two
architectures' numbers are NOT on a common policy and must never be pooled or compared directly.

    python3 scripts/rd_eval.py --gpu 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "repos" / "DELIVER"
RUNS = PROJECT / "results" / "removal_dropout"
CONDITIONS = ["clean", "zero_in", "removal_R"]


def find_ckpt(run_dir: Path):
    """Best-epoch checkpoint written by train_mm.py (weights only, not the _checkpoint.pth)."""
    cands = [p for p in (run_dir / "out").rglob("*.pth") if "_checkpoint" not in p.name]
    if not cands:
        return None
    # train_mm keeps exactly one best; if several survive, take the newest by mtime
    return max(cands, key=lambda p: p.stat().st_mtime)


def evaluate_one(ckpt: Path, condition: str, gpu: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(PROJECT / "scripts"))

    import torch
    import yaml
    from semseg.models import cmnext as MODEL
    from semseg.models.backbones import cmnext as BACKBONE
    import removal_dropout_cmnext as RD

    RD.install(BACKBONE)

    orig = MODEL.CMNeXt.forward

    def forward(self, x):
        if condition == "zero_in":
            RD.set_drop(None)
            x = [x[0], torch.zeros_like(x[1])]
        elif condition == "removal_R":
            RD.set_drop(torch.ones(x[0].shape[0], dtype=torch.bool))
        else:
            RD.set_drop(None)
        try:
            return orig(self, x)
        finally:
            RD.set_drop(None)

    MODEL.CMNeXt.forward = forward

    cfg = yaml.safe_load(open("configs/gate0_nyu_rgbd.yaml"))
    cfg["EVAL"]["MODEL_PATH"] = str(ckpt)
    cfg["EVAL"]["MSF"]["ENABLE"] = False           # single-scale, matching the sweep convention
    tmp = REPO / f"configs/_rdeval_{ckpt.stem[:40]}_{condition}.yaml"
    tmp.write_text(yaml.safe_dump(cfg))

    import io
    import contextlib
    import runpy
    sys.argv = ["val_mm.py", "--cfg", str(tmp)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(REPO / "tools" / "val_mm.py"), run_name="__main__")
    text = buf.getvalue()
    MODEL.CMNeXt.forward = orig

    # val_mm.py:163 prints exactly `mIoU : <value>`. Match that, and nothing else.
    # A looser "last float on a line containing mIoU" scan silently captures mean_Acc from
    # the trainer's `mean_IoU 55.37 mean_Acc 68.89` line — a wrong number, not an error.
    import re as _re
    hits = _re.findall(r"mIoU\s*:\s*([0-9]+\.?[0-9]*)", text)
    if not hits:
        raise SystemExit(f"REFUSING: no `mIoU :` line in val_mm output for {ckpt.name}/"
                         f"{condition}. Tail:\n{text[-600:]}")
    if len(set(hits)) > 1:
        raise SystemExit(f"REFUSING: {len(set(hits))} distinct mIoU values in one eval "
                         f"({sorted(set(hits))}) — cannot tell which is the result")
    return float(hits[-1]), text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args()

    runs = sorted(d for d in RUNS.glob("cmnext_*") if (d / "done.json").exists())
    if not runs:
        print("no finished arms yet")
        return 1
    print(f"{len(runs)} finished arm(s)\n")

    out = {}
    for d in runs:
        ck = find_ckpt(d)
        if ck is None:
            print(f"  SKIP {d.name}: no checkpoint written")
            continue
        out[d.name] = {"checkpoint": str(ck)}
        for cond in CONDITIONS:
            miou, _ = evaluate_one(ck, cond, a.gpu)
            out[d.name][cond] = miou
            print(f"  {d.name:28s} {cond:10s} mIoU = {miou}")

    p = RUNS / "eval.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
