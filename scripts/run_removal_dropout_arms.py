"""The 2x2's bottom row: Zero-Dropout vs Named-Removal Dropout on CMNeXt (NON-removal).

Pre-registered at plan/PREREG-removal-dropout.md. R is FROZEN there and implemented in
scripts/removal_dropout_cmnext.py, whose bitwise canary must pass before this runs.

THE ONLY DIFFERENCE BETWEEN THE ARMS is where "depth is missing" is instantiated:

    zero     depth tensor -> zeros for dropped samples, fused as usual   (ConD-style, the standard)
    removal  the named removal R: FRM bypassed, x_cam to the decoder     (ours)

Everything else is shared by construction, because both arms run this same file with the same
seed, schedule, optimizer, LR, augmentation, dropout probability and checkpoint policy. The
per-sample mask is drawn from a generator seeded identically in both arms, so arm `zero` and arm
`removal` drop THE SAME SAMPLES ON THE SAME STEPS. That removes "they saw different dropout
patterns" as an explanation of any difference, which a shared probability alone would not.

    python3 scripts/run_removal_dropout_arms.py --arm removal --seed 20260801 --gpu 1
    python3 scripts/run_removal_dropout_arms.py --smoke --gpu 1        # 2 steps, no claim
"""
import argparse
import json
import os
import runpy
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "repos" / "DELIVER"
RUNS = PROJECT / "results" / "removal_dropout"
CKPT = "checkpoints/cmnext_nyu/NYU_Depth_V2/cmnext_b4_nyu_rgbd.pth"

P_DROP = 0.475          # matched to the DFormerv2 dropout arm's P(alpha=0)
MASK_SEED = 20260721    # SHARED across arms so both drop the same samples on the same steps


def install_arm(arm: str, seed: int):
    """Wrap the top-level model forward to instantiate 'depth missing' one of two ways."""
    sys.path.insert(0, str(REPO))
    import torch
    from semseg.models import cmnext as MODEL
    from semseg.models.backbones import cmnext as BACKBONE

    sys.path.insert(0, str(PROJECT / "scripts"))
    import removal_dropout_cmnext as RD
    RD.install(BACKBONE)                       # per-sample removal mask in the backbone

    # RESUME is a full TRAINING resume: it restores optimizer, scheduler, epoch and best_miou
    # (train_mm.py:71-76). That is not a fine-tune. `init_pretrained` is the hook that loads
    # weights only, leaving a fresh optimizer at epoch 0 — identical for both arms.
    def init_pretrained(self, pretrained=None):
        sd = torch.load(str(REPO / CKPT), map_location="cpu")
        sd = sd.get("model_state_dict", sd.get("state_dict", sd))
        res = self.load_state_dict(sd, strict=False)
        n = len(sd) - len(res.unexpected_keys)
        print(f"[init] audited checkpoint: {n}/{len(sd)} tensors adopted, "
              f"{len(res.unexpected_keys)} unconsumed, {len(res.missing_keys)} left at init")
        if res.unexpected_keys:
            raise SystemExit(f"REFUSING: {len(res.unexpected_keys)} checkpoint tensors did not "
                             f"load (e.g. {res.unexpected_keys[:3]}) — the arms would start from "
                             f"a partially-initialised model and nothing could be attributed")

    MODEL.CMNeXt.init_pretrained = init_pretrained

    # cal_flops assumes a DDP-wrapped model (`model.module`); we run single-process so it is not
    # wrapped. It only prints a FLOPs table and touches nothing the experiment depends on.
    import semseg.utils.utils as _U
    _U.cal_flops = lambda *a, **k: None


    gen = torch.Generator().manual_seed(MASK_SEED + seed)
    orig = MODEL.CMNeXt.forward
    stats = {"steps": 0, "dropped": 0, "samples": 0}

    def forward(self, x):
        if self.training and arm in ("zero", "removal"):
            B = x[0].shape[0]
            mask = torch.rand(B, generator=gen) < P_DROP
            stats["steps"] += 1
            stats["dropped"] += int(mask.sum())
            stats["samples"] += B
            if arm == "zero":
                RD.set_drop(None)
                x = [x[0], torch.where(mask.reshape(-1, 1, 1, 1).to(x[1].device),
                                       torch.zeros_like(x[1]), x[1])]
            else:
                RD.set_drop(mask)
        else:
            RD.set_drop(None)                  # eval always sees the real forward
        try:
            return orig(self, x)
        finally:
            RD.set_drop(None)

    MODEL.CMNeXt.forward = forward
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["zero", "removal", "none"], default="removal")
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--smoke", action="store_true",
                    help="2 steps only, to prove the harness runs. Produces NO reportable number.")
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    # train_mm.py is written for a torchrun launch and reads these unconditionally.
    for k, v in (("LOCAL_RANK", "0"), ("RANK", "0"), ("WORLD_SIZE", "1"),
                 ("MASTER_ADDR", "127.0.0.1"), ("MASTER_PORT", str(29820 + int(a.gpu) * 7 + a.seed % 5))):
        os.environ.setdefault(k, v)
    os.chdir(REPO)

    import yaml
    cfg = yaml.safe_load(open("configs/nyu_rgbd.yaml"))
    cfg["MODEL"]["RESUME"] = ""      # weights come via init_pretrained; see install_arm
    cfg["TRAIN"]["EPOCHS"] = 2 if a.smoke else a.epochs
    cfg["TRAIN"]["DDP"] = False
    cfg["TRAIN"]["EVAL_START"] = 10**6 if a.smoke else cfg["TRAIN"]["EVAL_START"]
    tag = f"cmnext_{a.arm}_s{a.seed}" + ("_SMOKE" if a.smoke else "")
    d = RUNS / tag
    d.mkdir(parents=True, exist_ok=True)
    cfg["SAVE_DIR"] = str(d / "out")

    tmp = REPO / f"configs/_rd_{tag}.yaml"
    tmp.write_text(yaml.safe_dump(cfg))

    import torch
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    stats = install_arm(a.arm, a.seed)
    (d / "cmd.json").write_text(json.dumps({
        "arm": a.arm, "seed": a.seed, "p_drop": P_DROP, "mask_seed": MASK_SEED,
        "epochs": cfg["TRAIN"]["EPOCHS"], "resume": CKPT, "smoke": a.smoke,
        "config": str(tmp)}, indent=2) + "\n")

    t0 = time.time()
    # train_mm.py does `from val_mm import evaluate` — a sibling import that only resolves
    # when tools/ is on the path, which `runpy.run_path` does not arrange.
    sys.path.insert(0, str(REPO / "tools"))
    sys.argv = ["train_mm.py", "--cfg", str(tmp)]
    try:
        runpy.run_path(str(REPO / "tools" / "train_mm.py"), run_name="__main__")
        rc = 0
    except SystemExit as e:
        rc = int(e.code or 0)

    frac = stats["dropped"] / max(1, stats["samples"])
    print(f"\n[{tag}] steps={stats['steps']} dropped={stats['dropped']}/{stats['samples']} "
          f"= {frac:.4f} (target {P_DROP})")
    (d / "done.json").write_text(json.dumps({
        "tag": tag, "exit": rc, "gpu_hours": round((time.time() - t0) / 3600, 3),
        "realized_drop_fraction": frac, **stats}, indent=2) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
