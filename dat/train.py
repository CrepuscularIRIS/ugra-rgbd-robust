"""Single-GPU training for the DFormer family: conventional (C), RGB-only, DAT and the Table 3 ablations.

    python -m dat.train --host DFormerv2-S --preset DAT --pretrained <DFormerv2_Small_pretrained.pth> --out runs/dat_s
    python -m dat.train --host DFormerv2-S --preset RGB_ONLY --pretrained <...> --out runs/rgb_only_s
    python -m dat.train --host DFormerv2-S --preset DAT --pretrained <...> --steps 25000 --batch-size 16

Only the depth that enters the network changes between presets (see dat/presets.py); architecture, loss (one
pixelwise cross-entropy), optimizer, schedule and augmentation are the upstream ones, read from the upstream
config of each model (lr, batch size, epochs, warm-up). The default length is the upstream schedule
(`epochs * iters_per_epoch`); the paper fine-tunes DFormerv2-S for 25k steps with batch size 16.
fp32 by default: reduced precision is unstable for this architecture (docs/TROUBLESHOOTING.md).

`--pretrained` initialises the encoder from the upstream ImageNet RGB-D checkpoint (the original recipe);
`--init` instead warm-starts from a complete checkpoint (e.g. to build a reference model quickly).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch

from . import presets
from .availability import AvailabilitySampler


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--dataset", default="NYUDepthv2", choices=("NYUDepthv2", "SUNRGBD"))
    ap.add_argument("--preset", default="DAT", choices=sorted(presets.PRESETS))
    ap.add_argument("--pretrained", help="upstream pretrained encoder checkpoint (original training recipe)")
    ap.add_argument("--init", help="complete checkpoint to warm-start from")
    ap.add_argument("--steps", type=int, help="total optimizer steps (default: upstream epochs x iters/epoch)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=0, help="also save every N steps (0: only at the end)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-final-eval", action="store_true")
    a = ap.parse_args(argv)

    from .data import train_loader
    from .evaluate import evaluate
    from .data import eval_loader
    from .hosts.dformer import make_host

    seed_everything(a.seed)
    os.makedirs(a.out, exist_ok=True)
    host = make_host(a.host, a.dataset, pretrained=a.pretrained)
    sched = host.schedule()
    bs = a.batch_size or sched["batch_size"]
    lr = a.lr or sched["lr"]
    iters_per_epoch = sched["num_train"] // bs + 1
    total = a.steps or sched["epochs"] * iters_per_epoch
    warmup = sched["warmup_epochs"] * iters_per_epoch if not a.steps else min(total // 10, sched["warmup_epochs"] * iters_per_epoch)

    from utils.init_func import group_weight
    from utils.lr_policy import WarmUpPolyLR

    model = host.build(device=a.device)
    if a.init:
        host.load_weights(model, a.init)
    groups = group_weight([], model, torch.nn.BatchNorm2d, lr)
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), weight_decay=sched["weight_decay"])
    policy = WarmUpPolyLR(lr, 0.9, total, warmup)

    mixture = presets.get(a.preset)
    sampler = AvailabilitySampler(c=host.empty_value, mixture=mixture, seed=a.seed)
    loader = train_loader(host, bs, a.workers)
    json.dump({**vars(a), "mixture": mixture.as_tuple(), "batch_size": bs, "lr": lr, "total_steps": total,
               "warmup_steps": warmup}, open(os.path.join(a.out, "args.json"), "w"), indent=1)
    print(f"{a.host} / {a.dataset} / {a.preset} {mixture.as_tuple()}  steps {total}  bs {bs}  lr {lr}  "
          f"warmup {warmup}", flush=True)

    def save(name):
        torch.save({"state_dict": model.state_dict(), "step": step, "preset": a.preset, "host": a.host},
                   os.path.join(a.out, name))

    step, t0, run_loss, conds = 0, time.time(), 0.0, Counter()
    model.train()
    while step < total:
        for batch in loader:
            if step >= total:
                break
            rgb = batch["data"].to(a.device, non_blocking=True)
            depth = batch["modal_x"].to(a.device, non_blocking=True)
            label = batch["label"].to(a.device, non_blocking=True)
            depth_in, cs = sampler(depth)  # the only thing DAT changes (Algorithm 1)
            loss = host.loss(model, rgb, depth_in, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            cur = policy.get_lr(step)
            for g in opt.param_groups:
                g["lr"] = cur
            opt.step()
            step += 1
            run_loss += float(loss)
            conds.update(cs)
            if step % a.log_every == 0 or step == total:
                print(f"step {step:6d}/{total}  loss {run_loss / a.log_every:.4f}  lr {cur:.2e}  "
                      f"{step / (time.time() - t0):.2f} it/s  conditions {dict(conds)}", flush=True)
                run_loss, conds = 0.0, Counter()
            if a.save_every and step % a.save_every == 0:
                save("ckpt_last.pth")
    save("final.pth")
    print("saved", os.path.join(a.out, "final.pth"), flush=True)

    if not a.no_final_eval:
        res = evaluate(host, model, eval_loader(host), ("complete", "missing"), device=a.device)
        summary = {k: v["mIoU"] for k, v in res.items()}
        print(f"test mIoU  complete {summary['complete']:.2f}  missing {summary['missing']:.2f}", flush=True)
        json.dump(summary, open(os.path.join(a.out, "final_eval.json"), "w"))


if __name__ == "__main__":
    main()
