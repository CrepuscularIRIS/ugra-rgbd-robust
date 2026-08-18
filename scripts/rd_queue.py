#!/usr/bin/env python3
"""Launch the 2x2's bottom row: 2 arms x 3 seeds on CMNeXt, one job per GPU.

Pre-registered at plan/PREREG-removal-dropout.md. Restartable and idempotent: a run whose
done.json exists is skipped, so the driver can be killed and relaunched without losing work.

Measured cost from the smoke test: 0.05 GPU-h for 2 epochs => ~1 GPU-h per 40-epoch arm,
so ~6 GPU-h for the whole row.

  nohup python3 scripts/rd_queue.py run &
  python3 scripts/rd_queue.py status
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "results" / "removal_dropout"
PY = "/home/lingxufeng/miniconda3/envs/dformer/bin/python"
GPUS = [1, 0]                     # GPU1 primary, per the campaign rule
SEEDS = [20260801, 20260802, 20260803]
ARMS = ["removal", "zero"]
EPOCHS = 40


def jobs():
    # interleaved so an early stop still leaves a matched pair rather than three of one arm
    out = []
    for seed in SEEDS:
        for arm in ARMS:
            out.append({"arm": arm, "seed": seed, "tag": f"cmnext_{arm}_s{seed}"})
    return out


def launch(job, gpu):
    d = RUNS / job["tag"]
    d.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(PROJECT / "scripts" / "run_removal_dropout_arms.py"),
           "--arm", job["arm"], "--seed", str(job["seed"]),
           "--gpu", str(gpu), "--epochs", str(EPOCHS)]
    log = open(d / "train.log", "w")
    return subprocess.Popen(cmd, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)


def run():
    RUNS.mkdir(parents=True, exist_ok=True)
    # SINGLE-DRIVER LOCK. Two concurrent drivers put two jobs on one GPU and silently
    # contaminated the timings once; the second driver is the failure, not the port clash.
    lock = RUNS / "driver.lock"
    if lock.exists():
        pid = lock.read_text().strip()
        alive = Path(f"/proc/{pid}").exists()
        if alive:
            print(f"[rd] REFUSING: driver already running as pid {pid}")
            return
        print(f"[rd] stale lock from dead pid {pid}, taking over")
    lock.write_text(str(os.getpid()))
    try:
        _run()
    finally:
        lock.unlink(missing_ok=True)


def _run():
    RUNS.mkdir(parents=True, exist_ok=True)
    pending = [j for j in jobs() if not (RUNS / j["tag"] / "done.json").exists()]
    slots = {g: None for g in GPUS}
    print(f"[rd] {len(pending)} job(s) pending", flush=True)
    while pending or any(slots.values()):
        for gpu, cur in list(slots.items()):
            if cur is not None:
                job, proc, t0 = cur
                if proc.poll() is not None:
                    print(f"[rd] {job['tag']} exit={proc.returncode} "
                          f"{(time.time()-t0)/3600:.2f} GPU-h", flush=True)
                    slots[gpu] = None
            if slots[gpu] is None and pending:
                job = pending.pop(0)
                print(f"[rd] start {job['tag']} on GPU{gpu}", flush=True)
                slots[gpu] = (job, launch(job, gpu), time.time())
        time.sleep(20)
    print("[rd] all arms complete", flush=True)


def status():
    for j in jobs():
        d = RUNS / j["tag"]
        if (d / "done.json").exists():
            m = json.loads((d / "done.json").read_text())
            print(f"  DONE {j['tag']:28s} exit={m['exit']} {m['gpu_hours']}h "
                  f"drop={m.get('realized_drop_fraction', float('nan')):.4f}")
        elif d.exists():
            lg = d / "train.log"
            last = ""
            if lg.exists():
                ls = [x for x in lg.read_text(errors="ignore").splitlines() if "Epoch" in x]
                last = ls[-1][:80] if ls else "(starting)"
            print(f"  RUN  {j['tag']:28s} {last}")
        else:
            print(f"  WAIT {j['tag']}")


if __name__ == "__main__":
    (status if len(sys.argv) > 1 and sys.argv[1] == "status" else run)()
