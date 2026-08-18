#!/usr/bin/env python3
"""Write `done.json` for a run that finished WITHOUT its driver watching.

Needed by the Gate 0 handover (`plan/P0-RESUME.md`). The queue driver writes `done.json`
when it reaps a child; if the driver is stopped while its children keep running, those runs
finish untracked. A relaunched driver treats a directory without `done.json` as PARTIAL and
re-runs it, overwriting ~10 GPU-h of completed training. This closes that hole.

It refuses to mark a run done unless the log shows the final epoch was actually reached, so
it cannot be used to paper over a crashed run.

    python3 scripts/mark_done.py results/p0/mainB_baseline_s20260801
"""
import json
import re
import sys
from pathlib import Path

EVAL_LINE = re.compile(r"Epoch (\d+) validation result: mIoU ([0-9.]+)")


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        d = Path(arg)
        done = d / "done.json"
        if done.exists():
            print(f"SKIP {d.name}: done.json already exists")
            continue

        cmd = json.loads((d / "cmd.json").read_text())
        log = (d / "train.log").read_text(errors="ignore")
        evals = EVAL_LINE.findall(log)
        if not evals:
            raise SystemExit(f"{d.name}: no validation lines — run did not complete")
        final_epoch = int(evals[-1][0])
        planned = int(cmd["epochs"])

        # The last eval is NOT always the last epoch. `is_eval` (train.py:86) fires on
        # `epoch > checkpoint_start_epoch or epoch == 1 or epoch % 10 == 0`, and SUN sets
        # checkpoint_start_epoch=200 against a 75-epoch schedule -- so `epoch > 200` never
        # holds and the last eval lands at 70, not 75. NYU is unaffected (500 epochs vs a
        # 250 threshold, so it evals every epoch past 250 and ends exactly on `planned`).
        #
        # Requiring final_epoch == planned would therefore refuse every SUN run. Requiring
        # nothing would let a genuinely crashed run through. So compute the epoch the eval
        # cadence actually predicts, and check TRAINING reached `planned` separately.
        cse = 200 if cmd.get("dataset") == "SUNRGBD" else 250
        expected_last_eval = max(
            e for e in range(1, planned + 1)
            if e > cse or e == 1 or e % 10 == 0)
        if final_epoch != expected_last_eval:
            raise SystemExit(
                f"{d.name}: last eval at epoch {final_epoch}, cadence predicts "
                f"{expected_last_eval} (planned {planned}) — run is INCOMPLETE")
        # Independent check that training itself reached the final epoch, since the eval
        # cadence alone cannot show it.
        if f"Epoch {planned}/{planned}" not in log:
            raise SystemExit(
                f"{d.name}: no 'Epoch {planned}/{planned}' line — training did not reach "
                f"the planned final epoch and must not be marked done")

        stat = (d / "train.log").stat()
        hours = round((stat.st_mtime - (d / "cmd.json").stat().st_mtime) / 3600, 3)
        done.write_text(json.dumps(
            {"tag": d.name, "exit": 0, "gpu_hours": hours,
             "finished": "reconstructed from log mtime",
             "note": "marked by scripts/mark_done.py after a driver-less finish"},
            indent=2) + "\n")
        print(f"OK   {d.name}: epoch {final_epoch}/{planned}, {hours} GPU-h")


if __name__ == "__main__":
    main()
