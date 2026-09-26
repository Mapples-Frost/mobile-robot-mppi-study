"""Complete the paper-configuration fixed-H grid without changing prior runs.

The already audited ``paper_defaults`` directory contains the six RL models and
six fixed-H models.  This continuation trains only the fourteen missing fixed
H models in a separate directory, using the exact same run.py arguments.  It is
deliberately a small, resumable launcher: completed directories are skipped and
each child has its own lock/completion marker.
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runtime import ART, ROOT

OUT = ART / "results/paper_exact_grid_2026-09-23"
SCRIPTS = ROOT / "experiments/bohn2021_reproduction"
JOBS = [(task, h) for task, existing in [
    ("vehicle", {5, 10, 15}),
    ("pendulum", {20, 30, 40}),
] for h in range(5, 51, 5) if h not in existing]


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def run_one(job):
    task, horizon = job
    name = "%s_fixed_h%d" % (task, horizon)
    folder = OUT / name
    if (folder / "completed.json").exists():
        return {"name": name, "exit_code": 0, "skipped": True}
    log_path = OUT / (name + ".log")
    cmd = [
        sys.executable, "-u", str(SCRIPTS / "run.py"),
        "--task", task, "--seed", "0", "--steps", "15000",
        "--out", str(folder), "--aligned", "--scaled-obs",
        "--batch-size", "256", "--buffer-size", "1000000",
        "--ent-coef", "1.0", "--fixed-horizon", str(horizon),
        "--test-bank", str(ART / "configs" / (task + "_validation_bank.json")),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        code = subprocess.run(cmd, cwd=str(ROOT), stdout=log,
                              stderr=subprocess.STDOUT).returncode
    return {"name": name, "exit_code": code, "skipped": False}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "group": "paper_exact_grid_2026-09-23",
        "continuation_of": "results/paper_defaults",
        "jobs": [[t, 0, h] for t, h in JOBS],
        "training_steps": 15000,
        "batch_size": 256,
        "buffer_size": 1000000,
        "ent_coef": 1.0,
        "aligned": True,
        "scaled_obs": True,
        "fixed_horizons": list(range(5, 51, 5)),
        "seed": 0,
        "purpose": "Fill the fixed-H grid while preserving the audited paper_defaults runs.",
        "selection": "No selection on holdout; every H is retained.",
    }
    write(OUT / "protocol.json", protocol)
    done = []
    pending = []
    for job in JOBS:
        name = "%s_fixed_h%d" % job
        if (OUT / name / "completed.json").exists():
            done.append(name)
        else:
            pending.append(job)
    # Four workers matches the resource cap used for the original suite.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, pending))
    write(OUT / "status.json", {
        "complete": all(r["exit_code"] == 0 for r in results),
        "jobs": results,
        "already_complete": done,
        "timestamp": time.time(),
    })
    if not all(r["exit_code"] == 0 for r in results):
        raise SystemExit(1)
    write(OUT / "training_completed.json", {"complete": True, "jobs": results})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
