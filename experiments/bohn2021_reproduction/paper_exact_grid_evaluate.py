"""Evaluate the completed fixed-H continuation on the frozen 20-case holdout."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from runtime import ART, ROOT

OUT = ART / "results/paper_exact_grid_2026-09-23"
SCRIPTS = ROOT / "experiments/bohn2021_reproduction"
TASKS = ["vehicle", "pendulum"]


def run_one(args):
    folder, use_value = args
    mode = "holdout_value" if use_value else "holdout_no_value"
    dest = folder / mode
    if (dest / "completed.json").exists():
        return {"model": folder.name, "terminal_value": use_value, "exit_code": 0,
                "skipped": True}
    log = OUT / (folder.name + "_" + mode + ".log")
    cmd = [sys.executable, "-u", str(SCRIPTS / "evaluate_saved.py"),
           "--model-dir", str(folder),
           "--bank", str(ART / "configs" / (json.loads((folder / "manifest.json").read_text())["task"] + "_holdout_bank.json")),
           "--out", str(dest)]
    if not use_value:
        cmd.append("--no-value")
    with log.open("w", encoding="utf-8") as handle:
        code = subprocess.run(cmd, cwd=str(ROOT), stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    return {"model": folder.name, "terminal_value": use_value, "exit_code": code,
            "skipped": False}


def main():
    folders = [p for p in OUT.iterdir() if p.is_dir() and (p / "completed.json").exists()]
    jobs = [(folder, value) for folder in folders for value in [True, False]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, jobs))
    (OUT / "holdout_completed.json").write_text(json.dumps({
        "complete": all(r["exit_code"] == 0 for r in results), "jobs": results
    }, indent=2) + "\n")
    print(json.dumps(results, indent=2), flush=True)
    if not all(r["exit_code"] == 0 for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
