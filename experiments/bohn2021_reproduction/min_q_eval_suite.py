"""Evaluate every frozen comparator on the new registered banks."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from min_q_protocol import OUT, BASELINE_H, digest, protocol
from min_q_train_suite import verify_freeze
from runtime import ART, ROOT
from run import write


SCRIPTS = ROOT / "experiments/bohn2021_reproduction"


def model_dir(task, method, seed):
    if method == "min_q":
        return OUT / ("%s_min_q_s%d" % (task, seed))
    if method == "author_rl":
        return ART / "results/paper_defaults" / ("%s_rl_s%d" % (task, seed))
    assert method == "fixed" and seed in range(3)
    if seed:
        return OUT / ("%s_fixed_h%d_s%d" % (task, BASELINE_H[task], seed))
    group = "paper_defaults" if task == "pendulum" else "paper_exact_grid_2026-09-23"
    return ART / "results" / group / ("%s_fixed_h%d" % (task, BASELINE_H[task]))


JOBS = [(task, method, seed) for task in ("vehicle", "pendulum")
        for method in ("author_rl", "min_q", "fixed") for seed in range(3)]


def launch(split, job):
    task, method, seed = job
    source = model_dir(*job)
    done = json.loads((source / "completed.json").read_text())
    assert done["steps"] == 15000 and done["status"] == "complete"
    spec = json.loads((source / "manifest.json").read_text())
    assert spec["task"] == task and spec["seed"] == seed
    assert (spec["fixed_horizon"] is None) == (method != "fixed")
    if method == "min_q":
        assert spec["method_extension"]["source_sha256"] == protocol()["author_source_sha256"]
    else:
        assert "method_extension" not in spec
    bank = OUT / (task + "_" + split + "_bank.json")
    dest = OUT / "evaluations" / split / task / (method + "_s%d" % seed)
    if (dest / "completed.json").exists():
        result = json.loads((dest / "completed.json").read_text())
        assert result["model_hash"] == done["final_hash"] and result["bank"] == str(bank)
        return {"job": list(job), "status": "already_complete", "exit_code": 0}
    if dest.exists() and any(dest.iterdir()):
        raise RuntimeError("Incomplete evaluation needs inspection: " + str(dest))
    log = OUT / ("eval_%s_%s_%s_s%d.log" % (split, task, method, seed))
    cmd = [sys.executable, "-u", str(SCRIPTS / "evaluate_saved.py"),
           "--model-dir", str(source), "--bank", str(bank), "--out", str(dest)]
    with log.open("w") as stream:
        result = subprocess.run(cmd, cwd=str(ROOT), stdout=stream,
                                stderr=subprocess.STDOUT, timeout=7200)
    return {"job": list(job), "status": "complete" if result.returncode == 0 else "failed",
            "exit_code": result.returncode, "log": str(log.relative_to(ROOT))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("validation", "test"), required=True)
    args = ap.parse_args()
    verify_freeze()
    assert (OUT / "training_complete.json").exists()
    if args.split == "test":
        assert (OUT / "validation_audit_passed.json").exists(), "Audit validation before opening test outcomes"
    status = {"split": args.split, "registered_jobs": [list(j) for j in JOBS],
              "finished": [], "active": True, "workers": 4,
              "bank_hashes_sha256": digest(OUT / "bank_hashes.json")}
    write(OUT / (args.split + "_eval_status.json"), status)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(launch, args.split, job): job for job in JOBS}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                row = {"job": list(futures[future]), "status": "exception", "error": repr(error)}
            status["finished"].append(row)
            write(OUT / (args.split + "_eval_status.json"), status)
            print(json.dumps(row), flush=True)
    status["active"] = False
    status["complete"] = len(status["finished"]) == len(JOBS) and all(
        row.get("exit_code") == 0 for row in status["finished"])
    write(OUT / (args.split + "_eval_status.json"), status)
    if not status["complete"]:
        raise SystemExit(1)
    write(OUT / (args.split + "_eval_complete.json"), status)


if __name__ == "__main__":
    main()
