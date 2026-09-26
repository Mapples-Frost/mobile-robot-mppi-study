"""Train the pre-registered min-Q and matched fixed-H models."""
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from min_q_protocol import OUT, BASELINE_H, digest, protocol
from runtime import ART, ROOT
from run import write


SCRIPTS = ROOT / "experiments/bohn2021_reproduction"
JOBS = [(task, "min_q", seed) for task in ("vehicle", "pendulum") for seed in range(3)]
JOBS += [(task, "fixed", seed) for task in ("vehicle", "pendulum") for seed in (1, 2)]


def run_dir(task, variant, seed):
    suffix = "min_q_s%d" % seed if variant == "min_q" else "fixed_h%d_s%d" % (BASELINE_H[task], seed)
    return OUT / (task + "_" + suffix)


def verify_freeze():
    p = OUT / "protocol.json"
    assert json.loads(p.read_text()) == protocol()
    hashes = json.loads((OUT / "bank_hashes.json").read_text())
    for name, expected in hashes.items():
        assert digest(ROOT / name) == expected, name
    assert len(hashes) == 5
    for task in ("vehicle", "pendulum"):
        for seed in range(3):
            old = ART / "results/paper_defaults" / (task + "_rl_s%d" % seed)
            assert json.loads((old / "completed.json").read_text())["steps"] == 15000
        source = "paper_defaults" if task == "pendulum" else "paper_exact_grid_2026-09-23"
        fixed = ART / ("results/%s/%s_fixed_h%d" % (source, task, BASELINE_H[task]))
        assert json.loads((fixed / "completed.json").read_text())["steps"] == 15000


def launch(job):
    task, variant, seed = job
    folder = run_dir(*job)
    if (folder / "completed.json").exists():
        completed = json.loads((folder / "completed.json").read_text())
        assert completed["steps"] == 15000 and completed["status"] == "complete"
        return {"job": list(job), "status": "already_complete", "exit_code": 0}
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError("Incomplete output needs inspection before retry: " + str(folder))
    bank = OUT / (task + "_validation_bank.json")
    script = SCRIPTS / ("min_q_run.py" if variant == "min_q" else "run.py")
    cmd = [sys.executable, "-u", str(script), "--task", task, "--seed", str(seed),
           "--steps", "15000", "--out", str(folder), "--aligned", "--scaled-obs",
           "--batch-size", "256", "--buffer-size", "1000000", "--ent-coef", "1.0",
           "--test-bank", str(bank)]
    if variant == "fixed":
        cmd += ["--fixed-horizon", str(BASELINE_H[task])]
    log = OUT / (folder.name + ".log")
    with log.open("w") as stream:
        result = subprocess.run(cmd, cwd=str(ROOT), stdout=stream,
                                stderr=subprocess.STDOUT, timeout=10800)
    return {"job": list(job), "status": "complete" if result.returncode == 0 else "failed",
            "exit_code": result.returncode, "log": str(log.relative_to(ROOT))}


def main():
    verify_freeze()
    status = {"registered_jobs": [list(j) for j in JOBS], "finished": [],
              "active": True, "workers": 4}
    write(OUT / "train_status.json", status)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(launch, job): job for job in JOBS}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                row = {"job": list(futures[future]), "status": "exception", "error": repr(error)}
            status["finished"].append(row)
            write(OUT / "train_status.json", status)
            print(json.dumps(row), flush=True)
    status["active"] = False
    status["complete"] = len(status["finished"]) == len(JOBS) and all(
        row.get("exit_code") == 0 for row in status["finished"])
    write(OUT / "train_status.json", status)
    if not status["complete"]:
        raise SystemExit(1)
    write(OUT / "training_complete.json", status)


if __name__ == "__main__":
    main()
