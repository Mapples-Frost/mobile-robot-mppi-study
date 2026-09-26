"""Decompose the completed paper-grid validation comparison without using holdout."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "research_artifacts/bohn2021_reproduction_2026-09-17"
OLD = ART / "results/paper_defaults"
GRID = ART / "results/paper_exact_grid_2026-09-23"
OUT = ART / "results/paper_exact_grid_2026-09-23/validation_diagnosis.json"
RUNS = {
    "vehicle": (GRID / "vehicle_fixed_h25", [OLD / ("vehicle_rl_s%d" % i) for i in range(3)]),
    "pendulum": (OLD / "pendulum_fixed_h30", [OLD / ("pendulum_rl_s%d" % i) for i in range(3)]),
}


def read(path):
    return json.loads(path.read_text())


def inspect(folder):
    directory = folder / "eval_value"
    summary_path = directory / "summary.json"
    summary = read(summary_path)
    episodes = []
    sources = {str(summary_path.relative_to(ROOT)): hashlib.sha256(summary_path.read_bytes()).hexdigest()}
    for row in summary["episodes"]:
        trace_path = directory / ("trace_%02d.json" % row["episode"])
        trace = read(trace_path)
        sources[str(trace_path.relative_to(ROOT))] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        assert len(trace) == row["steps"]
        parts = {
            "performance": sum(t["performance"] for t in trace),
            "H": sum(t["compute"] for t in trace),
            "constraint": sum(t["constraint"] for t in trace),
        }
        for key, source in [("performance", "performance_cost"),
                            ("H", "computation_cost"), ("constraint", "constraint_cost")]:
            assert abs(parts[key] - row[source]) < 1e-6
        assert abs(sum(parts.values()) - row["total_cost"]) < 1e-6
        episodes.append({
            "scene": row["episode"], "steps": row["steps"], "termination": row["termination"],
            "cost": row["total_cost"], "cost_parts": parts,
            "solver_failures": sum(not t["solver_success"] for t in trace),
            "H_counts": {
                "1_9": sum(t["horizon"] < 10 for t in trace),
                "10_19": sum(10 <= t["horizon"] < 20 for t in trace),
                "20_29": sum(20 <= t["horizon"] < 30 for t in trace),
                "30_39": sum(30 <= t["horizon"] < 40 for t in trace),
                "40_50": sum(t["horizon"] >= 40 for t in trace),
            },
        })
    assert len(episodes) == 10
    assert abs(sum(e["cost"] for e in episodes) / 10 - summary["mean_total_cost"]) < 1e-6
    return episodes, sources


def main():
    result = {
        "scope": "Frozen 10-scene validation only; no holdout input or new simulation.",
        "tasks": {}, "source_sha256": {},
    }
    for task, (fixed_dir, rl_dirs) in RUNS.items():
        fixed, hashes = inspect(fixed_dir)
        result["source_sha256"].update(hashes)
        task_rows = {"fixed": fixed, "seeds": {}}
        for seed, folder in enumerate(rl_dirs):
            episodes, hashes = inspect(folder)
            result["source_sha256"].update(hashes)
            paired = []
            for rl, base in zip(episodes, fixed):
                assert rl["scene"] == base["scene"]
                paired.append({"scene": rl["scene"], "total": rl["cost"] - base["cost"],
                               "performance": rl["cost_parts"]["performance"] - base["cost_parts"]["performance"],
                               "H": rl["cost_parts"]["H"] - base["cost_parts"]["H"],
                               "constraint": rl["cost_parts"]["constraint"] - base["cost_parts"]["constraint"]})
            means = {key: sum(row[key] for row in paired) / len(paired)
                     for key in ["total", "performance", "H", "constraint"]}
            assert abs(means["total"] - sum(means[k] for k in ["performance", "H", "constraint"])) < 1e-6
            task_rows["seeds"][str(seed)] = {"episodes": episodes, "paired_differences": paired,
                                             "mean_difference": means}
        result["tasks"][task] = task_rows
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
