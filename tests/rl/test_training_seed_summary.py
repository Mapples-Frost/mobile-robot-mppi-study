import csv

import pytest

from experiments.rl.summarize_training_seed_ablation import (
    _parse_run,
    summarize,
)


FIELDS = [
    "success",
    "collision",
    "final_goal_distance",
    "trajectory_length",
    "minimum_clearance",
    "control_jerk",
    "safety_interventions",
    "planner_compute_ms_mean",
    "seed",
]


def _result(root, name, successes, seeds=(11, 12)):
    path = root / name
    path.mkdir()
    with (path / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for success, seed in zip(successes, seeds):
            writer.writerow({
                "success": bool(success),
                "collision": False,
                "final_goal_distance": 0.2 if success else 1.0,
                "trajectory_length": 4.0,
                "minimum_clearance": 0.3,
                "control_jerk": 0.1,
                "safety_interventions": 2,
                "planner_compute_ms_mean": 5.0,
                "seed": seed,
            })
    return path


def test_training_seed_summary_reports_mean_std_and_pooled_rate(tmp_path):
    first = _result(tmp_path, "first", (True, True))
    second = _result(tmp_path, "second", (False, False))
    records, aggregate = summarize([
        ("method", 1, first),
        ("method", 2, second),
    ])
    assert len(records) == 2
    result = aggregate["method"]
    assert result["success_rate_pooled"] == pytest.approx(0.5)
    assert result["success_rate_training_seed_mean"] == pytest.approx(0.5)
    assert result["success_rate_training_seed_std"] == pytest.approx(2 ** -0.5)


def test_training_seed_summary_rejects_inconsistent_evaluation_seeds(tmp_path):
    first = _result(tmp_path, "first", (True, True), seeds=(11, 12))
    second = _result(tmp_path, "second", (True, True), seeds=(11, 13))
    with pytest.raises(ValueError, match="inconsistent held-out seeds"):
        summarize([("method", 1, first), ("method", 2, second)])


def test_run_spec_parser_is_explicit():
    method, seed, path = _parse_run("l12=123=results/example")
    assert method == "l12"
    assert seed == 123
    assert path.name == "example"
    with pytest.raises(ValueError, match="METHOD=TRAINING_SEED"):
        _parse_run("ambiguous")
