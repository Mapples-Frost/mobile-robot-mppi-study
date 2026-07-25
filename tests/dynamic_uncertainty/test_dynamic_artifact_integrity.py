from pathlib import Path

import yaml

from experiments.dynamic_uncertainty.run_probability_smoke import (
    DEFAULT_CONFIG,
    ROOT,
    build_schedule,
    run_smoke,
)
from mobile_robot_mppi.obstacles.artifacts import validate_split_registry
from mobile_robot_mppi.obstacles.motion import PROCESS_NAMES


def test_seed_registry_is_disjoint_and_smoke_is_development_only():
    path = ROOT / "configs/seeds/dynamic_uncertainty_splits.yaml"
    with path.open("r", encoding="utf-8") as handle:
        registry = yaml.safe_load(handle)
    validate_split_registry(registry)
    assert not registry["sealed_registry"]["allowed_in_development"]


def test_schedule_has_exactly_40_unique_paired_jobs():
    schedule = build_schedule(
        PROCESS_NAMES,
        ("low", "medium"),
        (730100001, 730100002, 730100003, 730100004, 730100005),
        730199999,
    )
    assert len(schedule) == 40
    assert len({row["experimental_key"] for row in schedule}) == 40


def test_smoke_writes_complete_artifact_tree(tmp_path):
    result = run_smoke(DEFAULT_CONFIG, output_override=tmp_path / "smoke")
    assert result["audit"]["pass"]
    assert result["audit"]["completed_runs"] == 40
    assert not result["audit"]["sealed_seed_used"]
    assert result["audit"]["missing_required_files"] == []
    assert (
        Path(result["output_dir"]) / "figures/obstacle_processes_smoke.png"
    ).is_file()
