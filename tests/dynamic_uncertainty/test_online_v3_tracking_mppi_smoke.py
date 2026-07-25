from pathlib import Path

from mobile_robot_mppi.core.config import load_yaml

from experiments.dynamic_uncertainty.run_online_v3_tracking_mppi_smoke import (
    _obstacle_trajectory,
    _route_points,
    _validate_scope,
    build_schedule,
    route_crossing_count,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs/research/"
    "mujoco_v3_probabilistic_crossing_smoke_amendment5.yaml"
)


def test_schedule_is_paired_complete_randomized_and_reproducible():
    seeds = (730100001, 730100003, 730100005)
    arms = ("risk_disabled", "risk_enabled")
    first = build_schedule(seeds, arms, 730199988)
    second = build_schedule(seeds, arms, 730199988)
    assert first == second
    assert len(first) == 6
    assert {
        (job["seed"], job["arm"]) for job in first
    } == {(seed, arm) for seed in seeds for arm in arms}
    assert [job["run_order"] for job in first] == list(range(6))


def test_amendment3_uses_registered_smoke_seeds_and_40s_crossings():
    config = load_yaml(CONFIG_PATH)
    scope = _validate_scope(config)
    assert scope["sealed_registry_opened"] is False
    assert scope["selected_smoke_seeds"] == [
        730100001,
        730100003,
        730100005,
    ]

    start, goal = _route_points(config)
    counts = [
        route_crossing_count(
            _obstacle_trajectory(config, seed),
            start,
            goal,
            duration_s=40.0,
        )
        for seed in scope["selected_smoke_seeds"]
    ]
    assert counts == [5, 3, 4]
