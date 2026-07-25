import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from mobile_robot_mppi.obstacles.motion import (
    NoiseProfile,
    noise_profiles_from_mapping,
)
from mobile_robot_mppi.obstacles.patrol import (
    GENERATOR_VERSION_V3,
    V3_PROCESS_NAMES,
    audit_patrol_trajectory,
    generate_patrol_trajectory,
    validate_v3_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/research/dynamic_obstacle_process_v3.yaml"
SCRIPT_PATH = (
    ROOT
    / "experiments/dynamic_uncertainty/"
    "run_obstacle_generator_v3_qualification.py"
)


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_v3_config(value)
    return value


@pytest.fixture(scope="module")
def profiles(config):
    return noise_profiles_from_mapping(config["noise_profiles"])


def _load_qualification_module():
    spec = importlib.util.spec_from_file_location("v3_qualification", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("process", V3_PROCESS_NAMES)
def test_v3_is_exactly_reproducible(config, profiles, process):
    first = generate_patrol_trajectory(
        process, 730100021, profiles["medium"], config
    )
    second = generate_patrol_trajectory(
        process, 730100021, profiles["medium"], config
    )
    np.testing.assert_array_equal(first.states, second.states)
    np.testing.assert_array_equal(first.observations, second.observations)
    assert first.metadata == second.metadata
    assert first.metadata["generator_version"] == GENERATOR_VERSION_V3


@pytest.mark.parametrize("process", V3_PROCESS_NAMES)
@pytest.mark.parametrize("noise_name", ("low", "medium"))
def test_v3_representative_runs_obey_all_physical_limits(
    config, profiles, process, noise_name
):
    trajectory = generate_patrol_trajectory(
        process, 730100021, profiles[noise_name], config
    )
    audit = audit_patrol_trajectory(trajectory, config)
    assert audit["finite_truth"]
    assert audit["finite_available_observations"]
    assert audit["speed_within_limit"]
    assert audit["acceleration_within_limit"]
    assert audit["yaw_rate_within_limit"]
    assert audit["teleport_free"]


def test_every_development_shuttle_has_true_recurrence(config, profiles):
    for seed in range(730100021, 730100041):
        trajectory = generate_patrol_trajectory(
            "stochastic_shuttle", seed, profiles["medium"], config
        )
        audit = audit_patrol_trajectory(trajectory, config)
        assert audit["completed_legs"] >= 3
        assert audit["distinct_waypoints_visited"] == 2
        assert audit["round_trip_observed"]
        assert audit["waypoint_visits"][:4] == [0, 1, 0, 1]


def test_branching_patrol_has_route_diversity_and_waypoint_coverage(
    config, profiles
):
    routes = set()
    waypoints = set()
    for seed in range(730100021, 730100041):
        trajectory = generate_patrol_trajectory(
            "branching_patrol", seed, profiles["medium"], config
        )
        audit = audit_patrol_trajectory(trajectory, config)
        assert audit["completed_legs"] >= 4
        assert audit["distinct_waypoints_visited"] >= 3
        route = tuple(audit["waypoint_visits"])
        routes.add(route)
        waypoints.update(route)
    assert len(routes) >= 10
    assert waypoints == set(range(6))


def test_hybrid_patrol_is_not_a_fixed_event_template(config, profiles):
    intervention_kinds = set()
    first_times = set()
    routes = set()
    dropout_counts = set()
    for seed in range(730100021, 730100041):
        trajectory = generate_patrol_trajectory(
            "hybrid_patrol", seed, profiles["medium"], config
        )
        audit = audit_patrol_trajectory(trajectory, config)
        assert audit["completed_legs"] >= 3
        assert audit["intervention_count"] >= 2
        assert 2 <= audit["dropout_interval_count"] <= 5
        intervention_kinds.update(audit["intervention_kinds"])
        routes.add(tuple(audit["waypoint_visits"]))
        dropout_counts.add(audit["dropout_interval_count"])
        interventions = [
            event
            for event in trajectory.metadata["events"]
            if event["kind"]
            in ("early_retarget", "hesitation_stop", "speed_replan")
        ]
        first_times.add(float(interventions[0]["time_s"]))
    assert intervention_kinds == {
        "early_retarget",
        "hesitation_stop",
        "speed_replan",
    }
    assert len(first_times) >= 10
    assert len(routes) >= 10
    assert len(dropout_counts) >= 3


def test_observation_noise_changes_measurements_only(config):
    low_observation = NoiseProfile(
        "obs_low", acceleration_std=0.06, observation_std=0.01
    )
    high_observation = NoiseProfile(
        "obs_high", acceleration_std=0.06, observation_std=0.30
    )
    low = generate_patrol_trajectory(
        "hybrid_patrol", 730100029, low_observation, config
    )
    high = generate_patrol_trajectory(
        "hybrid_patrol", 730100029, high_observation, config
    )
    np.testing.assert_array_equal(low.states, high.states)
    np.testing.assert_array_equal(low.observed_mask, high.observed_mask)
    assert low.metadata["waypoint_visits"] == high.metadata["waypoint_visits"]
    assert not np.array_equal(
        np.nan_to_num(low.observations), np.nan_to_num(high.observations)
    )


def test_v3_schedule_is_unique_and_does_not_reuse_v1_v2_seeds(config):
    module = _load_qualification_module()
    seeds = tuple(range(730100021, 730100041))
    schedule = module.build_schedule(
        config["processes"],
        tuple(config["noise_profiles"]),
        seeds,
        config["schedule_seed"],
    )
    assert len(schedule) == 160
    assert len({job["experimental_key"] for job in schedule}) == 160
    assert all(job["seed"] >= 730100021 for job in schedule)


def test_v3_scope_keeps_every_algorithm_disabled(config):
    assert not any(config["scope_guards"].values())


def test_unknown_process_is_rejected(config, profiles):
    with pytest.raises(ValueError, match="unknown V3 obstacle process"):
        generate_patrol_trajectory(
            "arbitrary_process", 730100021, profiles["medium"], config
        )
