import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml

from mobile_robot_mppi.obstacles.motion import (
    PROCESS_NAMES,
    NoiseProfile,
    noise_profiles_from_mapping,
)
from mobile_robot_mppi.obstacles.state_machine import (
    GENERATOR_VERSION,
    audit_state_machine_trajectory,
    build_motion_program,
    generate_state_machine_trajectory,
    validate_v2_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/research/dynamic_obstacle_process_v2.yaml"
SCRIPT_PATH = (
    ROOT
    / "experiments/dynamic_uncertainty/"
    "run_obstacle_generator_v2_qualification.py"
)


@pytest.fixture(scope="module")
def config():
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_v2_config(value)
    return value


def _profiles(config):
    return noise_profiles_from_mapping(config["noise_profiles"])


def _load_qualification_module():
    spec = importlib.util.spec_from_file_location("v2_qualification", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("process", PROCESS_NAMES)
def test_v2_event_contracts_are_seeded_and_complete(config, process):
    module = _load_qualification_module()
    program = build_motion_program(process, 730100001, config)
    kinds = [event.kind for event in program.events]
    assert module._event_contract(process, kinds)
    assert program == build_motion_program(process, 730100001, config)
    assert GENERATOR_VERSION == config["generator_version"]


@pytest.mark.parametrize("process", PROCESS_NAMES)
@pytest.mark.parametrize("noise_name", ("low", "medium"))
def test_v2_trajectories_obey_hard_physical_limits(
    config, process, noise_name
):
    trajectory = generate_state_machine_trajectory(
        process, 730100001, _profiles(config)[noise_name], config
    )
    audit = audit_state_machine_trajectory(trajectory, config)
    assert audit["finite_truth"]
    assert audit["finite_available_observations"]
    assert audit["speed_within_limit"]
    assert audit["acceleration_within_limit"]
    assert audit["yaw_rate_within_limit"]
    position_steps = np.linalg.norm(
        np.diff(trajectory.states[:, :2], axis=0), axis=1
    )
    assert position_steps.max() <= (
        config["physical_limits"]["maximum_speed_mps"]
        * config["trajectory"]["dt"]
        + 1.0e-8
    )


def test_p4_has_two_missing_observation_intervals(config):
    trajectory = generate_state_machine_trajectory(
        "combined_change", 730100001, _profiles(config)["medium"], config
    )
    intervals = trajectory.metadata["dropout_intervals"]
    assert len(intervals) == config["observation"]["combined_dropout_count"]
    assert (~trajectory.observed_mask).sum() > 0
    assert np.isnan(
        trajectory.observations[~trajectory.observed_mask]
    ).all()


def test_observation_noise_cannot_change_hidden_truth(config):
    first = NoiseProfile("obs_low", acceleration_std=0.06, observation_std=0.01)
    second = NoiseProfile(
        "obs_high", acceleration_std=0.06, observation_std=0.30
    )
    low = generate_state_machine_trajectory(
        "combined_change", 730100004, first, config
    )
    high = generate_state_machine_trajectory(
        "combined_change", 730100004, second, config
    )
    np.testing.assert_array_equal(low.states, high.states)
    np.testing.assert_array_equal(low.observed_mask, high.observed_mask)
    assert not np.array_equal(
        np.nan_to_num(low.observations), np.nan_to_num(high.observations)
    )


def test_p4_distribution_covers_all_registered_event_variants(config):
    kinds = set()
    angles = set()
    resumes = set()
    for seed in range(730100001, 730100021):
        program = build_motion_program("combined_change", seed, config)
        for event in program.events:
            kinds.add(event.kind)
            if "angle_deg" in event.parameters:
                angles.add(float(event.parameters["angle_deg"]))
            if event.kind in ("restart_forward", "reverse"):
                resumes.add(event.kind)
    assert {"turn_left", "turn_right", "accelerate", "decelerate", "stop"} <= kinds
    assert resumes == {"restart_forward", "reverse"}
    assert angles == {-90.0, -45.0, 45.0, 90.0}


def test_primary_event_groups_remain_separated(config):
    minimum = float(config["event_program"]["minimum_group_separation_s"])
    for seed in range(730100001, 730100021):
        program = build_motion_program("combined_change", seed, config)
        primary_times = [
            event.time_s
            for event in program.events
            if event.kind not in ("restart_forward", "reverse")
        ]
        assert min(np.diff(primary_times)) >= minimum


def test_qualification_schedule_uses_160_unique_development_jobs(config):
    module = _load_qualification_module()
    profiles = tuple(config["noise_profiles"])
    seeds = tuple(range(730100001, 730100021))
    schedule = module.build_schedule(
        config["processes"], profiles, seeds, config["schedule_seed"]
    )
    assert len(schedule) == 160
    assert len({job["experimental_key"] for job in schedule}) == 160
    assert all(730100001 <= job["seed"] <= 730100020 for job in schedule)


def test_scope_guards_keep_all_algorithms_disabled(config):
    assert config["scope_guards"] == {
        "predictors_enabled": False,
        "imm_enabled": False,
        "collision_risk_enabled": False,
        "mppi_enabled": False,
        "rl_icode_hss_enabled": False,
        "sealed_registry_imported": False,
    }
