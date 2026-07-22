import math
from pathlib import Path

import numpy as np

from mobile_robot_mppi.core.config import load_yaml


ROOT = Path(__file__).resolve().parents[2]
SEEDS = (20262501, 20262502, 20262503)


def test_l260_training_contract_is_small_budget_rl_only_and_isolated():
    output_dirs = []
    for index, seed in enumerate(SEEDS):
        path = ROOT / f"configs/rl/l260_corrected_tracking_seed{seed}.yaml"
        config = load_yaml(path)
        training = config["rl"]["training"]
        assert training["seed"] == seed
        assert training["validation_seed_base"] == 20263501 + index
        assert training["total_steps"] == 20000
        assert training["checkpoint_interval"] == 10000
        assert training["evaluation_interval"] == 10000
        assert training["save_replay_buffer"] is True
        assert training["device"] == "cuda"
        assert training["bc_anchor"]["enabled"] is False
        assert training["physics_domain_config"] is None
        assert len(training["scene_configs"]) == 6
        assert len(training["validation_scene_configs"]) == 3
        assert set(training["scene_configs"]).isdisjoint(
            training["validation_scene_configs"]
        )
        assert all("l260_curriculum/train" in name for name in training["scene_configs"])
        assert all("l260_curriculum/validation" in name for name in training["validation_scene_configs"])
        assert "l258_tracking_curriculum" not in str(config).lower()
        phases = training["curriculum"]["phases"]
        assert [phase["until_step"] for phase in phases] == [
            1666, 3332, 4998, 6664, 8330, 10000,
            11666, 13332, 14998, 16664, 18330, 20000,
        ]
        for cycle in (phases[:6], phases[6:]):
            assert [int(np.argmax(phase["scene_weights"])) for phase in cycle] == list(range(6))
            assert all(sum(phase["scene_weights"]) == 1 for phase in cycle)
        initial = training["initial_state_curriculum"]
        assert initial["enabled"] is True
        assert [phase["until_step"] for phase in initial["phases"]] == [10000, 20000]
        for phase in initial["phases"]:
            assert set(phase["scenes"]) == {
                "l260_train_s_bend", "l260_train_offset_hairpin",
                "l260_train_near_double_loop", "l260_train_curvature_ramp",
                "l260_train_corridor_switchback", "l260_train_compound_turns",
            }
            for scene, candidates in phase["scenes"].items():
                assert [candidate["name"] for candidate in candidates] == [
                    "start", "quarter", "middle", "three_quarter"
                ]
                scene_config = load_yaml(
                    ROOT / f"configs/research/l260_curriculum/train/{scene}.yaml"
                )
                points = np.asarray(scene_config["task"]["points"], dtype=float)
                segments = points[1:] - points[:-1]
                for candidate in candidates:
                    state = np.asarray(candidate["state"], dtype=float)
                    assert state.shape == (5,)
                    assert np.isfinite(state).all()
                    starts = points[:-1]
                    squared = np.sum(segments ** 2, axis=1)
                    fractions = np.clip(
                        np.sum((state[:2] - starts) * segments, axis=1) / squared,
                        0.0, 1.0,
                    )
                    projections = starts + fractions[:, None] * segments
                    nearest = int(np.argmin(np.linalg.norm(projections - state[:2], axis=1)))
                    assert np.linalg.norm(projections[nearest] - state[:2]) < 2e-8
                    expected_yaw = math.atan2(segments[nearest, 1], segments[nearest, 0])
                    yaw_error = math.atan2(math.sin(state[2] - expected_yaw), math.cos(state[2] - expected_yaw))
                    assert abs(yaw_error) < 2e-8
        output_dirs.append(config["experiment"]["output_dir"])
    assert len(set(output_dirs)) == len(output_dirs)
