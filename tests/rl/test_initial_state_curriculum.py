import numpy as np
import pytest

from mobile_robot_mppi.rl.trainer import InitialStateCurriculum


def _config():
    return {
        "enabled": True,
        "phases": [
            {
                "name": "bootstrap",
                "until_step": 10,
                "scenes": {
                    "u_trap": [
                        {"name": "goal", "state": [3.0, 3.5, 0.0], "weight": 1.0},
                        {"name": "start", "state": [0.0, 0.0, 0.0], "weight": 0.0},
                    ]
                },
            },
            {
                "name": "full",
                "until_step": 20,
                "scenes": {
                    "u_trap": [
                        {"name": "start", "state": [0.0, 0.0, 0.0], "weight": 1.0},
                    ]
                },
            },
        ],
    }


def test_initial_state_curriculum_changes_distribution_by_step():
    curriculum = InitialStateCurriculum(_config())
    rng = np.random.RandomState(4)
    state, name, phase = curriculum.sample(0, "u_trap", rng)
    assert name == "goal"
    assert phase == "bootstrap"
    assert state.tolist() == [3.0, 3.5, 0.0]

    state, name, phase = curriculum.sample(10, "u_trap", rng)
    assert name == "start"
    assert phase == "full"


def test_initial_state_curriculum_does_not_override_unlisted_scene():
    curriculum = InitialStateCurriculum(_config())
    state, name, phase = curriculum.sample(
        0, "clean_single_obstacle", np.random.RandomState(1)
    )
    assert state is None
    assert name == "configured_initial"
    assert phase == "bootstrap"


def test_initial_state_curriculum_rejects_non_monotonic_phases():
    config = _config()
    config["phases"][1]["until_step"] = 5
    with pytest.raises(ValueError, match="strictly increase"):
        InitialStateCurriculum(config)
