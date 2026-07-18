import pytest

from mobile_robot_mppi.rl.competence import (
    ProgressCompetenceConfig,
    ProgressCompetenceGate,
)


def test_progress_gate_waits_for_coverage_and_activates_on_stagnation():
    gate = ProgressCompetenceGate({
        "observation_window_s": 1.0,
        "minimum_window_coverage_s": 1.0,
        "full_activation_progress_m": 0.04,
        "zero_activation_progress_m": 0.20,
        "activation_hold_s": 0.5,
    })
    first = gate.update(0.0, 5.0)
    assert not first.ready
    assert first.stagnation_activation == 0.0
    stagnant = gate.update(1.0, 4.99)
    assert stagnant.ready
    assert stagnant.progress_m == pytest.approx(0.01)
    assert stagnant.stagnation_activation == 1.0


def test_progress_gate_does_not_activate_when_goal_progress_is_adequate():
    gate = ProgressCompetenceGate({
        "observation_window_s": 1.0,
        "minimum_window_coverage_s": 1.0,
        "full_activation_progress_m": 0.04,
        "zero_activation_progress_m": 0.20,
        "activation_hold_s": 0.0,
    })
    gate.update(0.0, 5.0)
    result = gate.update(1.0, 4.70)
    assert result.ready
    assert result.progress_m == pytest.approx(0.30)
    assert result.stagnation_activation == 0.0


def test_progress_gate_hold_prevents_one_frame_chatter_and_reset_clears_state():
    gate = ProgressCompetenceGate({
        "observation_window_s": 1.0,
        "minimum_window_coverage_s": 1.0,
        "full_activation_progress_m": 0.04,
        "zero_activation_progress_m": 0.20,
        "activation_hold_s": 1.0,
    })
    gate.update(0.0, 5.0)
    gate.update(1.0, 4.99)
    held = gate.update(1.5, 4.50)
    assert held.held
    assert held.stagnation_activation == 1.0
    released = gate.update(2.1, 4.30)
    assert not released.held
    assert released.stagnation_activation == 0.0
    gate.reset()
    reset = gate.update(3.0, 4.30)
    assert not reset.ready
    assert reset.stagnation_activation == 0.0


@pytest.mark.parametrize(
    "mapping,match",
    [
        ({"observation_window_s": 0.0}, "window"),
        (
            {
                "observation_window_s": 1.0,
                "minimum_window_coverage_s": 2.0,
            },
            "coverage",
        ),
        (
            {
                "full_activation_progress_m": 0.2,
                "zero_activation_progress_m": 0.1,
            },
            "zero-activation",
        ),
        ({"activation_hold_s": -1.0}, "hold"),
    ],
)
def test_progress_gate_config_fails_closed(mapping, match):
    config = ProgressCompetenceConfig.from_mapping(mapping)
    with pytest.raises(ValueError, match=match):
        config.validate()


def test_progress_gate_rejects_non_monotonic_timestamps():
    gate = ProgressCompetenceGate()
    gate.update(1.0, 2.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        gate.update(1.0, 2.0)
