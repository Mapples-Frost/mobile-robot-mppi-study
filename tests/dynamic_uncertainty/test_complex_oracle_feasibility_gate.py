import numpy as np

from experiments.dynamic_uncertainty.audit_complex_oracle_feasibility_timing_gate import (
    _behavior_family,
    _expand_segments,
    _structured_parameters,
)
from mobile_robot_mppi.core.spaces import ActionSpec


def _action_spec():
    return ActionSpec(
        names=("v_cmd", "omega_cmd"),
        lower=np.asarray((-0.35, -0.90)),
        upper=np.asarray((0.70, 0.95)),
        rate_limits=np.asarray((0.60, 2.0)),
    )


def test_structured_oracle_search_is_deterministic_and_bounded():
    first_labels, first = _structured_parameters(_action_spec(), 2400, 417041)
    second_labels, second = _structured_parameters(_action_spec(), 2400, 417041)

    assert first_labels == second_labels
    np.testing.assert_allclose(first, second)
    assert first.shape == (2400, 8)
    assert np.all(first[:, 0::2] >= -0.35)
    assert np.all(first[:, 0::2] <= 0.70)
    assert np.all(first[:, 1::2] >= -0.90)
    assert np.all(first[:, 1::2] <= 0.95)


def test_four_stage_expansion_covers_entire_horizon():
    _, parameters = _structured_parameters(_action_spec(), 12, 417041)
    controls = _expand_segments(parameters, 80)

    assert controls.shape == (12, 80, 2)
    for start, end in ((0, 20), (20, 40), (40, 60), (60, 80)):
        np.testing.assert_allclose(
            controls[:, start:end],
            np.repeat(
                controls[:, start:start + 1],
                end - start,
                axis=1,
            ),
        )


def test_behavior_labels_collapse_to_three_training_families():
    assert _behavior_family("left") == "left"
    assert _behavior_family("s_right") == "right"
    assert _behavior_family("yield_left") == "yield"
    assert _behavior_family("cem2_yield_right_i1") == "yield"
