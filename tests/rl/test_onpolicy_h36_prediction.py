import numpy as np

from experiments.rl.analyze_onpolicy_h36_prediction import (
    _episode_windows,
    _wrapped_error,
)


def test_l39_windows_are_nonoverlapping_and_skip_collision():
    states = np.zeros((11, 5))
    controls = np.zeros((10, 2))
    collisions = np.zeros(10, dtype=bool)
    collisions[6] = True
    assert _episode_windows(states, controls, collisions, 5) == [(0, 5)]


def test_l39_heading_error_uses_shortest_arc():
    predicted = np.asarray([[[0.0, 0.0, np.pi - 0.01, 0.0, 0.0]]])
    target = np.asarray([[[0.0, 0.0, -np.pi + 0.01, 0.0, 0.0]]])
    error = _wrapped_error(predicted, target)
    assert np.isclose(error[0, 0, 2], -0.02)
