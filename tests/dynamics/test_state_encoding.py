import numpy as np
import pytest

from src.dynamics import StateEncoder


def test_raw_encoder_preserves_single_state_and_returns_copy():
    encoder = StateEncoder(mode="raw")
    state = np.asarray([1.0, -2.0, 0.7])

    encoded = encoder.encode(state)

    np.testing.assert_array_equal(encoded, state)
    assert encoded.shape == (3,)
    assert encoder.output_dim == 3
    assert not np.shares_memory(encoded, state)


def test_raw_encoder_preserves_batch_shape_and_values():
    encoder = StateEncoder(mode="raw")
    states = np.asarray([[1.0, -2.0, 0.7], [3.0, 4.0, -0.2]])

    encoded = encoder(states)

    np.testing.assert_array_equal(encoded, states)
    assert encoded.shape == (2, 3)
    assert not np.shares_memory(encoded, states)


def test_sincos_encoder_expands_single_periodic_state():
    encoder = StateEncoder(mode="sincos")
    theta = 0.7

    encoded = encoder([1.0, -2.0, theta])

    np.testing.assert_allclose(encoded, [1.0, -2.0, np.sin(theta), np.cos(theta)])
    assert encoded.shape == (4,)
    assert encoder.output_dim == encoder.feature_dim == 4


def test_sincos_encoder_expands_batch_along_last_axis():
    encoder = StateEncoder(mode="sincos")
    states = np.asarray([[1.0, -2.0, 0.0], [3.0, 4.0, np.pi / 2]])

    encoded = encoder.encode(states)

    np.testing.assert_allclose(
        encoded,
        [[1.0, -2.0, 0.0, 1.0], [3.0, 4.0, 1.0, 0.0]],
        atol=1e-14,
    )
    assert encoded.shape == (2, 4)


@pytest.mark.parametrize(
    "kwargs, exception",
    [
        ({"mode": "fourier"}, ValueError),
        ({"mode": 3}, TypeError),
        ({"state_dim": 0}, ValueError),
        ({"state_dim": 3, "angle_indices": (3,)}, ValueError),
        ({"state_dim": 3, "angle_indices": (2, 2)}, ValueError),
    ],
)
def test_encoder_constructor_rejects_invalid_configuration(kwargs, exception):
    with pytest.raises(exception):
        StateEncoder(**kwargs)


@pytest.mark.parametrize(
    "states",
    [
        np.zeros(2),
        np.zeros(4),
        np.zeros((2, 2)),
        np.zeros((1, 2, 3)),
        [0.0, np.nan, 0.0],
        [0.0, 0.0, np.inf],
        ["not-numeric", 0.0, 0.0],
    ],
)
def test_encoder_rejects_bad_shapes_and_nonfinite_or_nonnumeric_values(states):
    with pytest.raises(ValueError):
        StateEncoder(mode="sincos").encode(states)
