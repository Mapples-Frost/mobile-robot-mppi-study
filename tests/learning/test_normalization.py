import json

import numpy as np

from src.dynamics import StateEncoder
from src.learning.normalization import NormalizerBundle, StandardNormalizer
from src.learning.residual_dataset import ResidualDataset


def _training_dataset_with_extreme_holdout():
    state_t = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [2.0, 3.0, np.pi / 2.0],
            [1_000.0, 2_000.0, np.pi],
        ]
    )
    control = np.asarray([[1.0, -1.0], [3.0, 1.0], [900.0, 800.0]])
    residual = np.asarray(
        [[0.1, -0.2, 0.3], [0.5, 0.2, -0.1], [700.0, -600.0, 500.0]]
    )
    size = len(state_t)
    dt = np.full(size, 0.1)
    nominal = np.zeros((size, 3))
    observed = nominal + residual
    return ResidualDataset.from_mapping(
        {
            "episode_id": np.asarray(["train-0", "train-1", "test-0"]),
            "seed": np.asarray([0, 1, 2], dtype=np.int64),
            "step": np.zeros(size, dtype=np.int64),
            "time": np.zeros(size),
            "dt": dt,
            "state_t": state_t,
            "control_t": control,
            "applied_control_t": control.copy(),
            "state_t_plus_1": state_t + dt[:, None] * observed,
            "nominal_derivative": nominal,
            "observed_derivative": observed,
            "residual_target": residual,
            "disturbance_type": np.asarray(["seen", "seen", "held-out"]),
            "disturbance_parameters": np.asarray(["{}", "{}", "{}"]),
            "scene": np.asarray(["empty", "empty", "empty"]),
            "data_source": np.asarray(["pytest", "pytest", "pytest"]),
            "model_version": np.asarray(["v0", "v0", "v0"]),
        }
    )


def test_standard_normalizer_transform_inverse_roundtrip():
    values = np.asarray(
        [[-2.0, 10.0, 0.5], [0.0, 14.0, 1.5], [5.0, 18.0, -4.0]]
    )
    normalizer = StandardNormalizer().fit(values)

    normalized = normalizer.transform(values)

    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        normalizer.inverse_transform(normalized), values, atol=1e-15
    )
    np.testing.assert_allclose(normalizer.inverse(normalized), values, atol=1e-15)


def test_constant_feature_stays_finite_and_roundtrips_exactly():
    values = np.asarray([[1.0, 7.5], [3.0, 7.5], [5.0, 7.5]])
    normalizer = StandardNormalizer().fit(values)

    normalized = normalizer.transform(values)

    assert np.all(np.isfinite(normalized))
    np.testing.assert_allclose(normalized[:, 1], 0.0)
    np.testing.assert_allclose(normalizer.inverse_transform(normalized), values)


def test_standard_normalizer_state_dict_is_json_safe_and_restores_behavior():
    values = np.asarray([[1.0, -2.0], [4.0, 8.0], [10.0, 3.0]])
    original = StandardNormalizer().fit(values)

    state = original.state_dict()
    json.dumps(state)
    restored = StandardNormalizer.from_state_dict(state)

    probes = np.asarray([[-5.0, 0.25], [100.0, -50.0]])
    np.testing.assert_allclose(restored.transform(probes), original.transform(probes))
    np.testing.assert_allclose(
        restored.inverse_transform(original.transform(probes)), probes
    )
    assert restored.to_dict() == state


def test_normalizer_bundle_fit_uses_only_supplied_train_indices():
    dataset = _training_dataset_with_extreme_holdout()
    train_indices = np.asarray([0, 1], dtype=np.int64)
    train_dataset = dataset.subset(train_indices)
    encoder = StateEncoder(mode="sincos")

    bundle = NormalizerBundle.fit(train_dataset, state_encoder=encoder)

    encoded_train = encoder.encode(dataset.state_t[train_indices])
    normalized_states = bundle.transform_state(dataset.state_t[train_indices])
    normalized_controls = bundle.transform_control(
        dataset.control_t[train_indices]
    )
    normalized_residuals = bundle.transform_residual(
        dataset.residual_target[train_indices]
    )
    np.testing.assert_allclose(normalized_states.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalized_controls.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(normalized_residuals.mean(axis=0), 0.0, atol=1e-12)

    # If the extreme held-out row leaked into fitting, these recovered train means
    # would be close to the full-dataset means instead of the train-only means.
    np.testing.assert_allclose(
        bundle.inverse_state(normalized_states), encoded_train, atol=1e-15
    )
    np.testing.assert_allclose(
        bundle.inverse_control(normalized_controls),
        dataset.control_t[train_indices],
    )
    np.testing.assert_allclose(
        bundle.inverse_residual(normalized_residuals),
        dataset.residual_target[train_indices],
    )

    state = bundle.state_dict()
    json.dumps(state)
    restored = NormalizerBundle.from_state_dict(state)
    np.testing.assert_allclose(
        restored.transform_state(dataset.state_t[train_indices]), normalized_states
    )
