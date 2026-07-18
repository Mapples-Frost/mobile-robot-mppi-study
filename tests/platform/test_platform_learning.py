import numpy as np
import torch

from mobile_robot_mppi.learning.models import PlatformResidualDynamics, ResidualNetwork
from mobile_robot_mppi.learning.dataset_quality import (
    assert_episode_disjoint_splits,
    assert_residual_dataset_quality,
)
from mobile_robot_mppi.planning.dynamics import DynamicUnicyclePrediction, ResidualPrediction
from mobile_robot_mppi.planning.mppi import integrate_batch
from mobile_robot_mppi.core.spaces import dynamic_unicycle_state


def statistics():
    return {
        "feature_mean": np.zeros(6, dtype=np.float32),
        "feature_scale": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(2, dtype=np.float32),
        "control_scale": np.ones(2, dtype=np.float32),
        "residual_mean": np.zeros(5, dtype=np.float32),
        "residual_scale": np.ones(5, dtype=np.float32),
    }


def test_dynamic_five_icode_is_control_affine_and_batched():
    model = ResidualNetwork(
        5, 2,
        {"type": "icode_residual", "angle_indices": [2], "hidden_sizes": [16], "activation": "softplus"},
        statistics(),
    )
    state = torch.zeros((7, 5))
    control = torch.zeros((7, 2))
    residual, parts = model.components(state, control)
    assert residual.shape == (7, 5)
    assert parts["gain_normalized"].shape == (7, 5, 2)
    reconstructed = parts["drift_physical"] + parts["control_contribution_physical"]
    torch.testing.assert_close(residual, reconstructed)
    assert model.parameter_count() > 0


def test_icode_is_affine_in_raw_control_after_normalization():
    stats = statistics()
    stats["control_mean"] = np.asarray((0.2, -0.1), dtype=np.float32)
    stats["control_scale"] = np.asarray((0.5, 2.0), dtype=np.float32)
    model = ResidualNetwork(
        5, 2,
        {"type": "icode_residual", "angle_indices": [2], "hidden_sizes": [8]},
        stats,
    )
    state = torch.randn(4, 5)
    first = torch.randn(4, 2)
    second = torch.randn(4, 2)
    midpoint = 0.25 * first + 0.75 * second
    expected = 0.25 * model(state, first) + 0.75 * model(state, second)
    torch.testing.assert_close(model(state, midpoint), expected, rtol=1e-5, atol=1e-6)


def test_torchscript_residual_inference_matches_eager_cpu():
    model = ResidualNetwork(
        5, 2,
        {"type": "icode_residual", "angle_indices": [2], "hidden_sizes": [8]},
        statistics(),
    )
    eager = PlatformResidualDynamics(model, "cpu", use_torchscript=False)
    compiled = PlatformResidualDynamics(model, "cpu", use_torchscript=True)
    state = np.random.RandomState(2).normal(size=(7, 5))
    control = np.random.RandomState(3).normal(size=(7, 2))
    np.testing.assert_allclose(
        compiled.derivative(state, control), eager.derivative(state, control),
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_allclose(
        compiled.derivative(state[0], control[0]),
        eager.derivative(state[0], control[0]),
        rtol=1e-5, atol=1e-6,
    )


def test_five_state_residual_enters_rk4_rollout():
    class ZeroResidual:
        state_dim = 5
        control_dim = 2

        def derivative(self, state, control, time=None):
            del control, time
            return np.zeros_like(state)

    dynamics = ResidualPrediction(DynamicUnicyclePrediction(), ZeroResidual())
    state = np.zeros((3, 5))
    control = np.asarray(((0.2, 0.1),) * 3)
    following = integrate_batch(dynamics, state, control, 0.1, dynamic_unicycle_state(), "rk4")
    assert following.shape == (3, 5)
    assert np.all(following[:, 3] > 0.0)


def test_dataset_quality_gate_checks_label_identity_and_episode_leakage():
    state = np.zeros((4, 5), dtype=np.float64)
    following = state.copy()
    following[:, 0] = 0.1
    observed = np.zeros_like(state)
    observed[:, 0] = 1.0
    nominal = np.zeros_like(state)
    residual = observed - nominal
    mapping = {
        "episode_id": np.asarray(("a", "a", "b", "b")),
        "dt": np.full(4, 0.1),
        "state_t": state,
        "state_t_plus_1": following,
        "observed_derivative": observed,
        "nominal_derivative": nominal,
        "residual_target": residual,
        "control_t": np.zeros((4, 2)),
        "applied_control_t": np.zeros((4, 2)),
    }
    report = assert_residual_dataset_quality(mapping)
    assert report["residual_identity_max_abs"] == 0.0

    class Split:
        def __init__(self, values):
            self.episode_id = np.asarray(values)

    class Splits:
        train = Split(("a",))
        validation = Split(("b",))
        test = Split(())
        unseen = Split(())

    audit = assert_episode_disjoint_splits(Splits())
    assert audit["episode_counts"] == {"train": 1, "validation": 1, "test": 0, "unseen": 0}
