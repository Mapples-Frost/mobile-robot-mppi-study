import math

import numpy as np
import torch
from torch import nn

from src.dynamics.state_encoding import StateEncoder
from src.learning.normalization import NormalizerBundle
from src.learning.residual_losses import (
    CompositeResidualLoss,
    TorchNormalizerBundle,
    encode_state_torch,
    multistep_rollout_loss,
    one_step_prediction_loss,
    rollout_combined_dynamics,
    torch_integrate_step,
    wrapped_state_error,
)


class ZeroResidual(nn.Module):
    def forward(self, state_features, control):
        return state_features.new_zeros(state_features.shape[:-1] + (3,))


class ConstantResidual(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = nn.Parameter(torch.as_tensor(value, dtype=torch.float64))

    def forward(self, state_features, control):
        return self.value.expand(state_features.shape[:-1] + (self.value.numel(),))


def _identity(dtype=torch.float64):
    return TorchNormalizerBundle.identity(dtype=dtype)


def test_torch_normalizers_restore_bundle_state_and_sincos_is_differentiable():
    encoder = StateEncoder(mode="sincos")
    state = np.asarray(
        [[-1.0, 0.5, -2.0], [0.0, -0.5, 0.25], [2.0, 1.5, 2.5]],
        dtype=np.float64,
    )
    control = np.asarray([[-1.0, 0.5], [0.0, -0.25], [2.0, 1.0]])
    residual = np.asarray(
        [[-0.5, 0.0, 1.0], [0.25, -1.0, 0.0], [1.5, 2.0, -0.5]]
    )
    bundle = NormalizerBundle.fit(
        {"state_t": state, "control_t": control, "residual_target": residual},
        state_encoder=encoder,
    )
    torch_bundle = TorchNormalizerBundle.from_bundle(
        bundle, device="cpu", dtype=torch.float64
    )
    restored = TorchNormalizerBundle.from_state_dict(
        torch_bundle.state_dict(), device="cpu", dtype=torch.float64
    )

    state_tensor = torch.tensor(state, dtype=torch.float64, requires_grad=True)
    encoded = encode_state_torch(state_tensor, bundle.encoder_config)
    expected_encoded = torch.from_numpy(encoder.encode(state))
    torch.testing.assert_close(encoded, expected_encoded)
    torch.testing.assert_close(
        torch_bundle.normalize_state(state_tensor),
        torch.from_numpy(bundle.transform_state(state)),
    )
    torch.testing.assert_close(
        restored.normalize_control(torch.from_numpy(control)),
        torch.from_numpy(bundle.transform_control(control)),
    )
    normalized_residual = torch_bundle.normalize_residual(torch.from_numpy(residual))
    torch.testing.assert_close(
        torch_bundle.inverse_residual(normalized_residual), torch.from_numpy(residual)
    )

    # A non-constant combination avoids the sin^2 + cos^2 identity, whose
    # mathematically correct angle gradient is zero.
    (encoded[..., 2] + 2.0 * encoded[..., 3]).sum().backward()
    assert state_tensor.grad is not None
    assert torch.isfinite(state_tensor.grad).all()
    assert state_tensor.grad[:, 2].abs().sum() > 0.0


def test_zero_residual_combined_rollout_matches_straight_unicycle_motion():
    model = ZeroResidual()
    normalizers = _identity()
    initial = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -1.0, math.pi / 2.0]], dtype=torch.float64
    )
    controls = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.0], [2.0, 0.0]],
            [[0.25, 0.0], [1.0, 0.0], [0.5, 0.0]],
        ],
        dtype=torch.float64,
    )
    dt = torch.tensor([[0.1, 0.2, 0.05], [0.4, 0.1, 0.2]], dtype=torch.float64)

    trajectory = rollout_combined_dynamics(
        model, initial, controls, dt, normalizers, method="rk4"
    )

    distances = torch.cumsum(controls[..., 0] * dt, dim=1)
    expected = initial[:, None, :].expand(-1, controls.shape[1] + 1, -1).clone()
    expected[0, 1:, 0] += distances[0]
    expected[1, 1:, 1] += distances[1]
    torch.testing.assert_close(trajectory, expected, atol=1e-12, rtol=1e-12)


def test_rk4_integrates_constant_derivative_with_scalar_and_batch_dt():
    constant = torch.tensor([0.75, -0.5, 0.2], dtype=torch.float64)

    def derivative(state, held_control):
        return constant.expand_as(state)

    single_state = torch.tensor([1.0, 2.0, -0.4], dtype=torch.float64)
    single_control = torch.zeros(2, dtype=torch.float64)
    single = torch_integrate_step(
        derivative,
        single_state,
        single_control,
        0.2,
        method="rk4",
        angle_indices=(),
    )
    torch.testing.assert_close(single, single_state + 0.2 * constant)

    batch_state = torch.stack((single_state, -single_state))
    batch_control = torch.zeros(2, 2, dtype=torch.float64)
    batch_dt = torch.tensor([0.1, 0.35], dtype=torch.float64)
    batch = torch_integrate_step(
        derivative,
        batch_state,
        batch_control,
        batch_dt,
        method="rk4",
        angle_indices=(),
    )
    torch.testing.assert_close(
        batch, batch_state + batch_dt[:, None] * constant[None, :]
    )


def test_wrapped_theta_error_and_one_step_loss_use_short_periodic_difference():
    epsilon = 0.01
    prediction = torch.tensor([[0.0, 0.0, math.pi - epsilon]], dtype=torch.float64)
    target = torch.tensor([[0.0, 0.0, -math.pi + epsilon]], dtype=torch.float64)

    error = wrapped_state_error(prediction, target)

    torch.testing.assert_close(
        error, torch.tensor([[0.0, 0.0, -2.0 * epsilon]], dtype=torch.float64)
    )
    loss = one_step_prediction_loss(
        ZeroResidual(),
        prediction,
        torch.zeros(1, 2, dtype=torch.float64),
        torch.tensor([0.1], dtype=torch.float64),
        target,
        _identity(),
        method="rk4",
    )
    torch.testing.assert_close(
        loss, torch.tensor((2.0 * epsilon) ** 2 / 3.0, dtype=torch.float64)
    )


def _constant_residual_targets(initial, dt, residual):
    increments = dt[..., None] * residual
    return initial[:, None, :] + torch.cumsum(increments, dim=1)


def test_multistep_horizon_targets_are_exact_and_gradients_are_finite():
    normalizers = _identity()
    initial = torch.tensor(
        [[0.1, -0.2, 0.3], [-0.4, 0.5, -0.2]], dtype=torch.float64
    )
    horizon = 4
    controls = torch.zeros(2, horizon, 2, dtype=torch.float64)
    dt = torch.tensor(
        [[0.1, 0.2, 0.15, 0.05], [0.2, 0.1, 0.1, 0.25]], dtype=torch.float64
    )
    oracle_value = torch.tensor([0.4, -0.25, 0.2], dtype=torch.float64)
    targets = _constant_residual_targets(initial, dt, oracle_value)

    oracle = ConstantResidual(oracle_value)
    exact_loss, exact_trajectory = multistep_rollout_loss(
        oracle,
        initial,
        controls,
        dt,
        targets,
        normalizers,
        method="rk4",
        state_weights=torch.tensor([2.0, 1.0, 0.5]),
        horizon_weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )

    assert exact_trajectory.shape == (2, horizon + 1, 3)
    torch.testing.assert_close(exact_trajectory[:, 1:], targets, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(exact_loss, torch.zeros_like(exact_loss), atol=1e-24, rtol=0.0)

    learned = ConstantResidual([0.1, 0.05, -0.1])
    loss, trajectory = multistep_rollout_loss(
        learned, initial, controls, dt, targets, normalizers, method="rk4"
    )
    loss.backward()

    assert trajectory.shape == (2, horizon + 1, 3)
    assert loss > 0.0
    assert learned.value.grad is not None
    assert torch.isfinite(learned.value.grad).all()
    assert learned.value.grad.abs().sum() > 0.0

    targets_with_initial = torch.cat((initial[:, None, :], targets), dim=1)
    inclusive_loss, _ = multistep_rollout_loss(
        oracle,
        initial,
        controls,
        dt,
        targets_with_initial,
        normalizers,
        method="euler",
    )
    torch.testing.assert_close(
        inclusive_loss, torch.zeros_like(inclusive_loss), atol=1e-24, rtol=0.0
    )


def test_composite_loss_skips_absent_rollout_windows_with_stable_components():
    model = ZeroResidual()
    normalizers = _identity()
    state = torch.zeros(3, 3, dtype=torch.float64)
    control = torch.zeros(3, 2, dtype=torch.float64)
    batch = {
        "state_t": state,
        "control_t": control,
        "residual_target": torch.zeros_like(state),
        "state_t_plus_1": state.clone(),
        "dt": torch.full((3,), 0.1, dtype=torch.float64),
    }
    criterion = CompositeResidualLoss(
        model,
        normalizers,
        lambda_residual=1.0,
        lambda_one_step=1.0,
        lambda_multistep=1.0,
    )

    components = criterion(batch)

    for key in (
        "loss",
        "residual_loss",
        "one_step_loss",
        "multistep_loss",
        "regularization_loss",
        "rollout_rmse",
    ):
        assert key in components
        assert torch.isfinite(components[key])
    torch.testing.assert_close(
        components["multistep_loss"], torch.zeros_like(components["multistep_loss"])
    )

    batch_with_empty_windows = dict(batch)
    batch_with_empty_windows.update(
        {
            "rollout_initial_state": torch.empty(0, 3, dtype=torch.float64),
            "rollout_controls": torch.empty(0, 4, 2, dtype=torch.float64),
            "rollout_dt": torch.empty(0, 4, dtype=torch.float64),
            "rollout_target_states": torch.empty(0, 4, 3, dtype=torch.float64),
        }
    )
    empty_components = criterion(batch_with_empty_windows)
    torch.testing.assert_close(
        empty_components["multistep_loss"],
        torch.zeros_like(empty_components["multistep_loss"]),
    )
