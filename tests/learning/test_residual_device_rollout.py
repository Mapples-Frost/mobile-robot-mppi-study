import numpy as np
import pytest
import torch

from mobile_robot_mppi.core.spaces import (
    body_velocity_action,
    dynamic_unicycle_state,
)
from mobile_robot_mppi.learning.models import (
    PlatformResidualDynamics,
    ResidualComponentMaskedDynamics,
)
from mobile_robot_mppi.planning.dynamics import (
    DynamicUnicyclePrediction,
    ResidualPrediction,
)
from mobile_robot_mppi.planning.mppi import MppiConfig, MppiController


class _TinyResidual(torch.nn.Module):
    state_dim = 5
    control_dim = 2

    def forward(self, state, control):
        output = torch.zeros_like(state)
        output[..., 3] = (
            0.03 * control[..., 0] - 0.02 * state[..., 3]
        )
        output[..., 4] = (
            -0.04 * control[..., 1] + 0.01 * state[..., 4]
        )
        return output


def _controller(device_rollout_enabled):
    adapter = PlatformResidualDynamics(
        _TinyResidual(),
        device="cpu",
        use_torchscript=True,
        device_rollout_enabled=device_rollout_enabled,
    )
    residual = ResidualComponentMaskedDynamics(
        adapter, [0.0, 0.0, 0.0, 1.0, 1.0]
    )
    dynamics = ResidualPrediction(
        DynamicUnicyclePrediction(0.18, 0.12), residual
    )
    return MppiController(
        dynamics,
        dynamic_unicycle_state(),
        body_velocity_action((-0.15, 0.45), 1.2),
        MppiConfig(
            horizon=7,
            num_samples=13,
            dt=0.1,
            integrator="rk4",
            noise_sigma=(0.12, 0.5),
        ),
    )


def test_device_resident_rollout_matches_legacy_rk4_contract():
    rng = np.random.RandomState(730199908)
    controls = rng.uniform(
        [-0.15, -1.2], [0.45, 1.2], size=(13, 7, 2)
    )
    initial_state = np.asarray(
        [0.1, -0.2, 3.12, 0.2, -0.1], dtype=np.float64
    )
    legacy = _controller(False).rollout(initial_state, controls)
    optimized = _controller(True).rollout(initial_state, controls)

    np.testing.assert_allclose(optimized, legacy, atol=1.0e-10, rtol=0.0)
    assert np.max(np.abs(optimized[..., 2])) <= np.pi


def test_device_rollout_fast_path_is_opt_in():
    legacy = _controller(False)
    optimized = _controller(True)

    assert not legacy.dynamics.supports_rollout_batch
    assert optimized.dynamics.supports_rollout_batch


def test_cuda_graph_requires_cuda_device_resident_rollout():
    with pytest.raises(ValueError, match="requires CUDA"):
        PlatformResidualDynamics(
            _TinyResidual(),
            device="cpu",
            device_rollout_enabled=True,
            cuda_graph_enabled=True,
        )
