import numpy as np
import torch

from mobile_robot_mppi.learning.models import ResidualNetwork


def _statistics():
    return {
        "feature_mean": np.zeros(6, dtype=np.float32),
        "feature_scale": np.ones(6, dtype=np.float32),
        "control_mean": np.zeros(2, dtype=np.float32),
        "control_scale": np.ones(2, dtype=np.float32),
        "residual_mean": np.zeros(5, dtype=np.float32),
        "residual_scale": np.ones(5, dtype=np.float32),
    }


def test_structured_network_masks_known_kinematic_channels_for_batches():
    model = ResidualNetwork(5, 2, {
        "type": "icode_residual",
        "angle_indices": [2],
        "hidden_sizes": [8],
        "residual_output_mask": [0, 0, 0, 1, 1],
    }, _statistics())
    output = model(torch.randn(9, 5), torch.randn(9, 2))
    torch.testing.assert_close(output[:, :3], torch.zeros_like(output[:, :3]))
    assert tuple(output.shape) == (9, 5)
    assert model.checkpoint_config()["model"]["residual_output_mask"] == [0, 0, 0, 1, 1]

