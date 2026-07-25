import numpy as np

from mobile_robot_mppi.obstacles.collision_risk import (
    GaussianMixtureObstacleForecast,
)
from mobile_robot_mppi.visualization.probability_overlay import (
    confidence_ellipse_geometry,
    forecast_ellipse_specs,
    transform_gaussian_frame,
)


def _forecast():
    means = np.zeros((4, 2, 2), dtype=np.float64)
    means[:, 0, 0] = np.arange(1.0, 5.0)
    means[:, 1, 1] = np.arange(1.0, 5.0)
    covariances = np.broadcast_to(
        np.diag((0.04, 0.01)), (4, 2, 2, 2)
    ).copy()
    weights = np.tile((0.75, 0.25), (4, 1))
    return GaussianMixtureObstacleForecast(
        timestamp=0.0,
        dt=0.1,
        component_means=means,
        component_covariances=covariances,
        component_weights=weights,
        radius_m=0.25,
    )


def test_confidence_ellipse_uses_chi_square_95_percent_scale():
    major, minor, angle = confidence_ellipse_geometry(
        np.diag((0.04, 0.01))
    )
    scale = np.sqrt(-2.0 * np.log(0.05))
    np.testing.assert_allclose(major, 0.20 * scale, atol=1.0e-12)
    np.testing.assert_allclose(minor, 0.10 * scale, atol=1.0e-12)
    assert np.isfinite(angle)


def test_frame_transform_rotates_means_and_covariances():
    means = np.asarray([[[1.0, 0.0]]])
    covariances = np.diag((0.04, 0.01))[None, None, :, :]
    transformed_mean, transformed_covariance = transform_gaussian_frame(
        means,
        covariances,
        source_pose=(0.0, 0.0, 0.0),
        target_pose=(2.0, 3.0, np.pi / 2.0),
    )
    np.testing.assert_allclose(
        transformed_mean[0, 0], (2.0, 4.0), atol=1.0e-12
    )
    np.testing.assert_allclose(
        transformed_covariance[0, 0],
        np.diag((0.01, 0.04)),
        atol=1.0e-12,
    )


def test_forecast_specs_select_horizons_and_preserve_mode_weights():
    specs, means = forecast_ellipse_specs(
        _forecast(), horizon_indices=(0, 2, 99)
    )
    assert means.shape == (4, 2, 2)
    assert len(specs) == 6
    assert {item.horizon_index for item in specs} == {0, 2, 3}
    assert {
        item.weight for item in specs if item.horizon_index == 0
    } == {0.75, 0.25}
