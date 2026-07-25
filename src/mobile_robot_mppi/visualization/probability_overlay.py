"""MuJoCo overlays for online Gaussian-mixture obstacle forecasts."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ForecastEllipse:
    horizon_index: int
    horizon_time_s: float
    mode_index: int
    mean_xy: np.ndarray
    major_axis_m: float
    minor_axis_m: float
    angle_rad: float
    weight: float


def confidence_ellipse_geometry(
    covariance,
    probability=0.95,
    minimum_std_m=0.01,
):
    """Return major/minor radii and angle for a 2-D Gaussian contour."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (2, 2):
        raise ValueError("covariance must have shape (2,2)")
    if not np.isfinite(covariance).all():
        raise ValueError("covariance must be finite")
    if not np.allclose(
        covariance, covariance.T, atol=1.0e-10, rtol=0.0
    ):
        raise ValueError("covariance must be symmetric")
    if not 0.0 < float(probability) < 1.0:
        raise ValueError("probability must be in (0,1)")
    if (
        not np.isfinite(minimum_std_m)
        or float(minimum_std_m) <= 0.0
    ):
        raise ValueError("minimum std must be finite and positive")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues.min()) < -1.0e-9:
        raise ValueError("covariance must be positive semidefinite")
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(
        eigenvalues[order], float(minimum_std_m) ** 2
    )
    major_vector = eigenvectors[:, order[0]]
    scale = float(np.sqrt(-2.0 * np.log(1.0 - probability)))
    axes = scale * np.sqrt(eigenvalues)
    angle = float(np.arctan2(major_vector[1], major_vector[0]))
    return float(axes[0]), float(axes[1]), angle


def transform_gaussian_frame(
    means,
    covariances,
    source_pose,
    target_pose,
):
    """Rigidly transform Gaussian positions between two planar frames."""

    means = np.asarray(means, dtype=np.float64)
    covariances = np.asarray(covariances, dtype=np.float64)
    source = np.asarray(source_pose, dtype=np.float64).reshape(-1)
    target = np.asarray(target_pose, dtype=np.float64).reshape(-1)
    if means.shape[-1] != 2:
        raise ValueError("means must end in XY")
    if covariances.shape != means.shape[:-1] + (2, 2):
        raise ValueError("covariance shape must align with means")
    if source.shape != (3,) or target.shape != (3,):
        raise ValueError("source and target poses must be planar poses")
    if not (
        np.isfinite(means).all()
        and np.isfinite(covariances).all()
        and np.isfinite(source).all()
        and np.isfinite(target).all()
    ):
        raise ValueError("frame transform inputs must be finite")

    angle = float(target[2] - source[2])
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotation = np.asarray(
        ((cosine, -sine), (sine, cosine)), dtype=np.float64
    )
    transformed_means = (
        target[:2]
        + np.einsum(
            "ij,...j->...i", rotation, means - source[:2]
        )
    )
    transformed_covariances = np.einsum(
        "ij,...jk,lk->...il",
        rotation,
        covariances,
        rotation,
    )
    return transformed_means, transformed_covariances


def forecast_ellipse_specs(
    forecast,
    horizon_indices=(0, 9, 19, 35),
    probability=0.95,
    minimum_std_m=0.01,
    source_pose=None,
    target_pose=None,
):
    """Convert a planner forecast into renderable ellipse specifications."""

    means = np.asarray(forecast.component_means, dtype=np.float64)
    covariances = np.asarray(
        forecast.component_covariances, dtype=np.float64
    )
    if (source_pose is None) != (target_pose is None):
        raise ValueError(
            "source and target poses must be supplied together"
        )
    if source_pose is not None:
        means, covariances = transform_gaussian_frame(
            means, covariances, source_pose, target_pose
        )
    selected = tuple(
        sorted(
            {
                min(max(int(index), 0), forecast.horizon - 1)
                for index in horizon_indices
            }
        )
    )
    specs = []
    for horizon_index in selected:
        for mode_index in range(forecast.mode_count):
            major, minor, angle = confidence_ellipse_geometry(
                covariances[horizon_index, mode_index],
                probability=probability,
                minimum_std_m=minimum_std_m,
            )
            specs.append(
                ForecastEllipse(
                    horizon_index=horizon_index,
                    horizon_time_s=float(
                        (horizon_index + 1) * forecast.dt
                    ),
                    mode_index=mode_index,
                    mean_xy=means[
                        horizon_index, mode_index
                    ].copy(),
                    major_axis_m=major,
                    minor_axis_m=minor,
                    angle_rad=angle,
                    weight=float(
                        forecast.component_weights[
                            horizon_index, mode_index
                        ]
                    ),
                )
            )
    return tuple(specs), means


class MujocoProbabilityOverlay:
    """Render task markers and forecast confidence ellipses."""

    HORIZON_COLORS = (
        (0.00, 0.88, 1.00),
        (0.05, 0.55, 1.00),
        (0.48, 0.25, 0.95),
        (0.95, 0.12, 0.62),
    )

    def __init__(
        self,
        mujoco_module,
        viewer,
        start_xy,
        goal_xy,
        horizon_indices=(0, 9, 19, 35),
    ):
        self.mujoco = mujoco_module
        self.viewer = viewer
        self.start_xy = np.asarray(
            start_xy, dtype=np.float64
        ).reshape(2)
        self.goal_xy = np.asarray(
            goal_xy, dtype=np.float64
        ).reshape(2)
        self.horizon_indices = tuple(int(v) for v in horizon_indices)
        self.identity = np.eye(3, dtype=np.float64).reshape(-1)
        self.zero = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _pose_array(pose):
        if hasattr(pose, "as_array"):
            return np.asarray(
                pose.as_array(), dtype=np.float64
            ).reshape(3)
        return np.asarray(pose, dtype=np.float64).reshape(3)

    @staticmethod
    def _next_geom(scene):
        if scene.ngeom >= scene.maxgeom:
            return None
        geom = scene.geoms[scene.ngeom]
        scene.ngeom += 1
        return geom

    def _draw_cylinder(self, scene, xy, radius, rgba):
        geom = self._next_geom(scene)
        if geom is None:
            return
        self.mujoco.mjv_initGeom(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_CYLINDER),
            np.asarray((radius, 0.018, 0.0), dtype=np.float64),
            np.asarray((xy[0], xy[1], 0.02), dtype=np.float64),
            self.identity,
            np.asarray(rgba, dtype=np.float32),
        )

    def _draw_sphere(self, scene, xy, z, radius, rgba):
        geom = self._next_geom(scene)
        if geom is None:
            return
        self.mujoco.mjv_initGeom(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_SPHERE),
            np.asarray((radius, 0.0, 0.0), dtype=np.float64),
            np.asarray((xy[0], xy[1], z), dtype=np.float64),
            self.identity,
            np.asarray(rgba, dtype=np.float32),
        )

    def _draw_segment(self, scene, first, second, radius, rgba):
        geom = self._next_geom(scene)
        if geom is None:
            return
        self.mujoco.mjv_initGeom(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_CAPSULE),
            self.zero,
            self.zero,
            self.identity,
            np.asarray(rgba, dtype=np.float32),
        )
        self.mujoco.mjv_makeConnector(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_CAPSULE),
            float(radius),
            float(first[0]),
            float(first[1]),
            float(first[2]),
            float(second[0]),
            float(second[1]),
            float(second[2]),
        )

    def _draw_label(self, scene, xyz, label):
        geom = self._next_geom(scene)
        if geom is None:
            return
        self.mujoco.mjv_initGeom(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_LABEL),
            self.zero,
            self.zero,
            self.identity,
            np.ones(4, dtype=np.float32),
        )
        geom.pos = tuple(float(value) for value in xyz)
        geom.label = str(label)

    def _draw_ellipse(self, scene, spec, color, z_value):
        geom = self._next_geom(scene)
        if geom is None:
            return
        cosine = float(np.cos(spec.angle_rad))
        sine = float(np.sin(spec.angle_rad))
        rotation = np.asarray(
            (
                (cosine, -sine, 0.0),
                (sine, cosine, 0.0),
                (0.0, 0.0, 1.0),
            ),
            dtype=np.float64,
        ).reshape(-1)
        alpha = float(np.clip(0.07 + 0.30 * spec.weight, 0.07, 0.34))
        self.mujoco.mjv_initGeom(
            geom,
            int(self.mujoco.mjtGeom.mjGEOM_ELLIPSOID),
            np.asarray(
                (
                    spec.major_axis_m,
                    spec.minor_axis_m,
                    0.014,
                ),
                dtype=np.float64,
            ),
            np.asarray(
                (spec.mean_xy[0], spec.mean_xy[1], z_value),
                dtype=np.float64,
            ),
            rotation,
            np.asarray((*color, alpha), dtype=np.float32),
        )

    def update(
        self,
        forecasts: Sequence[object],
        online_pose,
        display_pose,
        tracker_diagnostics: Optional[dict] = None,
    ):
        source_pose = self._pose_array(online_pose)
        target_pose = self._pose_array(display_pose)
        with self.viewer.lock():
            scene = self.viewer.user_scn
            scene.ngeom = 0
            self._draw_segment(
                scene,
                (*self.start_xy, 0.025),
                (*self.goal_xy, 0.025),
                0.018,
                (0.38, 0.42, 0.48, 0.42),
            )
            self._draw_cylinder(
                scene, self.start_xy, 0.30, (0.10, 0.90, 0.28, 0.88)
            )
            self._draw_cylinder(
                scene, self.goal_xy, 0.30, (1.00, 0.72, 0.05, 0.92)
            )
            self._draw_label(
                scene, (*self.start_xy, 0.42), "START"
            )
            self._draw_label(scene, (*self.goal_xy, 0.42), "GOAL")

            if not forecasts:
                self._draw_label(
                    scene,
                    (-4.45, 3.05, 0.30),
                    "FORECAST UNAVAILABLE - FAIL CLOSED",
                )
                return

            forecast = forecasts[0]
            specs, transformed_means = forecast_ellipse_specs(
                forecast,
                horizon_indices=self.horizon_indices,
                source_pose=source_pose,
                target_pose=target_pose,
            )
            selected = sorted(
                {spec.horizon_index for spec in specs}
            )
            color_by_horizon = {
                horizon: self.HORIZON_COLORS[
                    min(index, len(self.HORIZON_COLORS) - 1)
                ]
                for index, horizon in enumerate(selected)
            }
            for spec in specs:
                color = color_by_horizon[spec.horizon_index]
                level = selected.index(spec.horizon_index)
                self._draw_ellipse(
                    scene, spec, color, 0.055 + 0.018 * level
                )
                self._draw_sphere(
                    scene,
                    spec.mean_xy,
                    0.085 + 0.018 * level,
                    0.025,
                    (*color, min(0.95, 0.35 + spec.weight)),
                )

            weighted_mean = np.sum(
                transformed_means
                * forecast.component_weights[..., None],
                axis=1,
            )
            for index in range(0, weighted_mean.shape[0] - 1, 2):
                next_index = min(index + 2, weighted_mean.shape[0] - 1)
                self._draw_segment(
                    scene,
                    (*weighted_mean[index], 0.115),
                    (*weighted_mean[next_index], 0.115),
                    0.012,
                    (0.02, 0.95, 0.88, 0.92),
                )
            self._draw_label(
                scene,
                (-4.45, 3.05, 0.30),
                "95% CENTER FORECAST | opacity = mode probability",
            )
            if tracker_diagnostics:
                availability = float(
                    tracker_diagnostics.get(
                        "forecast_availability", 0.0
                    )
                )
                self._draw_label(
                    scene,
                    (-4.45, 2.70, 0.30),
                    "online Change-Aware IMM | availability %.0f%%"
                    % (100.0 * availability),
                )


__all__ = [
    "ForecastEllipse",
    "MujocoProbabilityOverlay",
    "confidence_ellipse_geometry",
    "forecast_ellipse_specs",
    "transform_gaussian_frame",
]

