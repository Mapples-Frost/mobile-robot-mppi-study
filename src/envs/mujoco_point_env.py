import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


class MujocoPointEnv:
    def __init__(self, xml_path):
        self.xml_path = str(Path(xml_path).resolve())
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        self.pred_best_mocap_ids = []
        for i in range(40):
            body_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                f"pred_best_{i}",
            )
            if body_id == -1:
                continue

            mocap_id = self.model.body_mocapid[body_id]
            if mocap_id != -1:
                self.pred_best_mocap_ids.append(mocap_id)

        self.pred_aux_mocap_ids = [[], []]

        for path_idx in range(2):
            for i in range(20):
                body_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    f"pred_aux{path_idx}_{i}",
                )
                if body_id == -1:
                    continue

                mocap_id = self.model.body_mocapid[body_id]
                if mocap_id != -1:
                    self.pred_aux_mocap_ids[path_idx].append(mocap_id)

        self.history_points = deque(maxlen=60)
        self.history_z = 0.028
        self.history_segment_radius = 0.009
        self.history_rgba_start = np.array([0.78, 0.68, 0.54, 0.40], dtype=np.float32)
        self.history_rgba_end = np.array([0.60, 0.48, 0.36, 0.94], dtype=np.float32)
        self._overlay_init_size = np.ones(3, dtype=np.float64)
        self._overlay_init_pos = np.zeros(3, dtype=np.float64)
        self._overlay_init_mat = np.eye(3, dtype=np.float64).reshape(-1)
        self._overlay_dirty = True
        self._previous_visual_state = None
        self._current_visual_state = None
        self.visual_interpolation_frames = 8
        self.visual_frame_pause = 0.0018

        self.best_predicted_points = []
        self.aux_predicted_points = [[], []]
        self.predicted_max_length = 1.10
        self.best_predicted_z = 0.064
        self.aux_predicted_z = [0.058, 0.054]
        self.best_predicted_segment_radius = 0.0105
        self.aux_predicted_segment_radius = [0.0080, 0.0072]
        self.best_predicted_rgba = np.array([0.36, 0.60, 0.62, 1.00], dtype=np.float32)
        self.aux_predicted_rgba = [
            np.array([0.74, 0.63, 0.42, 0.98], dtype=np.float32),
            np.array([0.66, 0.54, 0.60, 0.98], dtype=np.float32),
        ]

    def _set_state(self, state):
        x, y, theta = state

        # freejoint: qpos[0:3] 是位置 qpos[3:7] 是四元数
        self.data.qpos[0] = x
        self.data.qpos[1] = y
        self.data.qpos[2] = 0.08

        half = theta * 0.5
        self.data.qpos[3] = math.cos(half)   # qw
        self.data.qpos[4] = 0.0              # qx
        self.data.qpos[5] = 0.0              # qy
        self.data.qpos[6] = math.sin(half)   # qz

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _compute_default_camera(self):
        center = np.array(self.model.stat.center, dtype=float)
        extent = float(self.model.stat.extent)

        lookat = center.copy()
        lookat[2] = max(0.08, lookat[2])

        return {
            "lookat": lookat,
            "distance": max(1.22 * extent, 4.8),
            "azimuth": 132.0,
            "elevation": -46.0,
        }

    def _configure_default_camera(self):
        if self.viewer is None:
            return

        camera_cfg = self._compute_default_camera()
        with self.viewer.lock():
            self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.viewer.cam.fixedcamid = -1
            self.viewer.cam.trackbodyid = -1
            self.viewer.cam.lookat[:] = camera_cfg["lookat"]
            self.viewer.cam.distance = camera_cfg["distance"]
            self.viewer.cam.azimuth = camera_cfg["azimuth"]
            self.viewer.cam.elevation = camera_cfg["elevation"]

    def _history_point_xyz(self, state):
        x, y, _ = state
        return np.array([x, y, self.history_z], dtype=np.float64)

    def _predicted_point_xyz(self, point, z):
        x, y, _ = point
        return np.array([x, y, z], dtype=np.float64)

    def _clip_polyline_length(self, points, max_length):
        if len(points) < 2 or max_length <= 0.0:
            return [point.copy() for point in points]

        clipped_points = [points[0].copy()]
        accumulated_length = 0.0

        for point in points[1:]:
            prev_point = clipped_points[-1]
            segment_xy = point[:2] - prev_point[:2]
            segment_length = float(np.linalg.norm(segment_xy))

            if segment_length <= 1e-9:
                continue

            remaining_length = max_length - accumulated_length
            if remaining_length <= 1e-9:
                break

            if segment_length <= remaining_length:
                clipped_points.append(point.copy())
                accumulated_length += segment_length
                continue

            blend = remaining_length / segment_length
            clipped_point = prev_point + blend * (point - prev_point)
            clipped_points.append(clipped_point)
            break

        return clipped_points

    def _cache_predicted_trajectory(self, trajectory_points, z):
        predicted_points = [self._predicted_point_xyz(point, z) for point in trajectory_points]
        return self._clip_polyline_length(predicted_points, self.predicted_max_length)

    def _reset_history(self, start_state):
        self.history_points.clear()
        self.history_points.append(self._history_point_xyz(start_state))
        self.best_predicted_points = []
        self.aux_predicted_points = [[], []]
        self._overlay_dirty = True
        self._clear_history_overlay()

    def _append_history_state(self, state):
        self.history_points.append(self._history_point_xyz(state))
        self._overlay_dirty = True

    def _clear_history_overlay(self):
        if self.viewer is not None and self.viewer.user_scn is not None:
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
        self._overlay_dirty = False

    def _hide_mocap_set(self, mocap_ids):
        for mocap_id in mocap_ids:
            self.data.mocap_pos[mocap_id] = np.array([0.0, 0.0, -1.0], dtype=float)
            self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _hide_all_predicted_mocaps(self):
        self._hide_mocap_set(self.pred_best_mocap_ids)
        for mocap_ids in self.pred_aux_mocap_ids:
            self._hide_mocap_set(mocap_ids)

    def _draw_polyline_overlay(
        self,
        user_scn,
        points,
        rgba,
        segment_radius,
    ):
        if len(points) < 2:
            return

        for idx in range(len(points) - 1):
            if user_scn.ngeom >= user_scn.maxgeom:
                return

            geom = user_scn.geoms[user_scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                self._overlay_init_size,
                self._overlay_init_pos,
                self._overlay_init_mat,
                rgba,
            )

            a = points[idx]
            b = points[idx + 1]
            mujoco.mjv_makeConnector(
                geom,
                int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                segment_radius,
                float(a[0]),
                float(a[1]),
                float(a[2]),
                float(b[0]),
                float(b[1]),
                float(b[2]),
            )
            user_scn.ngeom += 1

    def _rebuild_history_overlay(self):
        if self.viewer is None or self.viewer.user_scn is None:
            return

        history_points = list(self.history_points)
        with self.viewer.lock():
            user_scn = self.viewer.user_scn
            user_scn.ngeom = 0

            if len(history_points) >= 2:
                max_segments = min(len(history_points) - 1, user_scn.maxgeom)
                visible_points = history_points[-(max_segments + 1):]
                total_segments = len(visible_points) - 1

                for idx in range(total_segments):
                    geom = user_scn.geoms[user_scn.ngeom]
                    blend = (idx + 1) / max(total_segments, 1)
                    rgba = (
                        (1.0 - blend) * self.history_rgba_start
                        + blend * self.history_rgba_end
                    ).astype(np.float32, copy=False)

                    mujoco.mjv_initGeom(
                        geom,
                        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                        self._overlay_init_size,
                        self._overlay_init_pos,
                        self._overlay_init_mat,
                        rgba,
                    )

                    a = visible_points[idx]
                    b = visible_points[idx + 1]
                    mujoco.mjv_makeConnector(
                        geom,
                        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
                        self.history_segment_radius,
                        float(a[0]),
                        float(a[1]),
                        float(a[2]),
                        float(b[0]),
                        float(b[1]),
                        float(b[2]),
                    )
                    user_scn.ngeom += 1

            self._draw_polyline_overlay(
                user_scn=user_scn,
                points=self.aux_predicted_points[1],
                rgba=self.aux_predicted_rgba[1],
                segment_radius=self.aux_predicted_segment_radius[1],
            )
            self._draw_polyline_overlay(
                user_scn=user_scn,
                points=self.aux_predicted_points[0],
                rgba=self.aux_predicted_rgba[0],
                segment_radius=self.aux_predicted_segment_radius[0],
            )
            self._draw_polyline_overlay(
                user_scn=user_scn,
                points=self.best_predicted_points,
                rgba=self.best_predicted_rgba,
                segment_radius=self.best_predicted_segment_radius,
            )
        self._overlay_dirty = False

    def update_best_predicted_trajectory(self, trajectory_points):
        self.best_predicted_points = self._cache_predicted_trajectory(
            trajectory_points=trajectory_points,
            z=self.best_predicted_z,
        )
        self._hide_mocap_set(self.pred_best_mocap_ids)
        self._overlay_dirty = True

    def update_aux_predicted_trajectories(self, trajectory_list):
        for path_idx in range(2):
            if path_idx < len(trajectory_list):
                z_value = self.aux_predicted_z[path_idx]
                self.aux_predicted_points[path_idx] = self._cache_predicted_trajectory(
                    trajectory_points=trajectory_list[path_idx],
                    z=z_value,
                )
                self._hide_mocap_set(self.pred_aux_mocap_ids[path_idx])
            else:
                self.aux_predicted_points[path_idx] = []
                self._hide_mocap_set(self.pred_aux_mocap_ids[path_idx])
        self._overlay_dirty = True

    def launch_viewer(self):
        if self.viewer is None:
            try:
                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                    show_left_ui=False,
                    show_right_ui=False,
                )
            except TypeError:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._hide_all_predicted_mocaps()
            self._configure_default_camera()
            self._overlay_dirty = True
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self, start_state):
        self._set_state(start_state)
        self._previous_visual_state = tuple(start_state)
        self._current_visual_state = tuple(start_state)
        self._reset_history(start_state)

    def get_state(self):
        x = float(self.data.qpos[0])
        y = float(self.data.qpos[1])

        qw = float(self.data.qpos[3])
        qz = float(self.data.qpos[6])
        theta = 2.0 * math.atan2(qz, qw)

        return x, y, theta

    def step(self, control, dt):
        v, omega = control

        x, y, theta = self.get_state()
        previous_state = (x, y, theta)

        next_x = x + v * math.cos(theta) * dt
        next_y = y + v * math.sin(theta) * dt
        next_theta = theta + omega * dt

        next_state = (next_x, next_y, next_theta)
        self._set_state(next_state)
        current_state = self.get_state()
        self._previous_visual_state = previous_state
        self._current_visual_state = current_state
        self._append_history_state(current_state)
        return current_state

    def _interpolate_state(self, start_state, end_state, alpha):
        start_x, start_y, start_theta = start_state
        end_x, end_y, end_theta = end_state

        delta_theta = math.atan2(
            math.sin(end_theta - start_theta),
            math.cos(end_theta - start_theta),
        )

        return (
            start_x + (end_x - start_x) * alpha,
            start_y + (end_y - start_y) * alpha,
            start_theta + delta_theta * alpha,
        )

    def render(self, sleep_dt=0.0):
        if self.viewer is not None:
            with self.viewer.lock():
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.trackbodyid = -1

            if self._overlay_dirty:
                self._rebuild_history_overlay()

            if (
                self._previous_visual_state is not None
                and self._current_visual_state is not None
                and self.visual_interpolation_frames > 1
            ):
                for frame_idx in range(1, self.visual_interpolation_frames):
                    alpha = frame_idx / self.visual_interpolation_frames
                    interpolated_state = self._interpolate_state(
                        self._previous_visual_state,
                        self._current_visual_state,
                        alpha,
                    )
                    self._set_state(interpolated_state)
                    self.viewer.sync()
                    if self.visual_frame_pause > 0.0:
                        time.sleep(self.visual_frame_pause)

            if self._current_visual_state is not None:
                self._set_state(self._current_visual_state)
            self.viewer.sync()
            self._previous_visual_state = self._current_visual_state
            if sleep_dt > 0.0:
                time.sleep(sleep_dt)
