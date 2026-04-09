import math
import time
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

    def _update_predicted_mocap_set(self, mocap_ids, trajectory_points, z=0.03):
        visible_count = min(len(trajectory_points), len(mocap_ids))

        for i in range(visible_count):
            x, y, _ = trajectory_points[i]
            mocap_id = mocap_ids[i]

            self.data.mocap_pos[mocap_id] = np.array([x, y, z], dtype=float)
            self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        for i in range(visible_count, len(mocap_ids)):
            mocap_id = mocap_ids[i]

            self.data.mocap_pos[mocap_id] = np.array([0.0, 0.0, -1.0], dtype=float)
            self.data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def update_best_predicted_trajectory(self, trajectory_points):
        self._update_predicted_mocap_set(
            mocap_ids=self.pred_best_mocap_ids,
            trajectory_points=trajectory_points,
            z=0.040,
        )

    def update_aux_predicted_trajectories(self, trajectory_list):
        for path_idx in range(2):
            if path_idx < len(trajectory_list):
                z_value = 0.030 if path_idx == 0 else 0.022
                self._update_predicted_mocap_set(
                    mocap_ids=self.pred_aux_mocap_ids[path_idx],
                    trajectory_points=trajectory_list[path_idx],
                    z=z_value,
                )
            else:
                self._update_predicted_mocap_set(
                    mocap_ids=self.pred_aux_mocap_ids[path_idx],
                    trajectory_points=[],
                    z=0.030,
                )

    def launch_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def reset(self, start_state):
        x, y, theta = start_state

        # freejoint: qpos[0:3] 是位置, qpos[3:7] 是四元数
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

        next_x = x + v * math.cos(theta) * dt
        next_y = y + v * math.sin(theta) * dt
        next_theta = theta + omega * dt

        self.reset((next_x, next_y, next_theta))
        return self.get_state()

    def render(self, sleep_dt=0.01):
        if self.viewer is not None:
            self.viewer.sync()
            time.sleep(sleep_dt)