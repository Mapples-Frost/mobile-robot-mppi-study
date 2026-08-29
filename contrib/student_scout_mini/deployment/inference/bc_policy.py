"""
Car Inference Engine - Sparse Multiscale BC Policy (88% success rate)
============================================================
This module loads the trained BC policy and provides inference
for the Scout Mini differential drive robot.

Model: 9-dim input -> 128 -> 128 -> 64 -> 2 (wheel speeds)
Input: [x, y, theta, goal_x, goal_y, d_goal, theta_err, cos(theta), sin(theta)]
Output: [v_left, v_right] in [-V_MAX, V_MAX]
"""
import math
import json
import time
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Robot Parameters (Scout Mini)
# ============================================================
TRACK_WIDTH = 0.32       # m
COLLISION_RADIUS = 0.25  # m
V_MAX = 0.50             # m/s per wheel
OMEGA_MAX = 0.8          # rad/s (conservative for car)
DT = 0.2                 # s
GOAL_THRESHOLD = 0.3     # m

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return torch.tanh(self.net(x))


class BCPolicy:
    def __init__(self, model_path, device='cpu'):
        self.device = torch.device(device)
        self.model = PolicyNet().to(self.device)
        self.model.eval()

        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['policy_state'])

        self.sm = np.array(checkpoint['sm'], dtype=np.float32)
        self.ss = np.array(checkpoint['ss'], dtype=np.float32)
        self.am = np.array(checkpoint['am'], dtype=np.float32)
        self.asd = np.array(checkpoint['asd'], dtype=np.float32)

        self.last_vl = 0.0
        self.last_vr = 0.0
        self.infer_times = []

    def _wrap_angle(self, t):
        while t > math.pi: t -= 2*math.pi
        while t < -math.pi: t += 2*math.pi
        return t

    def state_to_array(self, state, goal):
        x, y, theta = state
        dx = goal[0] - x
        dy = goal[1] - y
        d_goal = math.hypot(dx, dy)
        goal_angle = math.atan2(dy, dx)
        theta_err = self._wrap_angle(goal_angle - theta)
        return np.array([
            x, y, theta,
            goal[0], goal[1],
            d_goal, theta_err,
            math.cos(theta), math.sin(theta)
        ], dtype=np.float32)

    def infer(self, state, goal):
        st = self.state_to_array(state, goal)
        snorm = (st - self.sm) / self.ss
        t0 = time.time()
        with torch.no_grad():
            anorm = self.model(torch.tensor(snorm, dtype=torch.float32).unsqueeze(0).to(self.device))[0].cpu().numpy()
        self.infer_times.append(time.time() - t0)

        vl = float(np.clip(anorm[0] * self.asd[0] + self.am[0], -V_MAX, V_MAX))
        vr = float(np.clip(anorm[1] * self.asd[1] + self.am[1], -V_MAX, V_MAX))

        self.last_vl = vl
        self.last_vr = vr
        return vl, vr

    def get_average_infer_time(self):
        if not self.infer_times:
            return 0.0
        return sum(self.infer_times) / len(self.infer_times)
