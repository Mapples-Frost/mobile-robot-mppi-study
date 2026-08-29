"""
Obstacle Avoidance Controller for Scout Mini
=============================================
Combines BC policy with safety checks for real-car deployment.

Usage:
    python car_controller.py --scene sparse --model_path ../model/sparse_multiscale_bc_policy.pth
"""
import math
import json
import time
import argparse
import sys
from pathlib import Path
from collections import deque

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bc_policy import BCPolicy, PolicyNet, TRACK_WIDTH, COLLISION_RADIUS, V_MAX, OMEGA_MAX, DT, GOAL_THRESHOLD

# ============================================================
# Scene Definitions (matches baseline_scenes.py)
# ============================================================
SCENES = {
    "sparse": {
        "start_state": (0.0, 0.0, 0.0),
        "goal": (6.0, 0.0),
        "obstacles": [
            (2.2, 0.9, 0.55),
            (2.2, -0.9, 0.55),
            (4.2, 0.9, 0.55),
            (4.2, -0.9, 0.55),
        ],
        "bounds": {"x_min": -0.5, "x_max": 6.5, "y_min": -1.8, "y_max": 1.8},
    },
    "dense": {
        "start_state": (0.0, 0.0, 0.0),
        "goal": (6.0, 0.0),
        "obstacles": [
            (1.0, 0.55, 0.33), (2.1, -0.35, 0.36), (3.2, 0.60, 0.36),
            (4.3, -0.50, 0.36), (5.1, 0.40, 0.30),
            (2.7, 1.35, 0.28), (3.8, -1.35, 0.28),
        ],
        "bounds": {"x_min": -0.6, "x_max": 6.6, "y_min": -1.5, "y_max": 2.2},
    },
    "narrow": {
        "start_state": (0.0, 0.0, 0.0),
        "goal": (6.0, 0.0),
        "obstacles": [
            (2.8, 0.9, 0.65), (2.8, -0.9, 0.65),
            (4.3, 0.9, 0.65), (4.3, -0.9, 0.65),
        ],
        "bounds": {"x_min": -0.5, "x_max": 6.5, "y_min": -1.5, "y_max": 1.5},
    },
}


def wrap_angle(t):
    while t > math.pi: t -= 2*math.pi
    while t < -math.pi: t += 2*math.pi
    return t


def check_collision(state, bounds, obstacles):
    x, y, _ = state
    if x < bounds['x_min'] + COLLISION_RADIUS or x > bounds['x_max'] - COLLISION_RADIUS:
        return True
    if y < bounds['y_min'] + COLLISION_RADIUS or y > bounds['y_max'] - COLLISION_RADIUS:
        return True
    for ox, oy, orr in obstacles:
        if math.hypot(x - ox, y - oy) < COLLISION_RADIUS + orr:
            return True
    return False


def v_omega_to_wheel(v, omega):
    vl = v - omega * TRACK_WIDTH / 2
    vr = v + omega * TRACK_WIDTH / 2
    return max(-V_MAX, min(V_MAX, vl)), max(-V_MAX, min(V_MAX, vr))


class CarController:
    """BC policy controller with safety layer for real car."""

    def __init__(self, model_path, scene_name='sparse', use_safety_filter=True):
        self.policy = BCPolicy(model_path)
        self.scene = SCENES[scene_name]
        self.goal = self.scene['goal']
        self.obstacles = self.scene['obstacles']
        self.bounds = self.scene['bounds']
        self.use_safety_filter = use_safety_filter
        self.state = list(self.scene['start_state'])
        self.history = deque(maxlen=100)
        self.stats = {'steps': 0, 'collision': False, 'success': False, 'timeout': False}

    def get_state(self):
        return tuple(self.state)

    def get_observation(self):
        """Get current observation from simulated sensors."""
        return self.policy.state_to_array(tuple(self.state), self.goal)

    def step(self):
        """One control step: infer + safety check + execute."""
        state = tuple(self.state)

        # BC policy inference
        vl, vr = self.policy.infer(state, self.goal)

        # Convert to v, omega for safety check
        v = (vl + vr) / 2
        omega = (vl - vr) / TRACK_WIDTH

        # Safety: check predicted next state
        if self.use_safety_filter:
            x, y, theta = state
            nx = x + v * math.cos(theta) * DT
            ny = y + v * math.sin(theta) * DT
            ntheta = wrap_angle(theta + omega * DT)

            if check_collision((nx, ny, ntheta), self.bounds, self.obstacles):
                # Emergency stop
                vl, vr = 0.0, 0.0
                v, omega = 0.0, 0.0
                self.stats['collision'] = True

        # Execute
        v_phys = (vl + vr) / 2
        omega_phys = (vl - vr) / TRACK_WIDTH
        x, y, theta = self.state
        nx = x + v_phys * math.cos(theta) * DT
        ny = y + v_phys * math.sin(theta) * DT
        ntheta = wrap_angle(theta + omega_phys * DT)

        self.state = [nx, ny, ntheta]
        self.history.append((nx, ny, ntheta, vl, vr))
        self.stats['steps'] += 1

        # Check goal
        if math.hypot(nx - self.goal[0], ny - self.goal[1]) < GOAL_THRESHOLD:
            self.stats['success'] = True

        return vl, vr

    def reset(self, start_state=None):
        if start_state:
            self.state = list(start_state)
        else:
            self.state = list(self.scene['start_state'])
        self.history.clear()
        self.stats = {'steps': 0, 'collision': False, 'success': False, 'timeout': False}

    def run_episode(self, max_steps=400):
        for _ in range(max_steps):
            vl, vr = self.step()
            if self.stats['success']:
                return True
            if self.stats['collision']:
                return False
        self.stats['timeout'] = True
        return False


def main():
    parser = argparse.ArgumentParser(description='Scout Mini BC Policy Controller')
    parser.add_argument('--scene', default='sparse', choices=['sparse', 'dense', 'narrow'])
    parser.add_argument('--model_path', default=None, help='Path to .pth model file')
    parser.add_argument('--episodes', type=int, default=20, help='Number of test episodes')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Determine model path
    if args.model_path is None:
        script_dir = Path(__file__).parent
        args.model_path = str(script_dir / '..' / 'model' / 'sparse_multiscale_bc_policy.pth')

    print(f"Loading model: {args.model_path}")
    controller = CarController(args.model_path, scene_name=args.scene)

    np.random.seed(args.seed)
    successes = 0
    all_times = []

    for ep in range(args.episodes):
        controller.reset()
        t0 = time.time()
        result = controller.run_episode(max_steps=400)
        elapsed = time.time() - t0

        if result:
            successes += 1
        all_times.append(elapsed)

        status = "SUCCESS" if result else ("COLLISION" if controller.stats['collision'] else "TIMEOUT")
        print(f"  Ep {ep+1:2d}: {status} | steps={controller.stats['steps']} | time={elapsed:.1f}s")

    rate = successes / args.episodes
    avg_time = sum(all_times) / len(all_times)
    avg_infer = controller.policy.get_average_infer_time() * 1000

    print(f"\n{'='*50}")
    print(f"Results: {successes}/{args.episodes} ({rate*100:.0f}%)")
    print(f"Avg episode time: {avg_time:.2f}s")
    print(f"Avg inference time: {avg_infer:.2f}ms")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
