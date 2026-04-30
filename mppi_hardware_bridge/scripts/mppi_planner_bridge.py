# -*- coding: utf-8 -*-
from __future__ import print_function

import importlib
import inspect
import os
import sys
import time
import types


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DEFAULT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "mppi_hardware_bridge",
    "config",
    "lab_runtime.yaml",
)

PLANNER_MODULE_NAME = "src.planners.mppi_mujoco_receding_horizon_experiment"


class MppiPlannerBridgeError(Exception):
    pass


def _ensure_project_root_on_path():
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)


def _install_mujoco_env_stub():
    module_name = "src.envs.mujoco_point_env"
    if module_name in sys.modules:
        return

    stub = types.ModuleType(module_name)

    class MujocoPointEnv(object):
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "MujocoPointEnv is unavailable in the MPPI hardware bridge. "
                "The bridge only imports planner helper functions and must not "
                "launch MuJoCo, viewer, render, or env.step."
            )

    stub.MujocoPointEnv = MujocoPointEnv
    sys.modules[module_name] = stub


def _import_existing_planner():
    _ensure_project_root_on_path()

    try:
        return importlib.import_module(PLANNER_MODULE_NAME), False
    except ImportError as exc:
        message = str(exc)
        if "mujoco" not in message.lower():
            raise MppiPlannerBridgeError(
                "Failed to import existing MPPI planner module '{}': {}".format(
                    PLANNER_MODULE_NAME,
                    exc,
                )
            )

    sys.modules.pop(PLANNER_MODULE_NAME, None)
    _install_mujoco_env_stub()

    try:
        return importlib.import_module(PLANNER_MODULE_NAME), True
    except Exception as exc:
        raise MppiPlannerBridgeError(
            "Failed to import existing MPPI planner with MuJoCo stub: {}".format(exc)
        )


def _get_arg_names(func):
    try:
        return list(inspect.getfullargspec(func).args)
    except AttributeError:
        return list(inspect.getargspec(func).args)


def _validate_planner_api(planner):
    required = {
        "sample_control_sequences": [
            "nominal_sequence",
            "current_state",
            "dt",
            "obstacles",
            "robot_radius",
            "num_samples",
            "v_std",
            "omega_std",
            "v_min",
            "v_max",
            "omega_max",
        ],
        "rollout_control_sequence": [
            "start_state",
            "control_sequence",
            "dt",
        ],
        "trajectory_cost": [
            "trajectory",
            "control_sequence",
            "goal",
            "obstacles",
            "robot_radius",
            "bounds",
        ],
        "compute_weights": [
            "costs",
            "temperature",
        ],
        "weighted_update_sequence": [
            "sampled_sequences",
            "weights",
        ],
        "shift_sequence": [
            "sequence",
        ],
        "initialize_control_sequence": [
            "nominal_control",
            "horizon",
        ],
    }

    for name, required_args in required.items():
        if not hasattr(planner, name):
            raise MppiPlannerBridgeError(
                "Existing planner is missing required function '{}'. "
                "Please confirm the planner API before wiring MPPI dry-run.".format(
                    name
                )
            )

        arg_names = _get_arg_names(getattr(planner, name))
        for required_arg in required_args:
            if required_arg not in arg_names:
                raise MppiPlannerBridgeError(
                    "Planner function signature needs manual confirmation: "
                    "{}() is missing argument '{}'.".format(name, required_arg)
                )


def _cfg_get(cfg, path, default=None):
    current = cfg
    for name in path:
        if current is None:
            return default
        if isinstance(current, dict):
            if name not in current:
                return default
            current = current[name]
        else:
            if not hasattr(current, name):
                return default
            current = getattr(current, name)
    return current


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_float_tuple(values, expected_len, name):
    if values is None or len(values) != expected_len:
        raise MppiPlannerBridgeError(
            "{} must be a sequence of length {}.".format(name, expected_len)
        )
    try:
        return tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise MppiPlannerBridgeError(
            "{} must contain numeric values: {}".format(name, exc)
        )


def _coerce_bounds(bounds):
    if bounds is None:
        return None

    key_sets = [
        ("x_min", "x_max", "y_min", "y_max"),
        ("xmin", "xmax", "ymin", "ymax"),
    ]

    for keys in key_sets:
        try:
            return {
                "x_min": float(bounds[keys[0]]),
                "x_max": float(bounds[keys[1]]),
                "y_min": float(bounds[keys[2]]),
                "y_max": float(bounds[keys[3]]),
            }
        except (KeyError, TypeError, ValueError):
            pass

    try:
        return {
            "x_min": float(bounds.x_min),
            "x_max": float(bounds.x_max),
            "y_min": float(bounds.y_min),
            "y_max": float(bounds.y_max),
        }
    except AttributeError:
        return {
            "x_min": float(bounds.xmin),
            "x_max": float(bounds.xmax),
            "y_min": float(bounds.ymin),
            "y_max": float(bounds.ymax),
        }


def _bounds_from_cfg(cfg):
    boundary = _cfg_get(cfg, ["boundary"], None)
    if boundary is None:
        return None

    return {
        "x_min": float(_cfg_get(cfg, ["boundary", "xmin"])),
        "x_max": float(_cfg_get(cfg, ["boundary", "xmax"])),
        "y_min": float(_cfg_get(cfg, ["boundary", "ymin"])),
        "y_max": float(_cfg_get(cfg, ["boundary", "ymax"])),
    }


def _coerce_obstacles(obstacles):
    if obstacles is None:
        return []

    coerced = []
    for obstacle in obstacles:
        if len(obstacle) != 3:
            raise MppiPlannerBridgeError(
                "Each obstacle must be (x, y, radius); got {}".format(obstacle)
            )
        coerced.append(
            (
                float(obstacle[0]),
                float(obstacle[1]),
                float(obstacle[2]),
            )
        )
    return coerced


class MppiPlannerBridge(object):
    def __init__(self, cfg, bounds=None, obstacles=None):
        self.cfg = cfg
        self.defaults_used = {}
        self.planner, self.used_mujoco_stub = _import_existing_planner()
        _validate_planner_api(self.planner)

        self.horizon = int(self._cfg_mppi("horizon", 15))
        self.num_samples = int(self._cfg_mppi("num_samples", 250))
        self.temperature = float(self._cfg_mppi("temperature", 8.0))
        self.dt = float(self._cfg_mppi("dt", 0.2))

        self.v_min = float(self._cfg_mppi("v_min", 0.0))
        self.v_max = float(
            self._cfg_mppi(
                "v_max",
                float(_cfg_get(cfg, ["limits", "v_max"], 1.4)),
            )
        )
        self.omega_max = float(
            self._cfg_mppi(
                "omega_max",
                float(_cfg_get(cfg, ["limits", "w_max"], 1.2)),
            )
        )
        self.v_std = float(self._cfg_mppi("v_std", 0.25))
        self.omega_std = float(self._cfg_mppi("omega_std", 0.35))
        self.robot_radius = float(self._cfg_mppi("robot_radius", 0.25))

        self.use_anisotropic_sampling = _as_bool(
            _cfg_get(cfg, ["mppi", "use_anisotropic_sampling"], False)
        )
        self.sdf_influence_distance = float(
            _cfg_get(cfg, ["mppi", "sdf_influence_distance"], 1.2)
        )
        self.sigma_parallel = float(self._cfg_mppi("sigma_parallel", 0.08))
        self.sigma_perp = float(self._cfg_mppi("sigma_perp", 0.01))

        default_nominal_v = min(max(self.v_min, self.v_max), self.v_max)
        self.nominal_control = (
            float(self._cfg_mppi("nominal_v", default_nominal_v)),
            float(self._cfg_mppi("nominal_omega", 0.0)),
        )

        if bounds is None:
            bounds = _bounds_from_cfg(cfg)
        self.bounds = _coerce_bounds(bounds)
        if self.bounds is None:
            self.bounds = {
                "x_min": -10.0,
                "x_max": 10.0,
                "y_min": -10.0,
                "y_max": 10.0,
            }
            self.defaults_used["bounds"] = "wide fallback [-10, 10]"

        self.obstacles = _coerce_obstacles(obstacles)
        self.nominal_sequence = None
        self.reset()

    def _cfg_mppi(self, name, default):
        sentinel = object()
        value = _cfg_get(self.cfg, ["mppi", name], sentinel)
        if value is sentinel:
            self.defaults_used[name] = default
            return default
        return value

    def reset(self):
        self.nominal_sequence = self.planner.initialize_control_sequence(
            self.nominal_control,
            self.horizon,
        )

    def set_obstacles(self, obstacles):
        self.obstacles = _coerce_obstacles(obstacles)

    def clear_obstacles(self):
        self.obstacles = []

    def compute_control(self, current_state_exp, goal_xy):
        current_state = _as_float_tuple(
            current_state_exp,
            3,
            "current_state_exp",
        )
        goal = _as_float_tuple(goal_xy, 2, "goal_xy")

        if self.nominal_sequence is None or len(self.nominal_sequence) != self.horizon:
            self.reset()

        nominal_first_before = self.nominal_sequence[0]
        plan_start = time.time()

        sampled_sequences = self.planner.sample_control_sequences(
            nominal_sequence=self.nominal_sequence,
            current_state=current_state,
            dt=self.dt,
            obstacles=self.obstacles,
            robot_radius=self.robot_radius,
            num_samples=self.num_samples,
            v_std=self.v_std,
            omega_std=self.omega_std,
            v_min=self.v_min,
            v_max=self.v_max,
            omega_max=self.omega_max,
            use_anisotropic_sampling=self.use_anisotropic_sampling,
            sdf_influence_distance=self.sdf_influence_distance,
            sigma_parallel=self.sigma_parallel,
            sigma_perp=self.sigma_perp,
            goal=goal,
            bounds=self.bounds,
        )

        costs = []
        collisions = []

        for control_sequence in sampled_sequences:
            trajectory = self.planner.rollout_control_sequence(
                start_state=current_state,
                control_sequence=control_sequence,
                dt=self.dt,
            )
            total_cost, collided = self.planner.trajectory_cost(
                trajectory=trajectory,
                control_sequence=control_sequence,
                goal=goal,
                obstacles=self.obstacles,
                robot_radius=self.robot_radius,
                bounds=self.bounds,
            )
            costs.append(float(total_cost))
            collisions.append(bool(collided))

        if not costs:
            raise MppiPlannerBridgeError(
                "Existing planner returned no sampled trajectories; cannot compute MPPI control."
            )

        weights = self.planner.compute_weights(costs, self.temperature)
        updated_sequence = self.planner.weighted_update_sequence(
            sampled_sequences,
            weights,
        )

        if not updated_sequence:
            raise MppiPlannerBridgeError(
                "Existing planner returned an empty updated control sequence."
            )

        raw_mppi_control = (
            float(updated_sequence[0][0]),
            float(updated_sequence[0][1]),
        )

        safety_intervened = False
        proposed_control = raw_mppi_control
        if hasattr(self.planner, "make_boundary_safe_control"):
            proposed_control, safety_intervened = self.planner.make_boundary_safe_control(
                current_state=current_state,
                proposed_control=raw_mppi_control,
                dt=self.dt,
                bounds=self.bounds,
                robot_radius=self.robot_radius,
            )
            proposed_control = (
                float(proposed_control[0]),
                float(proposed_control[1]),
            )

        tail_control = updated_sequence[-1]
        self.nominal_sequence = self.planner.shift_sequence(
            updated_sequence,
            tail_control=tail_control,
        )

        best_idx = min(range(len(costs)), key=lambda idx: costs[idx])
        plan_time_sec = time.time() - plan_start

        debug = {
            "planner_type": "mppi",
            "proposed_control": proposed_control,
            "raw_mppi_control": raw_mppi_control,
            "horizon": self.horizon,
            "num_samples": self.num_samples,
            "temperature": self.temperature,
            "dt": self.dt,
            "use_anisotropic_sampling": self.use_anisotropic_sampling,
            "sdf_influence_distance": self.sdf_influence_distance,
            "sigma_parallel": self.sigma_parallel,
            "sigma_perp": self.sigma_perp,
            "nominal_sequence_first": self.nominal_sequence[0],
            "nominal_sequence_first_before": nominal_first_before,
            "goal_xy": goal,
            "current_state_exp": current_state,
            "safety_intervened": bool(safety_intervened),
            "best_cost": float(costs[best_idx]),
            "best_collision": bool(collisions[best_idx]),
            "cost_min": float(min(costs)),
            "cost_max": float(max(costs)),
            "collision_count": int(sum(1 for collided in collisions if collided)),
            "plan_time_sec": plan_time_sec,
            "bounds": self.bounds,
            "obstacle_count": len(self.obstacles),
            "used_mujoco_stub": bool(self.used_mujoco_stub),
            "defaults_used": dict(self.defaults_used),
        }

        return proposed_control, debug


def _run_basic_tests():
    _ensure_project_root_on_path()

    from scenario_config import load_lab_runtime_config

    print("Loading config: {}".format(DEFAULT_CONFIG_PATH))
    cfg = load_lab_runtime_config(DEFAULT_CONFIG_PATH)

    print("Initializing MppiPlannerBridge")
    bridge = MppiPlannerBridge(cfg=cfg)
    bridge.set_obstacles([(2.0, 2.0, 0.08)])

    current_state_exp = (0.0, 0.0, 0.0)
    goal_xy = (0.5, 0.0)

    proposed_control, debug = bridge.compute_control(
        current_state_exp=current_state_exp,
        goal_xy=goal_xy,
    )

    print("proposed_control =", proposed_control)
    print("debug =", debug)

    if not isinstance(proposed_control, (tuple, list)):
        raise AssertionError("proposed_control must be tuple/list")
    if len(proposed_control) != 2:
        raise AssertionError("proposed_control must have length 2")

    float(proposed_control[0])
    float(proposed_control[1])
    if debug.get("obstacle_count") != 1:
        raise AssertionError("debug.obstacle_count should reflect set_obstacles")

    bridge.clear_obstacles()
    if bridge.obstacles:
        raise AssertionError("clear_obstacles should remove all obstacles")

    print("Basic MPPI planner bridge test passed.")
    return 0


def main():
    try:
        return _run_basic_tests()
    except Exception as exc:
        print("")
        print("MPPI planner bridge basic test failed.")
        print("Original error: {}".format(exc))
        print("")
        return 1


if __name__ == "__main__":
    sys.exit(main())
