# -*- coding: utf-8 -*-
from __future__ import print_function

import importlib
import inspect
import math
import os
import sys
import time

try:
    from mppi_memory_field import MppiMemoryField
except ImportError:
    MppiMemoryField = None
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
        self.base_horizon = int(self.horizon)
        self.base_num_samples = int(self.num_samples)
        self.effective_horizon = int(self.horizon)
        self.effective_num_samples = int(self.num_samples)
        self.runtime_profile = "normal"
        self.degraded_reason = "normal"
        self.temperature = float(self._cfg_mppi("temperature", 8.0))
        self.dt = float(self._cfg_mppi("dt", 0.2))
        self.omega_cost_weight = float(self._cfg_mppi("omega_cost_weight", 0.05))
        self.domega_cost_weight = float(self._cfg_mppi("domega_cost_weight", 0.08))
        self.spin_in_place_cost_weight = float(
            self._cfg_mppi("spin_in_place_cost_weight", 0.30)
        )
        self.wrong_way_spin_cost_weight = float(
            self._cfg_mppi("wrong_way_spin_cost_weight", 0.40)
        )
        self.spin_v_threshold = float(self._cfg_mppi("spin_v_threshold", 0.015))
        self.spin_omega_threshold = float(self._cfg_mppi("spin_omega_threshold", 0.15))
        self.memory_eval_stride = int(self._cfg_mppi("memory_eval_stride", 3))

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
        limit_w_max = float(_cfg_get(cfg, ["limits", "w_max"], self.omega_max))
        if self.omega_max > limit_w_max:
            self.defaults_used["omega_max_clamped_to_limits_w_max"] = limit_w_max
            self.omega_max = limit_w_max
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
        self.runtime_context = {}
        self.memory_field = MppiMemoryField(cfg) if MppiMemoryField is not None else None
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
            self.effective_horizon,
        )

    def resize_nominal_sequence(self, new_horizon):
        new_horizon = max(1, int(new_horizon))
        if self.nominal_sequence is None:
            self.effective_horizon = new_horizon
            self.reset()
            return

        current_sequence = list(self.nominal_sequence)
        if len(current_sequence) > new_horizon:
            self.nominal_sequence = current_sequence[:new_horizon]
        elif len(current_sequence) < new_horizon:
            if current_sequence:
                tail_control = current_sequence[-1]
            else:
                tail_control = self.nominal_control
            self.nominal_sequence = current_sequence + [
                tail_control for _ in range(new_horizon - len(current_sequence))
            ]
        self.effective_horizon = new_horizon

    def set_runtime_profile(self, profile_name, num_samples=None, horizon=None, reason="none"):
        profile_name = str(profile_name or "normal")
        if num_samples is None:
            num_samples = self.base_num_samples
        if horizon is None:
            horizon = self.base_horizon

        num_samples = max(1, int(num_samples))
        horizon = max(1, int(horizon))
        if horizon != int(self.effective_horizon):
            self.resize_nominal_sequence(horizon)

        self.runtime_profile = profile_name
        self.degraded_reason = str(reason or "none")
        self.effective_num_samples = num_samples
        self.num_samples = num_samples
        self.horizon = horizon

    def set_obstacles(self, obstacles):
        self.obstacles = _coerce_obstacles(obstacles)

    def clear_obstacles(self):
        self.obstacles = []

    def set_runtime_context(self, context):
        self.runtime_context = dict(context or {})

    def update_memory(self, state, goal_distance, control, min_front_range, avoidance_state):
        if self.memory_field is None:
            return {}
        return self.memory_field.update(
            state=state,
            goal_distance=goal_distance,
            control=control,
            min_front_range=min_front_range,
            avoidance_state=avoidance_state,
        )

    def memory_debug(self, state=None):
        if self.memory_field is None:
            return {
                "memory_enabled": False,
                "memory_feature_count": 0,
                "memory_nearest_type": "none",
                "memory_nearest_distance": None,
                "memory_nearest_strength": 0.0,
                "memory_cost": 0.0,
                "memory_escape_direction": None,
                "memory_temperature_scale": 1.0,
                "stuck_feature_added": False,
                "memory_feature_hit_count": 0,
                "memory_decay_applied": False,
            }
        return self.memory_field.debug_snapshot(state)

    def control_sequence_cost(self, control_sequence, current_state, goal):
        omega_cost = 0.0
        domega_cost = 0.0
        spin_cost = 0.0
        wrong_way_spin_cost = 0.0
        previous_omega = None
        horizon_steps = max(1, len(control_sequence))
        context = self.runtime_context or {}
        avoidance_state = str(context.get("avoidance_state", "CLEAR"))
        scan_reason = str(context.get("scan_reason", "none"))
        front_clear = scan_reason == "front_clear"
        goal_dx = float(goal[0]) - float(current_state[0])
        goal_dy = float(goal[1]) - float(current_state[1])
        goal_bearing_error = 0.0
        if abs(goal_dx) > 1e-9 or abs(goal_dy) > 1e-9:
            goal_bearing = math.atan2(goal_dy, goal_dx)
            goal_bearing_error = self.planner.wrap_angle(
                goal_bearing - float(current_state[2])
            ) if hasattr(self.planner, "wrap_angle") else goal_bearing - float(current_state[2])
        goal_sign = 1 if goal_bearing_error > 0.0 else -1 if goal_bearing_error < 0.0 else 0
        strong_spin_suppression = avoidance_state in ("CLEAR", "GOAL_REACQUIRE")

        for v_value, omega_value in control_sequence:
            v_value = float(v_value)
            omega_value = float(omega_value)
            omega_cost += self.omega_cost_weight * (omega_value ** 2)
            if previous_omega is not None:
                domega_cost += self.domega_cost_weight * (
                    (omega_value - previous_omega) ** 2
                )
            previous_omega = omega_value
            if abs(v_value) <= self.spin_v_threshold and abs(omega_value) >= self.spin_omega_threshold:
                multiplier = 1.6 if strong_spin_suppression else 0.6
                spin_cost += (
                    multiplier
                    * self.spin_in_place_cost_weight
                    * (abs(omega_value) - self.spin_omega_threshold) ** 2
                )
            if (
                front_clear
                and strong_spin_suppression
                and goal_sign != 0
                and abs(goal_bearing_error) > math.radians(20.0)
                and abs(omega_value) >= self.spin_omega_threshold
                and (1 if omega_value > 0.0 else -1) != goal_sign
            ):
                wrong_way_spin_cost += self.wrong_way_spin_cost_weight * (
                    abs(omega_value) ** 2
                )

        total = (
            omega_cost + domega_cost + spin_cost + wrong_way_spin_cost
        ) / float(horizon_steps)
        return total, {
            "omega_cost": omega_cost / float(horizon_steps),
            "domega_cost": domega_cost / float(horizon_steps),
            "spin_in_place_cost": spin_cost / float(horizon_steps),
            "wrong_way_spin_cost": wrong_way_spin_cost / float(horizon_steps),
            "total_control_smooth_cost": total,
        }

    def compute_control(self, current_state_exp, goal_xy):
        current_state = _as_float_tuple(
            current_state_exp,
            3,
            "current_state_exp",
        )
        goal = _as_float_tuple(goal_xy, 2, "goal_xy")

        if self.nominal_sequence is None or len(self.nominal_sequence) != self.effective_horizon:
            self.reset()

        nominal_first_before = self.nominal_sequence[0]
        plan_start = time.time()

        sample_start = time.time()
        sampled_sequences = self.planner.sample_control_sequences(
            nominal_sequence=self.nominal_sequence,
            current_state=current_state,
            dt=self.dt,
            obstacles=self.obstacles,
            robot_radius=self.robot_radius,
            num_samples=self.effective_num_samples,
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
        sample_time_sec = time.time() - sample_start

        costs = []
        collisions = []
        control_costs = []
        memory_costs = []
        cost_component_debug = {
            "omega_cost": 0.0,
            "domega_cost": 0.0,
            "spin_in_place_cost": 0.0,
            "wrong_way_spin_cost": 0.0,
            "total_control_smooth_cost": 0.0,
        }

        rollout_cost_start = time.time()
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
            control_cost, control_component_debug = self.control_sequence_cost(
                control_sequence,
                current_state,
                goal,
            )
            memory_cost = 0.0
            if self.memory_field is not None:
                memory_cost = self.memory_field.cost_for_trajectory(
                    trajectory,
                    step_stride=self.memory_eval_stride,
                )
            total_cost = float(total_cost) + control_cost + memory_cost
            costs.append(float(total_cost))
            collisions.append(bool(collided))
            control_costs.append(float(control_cost))
            memory_costs.append(float(memory_cost))
            for key in cost_component_debug:
                cost_component_debug[key] += float(control_component_debug.get(key, 0.0))
        rollout_cost_time_sec = time.time() - rollout_cost_start

        if not costs:
            raise MppiPlannerBridgeError(
                "Existing planner returned no sampled trajectories; cannot compute MPPI control."
            )

        update_start = time.time()
        memory_temperature_scale = 1.0
        if self.memory_field is not None:
            memory_temperature_scale = self.memory_field.temperature_scale(
                current_state,
                stuck_trap_active=bool(self.runtime_context.get("stuck_trap_active", False)),
            )
        effective_temperature = self.temperature * memory_temperature_scale
        weights = self.planner.compute_weights(costs, effective_temperature)
        updated_sequence = self.planner.weighted_update_sequence(
            sampled_sequences,
            weights,
        )
        update_time_sec = time.time() - update_start

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
        if sampled_sequences:
            for key in cost_component_debug:
                cost_component_debug[key] /= float(len(sampled_sequences))
        memory_debug = self.memory_debug(current_state)
        memory_debug["memory_cost"] = float(memory_costs[best_idx]) if memory_costs else 0.0

        debug = {
            "planner_type": "mppi",
            "proposed_control": proposed_control,
            "raw_mppi_control": raw_mppi_control,
            "horizon": self.effective_horizon,
            "num_samples": self.effective_num_samples,
            "effective_horizon": self.effective_horizon,
            "effective_num_samples": self.effective_num_samples,
            "base_horizon": self.base_horizon,
            "base_num_samples": self.base_num_samples,
            "runtime_profile": self.runtime_profile,
            "degraded_reason": self.degraded_reason,
            "temperature": self.temperature,
            "effective_temperature": effective_temperature,
            "memory_temperature_scale": memory_temperature_scale,
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
            "best_control_smooth_cost": float(control_costs[best_idx]),
            "best_memory_cost": float(memory_costs[best_idx]),
            "best_collision": bool(collisions[best_idx]),
            "cost_min": float(min(costs)),
            "cost_max": float(max(costs)),
            "collision_count": int(sum(1 for collided in collisions if collided)),
            "plan_time_sec": plan_time_sec,
            "planner_compute_ms": plan_time_sec * 1000.0,
            "sampling_ms": sample_time_sec * 1000.0,
            "rollout_cost_ms": rollout_cost_time_sec * 1000.0,
            "weights_update_ms": update_time_sec * 1000.0,
            "bounds": self.bounds,
            "obstacle_count": len(self.obstacles),
            "used_mujoco_stub": bool(self.used_mujoco_stub),
            "defaults_used": dict(self.defaults_used),
        }
        debug.update(cost_component_debug)
        debug.update(memory_debug)

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
