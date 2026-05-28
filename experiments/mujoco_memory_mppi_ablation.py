# -*- coding: utf-8 -*-
from __future__ import print_function

import argparse
import csv
import math
import os
import random
import sys
import time

if sys.version_info[0] < 3:
    import imp
else:
    imp = None

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


GOAL = (3.0, 3.0)
START_STATE = (0.0, 0.0, 0.0)
ROBOT_RADIUS = 0.25
V_MIN = 0.0
V_MAX = 0.45
OMEGA_MAX = 1.2
V_STD = 0.18
OMEGA_STD = 0.35


def timer_now():
    try:
        return time.perf_counter()
    except AttributeError:
        return time.time()


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def ensure_project_on_path():
    root = project_root()
    if root not in sys.path:
        sys.path.insert(0, root)


def load_module_from_path(module_name, path):
    if imp is not None:
        return imp.load_source(module_name, path)

    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_planner_helpers():
    ensure_project_on_path()
    try:
        from src.planners import mppi_mujoco_receding_horizon_experiment as helpers
        return helpers
    except Exception:
        helper_path = os.path.join(
            project_root(),
            "src",
            "planners",
            "mppi_mujoco_receding_horizon_experiment.py",
        )
        return load_module_from_path("_mppi_mujoco_receding_horizon_experiment", helper_path)


def load_memory_field_class():
    memory_path = os.path.join(
        project_root(),
        "mppi_hardware_bridge",
        "scripts",
        "mppi_memory_field.py",
    )
    memory_module = load_module_from_path("_mppi_memory_field_for_ablation", memory_path)
    return memory_module.MppiMemoryField


def wrap_angle(theta):
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta


def step_state(state, control, dt):
    x_value, y_value, theta = state
    v_value, omega_value = control
    return (
        x_value + v_value * math.cos(theta) * dt,
        y_value + v_value * math.sin(theta) * dt,
        wrap_angle(theta + omega_value * dt),
    )


class SimpleKinematicEnv(object):
    def __init__(self, start_state):
        self.state = tuple(start_state)

    def reset(self, start_state):
        self.state = tuple(start_state)
        return self.state

    def get_state(self):
        return self.state

    def step(self, control, dt):
        self.state = step_state(self.state, control, dt)
        return self.state

    def close(self):
        return None


def make_default_scene():
    obstacles = [
        (1.20, 0.75, 0.23),
        (1.20, 1.10, 0.23),
        (1.20, 1.45, 0.23),
        (1.55, 1.45, 0.23),
        (1.90, 1.45, 0.23),
        (1.90, 1.10, 0.23),
        (2.25, 2.05, 0.20),
        (2.55, 2.35, 0.20),
    ]
    bounds = {
        "x_min": -0.5,
        "x_max": 3.5,
        "y_min": -0.5,
        "y_max": 3.5,
    }
    return {
        "start_state": START_STATE,
        "goal": GOAL,
        "robot_radius": ROBOT_RADIUS,
        "bounds": bounds,
        "obstacles": obstacles,
    }


def obstacle_clearance(x_value, y_value, obstacle, robot_radius):
    obs_x, obs_y, obs_radius = obstacle
    return math.hypot(x_value - obs_x, y_value - obs_y) - (robot_radius + obs_radius)


def compute_min_obstacle_distance(state, obstacles, robot_radius):
    if not obstacles:
        return float("inf")
    x_value, y_value, _ = state
    return min(
        obstacle_clearance(x_value, y_value, obstacle, robot_radius)
        for obstacle in obstacles
    )


def is_inside_effective_bounds(state, bounds, robot_radius):
    x_value, y_value, _ = state
    return (
        bounds["x_min"] + robot_radius <= x_value <= bounds["x_max"] - robot_radius
        and bounds["y_min"] + robot_radius <= y_value <= bounds["y_max"] - robot_radius
    )


def check_collision(state, obstacles, robot_radius, bounds=None):
    if compute_min_obstacle_distance(state, obstacles, robot_radius) <= 0.0:
        return True
    if bounds is not None and not is_inside_effective_bounds(state, bounds, robot_radius):
        return True
    return False


def compute_goal_distance(state, goal):
    return math.hypot(goal[0] - state[0], goal[1] - state[1])


def clamp_value(value, lower, upper):
    return max(lower, min(upper, value))


def clamp_control(control):
    v_value, omega_value = control
    return (
        clamp_value(v_value, V_MIN, V_MAX),
        clamp_value(omega_value, -OMEGA_MAX, OMEGA_MAX),
    )


def would_step_out_of_bounds(state, control, dt, bounds, robot_radius):
    return not is_inside_effective_bounds(step_state(state, control, dt), bounds, robot_radius)


def make_boundary_safe_control(state, proposed_control, dt, bounds, robot_radius):
    proposed_v, proposed_omega = proposed_control
    candidate_controls = [
        (proposed_v, proposed_omega),
        (0.75 * proposed_v, proposed_omega),
        (0.50 * proposed_v, proposed_omega),
        (0.25 * proposed_v, proposed_omega),
        (0.10 * proposed_v, proposed_omega),
        (0.0, proposed_omega),
        (0.0, 0.0),
    ]
    for candidate in candidate_controls:
        if not would_step_out_of_bounds(state, candidate, dt, bounds, robot_radius):
            return candidate, candidate != proposed_control
    return (0.0, 0.0), True


class _Section(object):
    pass


class _Config(object):
    pass


def initialize_memory_field(args):
    MppiMemoryField = load_memory_field_class()
    cfg = _Config()
    cfg.memory = _Section()
    cfg.memory.enable = True
    cfg.memory.max_features = 40
    cfg.memory.stuck_window_sec = max(2.0, 20.0 * float(args.dt))
    cfg.memory.stuck_pos_radius = 0.18
    cfg.memory.stuck_goal_progress_threshold = 0.12
    cfg.memory.spin_omega_threshold = 0.80
    cfg.memory.spin_v_threshold = 0.08
    cfg.memory.feature_merge_distance = 0.30
    cfg.memory.feature_radius = 0.60
    cfg.memory.feature_strength_initial = 1.5
    cfg.memory.feature_strength_max = 6.0
    cfg.memory.feature_decay = 0.997
    cfg.memory.local_min_cost_weight = 100.0
    cfg.memory.spin_trap_cost_weight = 90.0
    cfg.memory.low_progress_cost_weight = 80.0
    cfg.memory.escape_direction_cost_weight = 70.0
    cfg.memory.temperature_boost_max = 2.0
    return MppiMemoryField(cfg)


def get_memory_debug(memory_field, state=None):
    if memory_field is None:
        return {
            "memory_feature_count": 0,
            "memory_nearest_type": "none",
            "memory_temperature_scale": 1.0,
        }
    try:
        return memory_field.debug_snapshot(state)
    except TypeError:
        return memory_field.debug_snapshot()


def format_memory_cost_by_type(by_type):
    if not by_type:
        return ""
    parts = []
    for key in sorted(by_type.keys()):
        value = float(by_type.get(key, 0.0))
        if abs(value) <= 1e-9:
            continue
        parts.append("%s:%.3f" % (key, value))
    return "|".join(parts)


def format_memory_feature_types(memory_debug):
    feature_types = memory_debug.get("memory_feature_types", {})
    if not feature_types:
        return ""
    parts = []
    for key in sorted(feature_types.keys()):
        count = int(feature_types.get(key, 0))
        if count > 0:
            parts.append("%s:%d" % (key, count))
    return "|".join(parts)


def get_memory_feature_positions(memory_field):
    if memory_field is None:
        return []
    positions = []
    for feature in getattr(memory_field, "features", []):
        try:
            positions.append((float(feature.position[0]), float(feature.position[1])))
        except (AttributeError, TypeError, ValueError, IndexError):
            continue
    return positions


def update_memory_field(
    memory_field,
    state,
    goal_distance,
    control,
    min_obstacle_distance,
    avoidance_state,
    now,
):
    if memory_field is None:
        return get_memory_debug(None, state)
    try:
        debug = memory_field.update(
            state=state,
            goal_distance=goal_distance,
            control=control,
            min_front_range=min_obstacle_distance,
            avoidance_state=avoidance_state,
            now=now,
        )
    except TypeError:
        debug = memory_field.update(
            state,
            goal_distance,
            control,
            min_obstacle_distance,
            avoidance_state,
            now,
        )
    if debug is None:
        debug = get_memory_debug(memory_field, state)
    return debug


def classify_step_state(
    recent_states,
    recent_goal_distances,
    control,
    min_obstacle_distance,
    collision,
):
    v_value, omega_value = control
    spin = abs(omega_value) > 0.8 and abs(v_value) < 0.08
    stuck = False

    if len(recent_states) >= 20:
        window_states = recent_states[-20:]
        window_goals = recent_goal_distances[-20:]
        xs = [state[0] for state in window_states]
        ys = [state[1] for state in window_states]
        position_span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        goal_improvement = window_goals[0] - window_goals[-1]
        stuck = position_span < 0.12 and goal_improvement < 0.08

    if collision:
        avoidance_state = "HARD_STOP_RECOVERY"
    elif min_obstacle_distance < 0.35:
        avoidance_state = "CREEP_ESCAPE"
    elif stuck or spin:
        avoidance_state = "CREEP_ESCAPE"
    else:
        avoidance_state = "CLEAR"

    return stuck, spin, avoidance_state


def memory_cost_for_trajectory(memory_field, trajectory, memory_stride):
    if memory_field is None:
        return 0.0

    if hasattr(memory_field, "memory_cost_for_trajectory"):
        try:
            return float(memory_field.memory_cost_for_trajectory(trajectory, stride=memory_stride))
        except TypeError:
            return float(memory_field.memory_cost_for_trajectory(trajectory, memory_stride))

    if hasattr(memory_field, "cost_for_trajectory"):
        try:
            return float(memory_field.cost_for_trajectory(trajectory, step_stride=memory_stride))
        except TypeError:
            return float(memory_field.cost_for_trajectory(trajectory, memory_stride))

    if hasattr(memory_field, "memory_cost_for_state"):
        cost_sum = 0.0
        count = 0
        stride = max(1, int(memory_stride))
        for index, state in enumerate(trajectory):
            if index != len(trajectory) - 1 and index % stride != 0:
                continue
            cost_sum += float(memory_field.memory_cost_for_state(state))
            count += 1
        return cost_sum / float(max(count, 1))

    return 0.0


def memory_breakdown_for_trajectory(memory_field, trajectory, control_sequence=None, memory_stride=3):
    if memory_field is None:
        return {
            "total": 0.0,
            "by_type": {},
            "nearest_type": "none",
            "nearest_distance": None,
            "nearest_strength": 0.0,
            "active_feature_count": 0,
            "temperature_scale": 1.0,
        }
    if hasattr(memory_field, "memory_cost_breakdown_for_trajectory"):
        try:
            return memory_field.memory_cost_breakdown_for_trajectory(
                trajectory,
                controls=control_sequence,
                stride=memory_stride,
            )
        except TypeError:
            return memory_field.memory_cost_breakdown_for_trajectory(
                trajectory,
                stride=memory_stride,
            )
    return {
        "total": memory_cost_for_trajectory(memory_field, trajectory, memory_stride),
        "by_type": {},
        "nearest_type": "none",
        "nearest_distance": None,
        "nearest_strength": 0.0,
        "active_feature_count": 0,
        "temperature_scale": 1.0,
    }


def compute_rollout_cost(
    helpers,
    trajectory,
    control_sequence,
    goal,
    obstacles,
    robot_radius,
    bounds,
    memory_field=None,
    memory_enabled=False,
    memory_stride=3,
):
    base_result = helpers.trajectory_cost(
        trajectory=trajectory,
        control_sequence=control_sequence,
        goal=goal,
        obstacles=obstacles,
        robot_radius=robot_radius,
        bounds=bounds,
    )
    if isinstance(base_result, tuple):
        base_cost = float(base_result[0])
        collided = bool(base_result[1])
    else:
        base_cost = float(base_result)
        collided = False

    memory_cost = 0.0
    if memory_enabled and memory_field is not None:
        memory_cost = memory_cost_for_trajectory(memory_field, trajectory, memory_stride)

    return base_cost + memory_cost, collided, memory_cost


def effective_temperature_for_state(args, memory_field, memory_enabled, current_state):
    scale = 1.0
    if memory_enabled and memory_field is not None:
        if hasattr(memory_field, "temperature_scale_for_state"):
            scale = memory_field.temperature_scale_for_state(current_state)
        elif hasattr(memory_field, "temperature_scale"):
            scale = memory_field.temperature_scale(current_state)
    scale = clamp_value(float(scale), 1.0, 3.0)
    return float(args.temperature) * scale


def memory_temperature_scale_for_state(memory_field, memory_enabled, current_state):
    if not memory_enabled or memory_field is None:
        return 1.0
    if hasattr(memory_field, "temperature_scale_for_state"):
        return clamp_value(float(memory_field.temperature_scale_for_state(current_state)), 1.0, 3.0)
    if hasattr(memory_field, "temperature_scale"):
        return clamp_value(float(memory_field.temperature_scale(current_state)), 1.0, 3.0)
    return 1.0


def plan_one_step(
    helpers,
    current_state,
    nominal_sequence,
    args,
    goal,
    obstacles,
    robot_radius,
    bounds,
    memory_field,
    memory_enabled,
):
    t0 = timer_now()
    effective_temperature = effective_temperature_for_state(
        args,
        memory_field,
        memory_enabled,
        current_state,
    )
    sampled_sequences = helpers.sample_control_sequences(
        nominal_sequence=nominal_sequence,
        current_state=current_state,
        dt=args.dt,
        obstacles=obstacles,
        robot_radius=robot_radius,
        num_samples=args.num_samples,
        v_std=V_STD,
        omega_std=OMEGA_STD,
        v_min=V_MIN,
        v_max=V_MAX,
        omega_max=OMEGA_MAX,
        goal=goal,
        bounds=bounds,
    )

    costs = []
    trajectories = []
    collisions = []
    for control_sequence in sampled_sequences:
        trajectory = helpers.rollout_control_sequence(current_state, control_sequence, args.dt)
        total_cost, collided, _ = compute_rollout_cost(
            helpers=helpers,
            trajectory=trajectory,
            control_sequence=control_sequence,
            goal=goal,
            obstacles=obstacles,
            robot_radius=robot_radius,
            bounds=bounds,
            memory_field=memory_field,
            memory_enabled=memory_enabled,
            memory_stride=3,
        )
        costs.append(total_cost)
        trajectories.append(trajectory)
        collisions.append(collided)

    weights = helpers.compute_weights(costs, effective_temperature)
    updated_sequence = helpers.weighted_update_sequence(sampled_sequences, weights)
    updated_trajectory = helpers.rollout_control_sequence(
        current_state,
        updated_sequence,
        args.dt,
    )
    _, updated_collided, updated_memory_cost = compute_rollout_cost(
        helpers=helpers,
        trajectory=updated_trajectory,
        control_sequence=updated_sequence,
        goal=goal,
        obstacles=obstacles,
        robot_radius=robot_radius,
        bounds=bounds,
        memory_field=memory_field,
        memory_enabled=memory_enabled,
        memory_stride=3,
    )
    updated_memory_breakdown = memory_breakdown_for_trajectory(
        memory_field,
        updated_trajectory,
        control_sequence=updated_sequence,
        memory_stride=3,
    )
    t1 = timer_now()

    best_index = min(range(len(costs)), key=lambda index: costs[index])
    sorted_indices = sorted(range(len(costs)), key=lambda index: costs[index])
    aux_predicted_trajectories = [
        trajectories[index]
        for index in sorted_indices
        if index != best_index
    ][:2]
    return {
        "updated_sequence": updated_sequence,
        "proposed_control": updated_sequence[0],
        "effective_temperature": effective_temperature,
        "memory_temperature_scale": memory_temperature_scale_for_state(
            memory_field,
            memory_enabled,
            current_state,
        ),
        "planner_compute_ms": (t1 - t0) * 1000.0,
        "memory_cost": updated_memory_cost,
        "memory_cost_total": float(updated_memory_breakdown.get("total", updated_memory_cost)),
        "memory_cost_by_type": updated_memory_breakdown.get("by_type", {}),
        "memory_nearest_type": updated_memory_breakdown.get("nearest_type", "none"),
        "memory_nearest_distance": updated_memory_breakdown.get("nearest_distance"),
        "memory_nearest_strength": updated_memory_breakdown.get("nearest_strength", 0.0),
        "best_cost": costs[best_index],
        "best_collision": collisions[best_index] or updated_collided,
        "best_trajectory": trajectories[best_index],
        "aux_predicted_trajectories": aux_predicted_trajectories,
    }


def shift_nominal_sequence(helpers, updated_sequence):
    tail_control = updated_sequence[-1] if updated_sequence else (0.25, 0.0)
    try:
        return helpers.shift_sequence(updated_sequence, tail_control=tail_control)
    except TypeError:
        return helpers.shift_sequence(updated_sequence, tail_control)


def update_env_predicted_trajectories(env, plan_info):
    try:
        if hasattr(env, "update_best_predicted_trajectory"):
            env.update_best_predicted_trajectory(plan_info["best_trajectory"])
        if hasattr(env, "update_aux_predicted_trajectories"):
            env.update_aux_predicted_trajectories(
                plan_info.get("aux_predicted_trajectories", [])
            )
    except Exception as exc:
        print("warning: predicted trajectory overlay update failed: %s" % exc)


def render_env_if_needed(env, args, viewer_enabled, step_index):
    if not viewer_enabled or not hasattr(env, "render"):
        return viewer_enabled
    if step_index % args.render_every != 0:
        return viewer_enabled
    try:
        env.render(sleep_dt=args.render_sleep)
        return True
    except TypeError:
        try:
            env.render()
            if args.render_sleep > 0.0:
                time.sleep(args.render_sleep)
            return True
        except Exception as exc:
            print("warning: MuJoCo render failed; disabling live viewer: %s" % exc)
    except Exception as exc:
        print("warning: MuJoCo render failed; disabling live viewer: %s" % exc)

    try:
        env.close()
    except Exception:
        pass
    return False


def make_env(args, start_state):
    wants_mujoco = bool(args.use_mujoco or args.viewer)
    wants_viewer = bool(wants_mujoco and not args.no_viewer)

    if not wants_mujoco:
        env = SimpleKinematicEnv(start_state)
        env.reset(start_state)
        return env, False, False, "not_requested"

    ensure_project_on_path()
    MujocoPointEnv = None
    fallback_reason = ""

    try:
        from src.envs.mujoco_point_env import MujocoPointEnv as ImportedMujocoPointEnv
        MujocoPointEnv = ImportedMujocoPointEnv
    except Exception as exc:
        fallback_reason = str(exc)
        try:
            env_path = os.path.join(project_root(), "src", "envs", "mujoco_point_env.py")
            env_module = load_module_from_path("_mujoco_point_env_for_ablation", env_path)
            MujocoPointEnv = env_module.MujocoPointEnv
        except Exception as path_exc:
            fallback_reason = fallback_reason or str(path_exc)
            if str(path_exc):
                fallback_reason = str(path_exc)

    if MujocoPointEnv is None:
        print("MuJoCo unavailable, falling back to SimpleKinematicEnv")
        env = SimpleKinematicEnv(start_state)
        env.reset(start_state)
        return env, False, False, fallback_reason

    xml_path = os.path.join(
        project_root(),
        "src",
        "models",
        "mujoco",
        "scene_minimal_robot.xml",
    )
    try:
        env = MujocoPointEnv(xml_path=xml_path)
        viewer_enabled = False
        if wants_viewer:
            try:
                env.launch_viewer()
                viewer_enabled = True
            except Exception as viewer_exc:
                print("warning: MuJoCo viewer unavailable; continuing without live viewer: %s" % viewer_exc)
        env.reset(start_state)
        return env, True, viewer_enabled, ""
    except Exception as exc:
        print("MuJoCo unavailable, falling back to SimpleKinematicEnv")
        env = SimpleKinematicEnv(start_state)
        env.reset(start_state)
        return env, False, False, str(exc)


def average(values, default=0.0):
    if not values:
        return default
    return sum(values) / float(len(values))


def run_episode(helpers, args, memory_enabled, episode_index):
    scene = make_default_scene()
    start_state = scene["start_state"]
    goal = scene["goal"]
    obstacles = scene["obstacles"]
    robot_radius = scene["robot_radius"]
    bounds = scene["bounds"]
    run_id = "%s_ep%03d" % ("memory_on" if memory_enabled else "memory_off", episode_index + 1)

    seed_value = getattr(args, "episode_seed_override", None)
    if seed_value is None:
        seed_value = args.seed + episode_index
    random.seed(seed_value)
    np.random.seed(seed_value)

    env, used_mujoco, viewer_enabled, fallback_reason = make_env(args, start_state)
    memory_field = initialize_memory_field(args) if memory_enabled else None

    current_state = env.get_state()
    current_state = (current_state[0], current_state[1], wrap_angle(current_state[2]))
    nominal_sequence = helpers.initialize_control_sequence((0.25, 0.0), args.horizon)
    trajectory_rows = []
    executed_trajectory = [current_state]
    recent_states = [current_state]
    recent_goal_distances = [compute_goal_distance(current_state, goal)]
    best_predicted_history = []
    memory_feature_history = []
    collision_any = False

    try:
        for step_index in range(args.max_steps):
            plan_info = plan_one_step(
                helpers=helpers,
                current_state=current_state,
                nominal_sequence=nominal_sequence,
                args=args,
                goal=goal,
                obstacles=obstacles,
                robot_radius=robot_radius,
                bounds=bounds,
                memory_field=memory_field,
                memory_enabled=memory_enabled,
            )
            best_predicted_history.append(plan_info["best_trajectory"])
            update_env_predicted_trajectories(env, plan_info)

            proposed_control = clamp_control(plan_info["proposed_control"])
            executed_control, _ = make_boundary_safe_control(
                current_state,
                proposed_control,
                args.dt,
                bounds,
                robot_radius,
            )
            executed_control = clamp_control(executed_control)

            next_state = env.step(executed_control, args.dt)
            if next_state is None:
                next_state = env.get_state()
            current_state = (
                float(next_state[0]),
                float(next_state[1]),
                wrap_angle(float(next_state[2])),
            )
            viewer_enabled = render_env_if_needed(
                env,
                args,
                viewer_enabled,
                step_index,
            )

            executed_trajectory.append(current_state)
            goal_distance = compute_goal_distance(current_state, goal)
            min_obstacle_distance = compute_min_obstacle_distance(
                current_state,
                obstacles,
                robot_radius,
            )
            collision = check_collision(current_state, obstacles, robot_radius, bounds)
            collision_any = collision_any or collision

            recent_states.append(current_state)
            recent_goal_distances.append(goal_distance)
            stuck, spin, avoidance_state = classify_step_state(
                recent_states,
                recent_goal_distances,
                executed_control,
                min_obstacle_distance,
                collision,
            )

            debug = update_memory_field(
                memory_field=memory_field,
                state=current_state,
                goal_distance=goal_distance,
                control=executed_control,
                min_obstacle_distance=min_obstacle_distance,
                avoidance_state=avoidance_state,
                now=(step_index + 1) * args.dt,
            )
            feature_count = int(debug.get("memory_feature_count", 0))
            nearest_type = debug.get("memory_nearest_type", plan_info.get("memory_nearest_type", "none"))
            nearest_distance = debug.get("memory_nearest_distance", plan_info.get("memory_nearest_distance"))
            memory_feature_history.append(get_memory_feature_positions(memory_field))

            trajectory_rows.append(
                {
                    "run_id": run_id,
                    "step": step_index + 1,
                    "time": (step_index + 1) * args.dt,
                    "x": current_state[0],
                    "y": current_state[1],
                    "theta": current_state[2],
                    "v": executed_control[0],
                    "omega": executed_control[1],
                    "goal_distance": goal_distance,
                    "min_obstacle_distance": min_obstacle_distance,
                    "memory_enabled": bool(memory_enabled),
                    "memory_feature_count": feature_count,
                    "memory_nearest_type": nearest_type,
                    "memory_nearest_distance": nearest_distance,
                    "memory_cost": plan_info["memory_cost"],
                    "memory_cost_total": plan_info.get("memory_cost_total", plan_info["memory_cost"]),
                    "memory_cost_by_type": format_memory_cost_by_type(
                        plan_info.get("memory_cost_by_type", {})
                    ),
                    "memory_temperature_scale": plan_info.get("memory_temperature_scale", 1.0),
                    "effective_temperature": plan_info["effective_temperature"],
                    "planner_compute_ms": plan_info["planner_compute_ms"],
                    "collision": bool(collision),
                    "stuck": bool(stuck),
                    "spin": bool(spin),
                    "avoidance_state": avoidance_state,
                }
            )

            nominal_sequence = shift_nominal_sequence(helpers, plan_info["updated_sequence"])

            if step_index % 10 == 0:
                print(
                    "step=%03d goal_distance=%.3f min_obstacle_distance=%.3f "
                    "v=%.3f omega=%.3f memory_feature_count=%d planner_compute_ms=%.2f"
                    % (
                        step_index + 1,
                        goal_distance,
                        min_obstacle_distance,
                        executed_control[0],
                        executed_control[1],
                        feature_count,
                        plan_info["planner_compute_ms"],
                    )
                )

            if goal_distance < 0.25 or collision:
                break
    finally:
        env.close()

    final_state = executed_trajectory[-1]
    final_goal_distance = compute_goal_distance(final_state, goal)
    min_distances = [row["min_obstacle_distance"] for row in trajectory_rows]
    compute_times = [row["planner_compute_ms"] for row in trajectory_rows]
    v_values = [row["v"] for row in trajectory_rows]
    omega_values = [row["omega"] for row in trajectory_rows]
    memory_debug = get_memory_debug(memory_field, final_state)
    success = final_goal_distance < 0.25 and not collision_any
    memory_total_costs = [float(row["memory_cost_total"]) for row in trajectory_rows]
    memory_temperature_scales = [
        float(row["memory_temperature_scale"]) for row in trajectory_rows
    ]

    summary = {
        "run_id": run_id,
        "memory_enabled": bool(memory_enabled),
        "success": bool(success),
        "final_goal_distance": final_goal_distance,
        "total_steps": len(trajectory_rows),
        "min_obstacle_distance": min(min_distances) if min_distances else compute_min_obstacle_distance(final_state, obstacles, robot_radius),
        "stuck_steps": sum(1 for row in trajectory_rows if row["stuck"]),
        "spin_steps": sum(1 for row in trajectory_rows if row["spin"]),
        "mean_abs_omega": average([abs(value) for value in omega_values]),
        "mean_v": average(v_values),
        "memory_feature_count_final": int(memory_debug.get("memory_feature_count", 0)),
        "memory_nearest_type_final": memory_debug.get("memory_nearest_type", "none"),
        "memory_total_cost_mean": average(memory_total_costs),
        "memory_total_cost_max": max(memory_total_costs) if memory_total_costs else 0.0,
        "memory_temperature_scale_mean": average(memory_temperature_scales, default=1.0),
        "memory_feature_types_final": format_memory_feature_types(memory_debug),
        "average_planner_compute_ms": average(compute_times),
        "collision": bool(collision_any),
        "used_mujoco": bool(used_mujoco),
        "fallback_reason": fallback_reason,
    }

    return {
        "summary": summary,
        "trajectory_rows": trajectory_rows,
        "executed_trajectory": executed_trajectory,
        "best_predicted_history": best_predicted_history,
        "memory_feature_positions": get_memory_feature_positions(memory_field),
        "memory_feature_history": memory_feature_history,
    }


TRAJECTORY_FIELDS = [
    "run_id",
    "step",
    "time",
    "x",
    "y",
    "theta",
    "v",
    "omega",
    "goal_distance",
    "min_obstacle_distance",
    "memory_enabled",
    "memory_feature_count",
    "memory_nearest_type",
    "memory_nearest_distance",
    "memory_cost",
    "memory_cost_total",
    "memory_cost_by_type",
    "memory_temperature_scale",
    "effective_temperature",
    "planner_compute_ms",
    "collision",
    "stuck",
    "spin",
    "avoidance_state",
]


SUMMARY_FIELDS = [
    "run_id",
    "memory_enabled",
    "success",
    "final_goal_distance",
    "total_steps",
    "min_obstacle_distance",
    "stuck_steps",
    "spin_steps",
    "mean_abs_omega",
    "mean_v",
    "memory_feature_count_final",
    "memory_nearest_type_final",
    "memory_total_cost_mean",
    "memory_total_cost_max",
    "memory_temperature_scale_mean",
    "memory_feature_types_final",
    "average_planner_compute_ms",
    "collision",
    "used_mujoco",
    "fallback_reason",
]


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def open_csv_for_write(path):
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")


def write_trajectory_csv(path, rows):
    with open_csv_for_write(path) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAJECTORY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary_csv(path, rows):
    with open_csv_for_write(path) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_ablation(path, scene, plot_trajectories, memory_feature_positions):
    fig, ax = plt.subplots(figsize=(7.0, 6.5))
    bounds = scene["bounds"]
    start_state = scene["start_state"]
    goal = scene["goal"]

    for obs_x, obs_y, obs_radius in scene["obstacles"]:
        ax.add_patch(
            Circle(
                (obs_x, obs_y),
                obs_radius,
                fill=False,
                linewidth=1.8,
                color="#3b3b3b",
            )
        )

    ax.scatter([start_state[0]], [start_state[1]], s=60, marker="o", label="start")
    ax.scatter([goal[0]], [goal[1]], s=80, marker="*", label="goal")

    labels = [
        (False, "memory off", "#4c78a8"),
        (True, "memory on", "#f58518"),
    ]
    for memory_enabled, label, color in labels:
        trajectories = plot_trajectories.get(memory_enabled, [])
        for index, trajectory in enumerate(trajectories):
            xs = [state[0] for state in trajectory]
            ys = [state[1] for state in trajectory]
            line_label = label if index == 0 else None
            ax.plot(xs, ys, linewidth=2.0, alpha=0.85, color=color, label=line_label)

    if memory_feature_positions:
        feature_xs = [position[0] for position in memory_feature_positions]
        feature_ys = [position[1] for position in memory_feature_positions]
        ax.scatter(
            feature_xs,
            feature_ys,
            s=48,
            marker="x",
            linewidths=1.8,
            color="#d62728",
            label="memory features",
        )

    ax.set_title("Memory-Augmented MPPI Ablation")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(bounds["x_min"], bounds["x_max"])
    ax.set_ylim(bounds["y_min"], bounds["y_max"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _feature_positions_for_frame(memory_feature_positions, state_index):
    if not memory_feature_positions:
        return []

    first = memory_feature_positions[0]
    try:
        float(first[0])
        float(first[1])
        return memory_feature_positions
    except (TypeError, ValueError, IndexError):
        history_index = max(0, min(state_index - 1, len(memory_feature_positions) - 1))
        return memory_feature_positions[history_index]


def plot_episode_gif(
    trajectory_rows,
    obstacles,
    start_state,
    goal,
    robot_radius,
    bounds,
    output_path,
    title,
    memory_feature_positions=None,
    best_predicted_history=None,
    fps=8,
):
    if best_predicted_history is None:
        best_predicted_history = []
    if memory_feature_positions is None:
        memory_feature_positions = []

    states = [start_state]
    for row in trajectory_rows:
        states.append((float(row["x"]), float(row["y"]), float(row["theta"])))
    if not states:
        return False

    max_frames = 120
    stride = max(1, int(math.ceil(len(states) / float(max_frames))))
    frame_indices = list(range(0, len(states), stride))
    if frame_indices[-1] != len(states) - 1:
        frame_indices.append(len(states) - 1)

    try:
        from matplotlib import animation
    except Exception as exc:
        print("warning: matplotlib animation unavailable; skipping GIF: %s" % exc)
        return False

    fig, ax = plt.subplots(figsize=(6.5, 6.0))

    for obs_x, obs_y, obs_radius in obstacles:
        ax.add_patch(
            Circle(
                (obs_x, obs_y),
                obs_radius,
                fill=False,
                linewidth=1.6,
                color="#3b3b3b",
            )
        )

    ax.scatter([start_state[0]], [start_state[1]], s=55, marker="o", label="start")
    ax.scatter([goal[0]], [goal[1]], s=80, marker="*", label="goal")

    executed_line, = ax.plot([], [], linewidth=2.0, color="#4c78a8", label="executed")
    predicted_line, = ax.plot([], [], "--", linewidth=1.4, color="#54a24b", label="best predicted")
    memory_scatter = ax.scatter(
        [],
        [],
        s=45,
        marker="x",
        linewidths=1.8,
        color="#d62728",
        label="memory features",
    )
    robot_circle = Circle(
        (states[0][0], states[0][1]),
        robot_radius,
        fill=False,
        linewidth=2.0,
        color="#f58518",
        label="robot",
    )
    ax.add_patch(robot_circle)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(bounds["x_min"], bounds["x_max"])
    ax.set_ylim(bounds["y_min"], bounds["y_max"])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="best")

    def animate(state_index):
        state = states[state_index]
        xs = [item[0] for item in states[: state_index + 1]]
        ys = [item[1] for item in states[: state_index + 1]]
        executed_line.set_data(xs, ys)
        robot_circle.center = (state[0], state[1])

        if best_predicted_history and state_index > 0:
            pred_index = max(0, min(state_index - 1, len(best_predicted_history) - 1))
            predicted = best_predicted_history[pred_index]
            predicted_line.set_data(
                [item[0] for item in predicted],
                [item[1] for item in predicted],
            )
        else:
            predicted_line.set_data([], [])

        feature_positions = _feature_positions_for_frame(
            memory_feature_positions,
            state_index,
        )
        if feature_positions:
            memory_scatter.set_offsets(
                np.array(
                    [[float(item[0]), float(item[1])] for item in feature_positions],
                    dtype=float,
                )
            )
        else:
            memory_scatter.set_offsets(np.empty((0, 2), dtype=float))
        return executed_line, predicted_line, memory_scatter, robot_circle

    try:
        anim = animation.FuncAnimation(
            fig,
            animate,
            frames=frame_indices,
            interval=1000.0 / float(max(1, int(fps))),
            blit=False,
        )
        ensure_dir(os.path.dirname(output_path))
        anim.save(output_path, writer="pillow", fps=max(1, int(fps)))
        plt.close(fig)
        return True
    except Exception as exc:
        plt.close(fig)
        print("warning: GIF save failed for %s: %s" % (output_path, exc))
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="MuJoCo/fallback memory MPPI ablation")
    parser.add_argument("--memory-disable", action="store_true", help="run memory off baseline only")
    parser.add_argument("--memory-enable", action="store_true", help="run memory on only")
    parser.add_argument("--run-both", action="store_true", help="run memory off and memory on")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=8.0)
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-dir", default=os.path.join("experiments", "results"))
    parser.add_argument("--no-viewer", action="store_true", default=False)
    parser.add_argument("--use-mujoco", action="store_true")
    parser.add_argument("--viewer", action="store_true", help="enable MuJoCo viewer and imply --use-mujoco")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-sleep", type=float, default=0.03)
    parser.add_argument("--save-gif", action="store_true")
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--gif-fps", type=int, default=8)
    parser.add_argument("--benchmark", action="store_true", help="run memory off/on over explicit seed list")
    parser.add_argument("--seeds", default="11,12,13", help="comma-separated seeds for --benchmark")
    args = parser.parse_args()
    if args.viewer:
        args.use_mujoco = True
        args.no_viewer = False
    args.render_every = max(1, int(args.render_every))
    args.gif_fps = max(1, int(args.gif_fps))
    return args


def parse_seed_list(seed_text):
    seeds = []
    for part in str(seed_text).split(","):
        part = part.strip()
        if not part:
            continue
        seeds.append(int(part))
    if not seeds:
        seeds = [11]
    return seeds


def select_memory_modes(args):
    if args.run_both or (not args.memory_disable and not args.memory_enable):
        return [False, True]
    modes = []
    if args.memory_disable:
        modes.append(False)
    if args.memory_enable:
        modes.append(True)
    return modes


def print_run_summary(summary, trajectory_csv, summary_csv, figure_path):
    print("success=%s" % summary["success"])
    print("final_goal_distance=%.3f" % summary["final_goal_distance"])
    print("total_steps=%d" % summary["total_steps"])
    print("collision=%s" % summary["collision"])
    print("stuck_steps=%d" % summary["stuck_steps"])
    print("spin_steps=%d" % summary["spin_steps"])
    print("average_planner_compute_ms=%.2f" % summary["average_planner_compute_ms"])
    print("trajectory_csv=%s" % trajectory_csv)
    print("summary_csv=%s" % summary_csv)
    print("figure_path=%s" % figure_path)
    print(
        "note=offline ablation uses fallback/global circular obstacles; "
        "live sim uses synthetic LaserScan -> local_obstacle_layer."
    )


def main():
    args = parse_args()
    helpers = load_planner_helpers()
    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    summary_csv = os.path.join(output_dir, "mujoco_memory_mppi_ablation.csv")
    trajectory_csv_by_mode = {
        False: os.path.join(output_dir, "mujoco_memory_mppi_trajectory_memory_off.csv"),
        True: os.path.join(output_dir, "mujoco_memory_mppi_trajectory_memory_on.csv"),
    }
    gif_path_by_mode = {
        False: os.path.join(output_dir, "mujoco_memory_mppi_ablation_memory_off.gif"),
        True: os.path.join(output_dir, "mujoco_memory_mppi_ablation_memory_on.gif"),
    }
    figure_path = os.path.join(output_dir, "mujoco_memory_mppi_ablation.png")

    modes = select_memory_modes(args)
    if args.benchmark and not args.run_both and not args.memory_disable and not args.memory_enable:
        modes = [False, True]
    episode_seeds = parse_seed_list(args.seeds) if args.benchmark else None
    all_summary_rows = []
    trajectory_rows_by_mode = {False: [], True: []}
    plot_trajectories = {False: [], True: []}
    memory_feature_positions = []
    gif_result_by_mode = {}

    for memory_enabled in modes:
        print("=== Running memory_enabled=%s ===" % memory_enabled)
        if episode_seeds is None:
            seed_values = [None for _ in range(max(1, int(args.episodes)))]
        else:
            seed_values = list(episode_seeds)
        for episode_index, seed_value in enumerate(seed_values):
            args.episode_seed_override = seed_value
            if seed_value is not None:
                print("seed=%d" % seed_value)
            result = run_episode(helpers, args, memory_enabled, episode_index)
            summary = result["summary"]
            all_summary_rows.append(summary)
            trajectory_rows_by_mode[memory_enabled].extend(result["trajectory_rows"])
            plot_trajectories[memory_enabled].append(result["executed_trajectory"])
            gif_result_by_mode[memory_enabled] = result
            if memory_enabled:
                memory_feature_positions.extend(result["memory_feature_positions"])
            print_run_summary(
                summary,
                trajectory_csv_by_mode[memory_enabled],
                summary_csv,
                figure_path,
            )
        args.episode_seed_override = None

    for memory_enabled in modes:
        write_trajectory_csv(
            trajectory_csv_by_mode[memory_enabled],
            trajectory_rows_by_mode[memory_enabled],
        )
    write_summary_csv(summary_csv, all_summary_rows)
    plot_ablation(figure_path, make_default_scene(), plot_trajectories, memory_feature_positions)

    if args.save_gif and not args.no_gif:
        scene = make_default_scene()
        for memory_enabled in modes:
            result = gif_result_by_mode.get(memory_enabled)
            if result is None:
                continue
            title = "Memory-Augmented MPPI Ablation - %s" % (
                "memory on" if memory_enabled else "memory off"
            )
            saved = plot_episode_gif(
                trajectory_rows=result["trajectory_rows"],
                obstacles=scene["obstacles"],
                start_state=scene["start_state"],
                goal=scene["goal"],
                robot_radius=scene["robot_radius"],
                bounds=scene["bounds"],
                output_path=gif_path_by_mode[memory_enabled],
                title=title,
                memory_feature_positions=result.get("memory_feature_history", []),
                best_predicted_history=result.get("best_predicted_history", []),
                fps=args.gif_fps,
            )
            if saved:
                print("gif_path=%s" % gif_path_by_mode[memory_enabled])

    print("summary_csv=%s" % summary_csv)
    for memory_enabled in modes:
        print("trajectory_csv=%s" % trajectory_csv_by_mode[memory_enabled])
        if args.save_gif and not args.no_gif:
            print("gif_path=%s" % gif_path_by_mode[memory_enabled])
    print("figure_path=%s" % figure_path)


if __name__ == "__main__":
    main()
