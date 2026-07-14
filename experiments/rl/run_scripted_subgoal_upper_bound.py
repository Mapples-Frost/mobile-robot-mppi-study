#!/usr/bin/env python3
"""Upper-bound diagnostic for the two-dimensional local-subgoal MPPI prior.

The offline collision-free route is privileged and diagnostic-only.  It is
used solely to generate the same normalized distance/bearing action expected
from SAC.  MPPI retains the original point-goal cost, receives obstacles only
from LaserScan/local_obstacle_layer, and remains behind scan_guard.
"""

import argparse
import copy
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Rectangle
from matplotlib.transforms import Affine2D

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import deep_merge, git_sha, load_yaml
from mobile_robot_mppi.evaluation.scene_feasibility import (
    audit_static_scene,
    point_clearance,
)
from mobile_robot_mppi.rl.environment import MppiPriorEnv
from mobile_robot_mppi.rl.scripted_subgoal import (
    ScriptedPolylineSubgoal,
    ScriptedSubgoalConfig,
)


DEFAULT_SCENES = (
    "configs/research/mujoco_strong_mppi_baseline.yaml",
    "configs/research/mujoco_u_trap_long_board.yaml",
)


def _csv_values(text, cast=float):
    return [cast(value.strip()) for value in str(text).split(",") if value.strip()]


def _resolved_scene(rl_config, scene_path, seed, num_samples, max_steps):
    scene = load_yaml(scene_path)
    config = deep_merge(scene, {
        "planner": copy.deepcopy(rl_config["planner"]),
        "rl": copy.deepcopy(rl_config["rl"]),
    })
    config["experiment"]["seed"] = int(seed)
    config["experiment"]["max_steps"] = int(max_steps)
    config["planner"].update({
        "sampling_prior": "rl",
        "prediction_mode": "nominal",
        "num_samples": int(num_samples),
    })
    config.setdefault("memory", {})["enable"] = False
    config["rl"]["enabled"] = True
    config["rl"].setdefault("gate", {})["mode"] = "none"
    training = config["rl"].setdefault("training", {})
    training["rl_prior_alpha"] = 1.0
    training["initial_state_noise"] = [
        0.0 for _ in config["experiment"]["initial_state"]
    ]
    return config


def _offline_route(config, margin, resolution, endpoint_margin=None):
    start = config["experiment"]["initial_state"][:2]
    goal = config["task"]["position"]
    radius = config["plant"]["robot"]["collision_radius"]
    endpoint_margin = (
        float(margin) if endpoint_margin is None else float(endpoint_margin)
    )
    if endpoint_margin < 0.0 or endpoint_margin > float(margin):
        raise ValueError("endpoint route margin must be in [0, route margin]")
    endpoint_audit = audit_static_scene(
        config["scene"], start, goal, radius,
        margin=endpoint_margin, resolution=float(resolution), include_path=True,
    )
    if not endpoint_audit["path_exists"]:
        raise ValueError(
            "no endpoint-feasible diagnostic route for %s"
            % endpoint_audit["scene"]
        )
    endpoint_points = np.asarray(endpoint_audit["path"], dtype=np.float64)
    transition_index = len(endpoint_points) - 1
    high_clearance_audit = endpoint_audit
    points = endpoint_points
    if endpoint_margin < float(margin) - 1e-12:
        obstacles = config["scene"].get("obstacles", ())
        high_clearance_audit = None
        # Work backward from the goal and switch to the relaxed suffix at the
        # closest low-margin waypoint that is itself valid at the high margin.
        for index in range(len(endpoint_points) - 1, -1, -1):
            candidate = endpoint_points[index]
            clearance = point_clearance(
                candidate[0], candidate[1], obstacles, radius
            )
            if clearance + 1e-12 < float(margin):
                continue
            candidate_audit = audit_static_scene(
                config["scene"], start, candidate, radius,
                margin=float(margin), resolution=float(resolution),
                include_path=True,
            )
            if candidate_audit["path_exists"]:
                high_clearance_audit = candidate_audit
                transition_index = index
                high_points = np.asarray(
                    candidate_audit["path"], dtype=np.float64
                )
                points = np.vstack((
                    high_points,
                    candidate[None, :],
                    endpoint_points[index + 1:],
                ))
                break
        if high_clearance_audit is None:
            raise ValueError(
                "no high-clearance prefix for diagnostic route in %s"
                % endpoint_audit["scene"]
            )
    exact_start = np.asarray(start, dtype=np.float64)
    exact_goal = np.asarray(goal, dtype=np.float64)
    if np.linalg.norm(points[0] - exact_start) > 1e-9:
        points = np.vstack((exact_start, points))
    else:
        points[0] = exact_start
    if np.linalg.norm(points[-1] - exact_goal) > 1e-9:
        points = np.vstack((points, exact_goal))
    else:
        points[-1] = exact_goal
    # Grid paths can contain duplicate endpoints after exact endpoint insertion.
    keep = np.concatenate((
        np.asarray((True,)),
        np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9,
    ))
    points = points[keep]
    audit = {
        "scene": str(config["scene"].get("name", "unknown")),
        "path_exists": True,
        "margin": float(margin),
        "endpoint_margin": endpoint_margin,
        "resolution": float(resolution),
        "transition_index_in_endpoint_route": int(transition_index),
        "high_clearance_audit": high_clearance_audit,
        "endpoint_audit": endpoint_audit,
        "path": points.tolist(),
    }
    return points, audit


def _run_episode(config, route, lookahead, seed, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = MppiPriorEnv(config, ROOT, seed=seed)
    with (run_dir / "config_resolved.json").open("w", encoding="utf-8") as handle:
        json.dump(
            environment.config, handle, indent=2, sort_keys=True, allow_nan=False
        )
    policy = ScriptedPolylineSubgoal(
        route,
        environment.config["rl"]["prior"],
        ScriptedSubgoalConfig(lookahead_distance=float(lookahead)),
    )
    environment.reset(seed=seed)
    policy.reset()
    rows = []
    last_info = {}
    previous_xy = None
    path_length = 0.0
    safety_interventions = 0
    minimum_clearance = float("inf")
    planner_times = []
    localization_position_errors = []
    localization_heading_errors = []
    try:
        while True:
            observed_pose = environment.perceived.observation.pose
            truth_before = environment.truth.pose
            position_error = float(np.hypot(
                observed_pose.x - truth_before.x,
                observed_pose.y - truth_before.y,
            ))
            heading_error = float(np.arctan2(
                np.sin(observed_pose.theta - truth_before.theta),
                np.cos(observed_pose.theta - truth_before.theta),
            ))
            localization_position_errors.append(position_error)
            localization_heading_errors.append(heading_error)
            action, diagnostic = policy.action(observed_pose)
            _, reward, terminated, truncated, last_info = environment.step(action)
            truth = environment.truth
            xy = np.asarray((truth.pose.x, truth.pose.y), dtype=np.float64)
            if previous_xy is not None:
                path_length += float(np.linalg.norm(xy - previous_xy))
            previous_xy = xy
            safety_interventions += int(bool(last_info["safety_override"]))
            minimum_clearance = min(
                minimum_clearance, float(last_info["minimum_clearance"])
            )
            planner_times.append(float(last_info["planner_compute_ms"]))
            prior = dict(last_info.get("prior", {}))
            rows.append({
                "step": environment.steps,
                "time": float(truth.timestamp),
                "x": float(truth.pose.x),
                "y": float(truth.pose.y),
                "theta": float(truth.pose.theta),
                "v": float(truth.twist.v),
                "omega": float(truth.twist.omega),
                "observed_x_before_step": float(observed_pose.x),
                "observed_y_before_step": float(observed_pose.y),
                "observed_theta_before_step": float(observed_pose.theta),
                "localization_position_error": position_error,
                "localization_heading_error": heading_error,
                "goal_distance": float(last_info["goal_distance"]),
                "policy_distance_action": float(action[0]),
                "policy_bearing_action": float(action[1]),
                "route_progress": diagnostic["route_progress"],
                "target_progress": diagnostic["target_progress"],
                "cross_track_error": diagnostic["cross_track_error"],
                "target_x": diagnostic["target_x"],
                "target_y": diagnostic["target_y"],
                "target_distance": diagnostic["target_distance"],
                "decoded_subgoal_distance": float(prior.get("subgoal_distance", 0.0)),
                "decoded_subgoal_bearing": float(prior.get("subgoal_bearing", 0.0)),
                "subgoal_decoder": str(prior.get("subgoal_decoder", "unknown")),
                "decoder_initial_v": float(prior.get("initial_v", 0.0)),
                "decoder_initial_omega": float(prior.get("initial_omega", 0.0)),
                "decoder_terminal_distance": float(
                    prior.get("predicted_terminal_distance", 0.0)
                ),
                "proposed_v": float(last_info["proposed_control"][0]),
                "proposed_omega": float(last_info["proposed_control"][1]),
                "executed_v": float(last_info["executed_control"][0]),
                "executed_omega": float(last_info["executed_control"][1]),
                "safety_override": bool(last_info["safety_override"]),
                "minimum_clearance": float(last_info["minimum_clearance"]),
                "planner_compute_ms": float(last_info["planner_compute_ms"]),
                "reward": float(reward),
            })
            if terminated or truncated:
                break
    finally:
        environment.close()
    with (run_dir / "trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "scene": str(config["scene"]["name"]),
        "method": "scripted_local_subgoal",
        "subgoal_decoder": str(
            environment.parameterization.config.subgoal_decoder
        ),
        "lookahead": float(lookahead),
        "seed": int(seed),
        "success": bool(last_info.get("success", False)),
        "collision": bool(last_info.get("collision", False)),
        "steps": len(rows),
        "final_goal_distance": float(last_info.get("goal_distance", float("inf"))),
        "trajectory_length": float(path_length),
        "minimum_clearance": float(minimum_clearance),
        "safety_interventions": int(safety_interventions),
        "planner_compute_ms_mean": float(np.mean(planner_times)),
        "planner_compute_ms_max": float(np.max(planner_times)),
        "localization_position_rmse": float(np.sqrt(np.mean(
            np.square(localization_position_errors)
        ))),
        "localization_position_error_max": float(np.max(
            localization_position_errors
        )),
        "localization_heading_rmse": float(np.sqrt(np.mean(
            np.square(localization_heading_errors)
        ))),
        "route_progress_final": float(policy.progress),
        "route_length": float(policy.total_length),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
    return summary


def _aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["scene"], row["lookahead"]), []).append(row)
    summaries = []
    for (scene, lookahead), values in sorted(groups.items()):
        summaries.append({
            "scene": scene,
            "method": "scripted_local_subgoal",
            "lookahead": lookahead,
            "seeds": len(values),
            "success_rate": float(np.mean([row["success"] for row in values])),
            "collision_rate": float(np.mean([row["collision"] for row in values])),
            "final_goal_distance_mean": float(np.mean([
                row["final_goal_distance"] for row in values
            ])),
            "final_goal_distance_std": float(np.std([
                row["final_goal_distance"] for row in values
            ])),
            "minimum_clearance_mean": float(np.mean([
                row["minimum_clearance"] for row in values
            ])),
            "safety_interventions_mean": float(np.mean([
                row["safety_interventions"] for row in values
            ])),
            "planner_compute_ms_mean": float(np.mean([
                row["planner_compute_ms_mean"] for row in values
            ])),
            "localization_position_rmse_mean": float(np.mean([
                row["localization_position_rmse"] for row in values
            ])),
            "localization_position_error_max": float(np.max([
                row["localization_position_error_max"] for row in values
            ])),
        })
    return summaries


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _draw_obstacles(axis, config):
    for obstacle in config.get("scene", {}).get("obstacles", ()):
        position = obstacle.get("position", (0.0, 0.0))
        if str(obstacle.get("type", "cylinder")) == "box":
            size = obstacle.get("size", (0.25, 0.25))
            patch = Rectangle(
                (-float(size[0]), -float(size[1])),
                2.0 * float(size[0]), 2.0 * float(size[1]),
                facecolor="0.75", edgecolor="0.25", alpha=0.7,
            )
            patch.set_transform(
                Affine2D().rotate(float(obstacle.get("yaw", 0.0))).translate(
                    float(position[0]), float(position[1])
                ) + axis.transData
            )
            axis.add_patch(patch)
        else:
            axis.add_patch(Circle(
                (float(position[0]), float(position[1])),
                float(obstacle.get("radius", 0.25)),
                facecolor="0.75", edgecolor="0.25", alpha=0.7,
            ))


def _plot(output, rows, configs, routes, representative_seed):
    scenes = list(configs)
    lookaheads = sorted(set(float(row["lookahead"]) for row in rows))
    figure, axes = plt.subplots(
        len(scenes), 1, figsize=(7.4, 5.2 * len(scenes)), squeeze=False,
        constrained_layout=True,
    )
    for scene_index, scene in enumerate(scenes):
        axis = axes[scene_index, 0]
        _draw_obstacles(axis, configs[scene])
        route = routes[scene]
        axis.plot(route[:, 0], route[:, 1], "--", color="0.35", label="privileged route")
        for lookahead in lookaheads:
            source = (
                output / "runs" / scene / ("lookahead_%03d" % round(100 * lookahead))
                / ("seed_%d" % representative_seed) / "trajectory.csv"
            )
            if not source.exists():
                continue
            data = np.genfromtxt(
                str(source), delimiter=",", names=True, dtype=None, encoding="utf-8"
            )
            axis.plot(data["x"], data["y"], linewidth=1.8, label="lookahead %.2f" % lookahead)
            axis.scatter(data["x"][-1], data["y"][-1], marker="x", s=45)
        goal = configs[scene]["task"]["position"]
        axis.scatter(goal[0], goal[1], marker="*", s=120, label="goal")
        axis.set_title("%s, seed=%d" % (scene, representative_seed))
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output / "scripted_subgoal_trajectories.png", dpi=180)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-config", default="configs/rl/sac_mppi_subgoal_l6.yaml")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="41,42,43,44,45")
    parser.add_argument("--lookaheads", default="0.70")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--route-margin", type=float, default=0.15)
    parser.add_argument("--endpoint-route-margin", type=float, default=None)
    parser.add_argument("--route-resolution", type=float, default=0.04)
    args = parser.parse_args(argv)
    seeds = _csv_values(args.seeds, int)
    lookaheads = _csv_values(args.lookaheads, float)
    if not seeds or not lookaheads or any(value <= 0.0 for value in lookaheads):
        raise ValueError("at least one seed and positive lookahead are required")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("output directory is not empty: %s" % output)
    output.mkdir(parents=True, exist_ok=True)
    rl_config = load_yaml(args.rl_config)
    rows = []
    configs = {}
    routes = {}
    audits = {}
    for scene_path in args.configs:
        template = _resolved_scene(
            rl_config, scene_path, seeds[0], args.num_samples, args.max_steps
        )
        scene = str(template["scene"]["name"])
        route, audit = _offline_route(
            template,
            args.route_margin,
            args.route_resolution,
            endpoint_margin=args.endpoint_route_margin,
        )
        configs[scene] = template
        routes[scene] = route
        audits[scene] = audit
        route_dir = output / "routes"
        route_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            str(route_dir / (scene + ".csv")),
            route,
            delimiter=",",
            header="x,y",
            comments="",
        )
        for lookahead in lookaheads:
            for seed in seeds:
                config = _resolved_scene(
                    rl_config, scene_path, seed, args.num_samples, args.max_steps
                )
                run_dir = (
                    output / "runs" / scene
                    / ("lookahead_%03d" % round(100 * lookahead))
                    / ("seed_%d" % seed)
                )
                row = _run_episode(config, route, lookahead, seed, run_dir)
                rows.append(row)
                print(
                    "scene=%s lookahead=%.2f seed=%d success=%s distance=%.3f collision=%s"
                    % (
                        scene, lookahead, seed, row["success"],
                        row["final_goal_distance"], row["collision"],
                    )
                )
    summaries = _aggregate(rows)
    _write_csv(output / "episodes.csv", rows)
    _write_csv(output / "summary.csv", summaries)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(ROOT),
        "diagnostic_only": True,
        "privileged_route_not_planner_obstacles": True,
        "rl_config": str(Path(args.rl_config)),
        "scene_configs": list(args.configs),
        "seeds": seeds,
        "lookaheads": lookaheads,
        "num_samples": int(args.num_samples),
        "max_steps": int(args.max_steps),
        "route_margin": float(args.route_margin),
        "endpoint_route_margin": (
            None
            if args.endpoint_route_margin is None
            else float(args.endpoint_route_margin)
        ),
        "route_resolution": float(args.route_resolution),
        "audits": audits,
        "summary": summaries,
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
    _plot(output, rows, configs, routes, seeds[0])
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
