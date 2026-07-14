#!/usr/bin/env python3
"""Plot geometry, offline diagnostic routes, and executed MuJoCo traces."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml


def _box_polygon(obstacle, padding=0.0):
    px, py = (float(value) for value in obstacle["position"])
    sx, sy = (float(value) + float(padding) for value in obstacle["size"])
    yaw = float(obstacle.get("yaw", 0.0))
    local = np.asarray(((-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)))
    rotation = np.asarray(
        ((math.cos(yaw), -math.sin(yaw)), (math.sin(yaw), math.cos(yaw)))
    )
    return local @ rotation.T + np.asarray((px, py))


def _draw_obstacles(axis, obstacles, robot_radius):
    from matplotlib.patches import Circle, Polygon

    for obstacle in obstacles:
        if str(obstacle.get("type", "cylinder")) == "box":
            axis.add_patch(Polygon(
                _box_polygon(obstacle, robot_radius), closed=True,
                facecolor="#f4b6b6", edgecolor="none", alpha=0.38,
            ))
            axis.add_patch(Polygon(
                _box_polygon(obstacle), closed=True,
                facecolor="#444444", edgecolor="#202020", linewidth=0.8,
            ))
        else:
            center = tuple(float(value) for value in obstacle["position"])
            radius = float(obstacle.get("radius", 0.25))
            axis.add_patch(Circle(
                center, radius + robot_radius, facecolor="#f4b6b6",
                edgecolor="none", alpha=0.38,
            ))
            axis.add_patch(Circle(
                center, radius, facecolor="#444444", edgecolor="#202020",
                linewidth=0.8,
            ))


def _read_trace(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([(float(row["x"]), float(row["y"])) for row in rows])


def plot_results(results_dir, config_paths):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir).resolve()
    with (results_dir / "metrics.json").open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    configs = {}
    for path in config_paths:
        config = load_yaml(path)
        configs[str(config["scene"]["name"])] = config
    figure_dir = results_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    styles = {
        ("direct_goal", "nominal"): ("#d62728", "--"),
        ("direct_goal", "icode"): ("#ff7f0e", "--"),
        ("offline_waypoints", "nominal"): ("#1f77b4", "-"),
        ("offline_waypoints", "icode"): ("#2ca02c", "-"),
        ("offline_polyline", "nominal"): ("#1f77b4", "-"),
        ("offline_polyline", "icode"): ("#2ca02c", "-"),
    }
    for scene, route in sorted(report["routes"].items()):
        if scene not in configs:
            raise KeyError("missing config for scene %s" % scene)
        config = configs[scene]
        fig, axis = plt.subplots(figsize=(7.2, 6.4), constrained_layout=True)
        robot_radius = float(config["plant"]["robot"]["collision_radius"])
        _draw_obstacles(axis, config["scene"].get("obstacles", ()), robot_radius)
        offline = np.asarray(route["geometry_audit"]["path"], dtype=np.float64)
        waypoints = np.asarray(route["waypoints"], dtype=np.float64)
        axis.plot(offline[:, 0], offline[:, 1], color="#73a9dc", linewidth=1.4,
                  alpha=0.75, label="offline feasibility path")
        axis.scatter(waypoints[:, 0], waypoints[:, 1], s=20, color="#174a7e",
                     zorder=5, label="diagnostic waypoints")
        for trace_path in sorted((results_dir / "runs" / scene).glob("*/*/seed_*/trajectory.csv")):
            relative = trace_path.relative_to(results_dir / "runs" / scene)
            mode, method = relative.parts[0], relative.parts[1]
            seed = relative.parts[2].replace("seed_", "")
            trace = _read_trace(trace_path)
            color, line_style = styles.get((mode, method), (None, "-"))
            axis.plot(trace[:, 0], trace[:, 1], color=color, linestyle=line_style,
                      linewidth=2.0, label="%s / %s / seed %s" % (mode, method, seed))
            axis.scatter(trace[-1, 0], trace[-1, 1], color=color, marker="x", s=55)
        start = np.asarray(config["experiment"]["initial_state"][:2], dtype=np.float64)
        goal = waypoints[-1]
        axis.scatter(start[0], start[1], marker="o", s=75, color="black", label="start", zorder=7)
        axis.scatter(goal[0], goal[1], marker="*", s=170, color="#f2c14e",
                     edgecolor="black", linewidth=0.7, label="goal", zorder=7)
        axis.set_title(
            "%s | goal=(%.2f, %.2f) | K=%d | route margin=%.2f m" % (
                scene, goal[0], goal[1], report["metadata"]["num_samples"],
                route["geometry_audit"]["margin"],
            )
        )
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, loc="best")
        output = figure_dir / (scene + "_waypoint_diagnostic.png")
        fig.savefig(str(output), dpi=180)
        plt.close(fig)
        generated.append(str(output))
    return generated


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args(argv)
    for path in plot_results(args.results_dir, args.configs):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
