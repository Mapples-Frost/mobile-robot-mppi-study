"""Static visual checks for stochastic obstacle trajectories."""

from pathlib import Path
from typing import Mapping

import numpy as np

from .motion import ObstacleTrajectory


def plot_obstacle_processes(
    trajectories: Mapping[str, ObstacleTrajectory],
    output_path,
    obstacle_radius: float = 0.25,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    order = (
        "noisy_cv",
        "speed_change",
        "direction_change",
        "combined_change",
    )
    available = [name for name in order if name in trajectories]
    columns = 2
    rows = int(np.ceil(len(available) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(11.0, 4.7 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    labels = {
        "noisy_cv": "P1 Noisy CV",
        "speed_change": "P2 Speed Change",
        "direction_change": "P3 Direction Change",
        "combined_change": "P4 Combined Change + Occlusion",
    }
    for axis, process in zip(axes.flat, available):
        trajectory = trajectories[process]
        truth = trajectory.states[:, :2]
        observed = trajectory.observed_mask
        axis.plot(
            truth[:, 0],
            truth[:, 1],
            color="#0072B2",
            linewidth=2.2,
            label="true obstacle path",
        )
        axis.scatter(
            trajectory.observations[observed, 0],
            trajectory.observations[observed, 1],
            s=13,
            alpha=0.55,
            color="#E69F00",
            label="noisy position observation",
            zorder=3,
        )
        changes = np.flatnonzero(trajectory.change_flags)
        if changes.size:
            axis.scatter(
                truth[changes, 0],
                truth[changes, 1],
                marker="X",
                s=90,
                color="#D55E00",
                edgecolors="white",
                linewidths=0.8,
                label="true change event",
                zorder=5,
            )
        if not observed.all():
            missing = np.flatnonzero(~observed)
            axis.plot(
                truth[missing, 0],
                truth[missing, 1],
                color="#CC79A7",
                linewidth=5.0,
                alpha=0.65,
                label="observation missing",
            )
        axis.scatter(
            truth[0, 0],
            truth[0, 1],
            marker="o",
            s=65,
            color="#009E73",
            label="start",
            zorder=6,
        )
        obstacle = Circle(
            (truth[-1, 0], truth[-1, 1]),
            radius=float(obstacle_radius),
            facecolor="#56B4E9",
            edgecolor="#003B5C",
            linewidth=1.5,
            alpha=0.75,
            label="obstacle footprint at final time",
            zorder=4,
        )
        axis.add_patch(obstacle)
        axis.set_title(labels[process])
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.grid(True, alpha=0.25)
        axis.set_aspect("equal", adjustable="datalim")
        axis.legend(fontsize=8, loc="best")
    for axis in axes.flat[len(available) :]:
        axis.set_visible(False)
    figure.suptitle(
        "Seeded stochastic dynamic-obstacle simulator (medium noise)",
        fontsize=15,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def plot_patrol_processes(trajectories, output_path):
    """Plot V3 paths with time color, registered waypoints, and visit order."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    order = (
        "stochastic_cruise",
        "stochastic_shuttle",
        "branching_patrol",
        "hybrid_patrol",
    )
    available = [name for name in order if name in trajectories]
    if not available:
        raise ValueError("at least one V3 trajectory is required")
    labels = {
        "stochastic_cruise": "P1 Stochastic Cruise (control)",
        "stochastic_shuttle": "P2 Stochastic A-B Shuttle",
        "branching_patrol": "P3 Branching Waypoint Patrol",
        "hybrid_patrol": "P4 Hybrid Intent Patrol + Occlusion",
    }
    figure, axes = plt.subplots(
        2, 2, figsize=(12.0, 10.0), constrained_layout=True
    )
    color_handle = None
    for axis, process in zip(axes.flat, available):
        trajectory = trajectories[process]
        truth = trajectory.states[:, :2]
        segments = np.stack((truth[:-1], truth[1:]), axis=1)
        color_handle = LineCollection(
            segments,
            cmap="viridis",
            norm=plt.Normalize(
                float(trajectory.times[0]), float(trajectory.times[-1])
            ),
            linewidths=2.8,
        )
        color_handle.set_array(trajectory.times[:-1])
        axis.add_collection(color_handle)
        waypoints = np.asarray(
            trajectory.metadata.get("waypoints_m", ()), dtype=np.float64
        )
        if waypoints.size:
            axis.scatter(
                waypoints[:, 0],
                waypoints[:, 1],
                marker="D",
                s=55,
                facecolor="white",
                edgecolor="#222222",
                linewidth=1.1,
                label="registered waypoint",
                zorder=4,
            )
            visits = trajectory.metadata["waypoint_visits"]
            for waypoint_index in sorted(set(visits)):
                visit_order = [
                    str(index)
                    for index, value in enumerate(visits)
                    if value == waypoint_index
                ]
                axis.annotate(
                    "W%d visits %s"
                    % (waypoint_index, ",".join(visit_order)),
                    waypoints[waypoint_index],
                    xytext=(5, 6),
                    textcoords="offset points",
                    fontsize=7,
                    color="#222222",
                )
        changes = np.flatnonzero(trajectory.change_flags)
        if changes.size:
            axis.scatter(
                truth[changes, 0],
                truth[changes, 1],
                marker="x",
                s=34,
                color="#D55E00",
                linewidth=1.2,
                label="intent/event change",
                zorder=5,
            )
        missing = np.flatnonzero(~trajectory.observed_mask)
        if missing.size:
            axis.scatter(
                truth[missing, 0],
                truth[missing, 1],
                s=18,
                color="#CC79A7",
                alpha=0.65,
                label="observation unavailable",
                zorder=3,
            )
        axis.scatter(
            truth[0, 0],
            truth[0, 1],
            s=70,
            color="#009E73",
            edgecolor="white",
            linewidth=0.8,
            label="episode start",
            zorder=6,
        )
        axis.set_title(labels[process])
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal", adjustable="datalim")
        axis.autoscale()
        axis.grid(True, alpha=0.22)
        axis.legend(fontsize=7, loc="best")
    for axis in axes.flat[len(available) :]:
        axis.set_visible(False)
    figure.suptitle(
        "V3 recurrent semi-Markov dynamic-obstacle processes", fontsize=15
    )
    if color_handle is not None:
        figure.colorbar(
            color_handle,
            ax=list(axes.flat),
            label="episode time [s]",
            shrink=0.72,
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


def plot_patrol_time_series(trajectories, output_path):
    """Plot x/y/speed histories so recurrent motion is visible despite overlap."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = (
        "stochastic_cruise",
        "stochastic_shuttle",
        "branching_patrol",
        "hybrid_patrol",
    )
    available = [name for name in order if name in trajectories]
    if not available:
        raise ValueError("at least one V3 trajectory is required")
    figure, axes = plt.subplots(
        len(available),
        1,
        figsize=(12.0, 2.7 * len(available)),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, process in zip(axes[:, 0], available):
        trajectory = trajectories[process]
        speed = np.linalg.norm(trajectory.states[:, 2:], axis=1)
        axis.plot(
            trajectory.times,
            trajectory.states[:, 0],
            color="#0072B2",
            linewidth=1.7,
            label="x [m]",
        )
        axis.plot(
            trajectory.times,
            trajectory.states[:, 1],
            color="#009E73",
            linewidth=1.5,
            label="y [m]",
        )
        axis.plot(
            trajectory.times,
            speed,
            color="#D55E00",
            linewidth=1.3,
            label="speed [m/s]",
        )
        for interval in trajectory.metadata["dropout_intervals"]:
            axis.axvspan(
                float(interval["start_time_s"]),
                float(interval["end_time_s"]),
                color="#CC79A7",
                alpha=0.22,
            )
        axis.set_ylabel(process.replace("_", "\n"), fontsize=8)
        axis.grid(True, alpha=0.22)
        axis.legend(ncol=3, fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("episode time [s]")
    figure.suptitle(
        "V3 recurrent motion histories (magenta shading = observation dropout)",
        fontsize=14,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output_path


__all__ = [
    "plot_obstacle_processes",
    "plot_patrol_processes",
    "plot_patrol_time_series",
]
