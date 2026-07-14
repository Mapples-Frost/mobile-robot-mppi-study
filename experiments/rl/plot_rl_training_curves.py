#!/usr/bin/env python3
"""Plot SAC training/validation evidence without inventing missing points."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rolling(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    result = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index + 1 - window)
        result[index] = values[start:index + 1].mean()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args(argv)
    run = Path(args.run_dir).resolve()
    episodes = _read(run / "episodes.csv")
    updates = _read(run / "updates.csv")
    validations = _read(run / "validation_episodes.csv")
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    episode_step = np.asarray([int(row["global_step"]) for row in episodes])
    episode_return = np.asarray([float(row["return"]) for row in episodes])
    episode_success = np.asarray([float(row["success"] == "True") for row in episodes])
    axes[0, 0].plot(episode_step, episode_return, alpha=0.25, label="episode")
    axes[0, 0].plot(episode_step, _rolling(episode_return, args.window), label="rolling mean")
    axes[0, 0].set_ylabel("Episode return")
    axes[0, 0].legend()
    axes[0, 1].plot(episode_step, _rolling(episode_success, args.window))
    axes[0, 1].set_ylabel("Rolling training success")
    update_step = np.asarray([int(row["global_step"]) for row in updates])
    axes[1, 0].plot(update_step, [float(row["critic1_loss"]) for row in updates], label="Q1")
    axes[1, 0].plot(update_step, [float(row["critic2_loss"]) for row in updates], label="Q2", alpha=0.8)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel("Critic MSE (log)")
    axes[1, 0].legend()
    scenes = sorted(set(row["scene"] for row in validations))
    for scene in scenes:
        rows = [row for row in validations if row["scene"] == scene]
        steps = sorted(set(int(row["global_step"]) for row in rows))
        distance = [
            np.mean([float(row["goal_distance"]) for row in rows if int(row["global_step"]) == step])
            for step in steps
        ]
        axes[1, 1].plot(steps, distance, marker="o", label=scene)
    axes[1, 1].set_ylabel("Validation final distance (m)")
    axes[1, 1].legend(fontsize=8)
    for axis in axes.reshape(-1):
        axis.set_xlabel("Environment steps")
        axis.grid(alpha=0.25)
    figure.suptitle("SAC MPPI-prior learnability evidence")
    destination = run / "training_curves.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
