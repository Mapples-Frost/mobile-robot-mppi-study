#!/usr/bin/env python3
"""Summarize and deterministically select L36 dynamic benchmark candidates."""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml


def _resolved_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value):
    return str(value).strip().lower() in ("1", "true", "yes")


def _candidate_rows(episodes, configured_order):
    grouped = defaultdict(list)
    for row in episodes:
        grouped[str(row["candidate"])].append(row)
    rows = []
    for order, name in enumerate(configured_order):
        selected = grouped[name]
        successes = int(sum(_bool(row["success"]) for row in selected))
        collisions = int(sum(_bool(row["collision"]) for row in selected))
        distances = np.asarray(
            [float(row["final_goal_distance"]) for row in selected], dtype=np.float64
        )
        rows.append({
            "candidate_order": order,
            "candidate": name,
            "episodes": len(selected),
            "successes": successes,
            "success_rate": successes / float(len(selected)),
            "collisions": collisions,
            "collision_rate": collisions / float(len(selected)),
            "timeouts": len(selected) - successes - collisions,
            "mean_final_goal_distance_m": float(np.mean(distances)),
            "std_final_goal_distance_m": float(np.std(distances, ddof=0)),
            "eligible": bool(successes >= 1 and collisions >= 1),
        })
    return rows


def _select_candidates(rows, targets):
    eligible = [row for row in rows if row["eligible"]]
    available = list(eligible)
    selected = []
    for label, target in zip(("easy", "moderate", "hard"), targets):
        if not available:
            break
        choice = min(
            available,
            key=lambda row: (
                abs(float(row["collision_rate"]) - float(target)),
                -float(row["std_final_goal_distance_m"]),
                int(row["candidate_order"]),
            ),
        )
        selected.append(dict(choice, difficulty_label=label, target_collision_rate=float(target)))
        available.remove(choice)
    return selected


def _plot(input_dir, rows, selected):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.16,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
    })
    labels = [row["candidate"].replace("_", "\n") for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.05, 3.15))
    ax.bar(
        x - width / 2,
        [row["success_rate"] for row in rows],
        width,
        color="#0072B2",
        label="Success",
    )
    ax.bar(
        x + width / 2,
        [row["collision_rate"] for row in rows],
        width,
        color="#D55E00",
        label="Collision",
    )
    selected_lookup = {row["candidate"]: row["difficulty_label"] for row in selected}
    for index, row in enumerate(rows):
        if row["candidate"] in selected_lookup:
            label_height = max(row["success_rate"], row["collision_rate"]) + 0.035
            ax.text(
                index,
                label_height,
                selected_lookup[row["candidate"]].upper(),
                ha="center",
                va="bottom",
                fontsize=6.5,
                fontweight="bold",
                color="#009E73",
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.1)
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Episode rate")
    ax.set_title(
        "L36 dynamic benchmark calibration with traditional MPPI only",
        loc="left",
    )
    ax.legend(ncol=2, loc="upper right")
    fig.text(
        0.5,
        -0.015,
        "Eight new development seeds per candidate; no RL or residual checkpoint is loaded.",
        ha="center",
        fontsize=7.2,
    )
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(input_dir / "fig_l36_dynamic_benchmark_calibration.png")
    fig.savefig(input_dir / "fig_l36_dynamic_benchmark_calibration.pdf")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", required=True)
    args = parser.parse_args(argv)
    config = load_yaml(_resolved_path(args.config))
    design = config["rl"]["benchmark_calibration"]
    input_dir = _resolved_path(args.input_dir)
    episodes = _read_csv(input_dir / "calibration_episodes.csv")
    configured_order = [str(item["name"]) for item in design["candidates"]]
    rows = _candidate_rows(episodes, configured_order)
    targets = [float(value) for value in design["selection_targets_collision_rate"]]
    selected = _select_candidates(rows, targets)
    finite_fields = (
        "final_goal_distance", "minimum_clearance", "control_jerk",
        "planner_compute_ms_mean",
    )
    nonfinite = sum(
        any(
            row.get(name) not in (None, "") and not np.isfinite(float(row[name]))
            for name in finite_fields
        )
        for row in episodes
    )
    expected = len(design["episode_seeds"]) * len(design["candidates"])
    unique_keys = {
        (str(row["candidate"]), int(row["episode_seed"])) for row in episodes
    }
    integrity = bool(
        len(episodes) == expected and len(unique_keys) == expected and nonfinite == 0
    )
    eligible_count = int(sum(row["eligible"] for row in rows))
    collision_span = (
        0.0 if len(selected) < 2 else float(
            max(row["collision_rate"] for row in selected)
            - min(row["collision_rate"] for row in selected)
        )
    )
    gate_config = design["gate"]
    gate = {
        "artifact_integrity": integrity,
        "eligible_candidates": eligible_count,
        "selected_candidates": len(selected),
        "selected_collision_rate_span": collision_span,
    }
    gate["passed"] = bool(
        integrity
        and eligible_count >= int(gate_config["minimum_eligible_candidates"])
        and len(selected) == 3
        and collision_span >= float(gate_config["minimum_selected_collision_rate_span"])
        and all(
            row["successes"] >= int(gate_config["minimum_successes_per_selected_candidate"])
            and row["collisions"] >= int(gate_config["minimum_collisions_per_selected_candidate"])
            for row in selected
        )
    )
    summary = {
        "design_id": str(design["design_id"]),
        "episodes": len(episodes),
        "candidate_summary": rows,
        "selected_candidates": selected,
        "calibration_gate": gate,
        "interpretation_guard": (
            "Selected candidates are development benchmarks chosen without RL/ICODE results; "
            "they are not final confirmation scenes."
        ),
    }
    _write_csv(input_dir / "candidate_summary.csv", rows)
    if selected:
        _write_csv(input_dir / "selected_candidates.csv", selected)
    (input_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _plot(input_dir, rows, selected)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
