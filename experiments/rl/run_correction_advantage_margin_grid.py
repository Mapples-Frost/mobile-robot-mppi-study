#!/usr/bin/env python3
"""Run one training checkpoint over a fixed target-critic margin grid."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rl.run_correction_advantage_diagnostic import (
    _condition_summary,
    _evaluation_config,
    _load_agent,
    _paired_rows,
    _resolved_path,
    _run_episode,
    _seeds,
    _write_csv,
)
from mobile_robot_mppi.core.config import git_sha


def _margins(text):
    values = [float(value) for value in str(text).split(",") if value.strip()]
    array = np.asarray(values, dtype=np.float64)
    if (
        array.size == 0
        or not np.isfinite(array).all()
        or np.any(array < 0.0)
        or np.any(np.diff(array) <= 0.0)
    ):
        raise ValueError(
            "margin grid must be finite, non-negative and strictly increasing"
        )
    return [float(value) for value in array]


def _margin_condition(margin):
    token = ("%.6f" % float(margin)).rstrip("0").rstrip(".")
    return "target_margin_" + token.replace(".", "p")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument("--training-seed", type=int, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--margins", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)

    seeds = _seeds(args.seeds)
    margins = _margins(args.margins)
    resolved = _evaluation_config(args.config, args.scene_config)
    baseline_path = _resolved_path(args.baseline_checkpoint)
    candidate_path = _resolved_path(args.candidate_checkpoint)
    baseline_agent, baseline_normalizer, baseline_payload = _load_agent(
        baseline_path, args.device
    )
    candidate_agent, candidate_normalizer, candidate_payload = _load_agent(
        candidate_path, args.device
    )
    if (
        baseline_agent.observation_dim != candidate_agent.observation_dim
        or baseline_agent.action_dim != candidate_agent.action_dim
    ):
        raise ValueError("baseline and candidate checkpoint dimensions differ")
    if (
        baseline_agent.frozen_base_actor_sha256()
        != candidate_agent.frozen_base_actor_sha256()
    ):
        raise ValueError("baseline and candidate do not share the same frozen BC")

    episodes = []
    steps = []
    condition_margin = {}
    for seed in seeds:
        episode, episode_steps = _run_episode(
            resolved,
            baseline_agent,
            baseline_normalizer,
            seed,
            "bc",
            "none",
            "target",
            0.0,
        )
        episodes.append(episode)
        steps.extend(episode_steps)
    for margin in margins:
        condition = _margin_condition(margin)
        condition_margin[condition] = margin
        for seed in seeds:
            episode, episode_steps = _run_episode(
                resolved,
                candidate_agent,
                candidate_normalizer,
                seed,
                condition,
                "hard",
                "target",
                margin,
            )
            episodes.append(episode)
            steps.extend(episode_steps)

    for episode in episodes:
        episode["training_seed"] = int(args.training_seed)
        episode["checkpoint"] = str(
            baseline_path
            if episode["condition"] == "bc"
            else candidate_path
        )
    for row in steps:
        row["training_seed"] = int(args.training_seed)
        row["checkpoint"] = str(
            baseline_path if row["condition"] == "bc" else candidate_path
        )
    paired = _paired_rows(episodes)
    summary = {
        "training_seed": int(args.training_seed),
        "calibration_seeds": seeds,
        "scene": str(resolved.get("scene", {}).get("name", "unknown")),
        "margins": margins,
        "condition_margin": condition_margin,
        "baseline_checkpoint": str(baseline_path),
        "candidate_checkpoint": str(candidate_path),
        "baseline_git_sha": baseline_payload.get("git_sha"),
        "candidate_git_sha": candidate_payload.get("git_sha"),
        "run_git_sha": git_sha(ROOT),
        "frozen_base_actor_sha256": (
            candidate_agent.frozen_base_actor_sha256()
        ),
        "conditions": _condition_summary(paired),
    }
    output = _resolved_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "episodes.csv", episodes)
    _write_csv(output / "paired_episodes.csv", paired)
    _write_csv(output / "steps.csv", steps)
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (output / "config_snapshot.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
