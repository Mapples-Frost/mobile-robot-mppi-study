#!/usr/bin/env python3
"""Analyze the frozen Gate 2 offline and paired closed-loop contrasts."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.evaluation.paired_checkpoint import (
    gate2_closed_loop_decision,
    paired_checkpoint_effects,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def offline_contrast(control, aligned):
    result = {}
    for split in ("validation", "test", "unseen"):
        before = control["evaluations"][split]["value_aligned"]
        after = aligned["evaluations"][split]["value_aligned"]
        values = {}
        for metric in (
            "terminal_value_rmse",
            "rollout_rmse",
            "terminal_position_rmse",
            "terminal_heading_rmse",
            "terminal_value_rank_correlation",
        ):
            control_value = float(before[metric])
            aligned_value = float(after[metric])
            higher_is_better = metric.endswith("rank_correlation")
            favorable = (
                aligned_value - control_value
                if higher_is_better
                else control_value - aligned_value
            )
            values[metric] = {
                "control": control_value,
                "aligned": aligned_value,
                "favorable_effect": favorable,
                "relative_favorable_change": (
                    None
                    if abs(control_value) < 1e-12
                    else favorable / abs(control_value)
                ),
            }
        result[split] = values
    test = result["test"]
    unseen = result["unseen"]
    result["gate"] = {
        "test_value_improved": (
            test["terminal_value_rmse"]["favorable_effect"] > 0.0
        ),
        "unseen_value_improved": (
            unseen["terminal_value_rmse"]["favorable_effect"] > 0.0
        ),
        "test_rollout_within_three_percent": (
            test["rollout_rmse"]["relative_favorable_change"] >= -0.03
        ),
        "unseen_rollout_within_three_percent": (
            unseen["rollout_rmse"]["relative_favorable_change"] >= -0.03
        ),
    }
    result["gate"]["offline_mechanism_passed"] = all(
        result["gate"].values()
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-training-summary", required=True)
    parser.add_argument("--aligned-training-summary", required=True)
    parser.add_argument("--control-episodes", required=True)
    parser.add_argument("--aligned-episodes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(argv)

    control_summary = load_json(args.control_training_summary)
    aligned_summary = load_json(args.aligned_training_summary)
    closed_loop = paired_checkpoint_effects(
        load_csv(args.control_episodes),
        load_csv(args.aligned_episodes),
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    closed_loop["gate"] = gate2_closed_loop_decision(closed_loop)
    result = {
        "schema_version": 1,
        "offline": offline_contrast(control_summary, aligned_summary),
        "closed_loop": closed_loop,
        "provenance": {
            "control_training_summary": str(
                Path(args.control_training_summary).resolve()
            ),
            "control_training_summary_sha256": sha256(
                args.control_training_summary
            ),
            "aligned_training_summary": str(
                Path(args.aligned_training_summary).resolve()
            ),
            "aligned_training_summary_sha256": sha256(
                args.aligned_training_summary
            ),
            "control_episodes": str(Path(args.control_episodes).resolve()),
            "control_episodes_sha256": sha256(args.control_episodes),
            "aligned_episodes": str(Path(args.aligned_episodes).resolve()),
            "aligned_episodes_sha256": sha256(args.aligned_episodes),
        },
    }
    result["gate2_development_passed"] = bool(
        result["offline"]["gate"]["offline_mechanism_passed"]
        and result["closed_loop"]["gate"]["closed_loop_development_passed"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output.resolve()),
        "gate2_development_passed": result["gate2_development_passed"],
        "offline_gate": result["offline"]["gate"],
        "closed_loop_gate": result["closed_loop"]["gate"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
