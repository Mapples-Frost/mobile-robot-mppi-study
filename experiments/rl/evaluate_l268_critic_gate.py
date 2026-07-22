#!/usr/bin/env python3
"""Aggregate the frozen paired L268 Critic Gate after all six arms finish."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobile_robot_mppi.core.config import git_sha


DEFAULT_CONFIG = ROOT / "configs/rl/l268_recovery_balanced_intervention.yaml"
DEFAULT_CRITIC = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/critic_only"
)
DEFAULT_OUTPUT = ROOT / (
    "results/research_platform/rl/l268_recovery_balanced_intervention/critic_gate"
)
L263_STATES = ROOT / (
    "results/research_platform/rl/l263_counterfactual_actor_diagnosis/"
    "state_manifest.json"
)


def _read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _number(value):
    if value in (None, "", "None", "null"):
        return None
    return float(value)


def _boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _paired(rows, field):
    by_seed = defaultdict(dict)
    for row in rows:
        by_seed[int(row["seed"])][row["arm"]] = _number(row[field])
    return {
        seed: values["recovery_balanced"] - values["control"]
        for seed, values in sorted(by_seed.items())
    }


def _gate_from_rows(rows, thresholds, scene_changes):
    treatment = [row for row in rows if row["arm"] == "recovery_balanced"]
    if len(rows) != 6 or len(treatment) != 3:
        raise ValueError("L268 Gate requires exactly 3 paired seeds x 2 arms")
    spearman_changes = _paired(rows, "in_support_spearman")
    top3_changes = _paired(rows, "in_support_top3")
    l263_pair_changes = _paired(rows, "in_support_pair_accuracy")
    heldout_pair_changes = _paired(rows, "heldout_pair_accuracy")
    treatment_spearman = float(np.mean([
        _number(row["in_support_spearman"]) for row in treatment
    ]))
    treatment_l263_pair = float(np.mean([
        _number(row["in_support_pair_accuracy"]) for row in treatment
    ]))
    treatment_heldout_pair = float(np.mean([
        _number(row["heldout_pair_accuracy"]) for row in treatment
    ]))
    median_spearman_change = float(statistics.median(spearman_changes.values()))
    median_top3_change = float(statistics.median(top3_changes.values()))
    median_l263_pair_change = float(statistics.median(l263_pair_changes.values()))
    median_heldout_pair_change = float(statistics.median(heldout_pair_changes.values()))
    improving_blocks = sum(value > 0.0 for value in spearman_changes.values())

    stable = all(
        _boolean(row["finite"])
        and _boolean(row["actor_alpha_unchanged"])
        and _number(row["maximum_absolute_q"])
        <= float(thresholds["maximum_absolute_mean_q"])
        and _number(row["mean_quantile_spread"])
        <= float(thresholds["maximum_mean_quantile_spread"])
        for row in rows
    )
    l263_pair_qualifies = (
        treatment_l263_pair >= float(thresholds["minimum_recovery_forward_accuracy"])
        and median_l263_pair_change
        >= float(thresholds["minimum_pair_accuracy_improvement"])
        and median_heldout_pair_change
        >= -float(thresholds["maximum_other_set_pair_accuracy_decrease"])
    )
    heldout_pair_qualifies = (
        treatment_heldout_pair
        >= float(thresholds["minimum_recovery_forward_accuracy"])
        and median_heldout_pair_change
        >= float(thresholds["minimum_pair_accuracy_improvement"])
        and median_l263_pair_change
        >= -float(thresholds["maximum_other_set_pair_accuracy_decrease"])
    )
    minimum_scene_change = min(scene_changes.values()) if scene_changes else -np.inf
    checks = {
        "all_runs_numerically_stable_and_actor_frozen": stable,
        "aggregate_in_support_spearman_positive": (
            treatment_spearman
            >= float(thresholds["minimum_aggregate_in_support_spearman"])
        ),
        "paired_median_spearman_improvement": (
            median_spearman_change
            >= float(thresholds["minimum_paired_median_spearman_improvement"])
        ),
        "improving_seed_blocks": (
            improving_blocks >= int(thresholds["minimum_improving_seed_blocks"])
        ),
        "recovery_forward_pair_accuracy": (
            l263_pair_qualifies or heldout_pair_qualifies
        ),
        "in_support_top3_improvement": (
            median_top3_change >= float(thresholds["minimum_top3_improvement"])
        ),
        "no_scene_spearman_collapse": (
            minimum_scene_change
            >= float(thresholds["minimum_scene_spearman_change"])
        ),
    }
    metrics = {
        "treatment_aggregate_in_support_spearman": treatment_spearman,
        "paired_spearman_changes": spearman_changes,
        "paired_median_spearman_improvement": median_spearman_change,
        "improving_seed_blocks": improving_blocks,
        "paired_top3_changes": top3_changes,
        "paired_median_top3_improvement": median_top3_change,
        "treatment_l263_pair_accuracy": treatment_l263_pair,
        "paired_l263_pair_accuracy_changes": l263_pair_changes,
        "paired_median_l263_pair_accuracy_improvement": median_l263_pair_change,
        "treatment_heldout_pair_accuracy": treatment_heldout_pair,
        "paired_heldout_pair_accuracy_changes": heldout_pair_changes,
        "paired_median_heldout_pair_accuracy_improvement": (
            median_heldout_pair_change
        ),
        "scene_spearman_changes": scene_changes,
        "minimum_scene_spearman_change": minimum_scene_change,
    }
    return checks, metrics


def _scene_changes(critic_dir: Path, rows):
    states = json.loads(L263_STATES.read_text(encoding="utf-8"))
    scene_by_state = {row["state_id"]: row["scene"] for row in states}
    arm_scene = defaultdict(list)
    for result in rows:
        run_dir = critic_dir / ("seed_%s" % result["seed"]) / result["arm"]
        metrics = _read_csv(run_dir / "l263_metrics_by_state.csv")
        for row in metrics:
            if row.get("support") != "in_support":
                continue
            value = _number(row.get("spearman"))
            if value is not None:
                arm_scene[(
                    int(result["seed"]), result["arm"], scene_by_state[row["state_id"]]
                )].append(value)
    paired = defaultdict(list)
    seeds = sorted({int(row["seed"]) for row in rows})
    scenes = sorted(set(scene_by_state.values()))
    for seed in seeds:
        for scene in scenes:
            control = arm_scene.get((seed, "control", scene), ())
            treatment = arm_scene.get((seed, "recovery_balanced", scene), ())
            if control and treatment:
                paired[scene].append(float(np.mean(treatment) - np.mean(control)))
    return {
        scene: float(np.mean(values)) for scene, values in sorted(paired.items())
    }


def evaluate(config_path: Path, critic_dir: Path, output: Path):
    integrity = json.loads((critic_dir / "integrity.json").read_text(encoding="utf-8"))
    if integrity.get("status") != "critic_only_complete_pending_gate_aggregation":
        raise ValueError("L268 Critic arms are not complete")
    rows = _read_csv(critic_dir / "results.csv")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["l268"]
    scene_changes = _scene_changes(critic_dir, rows)
    checks, metrics = _gate_from_rows(rows, config["critic_gate"], scene_changes)
    passed = all(checks.values())
    payload = {
        "protocol": "L268",
        "status": "critic_gate_pass" if passed else "critic_gate_fail",
        "critic_gate_pass": passed,
        "git_sha": git_sha(ROOT),
        "checks": checks,
        "metrics": metrics,
        "conditional_sac_authorized": bool(passed),
        "decision": (
            "start_preregistered_conditional_small_budget_sac"
            if passed else "stop_without_actor_training"
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    _json_dump(output / "gate.json", payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--critic-dir", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = evaluate(
        args.config.resolve(), args.critic_dir.resolve(), args.output_dir.resolve()
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
