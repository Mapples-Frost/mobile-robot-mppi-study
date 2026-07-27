"""Analyze the frozen three-map proposal-to-execution development round."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.run_complex_maneuver_proposal_execution_development import (
    DEFAULT_PROTOCOL,
    _load_protocol,
)


ARMS = ("margin0_control", "margin002_treatment")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _number(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _truth(row, key):
    return _number(row, key, 0.0) > 0.5


def _normalized_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    paper = config["planner"]["paper_rl_driven"]
    paper.pop("same_cycle_guided_relative_margin", None)
    config["experiment"].pop("name", None)
    config.pop("proposal_execution_contract", None)
    return config


def _rollout_contract(rows):
    values = sorted({
        int(_number(row, "paper_total_rollouts", -1)) for row in rows
    })
    valid = all(value in (0, 600) for value in values)
    head_quota = all(
        all(
            int(_number(row, f"supervised_head_{head}_proposal_count")) == 1
            for head in range(3)
        )
        for row in rows
        if int(_number(row, "paper_total_rollouts", -1)) == 600
    )
    zero_quota = all(
        all(
            int(_number(row, f"supervised_head_{head}_proposal_count")) == 0
            for head in range(3)
        )
        for row in rows
        if int(_number(row, "paper_total_rollouts", -1)) == 0
    )
    added = sorted({
        int(_number(row, "supervised_added_rollout_count", -1))
        for row in rows
    })
    return {
        "values": values,
        "permitted_values_only": valid,
        "valid_cycles_have_one_proposal_per_head": head_quota,
        "zero_cycles_have_no_supervised_proposals": zero_quota,
        "added_rollout_values": added,
        "pass": valid and head_quota and zero_quota and added == [0],
    }


def _mechanism_summary(metrics, rows):
    active = [
        row for row in rows
        if _truth(row, "supervised_counterfactual_available")
    ]
    influential = sum(
        _truth(row, "supervised_influence_survived_guard")
        for row in active
    )
    return {
        "collision": bool(metrics["collision"]),
        "success": bool(metrics["success"]),
        "steps": int(metrics["steps"]),
        "final_goal_distance": float(metrics["final_goal_distance"]),
        "safety_interventions": int(metrics["safety_interventions"]),
        "safety_intervention_fraction": (
            float(metrics["safety_interventions"]) / int(metrics["steps"])
        ),
        "selected_cycles": int(metrics["supervised_selected_count_total"]),
        "counterfactual_active_cycles": len(active),
        "influential_cycles": int(influential),
        "counterfactual_first_action_delta_norm_mean_active": (
            sum(_number(
                row,
                "supervised_counterfactual_first_action_delta_norm",
            ) for row in active) / len(active)
            if active else 0.0
        ),
        "influence_survived_guard_fraction_active": (
            influential / len(active) if active else 0.0
        ),
        "rollout_contract": _rollout_contract(rows),
    }


def _evaluate_pair(control, treatment):
    task_improved = (
        treatment["success"] and not control["success"]
    ) or (
        treatment["final_goal_distance"]
        < control["final_goal_distance"]
    )
    checks = {
        "collision_not_worse": (
            int(treatment["collision"]) <= int(control["collision"])
        ),
        "terminal_goal_distance_or_success_strictly_better": task_improved,
        "counterfactual_first_action_delta_norm_increases": (
            treatment[
                "counterfactual_first_action_delta_norm_mean_active"
            ]
            > control[
                "counterfactual_first_action_delta_norm_mean_active"
            ]
        ),
        "influence_survived_guard_fraction_increases": (
            treatment["influence_survived_guard_fraction_active"]
            > control["influence_survived_guard_fraction_active"]
        ),
        "rollout_counts_are_only_0_or_600": (
            control["rollout_contract"]["pass"]
            and treatment["rollout_contract"]["pass"]
        ),
    }
    return {
        "control": control,
        "treatment": treatment,
        "checks": checks,
        "pass": all(checks.values()),
    }


def analyze(protocol_path, input_root):
    protocol = _load_protocol(protocol_path)
    input_root = Path(input_root)
    pairs = {}
    for block in protocol["opened_development_blocks"]:
        map_name = str(block["map"])
        seed = int(block["seed"])
        arm_data = {}
        normalized = {}
        for arm in ARMS:
            episode = input_root / f"{map_name}_seed{seed}" / arm
            metrics_path = episode / "metrics.json"
            trajectory_path = episode / "trajectory.csv"
            config_path = episode / "config_resolved.yaml"
            if not (
                metrics_path.is_file()
                and trajectory_path.is_file()
                and config_path.is_file()
            ):
                raise FileNotFoundError(
                    "incomplete proposal-execution episode: %s" % episode
                )
            arm_data[arm] = _mechanism_summary(
                _read_json(metrics_path), _read_rows(trajectory_path)
            )
            normalized[arm] = _normalized_config(config_path)
        if normalized[ARMS[0]] != normalized[ARMS[1]]:
            raise ValueError(
                "pair changed configuration beyond the frozen margin"
            )
        pairs[map_name] = _evaluate_pair(
            arm_data["margin0_control"],
            arm_data["margin002_treatment"],
        )

    pooled_control_selected = sum(
        pair["control"]["selected_cycles"] for pair in pairs.values()
    )
    pooled_treatment_selected = sum(
        pair["treatment"]["selected_cycles"] for pair in pairs.values()
    )
    pooled_control_influential = sum(
        pair["control"]["influential_cycles"] for pair in pairs.values()
    )
    pooled_treatment_influential = sum(
        pair["treatment"]["influential_cycles"] for pair in pairs.values()
    )
    pooled_control_safety = sum(
        pair["control"]["safety_interventions"] for pair in pairs.values()
    ) / sum(pair["control"]["steps"] for pair in pairs.values())
    pooled_treatment_safety = sum(
        pair["treatment"]["safety_interventions"]
        for pair in pairs.values()
    ) / sum(pair["treatment"]["steps"] for pair in pairs.values())
    pooled_checks = {
        "actor_selected_or_influential_cycles_increase": (
            (
                pooled_treatment_selected + pooled_treatment_influential
            )
            > (pooled_control_selected + pooled_control_influential)
        ),
        "safety_intervention_fraction_does_not_increase": (
            pooled_treatment_safety <= pooled_control_safety
        ),
    }
    collision_regression = any(
        pair["treatment"]["collision"] and not pair["control"]["collision"]
        for pair in pairs.values()
    )
    passed = (
        not collision_regression
        and all(pair["pass"] for pair in pairs.values())
        and all(pooled_checks.values())
    )
    return {
        "schema_version": 1,
        "status": "pass" if passed else "fail",
        "scope": "development_not_held_out",
        "round": 1,
        "margin": 0.02,
        "pairs": pairs,
        "pooled": {
            "control_selected_cycles": pooled_control_selected,
            "treatment_selected_cycles": pooled_treatment_selected,
            "control_influential_cycles": pooled_control_influential,
            "treatment_influential_cycles": pooled_treatment_influential,
            "control_safety_intervention_fraction": pooled_control_safety,
            "treatment_safety_intervention_fraction": pooled_treatment_safety,
            "checks": pooled_checks,
        },
        "collision_regression": collision_regression,
        "next_round_consideration_allowed": (
            not passed
            and not collision_regression
            and pooled_checks[
                "safety_intervention_fraction_does_not_increase"
            ]
        ),
        "held_out_claim_authorized": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            "proposal-execution analysis output exists: %s" % output
        )
    result = analyze(args.protocol, args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = output.with_name("artifact_manifest.json")
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "result": str(output),
                "result_sha256": hashlib.sha256(
                    output.read_bytes()
                ).hexdigest(),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
