#!/usr/bin/env python3
"""Audit the preregistered L78 conservative-correction training blocks."""

import argparse
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mobile_robot_mppi.core.config import load_yaml  # noqa: E402
from experiments.rl.summarize_static_multigeometry_training import (  # noqa: E402
    _write_csv,
    summarize as summarize_training,
)


EXPECTED_TRAINING_SEEDS = (20260774, 20260775, 20260776)
EXPECTED_BC_TOKENS = (
    "bc_l13_seed20260721_20260714",
    "bc_l13_seed20260722_20260714",
    "bc_l13_seed20260723_20260714",
)
EXPECTED_ICODE_TOKENS = ("seed20261201", "seed20261202", "seed20261203")


def _resolved(path):
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _selected_blocks_satisfy_noninferiority(blocks):
    checks = []
    for block in blocks:
        if not bool(block["selected_nonzero"]):
            checks.append({
                "block": int(block["block"]),
                "selected_nonzero": False,
                "eligible": False,
                "clauses": {"selected_nonzero": False},
            })
            continue
        distance = float(block["paired_mean_goal_distance_improvement_m"])
        gains = int(block["paired_success_gains"])
        clauses = {
            "selected_nonzero": True,
            "success_losses": int(block["paired_success_losses"]) == 0,
            "collision_regressions": (
                int(block["paired_collision_regressions"]) == 0
            ),
            "mean_goal_distance_noninferior": (
                math.isfinite(distance) and distance >= 0.0
            ),
            "meaningful_task_improvement": (
                gains >= 1 or distance >= 0.005
            ),
        }
        checks.append({
            "block": int(block["block"]),
            "selected_nonzero": True,
            "eligible": bool(all(clauses.values())),
            "clauses": clauses,
        })
    return checks


def _loadable_training_checkpoint(path, expected_step):
    payload = torch.load(
        str(path), map_location=torch.device("cpu"), weights_only=False
    )
    required = {
        "agent",
        "normalizer",
        "resolved_config",
        "training_state",
        "git_sha",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            "%s missing checkpoint keys: %s" % (path, ", ".join(missing))
        )
    actual_step = int(payload["training_state"]["global_step"])
    if actual_step != int(expected_step):
        raise ValueError(
            "%s stores step %d, expected %d"
            % (path, actual_step, int(expected_step))
        )
    return payload


def _state_dict_exact(left, right):
    if set(left) != set(right):
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def summarize(
    config_paths,
    run_dirs,
    study_label="L78",
    expected_training_seeds=EXPECTED_TRAINING_SEEDS,
    expected_validation_seed_base=22300801,
    expected_group_robust=False,
    expected_critic_distribution="scalar",
    expected_num_quantiles=25,
    expected_cvar_fraction=1.0,
    interpretation_guard=None,
):
    if len(config_paths) != 3 or len(run_dirs) != 3:
        raise ValueError("%s requires exactly three model blocks" % study_label)
    base = summarize_training(
        config_paths,
        run_dirs,
        minimum_nonzero_blocks=2,
        study_label=study_label,
        interpretation_guard=interpretation_guard or (
            "%s is a development-training gate. Passing is not a deployment "
            "claim." % study_label
        ),
    )
    errors = list(base["artifact_errors"])
    schedule_exact = True
    actor_initialization_exact = True
    checkpoints_complete_and_loadable = True
    actor_delay_exact = True
    actor_update_started_exact = True
    frozen_base_actor_exact = True

    for index, (config_path, run_dir) in enumerate(
        zip(config_paths, run_dirs)
    ):
        config = load_yaml(_resolved(config_path))
        run = _resolved(run_dir)
        training = config["rl"]["training"]
        sac = config["rl"]["sac"]
        expected_seed = tuple(expected_training_seeds)[index]
        clauses = (
            int(training["seed"]) == expected_seed,
            int(training["total_steps"]) == 30000,
            int(training["actor_update_after"]) == 10000,
            str(training["replay_sampling"]) == "scene_outcome_balanced",
            float(training["replay_success_fraction"]) == 0.5,
            int(training["validation_seed_base"])
            == int(expected_validation_seed_base),
            float(sac["actor_lr"]) == 0.00005,
            float(sac["critic_lr"]) == 0.0002,
            float(sac["correction_penalty_weight"]) == 1.0,
            str(sac["policy_mode"]) == "frozen_bc_correction",
            bool(sac.get("actor_group_robust_enabled", False))
            == bool(expected_group_robust),
            str(sac.get("critic_distribution", "scalar"))
            == str(expected_critic_distribution),
            int(sac.get("critic_num_quantiles", 25))
            == int(expected_num_quantiles),
            float(sac.get("actor_cvar_fraction", 1.0))
            == float(expected_cvar_fraction),
            EXPECTED_ICODE_TOKENS[index] in str(
                config["planner"]["checkpoint"]
            ),
        )
        if not all(clauses):
            schedule_exact = False
            errors.append("block %d: conservative schedule mismatch" % index)

        try:
            metadata = json.loads(
                (run / "run_metadata.json").read_text(encoding="utf-8")
            )
            actor_source = str(
                metadata["actor_initialization"]["checkpoint"]
            )
            if EXPECTED_BC_TOKENS[index] not in actor_source:
                raise ValueError("frozen BC source mismatch")
            if (
                metadata["actor_initialization"]["mode"]
                != "frozen_bc_base_and_normalizer_only"
            ):
                raise ValueError("actor initialization mode mismatch")
        except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
            actor_initialization_exact = False
            errors.append("block %d: %s" % (index, error))

        try:
            initial_payload = _loadable_training_checkpoint(
                run / "checkpoints" / "initial.pt", 0
            )
            initial_base_hash = str(
                initial_payload["agent"]["base_actor_sha256"]
            )
            periodic_payloads = {}
            periodic_steps = range(
                5000, int(training["total_steps"]) + 1, 5000
            )
            for step in periodic_steps:
                checkpoint = (
                    run
                    / "checkpoints"
                    / ("step_%09d.pt" % int(step))
                )
                periodic_payloads[step] = _loadable_training_checkpoint(
                    checkpoint, step
                )
            _loadable_training_checkpoint(
                run / "checkpoints" / "latest.pt",
                int(training["total_steps"]),
            )
            selected_step = int(
                base["blocks"][index]["selected_global_step"]
            )
            selected_name = (
                "initial.pt"
                if selected_step == 0
                else "step_%09d.pt" % selected_step
            )
            _loadable_training_checkpoint(
                run / "checkpoints" / selected_name, selected_step
            )
            if not _state_dict_exact(
                initial_payload["agent"]["actor"],
                periodic_payloads[5000]["agent"]["actor"],
            ):
                actor_delay_exact = False
                errors.append(
                    "block %d: residual actor changed before 10k delay" % index
                )
            if _state_dict_exact(
                initial_payload["agent"]["actor"],
                periodic_payloads[10000]["agent"]["actor"],
            ):
                actor_update_started_exact = False
                errors.append(
                    "block %d: residual actor did not start at 10k" % index
                )
            if any(
                str(payload["agent"]["base_actor_sha256"])
                != initial_base_hash
                for payload in periodic_payloads.values()
            ):
                frozen_base_actor_exact = False
                errors.append(
                    "block %d: frozen BC actor hash changed" % index
                )
        except (
            FileNotFoundError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as error:
            checkpoints_complete_and_loadable = False
            errors.append("block %d: %s" % (index, error))

    eligibility = _selected_blocks_satisfy_noninferiority(base["blocks"])
    checks = dict(base["checks"])
    checks.update({
        "conservative_schedule_exact": schedule_exact,
        "actor_initialization_exact": actor_initialization_exact,
        "checkpoints_complete_and_loadable": (
            checkpoints_complete_and_loadable
        ),
        "actor_delay_exact": actor_delay_exact,
        "actor_update_started_exact": actor_update_started_exact,
        "frozen_base_actor_exact": frozen_base_actor_exact,
        "selected_checkpoint_noninferiority": (
            sum(bool(row["eligible"]) for row in eligibility) >= 2
        ),
    })
    base.update({
        "gate_passed": bool(not errors and all(checks.values())),
        "checks": checks,
        "artifact_errors": errors,
        "selected_block_eligibility": eligibility,
    })
    return base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    output = _resolved(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = summarize(args.config, args.run_dir)
    (output / "l78_training_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output / "l78_training_audit_blocks.csv", result["blocks"]
    )
    _write_csv(
        output / "l78_training_audit_eligibility.csv",
        result["selected_block_eligibility"],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
