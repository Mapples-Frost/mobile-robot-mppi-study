"""Collect safe-MPPI labels on states visited by a dynamic Actor roll-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from experiments.dynamic_uncertainty.collect_dynamic_actor_teacher import (
    DEFAULT_CONFIG,
    _actor_contract,
    _rows_to_arrays,
    _sha256,
    _write_json,
    configure_teacher_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_combined_safety_probe import (
    configure_combined_job,
)
from experiments.dynamic_uncertainty.run_rl_hss_exact_fallback_probe import (
    STAGE5_PROTOCOL,
)
from experiments.dynamic_uncertainty.run_rl_hss_stage4 import (
    _mapping,
    _resolve,
)
from mobile_robot_mppi.core.config import git_sha, load_yaml
from mobile_robot_mppi.core.types import SafetyDecision
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner
from mobile_robot_mppi.runtime.factories import make_components


DEFAULT_CHECKPOINT = (
    ROOT
    / "research_artifacts/dynamic_actor_correction_development_v2/"
    "checkpoints/best.pt"
)
DEFAULT_OUTPUT = ROOT / "research_artifacts/dynamic_actor_rollin_teacher_v1"
DEFAULT_SEEDS = tuple(range(730100118, 730100126))
DEFAULT_VALIDATION_SEEDS = frozenset((730100124, 730100125))


def configure_rollin_job(base, stage3, stage4, seed, checkpoint, maximum_steps):
    config = configure_combined_job(
        base, stage3, stage4, "combined_veto", int(seed)
    )
    config["planner"]["paper_rl_driven"]["proposal_advantage_gate"] = {
        "enabled": True,
        "mode": "shadow",
        "relative_disadvantage_margin": 0.0,
        "consecutive_disadvantages": 3,
    }
    config["planner"]["paper_rl_driven"][
        "standard_fallback_on_advantage_veto"
    ] = False
    config["rl"]["checkpoint"] = str(Path(checkpoint).resolve())
    config["rl"]["policy_id"] = "dynamic_actor_rollin"
    config["experiment"]["max_steps"] = int(maximum_steps)
    config["experiment"]["name"] += "__dynamic_actor_rollin"
    config["experiment"]["sealed_seeds_opened"] = False
    return config


class _RollinRecordingController:
    def __init__(
        self,
        student_controller,
        teacher_controller,
        teacher_reference,
        teacher_plant,
        encoder,
        action_spec,
        episode_seed,
    ):
        self._student = student_controller
        self._teacher = teacher_controller
        self._teacher_reference = teacher_reference
        self._teacher_plant = teacher_plant
        self._encoder = encoder
        self._action_spec = action_spec
        self._seed = int(episode_seed)
        self._previous_action = np.zeros(action_spec.dimension, dtype=np.float64)
        self._safety_override = False
        self._pending = None
        self.rows = []

    def __getattr__(self, name):
        return getattr(self._student, name)

    def reset(self, *args, **kwargs):
        self._encoder.reset()
        self._previous_action.fill(0.0)
        self._safety_override = False
        self._pending = None
        self.rows = []
        teacher_reference_reset = getattr(self._teacher_reference, "reset", None)
        if callable(teacher_reference_reset):
            teacher_reference_reset()
        self._teacher.reset(*args, **kwargs)
        return self._student.reset(*args, **kwargs)

    def plan(self, observation, reference):
        raw = self._encoder.encode(
            observation,
            reference,
            previous_action=self._previous_action,
            safety_override=self._safety_override,
        )
        teacher_plan = self._teacher.plan(
            observation, self._teacher_reference
        )
        student_plan = self._student.plan(observation, reference)
        diagnostics = dict(teacher_plan.diagnostics)
        target_x = float(diagnostics.get("target_x", observation.pose.x))
        target_y = float(diagnostics.get("target_y", observation.pose.y))
        teacher_action = np.asarray(
            teacher_plan.proposed_control.values, dtype=np.float32
        ).copy()
        self._pending = {
            "observation": raw.astype(np.float32, copy=True),
            "proposed_action": teacher_action.copy(),
            "executed_action": teacher_action.copy(),
            "target_action": np.clip(
                teacher_action, self._action_spec.lower, self._action_spec.upper
            ).astype(np.float32),
            "reverse_requested": bool(teacher_action[0] < -1.0e-6),
            "maximum_probability": float(diagnostics.get(
                "probabilistic_obstacle_maximum_step_probability", 0.0
            )),
            "probability_mass": float(diagnostics.get(
                "probabilistic_obstacle_probability_mass", 0.0
            )),
            "candidate_feasible_fraction": float(diagnostics.get(
                "probabilistic_obstacle_candidate_feasible_fraction", 1.0
            )),
            "nearest_dynamic_obstacle_distance": float(
                observation.auxiliary.get(
                    "nearest_dynamic_obstacle_center_distance", np.inf
                )
            ),
            "goal_distance": float(np.hypot(
                target_x - float(observation.pose.x),
                target_y - float(observation.pose.y),
            )),
            "step": len(self.rows),
            "episode_seed": self._seed,
        }
        return student_plan

    def observe_safety_decision(self, decision):
        if self._pending is None:
            raise RuntimeError("roll-in safety decision has no pending plans")
        actual = np.asarray(
            decision.executed_control.values, dtype=np.float64
        )
        teacher_proposed = self._pending["proposed_action"]
        teacher_feedback = SafetyDecision(
            proposed_control=type(decision.proposed_control)(
                teacher_proposed.astype(np.float64),
                decision.proposed_control.timestamp,
                "rollin_teacher",
            ),
            executed_control=decision.executed_control,
            overridden=bool(np.max(np.abs(
                teacher_proposed.astype(np.float64) - actual
            )) > 1.0e-12),
            reason="student_rollin_action",
            diagnostics={},
        )
        self._teacher.observe_safety_decision(teacher_feedback)
        self._student.observe_safety_decision(decision)
        row = dict(self._pending)
        row.update({
            "safety_override": bool(decision.overridden),
            "safety_reason": str(decision.reason),
        })
        self.rows.append(row)
        self._previous_action = actual.copy()
        self._safety_override = bool(decision.overridden)
        self._pending = None

    def close(self):
        for component in (self._student, self._teacher, self._teacher_plant):
            close = getattr(component, "close", None)
            if callable(close):
                close()


def collect(
    config_path=DEFAULT_CONFIG,
    checkpoint=DEFAULT_CHECKPOINT,
    output_dir=DEFAULT_OUTPUT,
    seeds=DEFAULT_SEEDS,
    validation_seeds=DEFAULT_VALIDATION_SEEDS,
):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    collection = dict(config["collection"])
    checkpoint = Path(checkpoint).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("dynamic Actor roll-in output already exists")
    checkpoint_sha = _sha256(checkpoint)
    _, encoder, action_spec, _ = _actor_contract(checkpoint, checkpoint_sha)
    protocol = _mapping(STAGE5_PROTOCOL)
    base = load_yaml(_resolve(protocol["base_config"]))
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    seeds = tuple(int(value) for value in seeds)
    validation_seeds = frozenset(int(value) for value in validation_seeds)
    if any(not 730100001 <= seed <= 730100300 for seed in seeds):
        raise ValueError("roll-in seeds must remain development-only")
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    for index, seed in enumerate(seeds):
        student_job = configure_rollin_job(
            base,
            stage3,
            stage4,
            seed,
            checkpoint,
            collection["maximum_steps"],
        )
        teacher_job = configure_teacher_job(
            base, stage3, stage4, seed, collection["maximum_steps"]
        )
        runner = ExperimentRunner(
            student_job,
            ROOT,
            output_dir=output / "runs" / ("seed_%d" % seed),
            headless=True,
        )
        teacher_components = make_components(teacher_job, ROOT)
        recorder = _RollinRecordingController(
            runner.components["controller"],
            teacher_components["controller"],
            teacher_components["reference"],
            teacher_components["plant"],
            encoder,
            action_spec,
            seed,
        )
        runner.components["controller"] = recorder
        print(json.dumps({
            "stage": "dynamic_actor_rollin",
            "episode": index + 1,
            "episodes": len(seeds),
            "seed": seed,
        }, sort_keys=True), flush=True)
        result = runner.run()
        arrays = _rows_to_arrays(
            recorder.rows,
            seed in validation_seeds,
            collection,
            result.summary,
        )
        shard_path = output / "shards" / ("seed_%d.npz" % seed)
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(shard_path, **arrays)
        descriptor = {
            "seed": seed,
            "validation": seed in validation_seeds,
            "samples": int(len(arrays["observations"])),
            "opportunity_samples": int(np.sum(arrays["low_risk_opportunity"])),
            "reverse_requested_samples": int(np.sum(arrays["reverse_requested"])),
            "success": bool(result.summary.get("success", False)),
            "collision": bool(result.summary.get("collision", False)),
            "file": str(shard_path.relative_to(output)),
            "sha256": _sha256(shard_path),
        }
        shards.append(descriptor)
        _write_json(output / "progress.json", {
            "completed": len(shards), "total": len(seeds), "shards": shards
        })
    manifest = {
        "schema_version": 1,
        "protocol": "dynamic_actor_closed_loop_rollin_v1",
        "git_sha": git_sha(ROOT),
        "source_checkpoint": str(checkpoint.relative_to(ROOT)),
        "source_checkpoint_sha256": checkpoint_sha,
        "observation_dim": encoder.dimension,
        "action_dim": action_spec.dimension,
        "action_names": list(action_spec.names),
        "action_lower": action_spec.lower.tolist(),
        "action_upper": action_spec.upper.tolist(),
        "student_observation_contract": "deployable_lidar_odometry_goal_safety_only",
        "future_truth_available_to_student": False,
        "teacher": "standard_risk_aware_mppi_on_dynamic_actor_rollin_states",
        "rollin_policy": "dynamic_actor_shadow_full_authority",
        "sealed_seeds_opened": False,
        "shards": shards,
        "totals": {
            "episodes": len(shards),
            "samples": int(sum(row["samples"] for row in shards)),
            "opportunity_samples": int(sum(
                row["opportunity_samples"] for row in shards
            )),
            "reverse_requested_samples": int(sum(
                row["reverse_requested_samples"] for row in shards
            )),
            "successes": int(sum(row["success"] for row in shards)),
            "collisions": int(sum(row["collision"] for row in shards)),
        },
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["totals"], sort_keys=True), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--validation-seeds",
        default=",".join(map(str, sorted(DEFAULT_VALIDATION_SEEDS))),
    )
    args = parser.parse_args(argv)
    collect(
        args.config,
        args.checkpoint,
        args.output_dir,
        tuple(int(value) for value in args.seeds.split(",") if value),
        frozenset(
            int(value) for value in args.validation_seeds.split(",") if value
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

