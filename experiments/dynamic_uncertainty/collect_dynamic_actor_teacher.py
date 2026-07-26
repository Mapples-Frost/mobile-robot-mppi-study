"""Collect deployable-observation demonstrations from safe probabilistic MPPI.

The teacher is standard risk-aware MPPI with the current stopping-feasibility
and emergency-motion safety fixes enabled.  The frozen L175 Actor is never
allowed to affect control; its encoder is reused only to produce the exact
48-dimensional observation available at deployment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

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
from mobile_robot_mppi.core.spaces import ActionSpec
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.observation import (
    ObservationEncoder,
    ObservationEncoderConfig,
)
from mobile_robot_mppi.runtime.experiment_runner import ExperimentRunner


DEFAULT_CONFIG = ROOT / "configs/research/dynamic_actor_correction_development.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _actor_contract(checkpoint: Path, expected_sha256: str):
    actual = _sha256(checkpoint)
    if actual != str(expected_sha256):
        raise ValueError("dynamic Actor source checkpoint SHA256 mismatch")
    payload = load_sac_checkpoint(checkpoint, map_location="cpu")
    if str(payload.get("action_mode")) != "direct_control":
        raise ValueError("dynamic teacher requires a direct-control Actor encoder")
    saved = payload["action_spec"]
    action_spec = ActionSpec(
        names=tuple(saved["names"]),
        lower=np.asarray(saved["lower"], dtype=np.float64),
        upper=np.asarray(saved["upper"], dtype=np.float64),
    )
    saved_encoder = payload.get("encoder_action_spec", saved)
    encoder_action_spec = ActionSpec(
        names=tuple(saved_encoder["names"]),
        lower=np.asarray(saved_encoder["lower"], dtype=np.float64),
        upper=np.asarray(saved_encoder["upper"], dtype=np.float64),
    )
    encoder = ObservationEncoder(
        ObservationEncoderConfig.from_mapping(payload["encoder_config"]),
        encoder_action_spec,
    )
    if encoder.dimension != int(payload["agent"]["observation_dim"]):
        raise ValueError("Actor checkpoint encoder dimension drifted")
    return payload, encoder, action_spec, actual


def configure_teacher_job(
    base,
    stage3,
    stage4,
    seed,
    maximum_steps,
    hard_violation_action="active_avoidance",
):
    """Return safe standard-MPPI teacher config without learned proposals."""

    config = configure_combined_job(
        base, stage3, stage4, "rl_hss_off", int(seed)
    )
    config["planner"].update({
        "probabilistic_obstacle_stopping_feasibility_enabled": True,
        "probabilistic_obstacle_emergency_candidates_enabled": True,
        "probabilistic_obstacle_emergency_candidate_prefix_steps": 3,
        "probabilistic_obstacle_hard_violation_action": str(
            hard_violation_action
        ),
    })
    config["perception"]["scan_guard"].update({
        "dynamic_escape_probability_mass_enabled": True,
        "dynamic_escape_hold_enabled": True,
        "dynamic_escape_hold_steps": 3,
        "dynamic_escape_hold_min_probability": 0.20,
    })
    config["experiment"]["max_steps"] = int(maximum_steps)
    config["experiment"]["name"] += "__dynamic_actor_teacher"
    config["experiment"]["dynamic_actor_teacher"] = True
    config["experiment"]["sealed_seeds_opened"] = False
    return config


class _RecordingController:
    """Transparent controller proxy that records causal Actor inputs/targets."""

    def __init__(self, controller, encoder, action_spec, episode_seed):
        self._controller = controller
        self._encoder = encoder
        self._action_spec = action_spec
        self._seed = int(episode_seed)
        self._previous_action = np.zeros(action_spec.dimension, dtype=np.float64)
        self._safety_override = False
        self._pending = None
        self.rows = []

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def reset(self, *args, **kwargs):
        self._encoder.reset()
        self._previous_action.fill(0.0)
        self._safety_override = False
        self._pending = None
        self.rows = []
        return self._controller.reset(*args, **kwargs)

    def plan(self, observation, reference):
        raw = self._encoder.encode(
            observation,
            reference,
            previous_action=self._previous_action,
            safety_override=self._safety_override,
        )
        plan = self._controller.plan(observation, reference)
        diagnostics = dict(plan.diagnostics)
        target_x = float(diagnostics.get("target_x", observation.pose.x))
        target_y = float(diagnostics.get("target_y", observation.pose.y))
        self._pending = {
            "observation": raw.astype(np.float32, copy=True),
            "proposed_action": np.asarray(
                plan.proposed_control.values, dtype=np.float32
            ).copy(),
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
        return plan

    def observe_safety_decision(self, decision):
        if self._pending is None:
            raise RuntimeError("teacher safety decision has no pending plan")
        executed = np.asarray(
            decision.executed_control.values, dtype=np.float32
        ).copy()
        row = dict(self._pending)
        row.update({
            "executed_action": executed,
            "target_action": np.clip(
                executed, self._action_spec.lower, self._action_spec.upper
            ).astype(np.float32),
            "reverse_requested": bool(executed[0] < -1.0e-6),
            "safety_override": bool(decision.overridden),
            "safety_reason": str(decision.reason),
        })
        self.rows.append(row)
        self._previous_action = executed.astype(np.float64, copy=True)
        self._safety_override = bool(decision.overridden)
        self._pending = None
        return self._controller.observe_safety_decision(decision)

    def close(self):
        close = getattr(self._controller, "close", None)
        if callable(close):
            return close()
        return None


def _rows_to_arrays(rows, validation, thresholds, episode_summary):
    if not rows:
        raise RuntimeError("dynamic Actor teacher episode produced no samples")
    arrays = {
        "observations": np.stack([row["observation"] for row in rows]),
        "proposed_actions": np.stack([row["proposed_action"] for row in rows]),
        "executed_actions": np.stack([row["executed_action"] for row in rows]),
        "target_actions": np.stack([row["target_action"] for row in rows]),
    }
    scalar_float = (
        "maximum_probability",
        "probability_mass",
        "candidate_feasible_fraction",
        "nearest_dynamic_obstacle_distance",
        "goal_distance",
    )
    for name in scalar_float:
        arrays[name] = np.asarray([row[name] for row in rows], dtype=np.float32)
    arrays["steps"] = np.asarray([row["step"] for row in rows], dtype=np.int32)
    arrays["episode_seeds"] = np.asarray(
        [row["episode_seed"] for row in rows], dtype=np.int64
    )
    arrays["validation"] = np.full(len(rows), bool(validation), dtype=np.bool_)
    arrays["safety_override"] = np.asarray(
        [row["safety_override"] for row in rows], dtype=np.bool_
    )
    arrays["reverse_requested"] = np.asarray(
        [row["reverse_requested"] for row in rows], dtype=np.bool_
    )
    arrays["episode_success"] = np.full(
        len(rows), bool(episode_summary.get("success", False)), dtype=np.bool_
    )
    arrays["episode_collision"] = np.full(
        len(rows), bool(episode_summary.get("collision", False)), dtype=np.bool_
    )
    arrays["low_risk_opportunity"] = (
        (arrays["maximum_probability"] <= float(
            thresholds["low_risk_maximum_probability"]
        ))
        & (arrays["candidate_feasible_fraction"] >= float(
            thresholds["minimum_candidate_feasible_fraction"]
        ))
        & (arrays["target_actions"][:, 0] >= float(
            thresholds["minimum_motion_speed_mps"]
        ))
        & ~arrays["safety_override"]
    )
    return arrays


def collect(config_path=DEFAULT_CONFIG, output_dir=None, maximum_episodes=None):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("protocol") not in {
        "dynamic_actor_correction_development_v1",
        "dynamic_actor_source_motion_development_v1",
    }:
        raise ValueError("dynamic Actor correction protocol mismatch")
    collection = dict(config["collection"])
    output = Path(output_dir or ROOT / collection["output_dir"]).resolve()
    if output.exists():
        raise FileExistsError("dynamic Actor teacher output already exists")
    checkpoint = ROOT / config["source_checkpoint"]
    _, encoder, action_spec, checkpoint_sha = _actor_contract(
        checkpoint, config["source_checkpoint_sha256"]
    )
    protocol = _mapping(STAGE5_PROTOCOL)
    base = load_yaml(_resolve(
        collection.get("base_config", protocol["base_config"])
    ))
    stage3 = _mapping(_resolve(protocol["stage3_protocol"]))
    stage4 = _mapping(_resolve(protocol["stage4_protocol"]))
    seeds = [int(value) for value in collection["episode_seeds"]]
    if maximum_episodes is not None:
        seeds = seeds[: int(maximum_episodes)]
    validation_seeds = {
        int(value) for value in collection.get("validation_seeds", ())
    }
    output.mkdir(parents=True, exist_ok=False)
    shards = []
    episode_rows = []
    for index, seed in enumerate(seeds):
        run_dir = output / "runs" / ("seed_%d" % seed)
        job = configure_teacher_job(
            base,
            stage3,
            stage4,
            seed,
            collection["maximum_steps"],
            collection.get(
                "hard_violation_action", "active_avoidance"
            ),
        )
        runner = ExperimentRunner(job, ROOT, output_dir=run_dir, headless=True)
        recorder = _RecordingController(
            runner.components["controller"], encoder, action_spec, seed
        )
        runner.components["controller"] = recorder
        print(json.dumps({
            "stage": "teacher_collection",
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
        episode_rows.append({**descriptor, "summary": result.summary})
        _write_json(output / "progress.json", {
            "completed": len(shards), "total": len(seeds), "shards": shards
        })
    manifest = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "git_sha": git_sha(ROOT),
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "source_checkpoint": config["source_checkpoint"],
        "source_checkpoint_sha256": checkpoint_sha,
        "observation_dim": encoder.dimension,
        "action_dim": action_spec.dimension,
        "action_names": list(action_spec.names),
        "action_lower": action_spec.lower.tolist(),
        "action_upper": action_spec.upper.tolist(),
        "student_observation_contract": "deployable_lidar_odometry_goal_safety_only",
        "future_truth_available_to_student": False,
        "teacher": "standard_risk_aware_mppi_combined_safety_v1",
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
        "episodes": episode_rows,
    }
    _write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest["totals"], sort_keys=True), flush=True)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--maximum-episodes", type=int)
    args = parser.parse_args(argv)
    collect(args.config, args.output_dir, args.maximum_episodes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
