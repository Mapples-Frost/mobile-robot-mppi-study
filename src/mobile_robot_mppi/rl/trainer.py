"""Reproducible multi-scene SAC training and validation loop."""

import copy
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from mobile_robot_mppi.core.config import git_sha
from .checkpointing import load_sac_checkpoint, save_sac_checkpoint
from .demonstrations import (
    demonstration_manifest_fingerprint,
    load_demonstration_manifest,
    load_demonstration_split,
)
from .environment import DirectControlEnv, MppiPriorEnv
from .observation import RunningNormalizer
from .parameterization import PriorParameterizationConfig
from .replay import ReplayBuffer
from .sac import SACAgent, SACConfig


_RESUME_OVERRIDE_FIELDS = frozenset((
    "total_steps",
    "checkpoint_interval",
))


def environment_class_from_config(config):
    """Resolve the RL action semantics without changing legacy defaults."""

    mode = str(
        dict(config.get("rl", {}))
        .get("training", {})
        .get("action_mode", "mppi_prior")
    )
    if mode == "mppi_prior":
        return MppiPriorEnv
    if mode == "direct_control":
        return DirectControlEnv
    raise ValueError(
        "rl.training.action_mode must be mppi_prior or direct_control"
    )


def _json_compatible(value):
    """Return a canonical JSON-safe representation for contract comparison."""

    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _resume_scene_contract(config):
    """Remove only explicitly permitted continuation-only overrides."""

    result = copy.deepcopy(dict(config))
    training = result.get("rl", {}).get("training", {})
    if isinstance(training, dict):
        for name in _RESUME_OVERRIDE_FIELDS:
            training.pop(name, None)
        anchor = training.get("bc_anchor", {})
        if isinstance(anchor, dict):
            # Dataset identity is locked by its manifest digest below; an
            # absolute/relative path change alone must not alter the contract.
            anchor.pop("dataset_dir", None)
    experiment = result.get("experiment", {})
    if isinstance(experiment, dict):
        # Output location is provenance, not environment or optimization state.
        experiment.pop("output_dir", None)
    return _json_compatible(result)


def _contract_digest(contract):
    encoded = json.dumps(
        contract, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 200000
    warmup_steps: int = 5000
    warmup_policy: str = "random"
    update_after: int = 1000
    actor_update_after: int = 0
    batch_size: int = 256
    replay_capacity: int = 500000
    replay_sampling: str = "uniform"
    replay_success_fraction: float = 0.25
    replay_require_all_scenes: bool = False
    updates_per_step: int = 1
    evaluation_interval: int = 10000
    evaluation_episodes: int = 3
    checkpoint_interval: int = 10000
    evaluate_initial_policy: bool = False
    save_replay_buffer: bool = True
    seed: int = 0
    validation_seed_base: Optional[int] = None
    validation_initial_state_noise: Optional[Sequence[float]] = None
    normalizer_update: str = "online"
    device: str = "auto"

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(**{
            name: values.get(name, field.default)
            for name, field in cls.__dataclass_fields__.items()
        })

    def validate(self):
        integer_positive = (
            self.total_steps,
            self.batch_size,
            self.replay_capacity,
            self.updates_per_step,
            self.evaluation_episodes,
        )
        if any(int(value) <= 0 for value in integer_positive):
            raise ValueError("positive RL training counts are required")
        if (
            self.warmup_steps < 0
            or self.update_after < 0
            or self.actor_update_after < 0
        ):
            raise ValueError(
                "RL warmup/update_after/actor_update_after cannot be negative"
            )
        if self.warmup_policy not in ("random", "actor"):
            raise ValueError("RL warmup_policy must be random or actor")
        if self.evaluation_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError("RL evaluation/checkpoint intervals cannot be negative")
        if self.replay_sampling not in (
            "uniform", "scene_balanced", "outcome_balanced",
            "scene_outcome_balanced",
        ):
            raise ValueError(
                "RL replay_sampling must be uniform, scene_balanced, or "
                "outcome_balanced, or scene_outcome_balanced"
            )
        if (
            not np.isfinite(self.replay_success_fraction)
            or not 0.0 <= self.replay_success_fraction <= 1.0
        ):
            raise ValueError("RL replay_success_fraction must be in [0, 1]")
        if self.validation_seed_base is not None and not (
            0 <= int(self.validation_seed_base) <= 2 ** 32 - 1
        ):
            raise ValueError("RL validation_seed_base must be in [0, 2**32 - 1]")
        if self.validation_initial_state_noise is not None:
            noise = np.asarray(
                self.validation_initial_state_noise, dtype=np.float64
            ).reshape(-1)
            if noise.size == 0 or not np.isfinite(noise).all() or np.any(noise < 0.0):
                raise ValueError(
                    "RL validation_initial_state_noise must be finite and non-negative"
                )
        if self.normalizer_update not in ("online", "frozen"):
            raise ValueError("RL normalizer_update must be online or frozen")


@dataclass(frozen=True)
class PrecisionConstraintConfig:
    """Episode-level dual control for a per-transition precision cost."""

    enabled: bool = False
    target_cross_track_rmse_m: float = 0.04
    dual_lr: float = 250.0
    initial_multiplier: float = 0.0
    maximum_multiplier: float = 100.0

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        return cls(**{
            name: values.get(name, field.default)
            for name, field in cls.__dataclass_fields__.items()
        })

    @property
    def target_mean_square(self):
        return float(self.target_cross_track_rmse_m) ** 2

    def validate(self):
        numeric = np.asarray((
            self.target_cross_track_rmse_m,
            self.dual_lr,
            self.initial_multiplier,
            self.maximum_multiplier,
        ), dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError("precision constraint settings must be finite")
        if self.target_cross_track_rmse_m <= 0.0:
            raise ValueError("precision constraint target must be positive")
        if self.dual_lr < 0.0:
            raise ValueError("precision constraint dual_lr cannot be negative")
        if self.initial_multiplier < 0.0 or self.maximum_multiplier <= 0.0:
            raise ValueError("precision constraint multiplier bounds are invalid")
        if self.initial_multiplier > self.maximum_multiplier:
            raise ValueError(
                "precision constraint initial multiplier exceeds maximum"
            )


class PrecisionConstraintController:
    """Projected dual ascent with explicit raw-cost reward shaping."""

    def __init__(self, config):
        self.config = (
            config
            if isinstance(config, PrecisionConstraintConfig)
            else PrecisionConstraintConfig.from_mapping(config)
        )
        self.config.validate()
        self.multiplier = float(self.config.initial_multiplier)
        self.updates = 0

    def shape_batch(self, batch):
        result = dict(batch)
        if not self.config.enabled:
            return result
        if "constraint_costs" not in batch:
            raise ValueError("constrained training batch has no constraint_costs")
        costs = np.asarray(batch["constraint_costs"], dtype=np.float32)
        if not np.isfinite(costs).all() or np.any(costs < 0.0):
            raise ValueError("constraint replay costs must be finite and non-negative")
        result["rewards"] = (
            np.asarray(batch["rewards"], dtype=np.float32)
            - np.float32(self.multiplier) * costs
        )
        return result

    def update(self, episode_mean_square):
        cost = float(episode_mean_square)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError("episode precision cost must be finite and non-negative")
        violation = cost - self.config.target_mean_square
        before = float(self.multiplier)
        if self.config.enabled:
            self.multiplier = float(np.clip(
                before + float(self.config.dual_lr) * violation,
                0.0,
                float(self.config.maximum_multiplier),
            ))
            self.updates += 1
        return {
            "precision_constraint_cost_mean_square": cost,
            "precision_constraint_target_mean_square": (
                self.config.target_mean_square
            ),
            "precision_constraint_violation": violation,
            "precision_constraint_multiplier_before": before,
            "precision_constraint_multiplier": float(self.multiplier),
            "precision_constraint_feasible": bool(violation <= 0.0),
        }

    def state_dict(self):
        return {
            "config": asdict(self.config),
            "multiplier": float(self.multiplier),
            "updates": int(self.updates),
        }

    def load_state_dict(self, state):
        saved = PrecisionConstraintConfig.from_mapping(state["config"])
        saved.validate()
        if saved != self.config:
            raise ValueError("precision constraint config does not match")
        multiplier = float(state["multiplier"])
        updates = int(state.get("updates", 0))
        if (
            not np.isfinite(multiplier)
            or not 0.0 <= multiplier <= self.config.maximum_multiplier
            or updates < 0
        ):
            raise ValueError("precision constraint checkpoint state is invalid")
        self.multiplier = multiplier
        self.updates = updates


@dataclass(frozen=True)
class CheckpointSelectionConfig:
    mode: str = "scalar"
    maximum_success_losses: int = 0
    maximum_collision_regressions: int = 0
    minimum_success_gains: int = 1
    minimum_mean_goal_distance_improvement: float = 0.005
    maximum_mean_goal_distance_increase: float = 0.0
    maximum_mean_cross_track_rmse: float = 0.04

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        return cls(**{
            name: values.get(name, field.default)
            for name, field in cls.__dataclass_fields__.items()
        })

    def validate(self):
        if self.mode not in (
            "scalar", "initial_noninferiority", "precision_constrained"
        ):
            raise ValueError(
                "checkpoint selection mode must be scalar, initial_noninferiority, "
                "or precision_constrained"
            )
        integer_values = (
            self.maximum_success_losses,
            self.maximum_collision_regressions,
            self.minimum_success_gains,
        )
        if any(int(value) != value or int(value) < 0 for value in integer_values):
            raise ValueError(
                "checkpoint selection paired-count thresholds must be non-negative integers"
            )
        if self.minimum_success_gains <= 0:
            raise ValueError(
                "checkpoint selection minimum_success_gains must be positive"
            )
        numeric = np.asarray((
            self.minimum_mean_goal_distance_improvement,
            self.maximum_mean_goal_distance_increase,
        ), dtype=np.float64)
        if not np.isfinite(numeric).all() or np.any(numeric < 0.0):
            raise ValueError(
                "checkpoint selection distance thresholds must be finite and non-negative"
            )
        if (
            not np.isfinite(self.maximum_mean_cross_track_rmse)
            or self.maximum_mean_cross_track_rmse <= 0.0
        ):
            raise ValueError(
                "checkpoint selection cross-track threshold must be positive"
            )


class ValidationCheckpointSelector:
    """Paired, fail-closed selection against the immutable step-zero policy."""

    def __init__(self, config):
        self.config = (
            config
            if isinstance(config, CheckpointSelectionConfig)
            else CheckpointSelectionConfig.from_mapping(config)
        )
        self.config.validate()
        if self.config.mode != "initial_noninferiority":
            raise ValueError(
                "ValidationCheckpointSelector requires initial_noninferiority mode"
            )
        self.reference_rows = None
        self.best_rank = None
        self.best_global_step = None

    @staticmethod
    def _keyed(rows):
        result = {}
        for row in rows:
            key = (str(row["scene"]), int(row["seed"]))
            if key in result:
                raise ValueError("validation rows contain a duplicate scene/seed")
            goal_distance = float(row["goal_distance"])
            return_value = float(row["return"])
            if not np.isfinite((goal_distance, return_value)).all():
                raise ValueError("validation rows contain NaN or Inf")
            result[key] = {
                "scene": key[0],
                "seed": key[1],
                "success": bool(row["success"]),
                "collision": bool(row["collision"]),
                "goal_distance": goal_distance,
                "return": return_value,
            }
        if not result:
            raise ValueError("checkpoint selection requires validation rows")
        return result

    @staticmethod
    def _rank(summary):
        return (
            float(summary["success_rate"]),
            -float(summary["collision_rate"]),
            -float(summary["mean_goal_distance"]),
            float(summary["mean_return"]),
        )

    def consider(self, rows, summary, global_step):
        current = self._keyed(rows)
        rank = self._rank(summary)
        if self.reference_rows is None:
            self.reference_rows = current
            self.best_rank = rank
            self.best_global_step = int(global_step)
            return {
                "selection_mode": self.config.mode,
                "selection_eligible": True,
                "selection_meaningful_improvement": True,
                "selection_selected": True,
                "selection_reason": "initial_reference",
                "selection_success_losses": 0,
                "selection_success_gains": 0,
                "selection_collision_regressions": 0,
                "selection_mean_goal_distance_improvement": 0.0,
                "selection_best_global_step": self.best_global_step,
            }
        if set(current) != set(self.reference_rows):
            raise ValueError(
                "validation scene/seed set changed after step-zero reference"
            )
        success_losses = sum(
            reference["success"] and not current[key]["success"]
            for key, reference in self.reference_rows.items()
        )
        success_gains = sum(
            not reference["success"] and current[key]["success"]
            for key, reference in self.reference_rows.items()
        )
        collision_regressions = sum(
            not reference["collision"] and current[key]["collision"]
            for key, reference in self.reference_rows.items()
        )
        reference_distance = float(np.mean([
            row["goal_distance"] for row in self.reference_rows.values()
        ]))
        current_distance = float(summary["mean_goal_distance"])
        distance_improvement = reference_distance - current_distance
        eligible = bool(
            success_losses <= self.config.maximum_success_losses
            and collision_regressions
            <= self.config.maximum_collision_regressions
            and current_distance
            <= reference_distance
            + self.config.maximum_mean_goal_distance_increase
            + 1e-12
        )
        meaningful = bool(
            success_gains >= self.config.minimum_success_gains
            or distance_improvement
            >= self.config.minimum_mean_goal_distance_improvement
        )
        better_than_best = bool(rank > tuple(self.best_rank))
        selected = bool(eligible and meaningful and better_than_best)
        if selected:
            self.best_rank = rank
            self.best_global_step = int(global_step)
            reason = "paired_noninferior_improvement"
        elif not eligible:
            reason = "noninferiority_failed"
        elif not meaningful:
            reason = "improvement_below_threshold"
        else:
            reason = "not_better_than_selected_checkpoint"
        return {
            "selection_mode": self.config.mode,
            "selection_eligible": eligible,
            "selection_meaningful_improvement": meaningful,
            "selection_selected": selected,
            "selection_reason": reason,
            "selection_success_losses": int(success_losses),
            "selection_success_gains": int(success_gains),
            "selection_collision_regressions": int(collision_regressions),
            "selection_mean_goal_distance_improvement": float(
                distance_improvement
            ),
            "selection_best_global_step": int(self.best_global_step),
        }

    def state_dict(self):
        rows = None
        if self.reference_rows is not None:
            rows = [
                self.reference_rows[key]
                for key in sorted(self.reference_rows)
            ]
        return {
            "config": asdict(self.config),
            "reference_rows": rows,
            "best_rank": self.best_rank,
            "best_global_step": self.best_global_step,
        }

    def load_state_dict(self, state):
        saved_config = CheckpointSelectionConfig.from_mapping(state["config"])
        saved_config.validate()
        if saved_config != self.config:
            raise ValueError("checkpoint selection config does not match")
        rows = state.get("reference_rows")
        self.reference_rows = None if rows is None else self._keyed(rows)
        rank = state.get("best_rank")
        self.best_rank = None if rank is None else tuple(float(value) for value in rank)
        step = state.get("best_global_step")
        self.best_global_step = None if step is None else int(step)
        if (self.reference_rows is None) != (self.best_rank is None):
            raise ValueError("checkpoint selection state is incomplete")


class PrecisionConstrainedCheckpointSelector:
    """Select the best feasible validation policy, failing closed if needed."""

    def __init__(self, config):
        self.config = (
            config
            if isinstance(config, CheckpointSelectionConfig)
            else CheckpointSelectionConfig.from_mapping(config)
        )
        self.config.validate()
        if self.config.mode != "precision_constrained":
            raise ValueError(
                "PrecisionConstrainedCheckpointSelector requires its named mode"
            )
        self.best_rank = None
        self.best_global_step = None
        self.best_feasible = False

    def _rank(self, summary):
        cross_track = float(summary["mean_cross_track_rmse"])
        feasible = bool(
            cross_track <= self.config.maximum_mean_cross_track_rmse + 1e-12
        )
        # Feasibility dominates.  Within the feasible set, retain the ordinary
        # control hierarchy; before feasibility, reduce violation first while
        # still rejecting unsafe solutions.
        if feasible:
            rank = (
                1.0,
                float(summary["success_rate"]),
                -float(summary["collision_rate"]),
                float(summary["mean_return"]),
                -cross_track,
            )
        else:
            rank = (
                0.0,
                float(summary["success_rate"]),
                -float(summary["collision_rate"]),
                -cross_track,
                float(summary["mean_return"]),
            )
        if not np.isfinite(np.asarray(rank, dtype=np.float64)).all():
            raise ValueError("precision checkpoint summary contains NaN or Inf")
        return feasible, rank

    def consider(self, rows, summary, global_step):
        del rows  # Aggregate constraint is the pre-registered selection unit.
        feasible, rank = self._rank(summary)
        selected = bool(self.best_rank is None or rank > tuple(self.best_rank))
        if selected:
            self.best_rank = rank
            self.best_global_step = int(global_step)
            self.best_feasible = feasible
        if selected and feasible:
            reason = "best_precision_feasible"
        elif selected:
            reason = "lowest_available_precision_violation"
        elif not feasible:
            reason = "constraint_infeasible_not_improved"
        else:
            reason = "feasible_but_not_better"
        return {
            "selection_mode": self.config.mode,
            "selection_eligible": feasible,
            "selection_meaningful_improvement": selected,
            "selection_selected": selected,
            "selection_reason": reason,
            "selection_success_losses": 0,
            "selection_success_gains": 0,
            "selection_collision_regressions": 0,
            "selection_mean_goal_distance_improvement": 0.0,
            "selection_cross_track_threshold": float(
                self.config.maximum_mean_cross_track_rmse
            ),
            "selection_cross_track_violation": float(max(
                0.0,
                float(summary["mean_cross_track_rmse"])
                - self.config.maximum_mean_cross_track_rmse,
            )),
            "selection_best_feasible": bool(self.best_feasible),
            "selection_best_global_step": self.best_global_step,
        }

    def state_dict(self):
        return {
            "config": asdict(self.config),
            "best_rank": self.best_rank,
            "best_global_step": self.best_global_step,
            "best_feasible": bool(self.best_feasible),
        }

    def load_state_dict(self, state):
        saved_config = CheckpointSelectionConfig.from_mapping(state["config"])
        saved_config.validate()
        if saved_config != self.config:
            raise ValueError("checkpoint selection config does not match")
        rank = state.get("best_rank")
        self.best_rank = None if rank is None else tuple(
            float(value) for value in rank
        )
        step = state.get("best_global_step")
        self.best_global_step = None if step is None else int(step)
        self.best_feasible = bool(state.get("best_feasible", False))
        if (self.best_rank is None) != (self.best_global_step is None):
            raise ValueError("precision checkpoint selection state is incomplete")


@dataclass(frozen=True)
class BehaviorCloningAnchorConfig:
    enabled: bool = False
    dataset_dir: Optional[str] = None
    batch_size: int = 256
    mean_weight: float = 1.0
    log_std_weight: float = 0.0
    target_log_std: float = -2.0

    @classmethod
    def from_mapping(cls, values=None):
        values = dict(values or {})
        return cls(**{
            name: values.get(name, field.default)
            for name, field in cls.__dataclass_fields__.items()
        })

    def validate(self):
        if not self.enabled:
            return
        if not self.dataset_dir:
            raise ValueError("enabled BC anchor requires dataset_dir")
        if int(self.batch_size) <= 0:
            raise ValueError("BC anchor batch_size must be positive")
        values = np.asarray(
            (self.mean_weight, self.log_std_weight, self.target_log_std),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("BC anchor loss settings must be finite")
        if self.mean_weight <= 0.0 or self.log_std_weight < 0.0:
            raise ValueError(
                "BC anchor mean_weight must be positive and log_std_weight non-negative"
            )


class BehaviorCloningAnchor:
    """Train-split-only supervised anchor sampled beside online replay."""

    def __init__(self, config, project_root, observation_dim, action_dim, seed):
        self.config = (
            config
            if isinstance(config, BehaviorCloningAnchorConfig)
            else BehaviorCloningAnchorConfig.from_mapping(config)
        )
        self.config.validate()
        self.enabled = bool(self.config.enabled)
        self.dataset_dir = None
        self.manifest_fingerprint = None
        self.observations = None
        self.actions = None
        self.rng = np.random.RandomState(int(seed) + 7919)
        if not self.enabled:
            return
        path = Path(self.config.dataset_dir)
        if not path.is_absolute():
            path = Path(project_root) / path
        self.dataset_dir = path.resolve()
        manifest = load_demonstration_manifest(self.dataset_dir)
        arrays = load_demonstration_split(self.dataset_dir, "train")
        self.manifest_fingerprint = demonstration_manifest_fingerprint(manifest)
        self.observations = arrays["observations"]
        self.actions = arrays["teacher_actions"]
        if (
            self.observations.shape[0] <= 0
            or self.observations.shape[1] != int(observation_dim)
        ):
            raise ValueError("BC anchor observation contract does not match actor")
        if self.actions.shape[1] != int(action_dim):
            raise ValueError("BC anchor action contract does not match actor")

    def sample(self, normalizer):
        if not self.enabled:
            return None
        count = int(self.observations.shape[0])
        indices = self.rng.randint(
            0, count, size=int(self.config.batch_size)
        )
        return {
            "observations": normalizer.normalize(self.observations[indices]),
            "actions": self.actions[indices],
        }

    def metadata(self):
        return {
            "enabled": self.enabled,
            "dataset_dir": (
                None if self.dataset_dir is None else str(self.dataset_dir)
            ),
            "dataset_manifest_sha256": self.manifest_fingerprint,
            "batch_size": int(self.config.batch_size),
            "mean_weight": float(self.config.mean_weight),
            "log_std_weight": float(self.config.log_std_weight),
            "target_log_std": float(self.config.target_log_std),
        }


def _validation_episode_seed(training_config, config_index, episode_index):
    """Return a model-selection seed independent of the training seed.

    A missing base preserves checkpoints produced before this option existed.
    New multi-training-seed experiments should set one fixed base so that
    checkpoint selection noise is not changed together with optimization
    noise.
    """

    base = training_config.validation_seed_base
    if base is None:
        base = int(training_config.seed) + 100000
    return int(base) + int(config_index) * 1000 + int(episode_index)


class SceneCurriculum:
    """Piecewise-constant episode sampling schedule over configured scenes."""

    def __init__(self, config, scene_count):
        self.scene_count = int(scene_count)
        if self.scene_count <= 0:
            raise ValueError("curriculum requires at least one scene")
        values = dict(config or {})
        self.enabled = bool(values.get("enabled", False))
        self.phases = []
        if not self.enabled:
            return
        previous_until = 0
        for index, item in enumerate(values.get("phases", ())):
            until_step = int(item["until_step"])
            weights = np.asarray(item["scene_weights"], dtype=np.float64).reshape(-1)
            if until_step <= previous_until:
                raise ValueError("curriculum until_step values must strictly increase")
            if (
                weights.shape != (self.scene_count,)
                or not np.isfinite(weights).all()
                or np.any(weights < 0.0)
                or float(weights.sum()) <= 0.0
            ):
                raise ValueError(
                    "curriculum scene_weights must be finite, non-negative and match scenes"
                )
            weights = weights / float(weights.sum())
            self.phases.append({
                "name": str(item.get("name", "phase_%d" % index)),
                "until_step": until_step,
                "weights": weights,
            })
            previous_until = until_step
        if not self.phases:
            raise ValueError("enabled curriculum requires at least one phase")

    def at(self, global_step):
        if not self.enabled:
            return "uniform", np.full(
                self.scene_count, 1.0 / float(self.scene_count), dtype=np.float64
            )
        step = int(global_step)
        for phase in self.phases:
            if step < phase["until_step"]:
                return phase["name"], phase["weights"].copy()
        last = self.phases[-1]
        return last["name"], last["weights"].copy()

    def metadata(self):
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "phases": [
                {
                    "name": phase["name"],
                    "until_step": phase["until_step"],
                    "scene_weights": phase["weights"].tolist(),
                }
                for phase in self.phases
            ],
        }


class InitialStateCurriculum:
    """Training-only reverse curriculum over collision-free reset states.

    Policies still receive only the ordinary odometry/LaserScan observation.
    Validation does not use this sampler and always starts from the scene's
    configured initial state.  The schedule is therefore an exploration aid,
    not privileged deployment information.
    """

    def __init__(self, config):
        values = dict(config or {})
        self.enabled = bool(values.get("enabled", False))
        self.phases = []
        if not self.enabled:
            return
        previous_until = 0
        for phase_index, item in enumerate(values.get("phases", ())):
            until_step = int(item["until_step"])
            if until_step <= previous_until:
                raise ValueError(
                    "initial-state curriculum until_step values must strictly increase"
                )
            scenes = {}
            for scene, candidates in dict(item.get("scenes", {})).items():
                parsed = []
                weights = []
                for candidate_index, candidate in enumerate(candidates):
                    candidate = dict(candidate)
                    state = np.asarray(candidate["state"], dtype=np.float64).reshape(-1)
                    weight = float(candidate.get("weight", 1.0))
                    if state.size == 0 or not np.isfinite(state).all():
                        raise ValueError("initial-state candidates must be finite vectors")
                    if not np.isfinite(weight) or weight < 0.0:
                        raise ValueError("initial-state candidate weights must be non-negative")
                    parsed.append({
                        "name": str(candidate.get(
                            "name", "candidate_%d" % candidate_index
                        )),
                        "state": state,
                    })
                    weights.append(weight)
                weights = np.asarray(weights, dtype=np.float64)
                if not parsed or float(weights.sum()) <= 0.0:
                    raise ValueError(
                        "each configured initial-state scene needs positive total weight"
                    )
                scenes[str(scene)] = {
                    "candidates": parsed,
                    "weights": weights / float(weights.sum()),
                }
            self.phases.append({
                "name": str(item.get("name", "reset_phase_%d" % phase_index)),
                "until_step": until_step,
                "scenes": scenes,
            })
            previous_until = until_step
        if not self.phases:
            raise ValueError(
                "enabled initial-state curriculum requires at least one phase"
            )

    def _phase(self, global_step):
        if not self.enabled:
            return None
        for phase in self.phases:
            if int(global_step) < phase["until_step"]:
                return phase
        return self.phases[-1]

    def sample(self, global_step, scene, rng):
        phase = self._phase(global_step)
        if phase is None:
            return None, "configured_initial", "disabled"
        scene = str(scene)
        distribution = phase["scenes"].get(scene)
        if distribution is None and "__" in scene:
            distribution = phase["scenes"].get(scene.split("__", 1)[0])
        if distribution is None:
            return None, "configured_initial", phase["name"]
        index = int(rng.choice(
            len(distribution["candidates"]), p=distribution["weights"]
        ))
        candidate = distribution["candidates"][index]
        return candidate["state"].copy(), candidate["name"], phase["name"]

    def metadata(self):
        if not self.enabled:
            return {"enabled": False}
        return {
            "enabled": True,
            "phases": [
                {
                    "name": phase["name"],
                    "until_step": phase["until_step"],
                    "scenes": {
                        scene: [
                            {
                                "name": candidate["name"],
                                "state": candidate["state"].tolist(),
                                "weight": float(distribution["weights"][index]),
                            }
                            for index, candidate in enumerate(
                                distribution["candidates"]
                            )
                        ]
                        for scene, distribution in phase["scenes"].items()
                    },
                }
                for phase in self.phases
            ],
        }


class EnvironmentPool:
    def __init__(self, configs, project_root, seed=0):
        if not configs:
            raise ValueError("RL training requires at least one scene config")
        self.configs = list(configs)
        self.project_root = project_root
        self.rng = np.random.RandomState(int(seed))
        self.environments = {}
        self.current_index = None
        self.current = None

    def _get(self, index):
        if index not in self.environments:
            environment_class = environment_class_from_config(
                self.configs[index]
            )
            self.environments[index] = environment_class(
                self.configs[index], self.project_root, seed=int(self.rng.randint(0, 2 ** 31 - 1))
            )
        return self.environments[index]

    def reset(
        self,
        episode_seed=None,
        sampling_weights=None,
        initial_state_curriculum=None,
        global_step=0,
    ):
        if sampling_weights is None:
            sampling_weights = np.full(
                len(self.configs), 1.0 / float(len(self.configs)), dtype=np.float64
            )
        probabilities = np.asarray(sampling_weights, dtype=np.float64).reshape(-1)
        if (
            probabilities.shape != (len(self.configs),)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0.0)
            or float(probabilities.sum()) <= 0.0
        ):
            raise ValueError("environment sampling weights are invalid")
        probabilities = probabilities / float(probabilities.sum())
        index = int(self.rng.choice(len(self.configs), p=probabilities))
        self.current_index = index
        self.current = self._get(index)
        scene = self.configs[index].get("scene", {}).get("name", "unknown")
        initial_state = None
        reset_name = "configured_initial"
        reset_phase = "disabled"
        if initial_state_curriculum is not None:
            initial_state, reset_name, reset_phase = initial_state_curriculum.sample(
                global_step, scene, self.rng
            )
        observation, info = self.current.reset(
            seed=episode_seed, initial_state=initial_state
        )
        info = dict(info)
        info.update({
            "initial_state_name": reset_name,
            "initial_state_phase": reset_phase,
        })
        return observation, info

    def close(self):
        for environment in self.environments.values():
            environment.close()


class SACTrainer:
    def __init__(
        self,
        training_configs: Sequence[Mapping[str, Any]],
        validation_configs: Sequence[Mapping[str, Any]],
        project_root,
        output_dir,
        resolved_config,
    ):
        self.resolved_config = dict(resolved_config)
        rl_config = dict(self.resolved_config.get("rl", {}))
        self.training_config = TrainingConfig.from_mapping(rl_config.get("training", {}))
        self.training_config.validate()
        training_mapping = dict(rl_config.get("training", {}))
        self.checkpoint_selection_config = CheckpointSelectionConfig.from_mapping(
            training_mapping.get("checkpoint_selection", {})
        )
        self.checkpoint_selection_config.validate()
        if self.checkpoint_selection_config.mode == "initial_noninferiority":
            self.checkpoint_selector = ValidationCheckpointSelector(
                self.checkpoint_selection_config
            )
        elif self.checkpoint_selection_config.mode == "precision_constrained":
            self.checkpoint_selector = PrecisionConstrainedCheckpointSelector(
                self.checkpoint_selection_config
            )
        else:
            self.checkpoint_selector = None
        if (
            isinstance(self.checkpoint_selector, ValidationCheckpointSelector)
            and not self.training_config.evaluate_initial_policy
        ):
            raise ValueError(
                "initial_noninferiority checkpoint selection requires "
                "evaluate_initial_policy=true"
            )
        self.precision_constraint = PrecisionConstraintController(
            training_mapping.get("precision_constraint", {})
        )
        if self.precision_constraint.config.enabled:
            reward_mapping = dict(rl_config.get("reward", {}))
            static_weight = float(
                reward_mapping.get("cross_track_penalty_weight", 0.0)
            )
            if abs(static_weight) > 1e-12:
                raise ValueError(
                    "enabled precision constraint requires "
                    "cross_track_penalty_weight=0 to avoid double counting"
                )
            if self.checkpoint_selection_config.mode != "precision_constrained":
                raise ValueError(
                    "enabled precision constraint requires precision_constrained "
                    "checkpoint selection"
                )
            if not np.isclose(
                self.checkpoint_selection_config.maximum_mean_cross_track_rmse,
                self.precision_constraint.config.target_cross_track_rmse_m,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "training and checkpoint precision thresholds must match"
                )
        self.project_root = Path(project_root).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.pool = EnvironmentPool(
            training_configs, self.project_root, self.training_config.seed
        )
        self.curriculum = SceneCurriculum(
            dict(rl_config.get("training", {})).get("curriculum", {}),
            len(training_configs),
        )
        self.initial_state_curriculum = InitialStateCurriculum(
            dict(rl_config.get("training", {})).get(
                "initial_state_curriculum", {}
            )
        )
        self.validation_configs = list(validation_configs or training_configs[:1])
        probe_class = environment_class_from_config(training_configs[0])
        probe = probe_class(
            training_configs[0], self.project_root, self.training_config.seed
        )
        try:
            observation_dim = probe.observation_dim
            action_dim = probe.policy_action_dim
            self.encoder_config = probe.encoder.config
            self.parameterization_config = probe.parameterization.config
            self.intrinsic_exploration_config = (
                probe.intrinsic_exploration.config
            )
            self.action_spec = probe.action_spec
            self.action_mode = probe.action_mode
        finally:
            probe.close()
        sac_config = SACConfig.from_mapping(rl_config.get("sac", {}))
        requested_device = str(self.training_config.device)
        if requested_device == "cpu" or (
            requested_device == "auto"
            and not torch.cuda.is_available()
        ):
            thread_count = int(rl_config.get("torch_num_threads", 1))
            if thread_count <= 0:
                raise ValueError("rl.torch_num_threads must be positive")
            torch.set_num_threads(thread_count)
        self.agent = SACAgent(
            observation_dim,
            action_dim,
            sac_config,
            device=self.training_config.device,
            seed=self.training_config.seed,
        )
        if self.agent.is_correction_policy:
            if self.training_config.warmup_policy != "actor":
                raise ValueError(
                    "frozen BC correction mode requires warmup_policy=actor"
                )
            if self.training_config.normalizer_update != "frozen":
                raise ValueError(
                    "frozen BC correction mode requires normalizer_update=frozen"
                )
        self.normalizer = RunningNormalizer(observation_dim)
        self.bc_anchor = BehaviorCloningAnchor(
            dict(rl_config.get("training", {})).get("bc_anchor", {}),
            self.project_root,
            observation_dim,
            action_dim,
            self.training_config.seed,
        )
        if self.agent.is_correction_policy and self.bc_anchor.enabled:
            raise ValueError(
                "frozen BC correction mode cannot enable the full-actor BC anchor"
            )
        self.replay = ReplayBuffer(
            self.training_config.replay_capacity,
            observation_dim,
            action_dim,
            seed=self.training_config.seed,
        )
        self.global_step = 0
        self.episodes = 0
        self.best_validation_score = -float("inf")
        self.actor_initialization = None
        self.episode_records = []
        self.update_records = []
        self.validation_records = []
        self.rng = np.random.RandomState(self.training_config.seed)
        self._episode_in_progress = False
        self.interrupted_episodes = 0
        self.resume_provenance = None
        self._write_run_metadata()

    def _resume_contract(self):
        training = asdict(self.training_config)
        for name in _RESUME_OVERRIDE_FIELDS:
            training.pop(name, None)
        anchor = self.bc_anchor.metadata()
        anchor.pop("dataset_dir", None)
        return _json_compatible({
            "training": training,
            "checkpoint_selection": asdict(
                self.checkpoint_selection_config
            ),
            "precision_constraint": asdict(
                self.precision_constraint.config
            ),
            "sac": self.agent.config.to_dict(),
            "resolved_device": str(self.agent.device),
            "encoder": self.encoder_config.to_dict(),
            "action_mode": getattr(
                self, "action_mode", "mppi_prior"
            ),
            "parameterization": self.parameterization_config.to_dict(),
            "action_spec": {
                "names": tuple(self.action_spec.names),
                "lower": self.action_spec.lower,
                "upper": self.action_spec.upper,
            },
            "behavior_cloning_anchor": anchor,
            "curriculum": self.curriculum.metadata(),
            "initial_state_curriculum": self.initial_state_curriculum.metadata(),
            "training_scenes": [
                _resume_scene_contract(config) for config in self.pool.configs
            ],
            "validation_scenes": [
                _resume_scene_contract(config)
                for config in self.validation_configs
            ],
        })

    def _write_run_metadata(self):
        metadata = {
            "git_sha": git_sha(self.project_root),
            "training": asdict(self.training_config),
            "checkpoint_selection": {
                "config": asdict(self.checkpoint_selection_config),
                "state": (
                    None
                    if self.checkpoint_selector is None
                    else self.checkpoint_selector.state_dict()
                ),
            },
            "precision_constraint": self.precision_constraint.state_dict(),
            "observation_dim": self.agent.observation_dim,
            "policy_action_dim": self.agent.action_dim,
            "sac": self.agent.config.to_dict(),
            "policy_mode": self.agent.config.policy_mode,
            "action_mode": getattr(
                self, "action_mode", "mppi_prior"
            ),
            "encoder": self.encoder_config.to_dict(),
            "parameterization": self.parameterization_config.to_dict(),
            "intrinsic_exploration": self.intrinsic_exploration_config.to_dict(),
            "curriculum": self.curriculum.metadata(),
            "initial_state_curriculum": self.initial_state_curriculum.metadata(),
            "behavior_cloning_anchor": self.bc_anchor.metadata(),
            "replay_scene_groups": {
                str(index): config.get("scene", {}).get("name", "unknown")
                for index, config in enumerate(self.pool.configs)
            },
            "actor_initialization": self.actor_initialization,
            "resume_provenance": self.resume_provenance,
            "resume_semantics": "restart_interrupted_episode_v1",
        }
        with (self.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def _normalized_batch(self, batch):
        result = self.precision_constraint.shape_batch(batch)
        result["observations"] = self.normalizer.normalize(batch["observations"])
        result["next_observations"] = self.normalizer.normalize(batch["next_observations"])
        return result

    def _save_csv(self, path, records):
        if not records:
            return
        fieldnames = []
        seen = set()
        for record in records:
            for name in record:
                if name not in seen:
                    seen.add(name)
                    fieldnames.append(name)
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def _load_csv_until(path, global_step):
        source = Path(path)
        if not source.exists():
            return []
        with source.open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return [
            row for row in rows
            if int(float(row.get("global_step", 0))) <= int(global_step)
        ]

    def save(self, name="latest.pt"):
        path = self.checkpoint_dir / name
        resume_contract = self._resume_contract()
        replay_data_included = bool(self.training_config.save_replay_buffer)
        return save_sac_checkpoint(
            path,
            self.agent,
            self.normalizer,
            self.encoder_config,
            self.parameterization_config,
            self.action_spec,
            self.resolved_config,
            self.project_root,
            {
                "global_step": self.global_step,
                "episodes": self.episodes,
                "best_validation_score": self.best_validation_score,
                "checkpoint_selector": (
                    None
                    if self.checkpoint_selector is None
                    else self.checkpoint_selector.state_dict()
                ),
                "precision_constraint": self.precision_constraint.state_dict(),
                "rng_state": self.rng.get_state(),
                "pool_rng_state": self.pool.rng.get_state(),
                "bc_anchor_rng_state": self.bc_anchor.rng.get_state(),
                "bc_anchor_dataset_manifest_sha256": (
                    self.bc_anchor.manifest_fingerprint
                ),
                "actor_initialization": self.actor_initialization,
                "resume_contract": resume_contract,
                "resume_contract_sha256": _contract_digest(resume_contract),
                "resume_semantics": "restart_interrupted_episode_v1",
                "episode_in_progress": bool(self._episode_in_progress),
                "interrupted_episodes": int(self.interrupted_episodes),
                "replay_data_included": replay_data_included,
                "torch_cpu_rng_state": torch.get_rng_state().cpu(),
                "torch_cuda_rng_state_all": (
                    [state.cpu() for state in torch.cuda.get_rng_state_all()]
                    if self.agent.device.type == "cuda" else None
                ),
                "resume_provenance": self.resume_provenance,
                "action_mode": getattr(
                    self, "action_mode", "mppi_prior"
                ),
            },
            replay_buffer=self.replay,
            include_replay=replay_data_included,
        )

    def resume(self, checkpoint_path, allow_legacy=False):
        payload = load_sac_checkpoint(checkpoint_path, map_location=self.agent.device)
        training = payload["training_state"]
        saved_contract = training.get("resume_contract")
        current_contract = self._resume_contract()
        if saved_contract is None:
            if not allow_legacy:
                raise ValueError(
                    "legacy RL checkpoint has no resume contract; pass the explicit "
                    "legacy override only for a documented non-exact continuation"
                )
        elif _json_compatible(saved_contract) != current_contract:
            raise ValueError(
                "resume experiment contract does not match checkpoint "
                "(only total_steps and checkpoint_interval may change)"
            )
        if dict(payload["encoder_config"]) != self.encoder_config.to_dict():
            raise ValueError("resume encoder config does not match experiment")
        if dict(payload["parameterization_config"]) != (
            self.parameterization_config.to_dict()
        ):
            raise ValueError(
                "resume parameterization config does not match experiment"
            )
        saved_action = payload["action_spec"]
        if tuple(saved_action.get("names", ())) != tuple(self.action_spec.names):
            raise ValueError("resume action names do not match experiment")
        if not np.array_equal(saved_action["lower"], self.action_spec.lower) or not (
            np.array_equal(saved_action["upper"], self.action_spec.upper)
        ):
            raise ValueError("resume action bounds do not match experiment")
        source_agent_config = dict(payload["agent"].get("config", {}))
        target_agent_config = self.agent.config.to_dict()
        source_agent_config["hidden_sizes"] = tuple(
            source_agent_config.get("hidden_sizes", ())
        )
        target_agent_config["hidden_sizes"] = tuple(
            target_agent_config.get("hidden_sizes", ())
        )
        if source_agent_config != target_agent_config:
            raise ValueError("resume SAC config does not match experiment")
        self.agent.load_state_dict(payload["agent"], load_optimizers=True)
        self.normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
        if self.normalizer.dimension != self.agent.observation_dim:
            raise ValueError("resume normalizer dimension does not match actor")
        replay_state = payload.get("replay_buffer")
        replay_has_data = bool(
            replay_state is not None and "observations" in replay_state
        )
        saved_global_step = int(training.get("global_step", 0))
        if saved_global_step > 0 and not replay_has_data:
            raise ValueError(
                "RL resume requires checkpoint replay data; use actor-only "
                "initialization for an intentional warm start"
            )
        if replay_has_data:
            self.replay = ReplayBuffer.from_state_dict(
                replay_state, seed=self.training_config.seed
            )
        self.global_step = int(training.get("global_step", 0))
        self.episodes = int(training.get("episodes", 0))
        self.interrupted_episodes = int(training.get("interrupted_episodes", 0))
        self.best_validation_score = float(
            training.get("best_validation_score", -float("inf"))
        )
        saved_selector = training.get("checkpoint_selector")
        if self.checkpoint_selector is None:
            if saved_selector is not None:
                raise ValueError(
                    "resume checkpoint contains a paired selector but the "
                    "experiment requests scalar selection"
                )
        else:
            if saved_selector is None:
                raise ValueError(
                    "paired checkpoint selection cannot resume without its "
                    "step-zero reference state"
                )
            self.checkpoint_selector.load_state_dict(saved_selector)
        saved_constraint = training.get("precision_constraint")
        if saved_constraint is None:
            if self.precision_constraint.config.enabled:
                raise ValueError(
                    "constrained training cannot resume without dual state"
                )
        else:
            self.precision_constraint.load_state_dict(saved_constraint)
        if "rng_state" in training:
            self.rng.set_state(training["rng_state"])
        if "pool_rng_state" in training:
            self.pool.rng.set_state(training["pool_rng_state"])
        if "bc_anchor_rng_state" in training:
            self.bc_anchor.rng.set_state(training["bc_anchor_rng_state"])
        saved_anchor_fingerprint = training.get(
            "bc_anchor_dataset_manifest_sha256"
        )
        if saved_anchor_fingerprint != self.bc_anchor.manifest_fingerprint:
            raise ValueError("resume BC anchor dataset does not match checkpoint")
        self.actor_initialization = training.get("actor_initialization")
        cpu_rng_state = training.get("torch_cpu_rng_state")
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state.cpu())
        cuda_rng_states = training.get("torch_cuda_rng_state_all")
        if cuda_rng_states is not None:
            if not torch.cuda.is_available():
                raise ValueError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_rng_states])
        episode_was_interrupted = bool(
            training.get("episode_in_progress", self.global_step > 0)
        )
        if episode_was_interrupted:
            # MuJoCo/controller state is deliberately not serialized.  Resume
            # therefore starts a fresh episode and records that discontinuity
            # instead of pretending to be bit-exact mid-episode continuation.
            self.interrupted_episodes += 1
        self._episode_in_progress = False
        source_contract_digest = training.get("resume_contract_sha256")
        if source_contract_digest is None and saved_contract is not None:
            source_contract_digest = _contract_digest(saved_contract)
        self.resume_provenance = {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "source_git_sha": payload.get("git_sha"),
            "source_global_step": self.global_step,
            "source_resume_contract_sha256": source_contract_digest,
            "legacy_override": bool(saved_contract is None),
            "episode_restart_applied": episode_was_interrupted,
        }
        self.episode_records = self._load_csv_until(
            self.output_dir / "episodes.csv", self.global_step
        )
        self.update_records = self._load_csv_until(
            self.output_dir / "updates.csv", self.global_step
        )
        self.validation_records = self._load_csv_until(
            self.output_dir / "validation_episodes.csv", self.global_step
        )
        self._write_run_metadata()

    def initialize_actor_from(self, checkpoint_path):
        """Initialize only policy weights and observation statistics.

        This is intentionally different from :meth:`resume`: critics, target
        critics, entropy temperature, optimizers, replay, RNG state and all
        SAC counters remain fresh.  The source must be a compatible direct
        policy; it may be a BC checkpoint or a previous SAC actor warm start.
        """

        payload = load_sac_checkpoint(
            checkpoint_path, map_location=self.agent.device
        )
        mismatches = []
        source_agent = payload["agent"]
        if int(source_agent.get("observation_dim", -1)) != self.agent.observation_dim:
            mismatches.append("observation_dim")
        if int(source_agent.get("action_dim", -1)) != self.agent.action_dim:
            mismatches.append("action_dim")
        if dict(payload["encoder_config"]) != self.encoder_config.to_dict():
            mismatches.append("encoder_config")
        # Canonicalize older checkpoints through the current defaults.  This
        # accepts fields that were absent before the dynamic subgoal decoder
        # was introduced only when they resolve to exactly the current
        # semantics; any actual parameterization change still fails closed.
        source_parameterization = PriorParameterizationConfig.from_mapping(
            payload["parameterization_config"]
        ).to_dict()
        if source_parameterization != self.parameterization_config.to_dict():
            mismatches.append("parameterization_config")
        source_spec = payload["action_spec"]
        if tuple(source_spec.get("names", ())) != tuple(self.action_spec.names):
            mismatches.append("action_spec.names")
        for bound in ("lower", "upper"):
            value = np.asarray(source_spec.get(bound, ()), dtype=np.float64)
            expected = np.asarray(getattr(self.action_spec, bound), dtype=np.float64)
            if value.shape != expected.shape or not np.array_equal(value, expected):
                mismatches.append("action_spec.%s" % bound)
        source_config = dict(source_agent.get("config", {}))
        if str(source_config.get("policy_mode", "direct")) != "direct":
            mismatches.append("source_policy_mode")
        target_config = self.agent.config.to_dict()
        for name in ("hidden_sizes", "activation", "log_std_min", "log_std_max"):
            source_value = source_config.get(name)
            target_value = target_config.get(name)
            if name == "hidden_sizes":
                source_value = tuple(source_value or ())
                target_value = tuple(target_value or ())
            if source_value != target_value:
                mismatches.append("actor_config.%s" % name)
        normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
        if normalizer.dimension != self.agent.observation_dim:
            mismatches.append("normalizer.dimension")
        if normalizer.count <= 0:
            mismatches.append("normalizer.count")
        if self.bc_anchor.enabled:
            source_fingerprint = payload.get("training_state", {}).get(
                "dataset_manifest_sha256"
            )
            if source_fingerprint != self.bc_anchor.manifest_fingerprint:
                mismatches.append("bc_anchor.dataset_manifest_sha256")
        if mismatches:
            raise ValueError(
                "actor initialization checkpoint contract mismatch: %s"
                % ", ".join(sorted(set(mismatches)))
            )
        # A direct policy copies the BC actor.  Correction mode instead locks
        # that actor as an immutable base and keeps a separately initialized,
        # trainable residual policy.  Neither path imports critic, optimizer,
        # replay, entropy-temperature, RNG or counter state from the source.
        if self.agent.is_correction_policy:
            self.agent.initialize_frozen_base_actor(source_agent["actor"])
            initialization_mode = "frozen_bc_base_and_normalizer_only"
        else:
            self.agent.actor.load_state_dict(source_agent["actor"])
            initialization_mode = "actor_and_normalizer_only"
        self.normalizer = normalizer
        self.actor_initialization = {
            "mode": initialization_mode,
            "target_policy_mode": self.agent.config.policy_mode,
            "frozen_base_actor_sha256": (
                self.agent.frozen_base_actor_sha256()
                if self.agent.is_correction_policy else None
            ),
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "source_git_sha": payload.get("git_sha"),
            "source_phase": payload.get("training_state", {}).get("phase"),
            "source_bc_epoch": payload.get("training_state", {}).get("bc_epoch"),
        }
        self._write_run_metadata()

    def _update_normalizer(self, observations):
        if self.training_config.normalizer_update == "online":
            self.normalizer.update(observations)

    def _reset_training_episode(self):
        phase_name, weights = self.curriculum.at(self.global_step)
        observation, info = self.pool.reset(
            episode_seed=(
                self.training_config.seed
                + self.episodes
                + self.interrupted_episodes
            ),
            sampling_weights=weights,
            initial_state_curriculum=self.initial_state_curriculum,
            global_step=self.global_step,
        )
        info = dict(info)
        info["curriculum_phase"] = phase_name
        info["curriculum_weights"] = weights.tolist()
        return observation, info

    def _run_validation_episode(self, config, seed):
        validation_config = copy.deepcopy(config)
        # Intrinsic novelty is a training aid, never part of validation return
        # or checkpoint selection.  Task success/collision metrics therefore
        # remain comparable with runs that have exploration disabled.
        validation_config.setdefault("rl", {}).setdefault(
            "intrinsic_exploration", {}
        )["enabled"] = False
        if self.training_config.validation_initial_state_noise is not None:
            validation_config.setdefault("rl", {}).setdefault(
                "training", {}
            )["initial_state_noise"] = list(
                self.training_config.validation_initial_state_noise
            )
        environment_class = environment_class_from_config(validation_config)
        environment = environment_class(
            validation_config, self.project_root, seed=seed
        )
        observation, _ = environment.reset(seed=seed)
        total_reward = 0.0
        gate_alpha_sum = 0.0
        distance_gate_alpha_sum = 0.0
        correction_sums = {
            "correction_gate_alpha": 0.0,
            "base_action_abs_mean": 0.0,
            "unit_correction_abs_mean": 0.0,
            "applied_correction_abs_mean": 0.0,
            "raw_applied_correction_abs_mean": 0.0,
            "correction_advantage_gate_alpha": 0.0,
            "online_conservative_advantage": 0.0,
            "target_conservative_advantage": 0.0,
        }
        applied_correction_abs_max = 0.0
        online_advantage_min = float("inf")
        online_advantage_max = float("-inf")
        target_advantage_min = float("inf")
        target_advantage_max = float("-inf")
        online_positive_advantage_steps = 0
        target_positive_advantage_steps = 0
        covariance_scale_sum = np.zeros(
            environment.action_spec.dimension, dtype=np.float64
        )
        covariance_scale_square_sum = np.zeros_like(covariance_scale_sum)
        covariance_scale_min = np.full_like(covariance_scale_sum, np.inf)
        covariance_scale_max = np.full_like(covariance_scale_sum, -np.inf)
        cross_track_square_sum = 0.0
        steps = 0
        last_info = {}
        try:
            while True:
                normalized = self.normalizer.normalize(observation)
                action, policy_diagnostics = self.agent.select_action(
                    normalized, deterministic=True
                )
                if self.agent.is_correction_policy:
                    action, advantage_diagnostics = (
                        self.agent.filter_correction_by_advantage(
                            normalized,
                            action,
                            gate_mode="none",
                            critic_source="online",
                            threshold=0.0,
                        )
                    )
                    policy_diagnostics.update(advantage_diagnostics)
                for name in correction_sums:
                    correction_sums[name] += float(
                        policy_diagnostics.get(name, 0.0)
                    )
                online_advantage = float(policy_diagnostics.get(
                    "online_conservative_advantage", 0.0
                ))
                target_advantage = float(policy_diagnostics.get(
                    "target_conservative_advantage", 0.0
                ))
                online_advantage_min = min(
                    online_advantage_min, online_advantage
                )
                online_advantage_max = max(
                    online_advantage_max, online_advantage
                )
                target_advantage_min = min(
                    target_advantage_min, target_advantage
                )
                target_advantage_max = max(
                    target_advantage_max, target_advantage
                )
                online_positive_advantage_steps += int(online_advantage >= 0.0)
                target_positive_advantage_steps += int(target_advantage >= 0.0)
                applied_correction_abs_max = max(
                    applied_correction_abs_max,
                    float(policy_diagnostics.get(
                        "applied_correction_abs_max", 0.0
                    )),
                )
                observation, reward, terminated, truncated, last_info = environment.step(action)
                total_reward += reward
                prior = dict(last_info.get("prior", {}))
                gate_alpha_sum += float(prior.get("gate_alpha", 1.0))
                distance_gate_alpha_sum += float(
                    prior.get("distance_gate_alpha", 1.0)
                )
                covariance_scale = np.asarray(
                    prior.get(
                        "covariance_scale",
                        np.ones(environment.action_spec.dimension),
                    ),
                    dtype=np.float64,
                ).reshape(-1)
                if (
                    covariance_scale.shape
                    != (environment.action_spec.dimension,)
                    or not np.isfinite(covariance_scale).all()
                    or np.any(covariance_scale <= 0.0)
                ):
                    raise FloatingPointError(
                        "validation covariance scale is invalid"
                    )
                covariance_scale_sum += covariance_scale
                covariance_scale_square_sum += covariance_scale ** 2
                covariance_scale_min = np.minimum(
                    covariance_scale_min, covariance_scale
                )
                covariance_scale_max = np.maximum(
                    covariance_scale_max, covariance_scale
                )
                cross_track_error = float(
                    last_info.get("cross_track_error", 0.0)
                )
                if not np.isfinite(cross_track_error) or cross_track_error < 0.0:
                    raise FloatingPointError(
                        "validation cross-track error is invalid"
                    )
                cross_track_square_sum += cross_track_error ** 2
                steps += 1
                if terminated or truncated:
                    break
        finally:
            environment.close()
        covariance_mean = covariance_scale_sum / max(steps, 1)
        covariance_variance = np.maximum(
            covariance_scale_square_sum / max(steps, 1)
            - covariance_mean ** 2,
            0.0,
        )
        result = {
            "return": float(total_reward),
            "success": bool(last_info.get("success", False)),
            "collision": bool(last_info.get("collision", False)),
            "goal_distance": float(last_info.get("goal_distance", float("inf"))),
            "gate_alpha_mean": gate_alpha_sum / max(steps, 1),
            "distance_gate_alpha_mean": distance_gate_alpha_sum / max(steps, 1),
            "correction_gate_alpha_mean": (
                correction_sums["correction_gate_alpha"] / max(steps, 1)
            ),
            "base_action_abs_mean": (
                correction_sums["base_action_abs_mean"] / max(steps, 1)
            ),
            "unit_correction_abs_mean": (
                correction_sums["unit_correction_abs_mean"] / max(steps, 1)
            ),
            "applied_correction_abs_mean": (
                correction_sums["applied_correction_abs_mean"]
                / max(steps, 1)
            ),
            "applied_correction_abs_max": applied_correction_abs_max,
            "raw_applied_correction_abs_mean": (
                correction_sums["raw_applied_correction_abs_mean"]
                / max(steps, 1)
            ),
            "correction_advantage_gate_alpha_mean": (
                correction_sums["correction_advantage_gate_alpha"]
                / max(steps, 1)
            ),
            "online_conservative_advantage_mean": (
                correction_sums["online_conservative_advantage"]
                / max(steps, 1)
            ),
            "online_conservative_advantage_min": (
                online_advantage_min if steps else 0.0
            ),
            "online_conservative_advantage_max": (
                online_advantage_max if steps else 0.0
            ),
            "online_positive_advantage_fraction": (
                online_positive_advantage_steps / max(steps, 1)
            ),
            "target_conservative_advantage_mean": (
                correction_sums["target_conservative_advantage"]
                / max(steps, 1)
            ),
            "target_conservative_advantage_min": (
                target_advantage_min if steps else 0.0
            ),
            "target_conservative_advantage_max": (
                target_advantage_max if steps else 0.0
            ),
            "target_positive_advantage_fraction": (
                target_positive_advantage_steps / max(steps, 1)
            ),
            "cross_track_rmse": float(
                np.sqrt(cross_track_square_sum / max(steps, 1))
            ),
        }
        for index in range(environment.action_spec.dimension):
            result["covariance_scale_%d_mean" % index] = float(
                covariance_mean[index]
            )
            result["covariance_scale_%d_std" % index] = float(
                np.sqrt(covariance_variance[index])
            )
            result["covariance_scale_%d_min" % index] = float(
                covariance_scale_min[index] if steps else 1.0
            )
            result["covariance_scale_%d_max" % index] = float(
                covariance_scale_max[index] if steps else 1.0
            )
        return result

    def evaluate(self):
        rows = []
        for config_index, config in enumerate(self.validation_configs):
            for episode_index in range(self.training_config.evaluation_episodes):
                seed = _validation_episode_seed(
                    self.training_config, config_index, episode_index
                )
                result = self._run_validation_episode(config, seed)
                result.update({
                    "global_step": self.global_step,
                    "scene": config.get("scene", {}).get("name", "unknown"),
                    "seed": seed,
                })
                rows.append(result)
        success_rate = float(np.mean([float(row["success"]) for row in rows]))
        collision_rate = float(np.mean([float(row["collision"]) for row in rows]))
        mean_return = float(np.mean([row["return"] for row in rows]))
        mean_distance = float(np.mean([row["goal_distance"] for row in rows]))
        summary = {
            "global_step": self.global_step,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "mean_return": mean_return,
            "mean_goal_distance": mean_distance,
            "mean_cross_track_rmse": float(np.mean([
                row["cross_track_rmse"] for row in rows
            ])),
            "mean_applied_correction_abs": float(np.mean([
                row["applied_correction_abs_mean"] for row in rows
            ])),
            "max_applied_correction_abs": float(np.max([
                row["applied_correction_abs_max"] for row in rows
            ])),
            "mean_online_conservative_advantage": float(np.mean([
                row["online_conservative_advantage_mean"] for row in rows
            ])),
            "mean_target_conservative_advantage": float(np.mean([
                row["target_conservative_advantage_mean"] for row in rows
            ])),
            "mean_online_positive_advantage_fraction": float(np.mean([
                row["online_positive_advantage_fraction"] for row in rows
            ])),
            "mean_target_positive_advantage_fraction": float(np.mean([
                row["target_positive_advantage_fraction"] for row in rows
            ])),
        }
        for index in range(self.action_spec.dimension):
            for statistic in ("mean", "std", "min", "max"):
                field = "covariance_scale_%d_%s" % (index, statistic)
                summary["mean_" + field] = float(np.mean([
                    row[field] for row in rows
                ]))
        score = 1000.0 * success_rate - 1000.0 * collision_rate + mean_return
        if self.checkpoint_selector is None:
            selected = bool(score > self.best_validation_score)
            decision = {
                "selection_mode": "scalar",
                "selection_eligible": True,
                "selection_meaningful_improvement": selected,
                "selection_selected": selected,
                "selection_reason": (
                    "scalar_score_improved"
                    if selected else "scalar_score_not_improved"
                ),
                "selection_success_losses": 0,
                "selection_success_gains": 0,
                "selection_collision_regressions": 0,
                "selection_mean_goal_distance_improvement": 0.0,
                "selection_best_global_step": (
                    self.global_step if selected else None
                ),
            }
        else:
            decision = self.checkpoint_selector.consider(
                rows, summary, self.global_step
            )
            selected = bool(decision["selection_selected"])
        summary.update(decision)
        for row in rows:
            row.update(decision)
        self.validation_records.extend(rows)
        if selected:
            self.best_validation_score = score
            self.save("best.pt")
        return summary

    def run(self):
        target_steps = int(self.training_config.total_steps)
        if self.agent.is_correction_policy and not self.agent.base_actor_initialized:
            raise ValueError(
                "frozen BC correction training requires --initialize-actor-from "
                "or a correction-policy --resume checkpoint"
            )
        observation, reset_info = self._reset_training_episode()
        self._episode_in_progress = True
        if (
            self.training_config.normalizer_update == "frozen"
            and self.normalizer.count <= 0
        ):
            raise ValueError(
                "frozen RL normalizer requires a fitted initialization checkpoint"
            )
        if self.bc_anchor.enabled and (
            self.training_config.normalizer_update != "frozen"
            or self.actor_initialization is None
        ):
            raise ValueError(
                "BC anchor requires frozen normalizer and actor-only initialization"
            )
        initial_policy_evaluated = False
        if (
            self.global_step == 0
            and self.training_config.evaluate_initial_policy
            and self.normalizer.count > 0
        ):
            # A warm-start checkpoint already owns a fitted observation
            # distribution.  Evaluate and save that exact policy before the
            # first training reset is allowed to add a seed-specific sample;
            # otherwise nominally common step-zero references differ across
            # independent training seeds.
            if not (self.checkpoint_dir / "initial.pt").exists():
                self.save("initial.pt")
            summary = self.evaluate()
            print(json.dumps({"validation": summary}, sort_keys=True))
            initial_policy_evaluated = True
        self._update_normalizer(observation)
        if self.global_step == 0 and not (self.checkpoint_dir / "initial.pt").exists():
            # Preserve the untrained policy and its first valid observation
            # statistics so every learning run has a reproducible step-zero
            # control comparison rather than only loss curves.
            self.save("initial.pt")
        if (
            self.global_step == 0
            and self.training_config.evaluate_initial_policy
            and not initial_policy_evaluated
        ):
            summary = self.evaluate()
            print(json.dumps({"validation": summary}, sort_keys=True))
        episode_return = 0.0
        episode_length = 0
        episode_gate_alpha = 0.0
        episode_distance_gate_alpha = 0.0
        episode_intrinsic_return = 0.0
        episode_cross_track_square_sum = 0.0
        episode_covariance_scale_sum = np.zeros(
            self.action_spec.dimension, dtype=np.float64
        )
        episode_covariance_scale_square_sum = np.zeros_like(
            episode_covariance_scale_sum
        )
        episode_replay_handles = []
        last_update = {}
        while self.global_step < target_steps:
            if (
                self.global_step < self.training_config.warmup_steps
                and self.training_config.warmup_policy == "random"
            ):
                action = self.rng.uniform(-1.0, 1.0, size=self.agent.action_dim).astype(np.float32)
            else:
                action, _ = self.agent.select_action(
                    self.normalizer.normalize(observation), deterministic=False
                )
            next_observation, reward, terminated, truncated, info = self.pool.current.step(action)
            prior_diagnostics = dict(info.get("prior", {}))
            reward_terms = dict(info.get("reward_terms", {}))
            episode_intrinsic_return += float(
                reward_terms.get("intrinsic_exploration", 0.0)
            )
            episode_gate_alpha += float(prior_diagnostics.get("gate_alpha", 1.0))
            episode_distance_gate_alpha += float(
                prior_diagnostics.get("distance_gate_alpha", 1.0)
            )
            covariance_scale = np.asarray(
                prior_diagnostics.get(
                    "covariance_scale", np.ones(self.action_spec.dimension)
                ),
                dtype=np.float64,
            ).reshape(-1)
            if (
                covariance_scale.shape != (self.action_spec.dimension,)
                or not np.isfinite(covariance_scale).all()
                or np.any(covariance_scale <= 0.0)
            ):
                raise FloatingPointError("training covariance scale is invalid")
            episode_covariance_scale_sum += covariance_scale
            episode_covariance_scale_square_sum += covariance_scale ** 2
            cross_track_error = float(info.get("cross_track_error", 0.0))
            if not np.isfinite(cross_track_error) or cross_track_error < 0.0:
                raise FloatingPointError("training cross-track error is invalid")
            episode_cross_track_square_sum += cross_track_error ** 2
            self._update_normalizer(next_observation)
            # Time-limit truncation is bootstrappable; true terminal states are not.
            replay_handle = self.replay.add(
                observation,
                action,
                reward,
                next_observation,
                terminated,
                group=self.pool.current_index,
                constraint_cost=cross_track_error ** 2,
            )
            episode_replay_handles.append(replay_handle)
            observation = next_observation
            episode_return += reward
            episode_length += 1
            self.global_step += 1
            if (
                self.global_step >= self.training_config.update_after
                and self.replay.size >= self.training_config.batch_size
                and (
                    not self.training_config.replay_require_all_scenes
                    or len(self.replay.group_counts()) == len(self.pool.configs)
                )
            ):
                for _ in range(self.training_config.updates_per_step):
                    raw_batch = self.replay.sample(
                        self.training_config.batch_size,
                        strategy=self.training_config.replay_sampling,
                        success_fraction=(
                            self.training_config.replay_success_fraction
                        ),
                    )
                    batch = self._normalized_batch(raw_batch)
                    update_actor = bool(
                        self.global_step >= self.training_config.actor_update_after
                    )
                    behavior_batch = (
                        self.bc_anchor.sample(self.normalizer)
                        if update_actor else None
                    )
                    last_update = self.agent.update(
                        batch,
                        update_actor=update_actor,
                        behavior_batch=behavior_batch,
                        behavior_cloning_weight=(
                            self.bc_anchor.config.mean_weight
                            if self.bc_anchor.enabled and update_actor else 0.0
                        ),
                        behavior_log_std_weight=(
                            self.bc_anchor.config.log_std_weight
                            if self.bc_anchor.enabled and update_actor else 0.0
                        ),
                        behavior_target_log_std=(
                            self.bc_anchor.config.target_log_std
                        ),
                    )
                record = {"global_step": self.global_step}
                record.update(last_update)
                record["precision_constraint_multiplier"] = float(
                    self.precision_constraint.multiplier
                )
                record["batch_constraint_cost_mean"] = float(np.mean(
                    raw_batch["constraint_costs"]
                ))
                record["batch_task_reward_mean"] = float(np.mean(
                    raw_batch["rewards"]
                ))
                record["batch_constrained_reward_mean"] = float(np.mean(
                    batch["rewards"]
                ))
                for group_index in range(len(self.pool.configs)):
                    record["batch_scene_fraction_%d" % group_index] = float(
                        np.mean(raw_batch["groups"] == group_index)
                    )
                record["batch_success_fraction"] = float(
                    np.mean(raw_batch["outcomes"] == 1)
                )
                record["batch_labeled_fraction"] = float(
                    np.mean(raw_batch["outcomes"] >= 0)
                )
                self.update_records.append(record)
            if terminated or truncated:
                episode_mean_square = (
                    episode_cross_track_square_sum / max(episode_length, 1)
                )
                constraint_diagnostics = self.precision_constraint.update(
                    episode_mean_square
                )
                labeled_transitions = self.replay.mark_episode_outcome(
                    episode_replay_handles,
                    success=bool(info.get("success", False)),
                )
                record = {
                    "episode": self.episodes,
                    "global_step": self.global_step,
                    "scene": info.get("scene", reset_info.get("scene", "unknown")),
                    "length": episode_length,
                    "return": float(episode_return),
                    "success": bool(info.get("success", False)),
                    "collision": bool(info.get("collision", False)),
                    "goal_distance": float(info.get("goal_distance", float("inf"))),
                    "replay_size": self.replay.size,
                    "curriculum_phase": reset_info.get(
                        "curriculum_phase", "uniform"
                    ),
                    "initial_state_phase": reset_info.get(
                        "initial_state_phase", "disabled"
                    ),
                    "initial_state_name": reset_info.get(
                        "initial_state_name", "configured_initial"
                    ),
                    "rl_gate_alpha_mean": episode_gate_alpha
                    / max(episode_length, 1),
                    "distance_gate_alpha_mean": episode_distance_gate_alpha
                    / max(episode_length, 1),
                    "intrinsic_exploration_return": float(
                        episode_intrinsic_return
                    ),
                    "replay_labeled_transitions": int(labeled_transitions),
                    "cross_track_rmse": float(np.sqrt(
                        episode_mean_square
                    )),
                }
                record.update(constraint_diagnostics)
                covariance_mean = (
                    episode_covariance_scale_sum / max(episode_length, 1)
                )
                covariance_std = np.sqrt(np.maximum(
                    episode_covariance_scale_square_sum
                    / max(episode_length, 1)
                    - covariance_mean ** 2,
                    0.0,
                ))
                for index in range(self.action_spec.dimension):
                    record["covariance_scale_%d_mean" % index] = float(
                        covariance_mean[index]
                    )
                    record["covariance_scale_%d_std" % index] = float(
                        covariance_std[index]
                    )
                self.episode_records.append(record)
                self.episodes += 1
                self._episode_in_progress = False
                observation, reset_info = self._reset_training_episode()
                self._episode_in_progress = True
                self._update_normalizer(observation)
                episode_return = 0.0
                episode_length = 0
                episode_gate_alpha = 0.0
                episode_distance_gate_alpha = 0.0
                episode_intrinsic_return = 0.0
                episode_cross_track_square_sum = 0.0
                episode_covariance_scale_sum.fill(0.0)
                episode_covariance_scale_square_sum.fill(0.0)
                episode_replay_handles = []
            if (
                self.training_config.evaluation_interval > 0
                and self.global_step % self.training_config.evaluation_interval == 0
            ):
                summary = self.evaluate()
                print(json.dumps({"validation": summary}, sort_keys=True))
            if (
                self.training_config.checkpoint_interval > 0
                and self.global_step % self.training_config.checkpoint_interval == 0
            ):
                self.save("step_%09d.pt" % self.global_step)
                self.save("latest.pt")
                self._flush_logs()
        self.save("latest.pt")
        if self.training_config.evaluation_interval == 0 or (
            self.global_step % self.training_config.evaluation_interval != 0
        ):
            self.evaluate()
        self._flush_logs()
        self.pool.close()
        return {
            "global_step": self.global_step,
            "episodes": self.episodes,
            "interrupted_episodes": self.interrupted_episodes,
            "best_validation_score": self.best_validation_score,
            "checkpoint_selection": (
                None
                if self.checkpoint_selector is None
                else self.checkpoint_selector.state_dict()
            ),
            "precision_constraint": self.precision_constraint.state_dict(),
            "latest_checkpoint": str(self.checkpoint_dir / "latest.pt"),
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
        }

    def _flush_logs(self):
        self._save_csv(self.output_dir / "episodes.csv", self.episode_records)
        self._save_csv(self.output_dir / "updates.csv", self.update_records)
        self._save_csv(self.output_dir / "validation_episodes.csv", self.validation_records)
        summary = {
            "global_step": self.global_step,
            "episodes": self.episodes,
            "interrupted_episodes": self.interrupted_episodes,
            "best_validation_score": self.best_validation_score,
            "checkpoint_selection": (
                None
                if self.checkpoint_selector is None
                else self.checkpoint_selector.state_dict()
            ),
            "precision_constraint": self.precision_constraint.state_dict(),
            "episode_records": len(self.episode_records),
            "update_records": len(self.update_records),
            "validation_records": len(self.validation_records),
            "replay_sampling": self.training_config.replay_sampling,
            "replay_group_counts": self.replay.group_counts(),
            "replay_outcome_counts": self.replay.outcome_counts(),
        }
        with (self.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
