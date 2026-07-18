"""Offline behavior cloning for a privileged local-subgoal teacher.

The teacher may use an offline collision-free route while producing the same
two normalized parameters accepted by the deployable SAC actor.  Only encoded
LaserScan/odometry/goal observations and those two targets enter this module;
route geometry and simulator truth stay in the collector's separate audit
artifacts.
"""

import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from mobile_robot_mppi.core.config import git_sha
from .checkpointing import load_sac_checkpoint, save_sac_checkpoint
from .demonstrations import (
    demonstration_manifest_fingerprint,
    load_demonstration_manifest,
    load_demonstration_splits,
)
from .environment import MppiPriorEnv
from .observation import RunningNormalizer
from .sac import SACAgent, SACConfig


@dataclass(frozen=True)
class BehaviorCloningConfig:
    epochs: int = 100
    batch_size: int = 256
    early_stopping_patience: int = 15
    minimum_delta: float = 1e-6
    log_std_weight: float = 1e-3
    target_log_std: float = -2.0
    seed: int = 0
    device: str = "auto"

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(**{
            name: values.get(name, field.default)
            for name, field in cls.__dataclass_fields__.items()
        })

    def validate(self):
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("BC epochs and batch_size must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("BC early_stopping_patience cannot be negative")
        numeric = np.asarray(
            (self.minimum_delta, self.log_std_weight, self.target_log_std),
            dtype=np.float64,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("BC loss settings must be finite")
        if self.minimum_delta < 0.0 or self.log_std_weight < 0.0:
            raise ValueError("BC minimum_delta/log_std_weight cannot be negative")
        if not 0 <= int(self.seed) <= 2 ** 32 - 1:
            raise ValueError("BC seed must be in [0, 2**32 - 1]")
        if self.device != "auto" and self.device != "cpu" and not str(
            self.device
        ).startswith("cuda"):
            raise ValueError("BC device must be auto, cpu or cuda")


def evaluate_behavior_cloning(agent, normalizer, arrays):
    observations = normalizer.normalize(arrays["observations"])
    targets = np.asarray(arrays["teacher_actions"], dtype=np.float32)
    if observations.shape[0] == 0:
        raise ValueError("BC evaluation split cannot be empty")
    with torch.no_grad():
        tensor = torch.as_tensor(
            observations, dtype=torch.float32, device=agent.device
        )
        predicted = agent.actor.mean_action(tensor)
        _, log_std = agent.actor.distribution(tensor)
    prediction = predicted.cpu().numpy().astype(np.float64)
    target = targets.astype(np.float64)
    error = prediction - target
    per_dimension_rmse = np.sqrt(np.mean(np.square(error), axis=0))
    metrics = {
        "samples": int(target.shape[0]),
        "mse": float(np.mean(np.square(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "distance_action_rmse": float(per_dimension_rmse[0]),
        "bearing_action_rmse": float(per_dimension_rmse[1]),
        "prediction_saturation_rate": float(
            np.mean(np.abs(prediction) >= 0.99)
        ),
        "teacher_saturation_rate": float(np.mean(np.abs(target) >= 0.99)),
        "policy_log_std_mean": float(log_std.mean().cpu()),
    }
    if not np.isfinite(tuple(metrics.values())).all():
        raise FloatingPointError("BC evaluation produced NaN or Inf")
    return metrics


class BehaviorCloningTrainer:
    """Reproducible, episode-split actor pretraining with exact resume."""

    def __init__(
        self,
        environment_config,
        resolved_config,
        dataset_dir,
        project_root,
        output_dir,
    ):
        self.environment_config = dict(environment_config)
        self.resolved_config = dict(resolved_config)
        self.project_root = Path(project_root).resolve()
        self.dataset_dir = Path(dataset_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        values = self.resolved_config.get("rl", {}).get(
            "behavior_cloning", {}
        )
        self.config = BehaviorCloningConfig.from_mapping(values)
        self.config.validate()
        self.manifest = load_demonstration_manifest(self.dataset_dir)
        self.manifest_fingerprint = demonstration_manifest_fingerprint(
            self.manifest
        )
        demonstration_splits = load_demonstration_splits(self.dataset_dir)
        self.train_data = demonstration_splits["train"]
        self.validation_data = demonstration_splits["validation"]
        self.test_data = demonstration_splits["test"]

        probe = MppiPriorEnv(
            self.environment_config, self.project_root, self.config.seed
        )
        try:
            observation_dim = probe.observation_dim
            action_dim = probe.policy_action_dim
            self.encoder_config = probe.encoder.config
            self.parameterization_config = probe.parameterization.config
            self.action_spec = probe.action_spec
        finally:
            probe.close()
        if self.parameterization_config.kind != "local_subgoal":
            raise ValueError("BC v1 requires the local_subgoal prior")
        if self.parameterization_config.learn_covariance:
            raise ValueError("BC v1 does not supervise learned covariance")
        if action_dim != 2:
            raise ValueError("BC v1 teacher requires exactly two subgoal actions")
        if self.encoder_config.include_absolute_pose:
            raise ValueError("BC student encoder cannot include absolute pose")
        if dict(self.manifest["observation_encoder"]) != (
            self.encoder_config.to_dict()
        ):
            raise ValueError(
                "demonstration observation_encoder does not match current environment"
            )
        if dict(self.manifest["prior_parameterization"]) != (
            self.parameterization_config.to_dict()
        ):
            raise ValueError(
                "demonstration prior_parameterization does not match current environment"
            )
        for split_name, arrays in (
            ("train", self.train_data),
            ("validation", self.validation_data),
            ("test", self.test_data),
        ):
            if arrays["observations"].shape[1] != observation_dim:
                raise ValueError(
                    "%s demonstration observation dimension does not match config"
                    % split_name
                )
            if arrays["teacher_actions"].shape[1] != action_dim:
                raise ValueError(
                    "%s teacher action dimension does not match config" % split_name
                )
        requested_device = str(self.config.device)
        if requested_device == "cpu" or (
            requested_device == "auto" and not torch.cuda.is_available()
        ):
            thread_count = int(
                self.resolved_config.get("rl", {}).get("torch_num_threads", 1)
            )
            if thread_count <= 0:
                raise ValueError("rl.torch_num_threads must be positive")
            torch.set_num_threads(thread_count)
        sac_config = SACConfig.from_mapping(
            self.resolved_config.get("rl", {}).get("sac", {})
        )
        self.agent = SACAgent(
            observation_dim,
            action_dim,
            sac_config,
            device=self.config.device,
            seed=self.config.seed,
        )
        # This is the only normalizer update in the BC pipeline.  Validation
        # and test episodes are never permitted to influence these statistics.
        self.normalizer = RunningNormalizer(observation_dim)
        self.normalizer.update(self.train_data["observations"])
        self.rng = np.random.RandomState(self.config.seed)
        self.epoch = 0
        self.best_validation_mse = float("inf")
        self.epochs_without_improvement = 0
        self.best_actor_state = None
        self.records = []
        self.resume_provenance = None
        self._write_metadata()

    def _resume_contract(self):
        behavior_cloning = asdict(self.config)
        # Extending the requested epoch budget is the only permitted
        # continuation override.  Changing CPU/CUDA changes the stochastic
        # execution contract and is therefore an explicit new run.
        behavior_cloning.pop("epochs", None)
        return {
            "behavior_cloning": behavior_cloning,
            "sac": self.agent.config.to_dict(),
            "resolved_device": str(self.agent.device),
            "encoder": self.encoder_config.to_dict(),
            "parameterization": self.parameterization_config.to_dict(),
            "action_spec": {
                "names": tuple(self.action_spec.names),
                "lower": self.action_spec.lower.tolist(),
                "upper": self.action_spec.upper.tolist(),
            },
        }

    def _write_metadata(self):
        metadata = {
            "phase": "behavior_cloning",
            "git_sha": git_sha(self.project_root),
            "dataset_dir": str(self.dataset_dir),
            "dataset_manifest_sha256": self.manifest_fingerprint,
            "behavior_cloning": asdict(self.config),
            "observation_dim": self.agent.observation_dim,
            "policy_action_dim": self.agent.action_dim,
            "normalizer_source": "train_split_only",
            "teacher_privilege": (
                "offline route is collector-only and absent from model input"
            ),
            "resume_provenance": self.resume_provenance,
        }
        with (self.output_dir / "run_metadata.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)

    def _training_state(self):
        return {
            "phase": "behavior_cloning",
            "global_step": 0,
            "episodes": 0,
            "bc_epoch": self.epoch,
            "bc_update_steps": int(getattr(self.agent, "bc_update_steps", 0)),
            "best_bc_validation_mse": self.best_validation_mse,
            "epochs_without_improvement": self.epochs_without_improvement,
            "rng_state": self.rng.get_state(),
            "dataset_dir": str(self.dataset_dir),
            "dataset_manifest_sha256": self.manifest_fingerprint,
            "resume_contract": self._resume_contract(),
            "torch_cpu_rng_state": torch.get_rng_state().cpu(),
            "torch_cuda_rng_state_all": (
                [state.cpu() for state in torch.cuda.get_rng_state_all()]
                if self.agent.device.type == "cuda" else None
            ),
            "best_actor_state": self.best_actor_state,
            "resume_provenance": self.resume_provenance,
        }

    def save(self, name):
        return save_sac_checkpoint(
            self.checkpoint_dir / name,
            self.agent,
            self.normalizer,
            self.encoder_config,
            self.parameterization_config,
            self.action_spec,
            self.resolved_config,
            self.project_root,
            self._training_state(),
            replay_buffer=None,
            include_replay=False,
        )

    def resume(self, checkpoint_path, allow_legacy=False):
        payload = load_sac_checkpoint(
            checkpoint_path, map_location=self.agent.device
        )
        state = payload["training_state"]
        if state.get("phase") != "behavior_cloning":
            raise ValueError("--resume-bc requires a behavior-cloning checkpoint")
        saved_contract = state.get("resume_contract")
        if saved_contract is None:
            if not allow_legacy:
                raise ValueError(
                    "legacy BC checkpoint has no resume contract; use the explicit "
                    "legacy override only for a documented continuation"
                )
        elif saved_contract != self._resume_contract():
            raise ValueError(
                "BC resume experiment contract does not match checkpoint "
                "(only epochs may change)"
            )
        if state.get("dataset_manifest_sha256") != self.manifest_fingerprint:
            raise ValueError("BC resume dataset manifest does not match checkpoint")
        if dict(payload["encoder_config"]) != self.encoder_config.to_dict():
            raise ValueError("BC resume encoder config does not match")
        if dict(payload["parameterization_config"]) != (
            self.parameterization_config.to_dict()
        ):
            raise ValueError("BC resume parameterization config does not match")
        source_normalizer = RunningNormalizer.from_state_dict(
            payload["normalizer"]
        )
        source_normalizer_state = source_normalizer.state_dict()
        expected_normalizer_state = self.normalizer.state_dict()
        scalar_fields = ("dimension", "min_std", "clip", "count")
        if any(
            source_normalizer_state[name] != expected_normalizer_state[name]
            for name in scalar_fields
        ) or any(
            not np.array_equal(
                source_normalizer_state[name], expected_normalizer_state[name]
            )
            for name in ("mean", "m2")
        ):
            raise ValueError(
                "BC resume train-only normalizer state does not match dataset"
            )
        self.agent.load_state_dict(payload["agent"], load_optimizers=True)
        self.normalizer = source_normalizer
        self.epoch = int(state.get("bc_epoch", 0))
        self.best_validation_mse = float(
            state.get("best_bc_validation_mse", float("inf"))
        )
        self.epochs_without_improvement = int(
            state.get("epochs_without_improvement", 0)
        )
        if "rng_state" in state:
            self.rng.set_state(state["rng_state"])
        cpu_rng_state = state.get("torch_cpu_rng_state")
        if cpu_rng_state is not None:
            torch.set_rng_state(cpu_rng_state.cpu())
        cuda_rng_states = state.get("torch_cuda_rng_state_all")
        if cuda_rng_states is not None:
            if not torch.cuda.is_available():
                raise ValueError(
                    "BC checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all([item.cpu() for item in cuda_rng_states])
        self.best_actor_state = state.get("best_actor_state")
        if self.best_actor_state is None:
            local_best = self.checkpoint_dir / "best.pt"
            if local_best.exists():
                best_payload = load_sac_checkpoint(
                    local_best, map_location=self.agent.device
                )
                self.best_actor_state = copy.deepcopy(
                    best_payload["agent"]["actor"]
                )
            elif Path(checkpoint_path).name == "best.pt":
                self.best_actor_state = copy.deepcopy(payload["agent"]["actor"])
            elif not allow_legacy:
                raise ValueError(
                    "BC resume checkpoint is missing the best actor snapshot"
                )
        self.resume_provenance = {
            "checkpoint": str(Path(checkpoint_path).resolve()),
            "source_git_sha": payload.get("git_sha"),
            "source_epoch": self.epoch,
            "legacy_override": bool(saved_contract is None),
        }
        log_path = self.output_dir / "training.csv"
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", newline="") as handle:
                self.records = [
                    row for row in csv.DictReader(handle)
                    if int(float(row.get("epoch", 0))) <= self.epoch
                ]
        self._write_metadata()

    def _flush_records(self):
        if not self.records:
            return
        with (self.output_dir / "training.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(self.records[0].keys())
            )
            writer.writeheader()
            writer.writerows(self.records)

    def _train_epoch(self):
        count = int(self.train_data["observations"].shape[0])
        order = self.rng.permutation(count)
        metrics = []
        for start in range(0, count, self.config.batch_size):
            indices = order[start:start + self.config.batch_size]
            observations = self.normalizer.normalize(
                self.train_data["observations"][indices]
            )
            targets = self.train_data["teacher_actions"][indices]
            metrics.append(self.agent.behavior_cloning_update(
                observations,
                targets,
                log_std_weight=self.config.log_std_weight,
                target_log_std=self.config.target_log_std,
            ))
        return {
            name: float(np.mean([item[name] for item in metrics]))
            for name in metrics[0]
        }

    def run(self):
        early_stopped = bool(
            self.config.early_stopping_patience > 0
            and self.epochs_without_improvement
            >= self.config.early_stopping_patience
        )
        while self.epoch < self.config.epochs and not early_stopped:
            train_metrics = self._train_epoch()
            self.epoch += 1
            validation = evaluate_behavior_cloning(
                self.agent, self.normalizer, self.validation_data
            )
            improved = (
                validation["mse"]
                < self.best_validation_mse - self.config.minimum_delta
            )
            if improved:
                self.best_validation_mse = validation["mse"]
                self.epochs_without_improvement = 0
                self.best_actor_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.agent.actor.state_dict().items()
                }
            else:
                self.epochs_without_improvement += 1
            row = {"epoch": self.epoch}
            row.update({"train_%s" % key: value for key, value in train_metrics.items()})
            row.update({"validation_%s" % key: value for key, value in validation.items()})
            row["best_validation_mse"] = self.best_validation_mse
            self.records.append(row)
            if improved:
                self.save("best.pt")
            self.save("latest.pt")
            self._flush_records()
            print(json.dumps({"behavior_cloning": row}, sort_keys=True))
            if (
                self.config.early_stopping_patience > 0
                and self.epochs_without_improvement
                >= self.config.early_stopping_patience
            ):
                early_stopped = True
                break
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            self.save("latest.pt")
        best_path = self.checkpoint_dir / "best.pt"
        if self.best_actor_state is None:
            if not best_path.exists():
                raise ValueError(
                    "BC training has no best actor snapshot for final evaluation"
                )
            best_payload = load_sac_checkpoint(
                best_path, map_location=self.agent.device
            )
            self.best_actor_state = copy.deepcopy(
                best_payload["agent"]["actor"]
            )
        self.agent.actor.load_state_dict(self.best_actor_state)
        if not best_path.exists():
            # A resumed run may intentionally use a new output directory and
            # produce no new improvement.  Materialize a self-contained best
            # inference checkpoint from the snapshot carried by latest.pt.
            self.save("best.pt")
        test_metrics = evaluate_behavior_cloning(
            self.agent, self.normalizer, self.test_data
        )
        summary = {
            "phase": "behavior_cloning",
            "epochs_completed": self.epoch,
            "best_validation_mse": self.best_validation_mse,
            "test": test_metrics,
            "best_checkpoint": str(self.checkpoint_dir / "best.pt"),
            "latest_checkpoint": str(self.checkpoint_dir / "latest.pt"),
            "dataset_manifest_sha256": self.manifest_fingerprint,
            "normalizer_count": self.normalizer.count,
        }
        with (self.output_dir / "summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
        return summary
