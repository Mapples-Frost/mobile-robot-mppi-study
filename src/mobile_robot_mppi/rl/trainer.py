"""Reproducible multi-scene SAC training and validation loop."""

import copy
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from mobile_robot_mppi.core.config import git_sha
from .checkpointing import load_sac_checkpoint, save_sac_checkpoint
from .environment import MppiPriorEnv
from .observation import RunningNormalizer
from .replay import ReplayBuffer
from .sac import SACAgent, SACConfig


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int = 200000
    warmup_steps: int = 5000
    warmup_policy: str = "random"
    update_after: int = 1000
    batch_size: int = 256
    replay_capacity: int = 500000
    replay_sampling: str = "uniform"
    replay_success_fraction: float = 0.25
    replay_require_all_scenes: bool = False
    updates_per_step: int = 1
    evaluation_interval: int = 10000
    evaluation_episodes: int = 3
    checkpoint_interval: int = 10000
    save_replay_buffer: bool = True
    seed: int = 0
    validation_seed_base: Optional[int] = None
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
        if self.warmup_steps < 0 or self.update_after < 0:
            raise ValueError("RL warmup/update_after cannot be negative")
        if self.warmup_policy not in ("random", "actor"):
            raise ValueError("RL warmup_policy must be random or actor")
        if self.evaluation_interval < 0 or self.checkpoint_interval < 0:
            raise ValueError("RL evaluation/checkpoint intervals cannot be negative")
        if self.replay_sampling not in (
            "uniform", "scene_balanced", "outcome_balanced"
        ):
            raise ValueError(
                "RL replay_sampling must be uniform, scene_balanced, or "
                "outcome_balanced"
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
        if self.normalizer_update not in ("online", "frozen"):
            raise ValueError("RL normalizer_update must be online or frozen")


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
            self.environments[index] = MppiPriorEnv(
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
        probe = MppiPriorEnv(training_configs[0], self.project_root, self.training_config.seed)
        try:
            observation_dim = probe.observation_dim
            action_dim = probe.policy_action_dim
            self.encoder_config = probe.encoder.config
            self.parameterization_config = probe.parameterization.config
            self.intrinsic_exploration_config = (
                probe.intrinsic_exploration.config
            )
            self.action_spec = probe.action_spec
        finally:
            probe.close()
        sac_config = SACConfig.from_mapping(rl_config.get("sac", {}))
        requested_device = str(self.training_config.device)
        if requested_device == "cpu" or (
            requested_device == "auto"
            and not __import__("torch").cuda.is_available()
        ):
            import torch
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
        self.normalizer = RunningNormalizer(observation_dim)
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
        self._write_run_metadata()

    def _write_run_metadata(self):
        metadata = {
            "git_sha": git_sha(self.project_root),
            "training": asdict(self.training_config),
            "observation_dim": self.agent.observation_dim,
            "policy_action_dim": self.agent.action_dim,
            "encoder": self.encoder_config.to_dict(),
            "parameterization": self.parameterization_config.to_dict(),
            "intrinsic_exploration": self.intrinsic_exploration_config.to_dict(),
            "curriculum": self.curriculum.metadata(),
            "initial_state_curriculum": self.initial_state_curriculum.metadata(),
            "replay_scene_groups": {
                str(index): config.get("scene", {}).get("name", "unknown")
                for index, config in enumerate(self.pool.configs)
            },
            "actor_initialization": self.actor_initialization,
        }
        with (self.output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

    def _normalized_batch(self, batch):
        result = dict(batch)
        result["observations"] = self.normalizer.normalize(batch["observations"])
        result["next_observations"] = self.normalizer.normalize(batch["next_observations"])
        return result

    def _save_csv(self, path, records):
        if not records:
            return
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    def save(self, name="latest.pt"):
        path = self.checkpoint_dir / name
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
                "rng_state": self.rng.get_state(),
                "pool_rng_state": self.pool.rng.get_state(),
            },
            replay_buffer=self.replay,
            include_replay=self.training_config.save_replay_buffer,
        )

    def resume(self, checkpoint_path):
        payload = load_sac_checkpoint(checkpoint_path, map_location=self.agent.device)
        self.agent.load_state_dict(payload["agent"], load_optimizers=True)
        self.normalizer = RunningNormalizer.from_state_dict(payload["normalizer"])
        replay_state = payload.get("replay_buffer")
        if replay_state is not None and "observations" in replay_state:
            self.replay = ReplayBuffer.from_state_dict(
                replay_state, seed=self.training_config.seed
            )
        training = payload["training_state"]
        self.global_step = int(training.get("global_step", 0))
        self.episodes = int(training.get("episodes", 0))
        self.best_validation_score = float(
            training.get("best_validation_score", -float("inf"))
        )
        if "rng_state" in training:
            self.rng.set_state(training["rng_state"])
        if "pool_rng_state" in training:
            self.pool.rng.set_state(training["pool_rng_state"])

    def initialize_actor_from(self, checkpoint_path):
        """Initialize only the policy and observation statistics from BC.

        This is intentionally different from :meth:`resume`: critics, target
        critics, entropy temperature, optimizers, replay, RNG state and all
        SAC counters remain fresh.  Consequently a BC checkpoint cannot leak
        training state into the controlled BC-to-SAC comparison.
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
        if dict(payload["parameterization_config"]) != self.parameterization_config.to_dict():
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
        if mismatches:
            raise ValueError(
                "BC actor checkpoint contract mismatch: %s"
                % ", ".join(sorted(set(mismatches)))
            )
        # The freshly constructed optimizer still owns these same parameter
        # objects and has no Adam moments, so copying weights is sufficient.
        self.agent.actor.load_state_dict(source_agent["actor"])
        self.normalizer = normalizer
        self.actor_initialization = {
            "mode": "actor_and_normalizer_only",
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
            episode_seed=self.training_config.seed + self.episodes,
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
        environment = MppiPriorEnv(
            validation_config, self.project_root, seed=seed
        )
        observation, _ = environment.reset(seed=seed)
        total_reward = 0.0
        gate_alpha_sum = 0.0
        distance_gate_alpha_sum = 0.0
        steps = 0
        last_info = {}
        try:
            while True:
                normalized = self.normalizer.normalize(observation)
                action, _ = self.agent.select_action(normalized, deterministic=True)
                observation, reward, terminated, truncated, last_info = environment.step(action)
                total_reward += reward
                prior = dict(last_info.get("prior", {}))
                gate_alpha_sum += float(prior.get("gate_alpha", 1.0))
                distance_gate_alpha_sum += float(
                    prior.get("distance_gate_alpha", 1.0)
                )
                steps += 1
                if terminated or truncated:
                    break
        finally:
            environment.close()
        return {
            "return": float(total_reward),
            "success": bool(last_info.get("success", False)),
            "collision": bool(last_info.get("collision", False)),
            "goal_distance": float(last_info.get("goal_distance", float("inf"))),
            "gate_alpha_mean": gate_alpha_sum / max(steps, 1),
            "distance_gate_alpha_mean": distance_gate_alpha_sum / max(steps, 1),
        }

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
        }
        self.validation_records.extend(rows)
        score = 1000.0 * success_rate - 1000.0 * collision_rate + mean_return
        if score > self.best_validation_score:
            self.best_validation_score = score
            self.save("best.pt")
        return summary

    def run(self):
        target_steps = int(self.training_config.total_steps)
        observation, reset_info = self._reset_training_episode()
        if (
            self.training_config.normalizer_update == "frozen"
            and self.normalizer.count <= 0
        ):
            raise ValueError(
                "frozen RL normalizer requires a fitted initialization checkpoint"
            )
        self._update_normalizer(observation)
        if self.global_step == 0 and not (self.checkpoint_dir / "initial.pt").exists():
            # Preserve the untrained policy and its first valid observation
            # statistics so every learning run has a reproducible step-zero
            # control comparison rather than only loss curves.
            self.save("initial.pt")
        episode_return = 0.0
        episode_length = 0
        episode_gate_alpha = 0.0
        episode_distance_gate_alpha = 0.0
        episode_intrinsic_return = 0.0
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
            self._update_normalizer(next_observation)
            # Time-limit truncation is bootstrappable; true terminal states are not.
            replay_handle = self.replay.add(
                observation,
                action,
                reward,
                next_observation,
                terminated,
                group=self.pool.current_index,
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
                    last_update = self.agent.update(batch)
                record = {"global_step": self.global_step}
                record.update(last_update)
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
                }
                self.episode_records.append(record)
                self.episodes += 1
                observation, reset_info = self._reset_training_episode()
                self._update_normalizer(observation)
                episode_return = 0.0
                episode_length = 0
                episode_gate_alpha = 0.0
                episode_distance_gate_alpha = 0.0
                episode_intrinsic_return = 0.0
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
            "best_validation_score": self.best_validation_score,
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
            "best_validation_score": self.best_validation_score,
            "episode_records": len(self.episode_records),
            "update_records": len(self.update_records),
            "validation_records": len(self.validation_records),
            "replay_sampling": self.training_config.replay_sampling,
            "replay_group_counts": self.replay.group_counts(),
            "replay_outcome_counts": self.replay.outcome_counts(),
        }
        with (self.output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
