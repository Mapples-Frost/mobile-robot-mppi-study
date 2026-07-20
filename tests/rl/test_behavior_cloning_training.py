import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import mobile_robot_mppi.rl.behavior_cloning as bc_module
from mobile_robot_mppi.core.spaces import body_velocity_action
from mobile_robot_mppi.rl.behavior_cloning import (
    BehaviorCloningConfig,
    BehaviorCloningTrainer,
)
from mobile_robot_mppi.rl.checkpointing import load_sac_checkpoint
from mobile_robot_mppi.rl.demonstrations import (
    DEMONSTRATION_SCHEMA,
    DEMONSTRATION_SCHEMA_VERSION,
    write_demonstration_manifest,
    write_demonstration_shard,
)
from mobile_robot_mppi.rl.observation import ObservationEncoderConfig
from mobile_robot_mppi.rl.parameterization import PriorParameterizationConfig


def _write_dataset(root):
    split_plan = {"train": [1, 2], "validation": [3], "test": [4]}
    descriptors = {}
    for split_index, split in enumerate(("train", "validation", "test")):
        episodes = 2 if split == "train" else 1
        observations = []
        actions = []
        episode_ids = []
        steps = []
        for episode in range(episodes):
            identifier = 10 * split_index + episode
            for step in range(2):
                value = 0.1 * (identifier + step)
                observations.append((value, value + 0.2, 1.0 - value))
                actions.append((np.tanh(value), np.tanh(-value - 0.1)))
                episode_ids.append(identifier)
                steps.append(step)
        arrays = {
            "observation": np.asarray(observations, dtype=np.float32),
            "teacher_action": np.asarray(actions, dtype=np.float32),
            "episode_id": np.asarray(episode_ids, dtype=np.int64),
            "step": np.asarray(steps, dtype=np.int64),
        }
        path = root / "splits" / (split + ".npz")
        result = write_demonstration_shard(path, arrays)
        descriptors[split] = {
            "file": "splits/%s.npz" % split,
            "sha256": result["sha256"],
            "bytes": result["bytes"],
            "samples": result["sample_count"],
            "episodes": result["episode_count"],
            "seeds": split_plan[split],
        }
    (root / "audit").mkdir(exist_ok=True)
    manifest = {
        "schema": DEMONSTRATION_SCHEMA,
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "created_utc": "2026-07-14T00:00:00+00:00",
        "git_sha": "b" * 40,
        "config": {"unit": True},
        "observation_dim": 3,
        "action_dim": 2,
        "action_mode": "mppi_prior",
        "observation_encoder": ObservationEncoderConfig(
            lidar_sectors=1,
            include_previous_action=False,
            include_safety_state=False,
        ).to_dict(),
        "prior_parameterization": PriorParameterizationConfig(
            kind="local_subgoal", num_knots=2, learn_covariance=False
        ).to_dict(),
        "teacher": {
            "class": "ScriptedPolylineSubgoal",
            "action_space": "normalized_local_subgoal_distance_bearing",
            "student_observation_source": "MppiPriorEnv.reset_and_step",
        },
        "split_plan": split_plan,
        "splits": descriptors,
        "counts": {"successful_episodes": 4, "failed_episodes": 0},
        "audit": {
            "directory": "audit",
            "included_in_training_shards": False,
        },
    }
    write_demonstration_manifest(root, manifest)


class _Probe:
    def __init__(self, *_args, **_kwargs):
        self.observation_dim = 3
        self.policy_action_dim = 2
        self.encoder = SimpleNamespace(config=ObservationEncoderConfig(
            lidar_sectors=1,
            include_previous_action=False,
            include_safety_state=False,
        ))
        self.parameterization = SimpleNamespace(
            config=PriorParameterizationConfig(
                kind="local_subgoal", num_knots=2, learn_covariance=False
            )
        )
        self.action_spec = body_velocity_action((0.0, 0.4), 1.0)

    def close(self):
        pass


def test_behavior_cloning_config_rejects_invalid_values():
    BehaviorCloningConfig(epochs=1, batch_size=1).validate()
    with pytest.raises(ValueError, match="epochs"):
        BehaviorCloningConfig(epochs=0).validate()
    with pytest.raises(ValueError, match="log_std_weight"):
        BehaviorCloningConfig(log_std_weight=-1.0).validate()


def test_bc_trainer_uses_train_only_normalizer_and_writes_resume_checkpoint(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    monkeypatch.setattr(bc_module, "MppiPriorEnv", _Probe)
    config = {
        "rl": {
            "torch_num_threads": 1,
            "sac": {"hidden_sizes": [8, 8], "activation": "relu"},
            "behavior_cloning": {
                "epochs": 2,
                "batch_size": 2,
                "early_stopping_patience": 0,
                "seed": 6,
                "device": "cpu",
            },
        }
    }
    output = tmp_path / "run"
    trainer = BehaviorCloningTrainer(
        {}, config, dataset, tmp_path, output
    )

    assert trainer.normalizer.count == 4
    result = trainer.run()

    assert trainer.normalizer.count == 4
    assert result["test"]["samples"] == 2
    payload = load_sac_checkpoint(result["latest_checkpoint"])
    assert payload["training_state"]["phase"] == "behavior_cloning"
    assert payload["training_state"]["bc_epoch"] == 2
    assert payload["normalizer"]["count"] == 4
    with (output / "run_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["normalizer_source"] == "train_split_only"


def test_bc_resume_is_epoch_exact_and_truncates_future_log_rows(
    tmp_path, monkeypatch
):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    monkeypatch.setattr(bc_module, "MppiPriorEnv", _Probe)

    def config(epochs):
        return {
            "rl": {
                "torch_num_threads": 1,
                "sac": {"hidden_sizes": [8, 8], "activation": "relu"},
                "behavior_cloning": {
                    "epochs": epochs,
                    "batch_size": 2,
                    "early_stopping_patience": 0,
                    "seed": 17,
                    "device": "cpu",
                },
            }
        }

    baseline = BehaviorCloningTrainer(
        {}, config(4), dataset, tmp_path, tmp_path / "baseline"
    )
    baseline_result = baseline.run()

    interrupted_output = tmp_path / "interrupted"
    first = BehaviorCloningTrainer(
        {}, config(2), dataset, tmp_path, interrupted_output
    )
    first_result = first.run()
    with (interrupted_output / "training.csv").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write("99,corrupt_future_row\n")

    resumed = BehaviorCloningTrainer(
        {}, config(4), dataset, tmp_path, interrupted_output
    )
    resumed.resume(first_result["latest_checkpoint"])
    resumed_result = resumed.run()

    baseline_payload = load_sac_checkpoint(baseline_result["latest_checkpoint"])
    resumed_payload = load_sac_checkpoint(resumed_result["latest_checkpoint"])
    for name, value in baseline_payload["agent"]["actor"].items():
        torch.testing.assert_close(
            value, resumed_payload["agent"]["actor"][name], rtol=0.0, atol=0.0
        )
    assert baseline_payload["agent"]["bc_update_steps"] == (
        resumed_payload["agent"]["bc_update_steps"]
    )
    with (interrupted_output / "training.csv").open(
        "r", encoding="utf-8"
    ) as handle:
        epochs = [int(line.split(",", 1)[0]) for line in handle.readlines()[1:]]
    assert epochs == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "contract,key,value,match",
    [
        (
            "observation_encoder",
            "lidar_max_range",
            9.0,
            "observation_encoder does not match",
        ),
        (
            "prior_parameterization",
            "subgoal_max_distance",
            2.5,
            "prior_parameterization does not match",
        ),
    ],
)
def test_bc_trainer_rejects_same_shape_but_different_semantic_contract(
    tmp_path, monkeypatch, contract, key, value, match
):
    dataset = tmp_path / "dataset"
    _write_dataset(dataset)
    manifest_path = dataset / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest[contract][key] = value
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    monkeypatch.setattr(bc_module, "MppiPriorEnv", _Probe)
    config = {
        "rl": {
            "torch_num_threads": 1,
            "sac": {"hidden_sizes": [8, 8], "activation": "relu"},
            "behavior_cloning": {
                "epochs": 1,
                "batch_size": 2,
                "seed": 6,
                "device": "cpu",
            },
        }
    }

    with pytest.raises(ValueError, match=match):
        BehaviorCloningTrainer({}, config, dataset, tmp_path, tmp_path / "run")
