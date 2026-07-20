import json
from types import SimpleNamespace

import numpy as np
import pytest

import experiments.rl.collect_scripted_subgoal_demonstrations as collector_module
from experiments.rl.collect_scripted_subgoal_demonstrations import _collect_episode
from mobile_robot_mppi.rl.demonstrations import (
    DEMONSTRATION_SCHEMA,
    DEMONSTRATION_SCHEMA_VERSION,
    demonstration_manifest_fingerprint,
    load_demonstration_manifest,
    load_demonstration_split,
    load_demonstration_splits,
    sha256_file,
    split_episode_seeds,
    validate_demonstration_arrays,
    write_demonstration_manifest,
    write_demonstration_shard,
)
from mobile_robot_mppi.rl.observation import ObservationEncoderConfig
from mobile_robot_mppi.rl.parameterization import PriorParameterizationConfig
from mobile_robot_mppi.rl.trainer import BehaviorCloningAnchor


def _encoder_contract():
    return ObservationEncoderConfig(
        lidar_sectors=1,
        include_previous_action=False,
        include_safety_state=False,
    ).to_dict()


def _prior_contract():
    return PriorParameterizationConfig(
        kind="local_subgoal", num_knots=2, learn_covariance=False
    ).to_dict()


def _arrays(episode_id, observation_dim=3):
    return {
        "observation": np.asarray(
            [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=np.float32
        )[:, :observation_dim],
        "teacher_action": np.asarray(
            [[-0.2, 0.5], [0.1, -0.4]], dtype=np.float32
        ),
        "episode_id": np.asarray([episode_id, episode_id], dtype=np.int64),
        "step": np.asarray([0, 1], dtype=np.int64),
    }


def _dataset(tmp_path):
    descriptors = {}
    split_plan = {"train": [11], "validation": [12], "test": [13]}
    for index, split in enumerate(("train", "validation", "test")):
        path = tmp_path / "splits" / (split + ".npz")
        result = write_demonstration_shard(path, _arrays(index + 1))
        descriptors[split] = {
            "file": "splits/%s.npz" % split,
            "sha256": result["sha256"],
            "bytes": result["bytes"],
            "samples": result["sample_count"],
            "episodes": result["episode_count"],
            "seeds": list(split_plan[split]),
        }
    (tmp_path / "audit").mkdir(exist_ok=True)
    manifest = {
        "schema": DEMONSTRATION_SCHEMA,
        "schema_version": DEMONSTRATION_SCHEMA_VERSION,
        "created_utc": "2026-07-14T00:00:00+00:00",
        "git_sha": "a" * 40,
        "config": {"unit": True},
        "observation_dim": 3,
        "action_dim": 2,
        "action_mode": "mppi_prior",
        "observation_encoder": _encoder_contract(),
        "prior_parameterization": _prior_contract(),
        "teacher": {
            "class": "ScriptedPolylineSubgoal",
            "action_space": "normalized_local_subgoal_distance_bearing",
            "student_observation_source": "MppiPriorEnv.reset_and_step",
        },
        "split_plan": split_plan,
        "splits": descriptors,
        "counts": {
            "successful_episodes": 3,
            "failed_episodes": 0,
        },
        "audit": {
            "directory": "audit",
            "included_in_training_shards": False,
        },
    }
    write_demonstration_manifest(tmp_path, manifest)
    return manifest


def _legacy_dataset(tmp_path, mismatch=None, include_audit=True):
    manifest = _dataset(tmp_path)
    manifest["schema_version"] = 1
    manifest["config"]["observation_encoder"] = manifest.pop(
        "observation_encoder"
    )
    prior = manifest.pop("prior_parameterization")
    manifest["counts"]["requested_episodes"] = 3
    if include_audit:
        resolved_dir = tmp_path / "audit" / "resolved_configs"
        resolved_dir.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            encoder = _encoder_contract()
            current_prior = dict(prior)
            if index == 2 and mismatch == "encoder":
                encoder["lidar_max_range"] = 8.0
            if index == 2 and mismatch == "prior":
                current_prior["subgoal_max_distance"] = 2.5
            with (resolved_dir / ("episode_%06d.json" % index)).open(
                "w", encoding="utf-8"
            ) as handle:
                json.dump({
                    "rl": {
                        "observation": encoder,
                        "prior": current_prior,
                    }
                }, handle)
    with (tmp_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return manifest


def test_seed_split_is_deterministic_disjoint_and_never_splits_timesteps():
    first = split_episode_seeds(
        range(20, 30), validation_fraction=0.2, test_fraction=0.2, seed=77
    )
    second = split_episode_seeds(
        range(20, 30), validation_fraction=0.2, test_fraction=0.2, seed=77
    )

    assert first == second
    assert len(first["train"]) == 6
    assert len(first["validation"]) == 2
    assert len(first["test"]) == 2
    all_values = [value for values in first.values() for value in values]
    assert len(all_values) == len(set(all_values)) == 10


def test_seed_split_rejects_duplicates_bad_fractions_and_too_few_seeds():
    with pytest.raises(ValueError, match="unique"):
        split_episode_seeds([1, 1])
    with pytest.raises(ValueError, match="sum below one"):
        split_episode_seeds([1, 2, 3], 0.5, 0.5)
    with pytest.raises(ValueError, match="too few"):
        split_episode_seeds([1, 2], 0.2, 0.2)


def test_direct_teacher_scene_retains_residual_prediction_context(monkeypatch):
    monkeypatch.setattr(
        collector_module,
        "_resolved_scene",
        lambda *_args: {"planner": {"prediction_mode": "nominal"}},
    )
    rl_config = {"planner": {"prediction_mode": "icode_residual"}}

    direct = collector_module._teacher_scene(
        rl_config, "scene.yaml", 1, 2, 3, "direct_control"
    )
    legacy = collector_module._teacher_scene(
        rl_config, "scene.yaml", 1, 2, 3, "local_subgoal"
    )

    assert direct["planner"]["prediction_mode"] == "icode_residual"
    assert legacy["planner"]["prediction_mode"] == "nominal"


def test_demonstration_loader_roundtrip_exposes_only_student_arrays(tmp_path):
    _dataset(tmp_path)

    manifest = load_demonstration_manifest(tmp_path)
    split = load_demonstration_split(tmp_path, "train")

    assert manifest["audit"]["included_in_training_shards"] is False
    assert set(split) == {
        "observations", "teacher_actions", "episode_ids", "steps"
    }
    np.testing.assert_allclose(split["observations"], _arrays(1)["observation"])
    np.testing.assert_allclose(
        split["teacher_actions"], _arrays(1)["teacher_action"]
    )
    assert split["teacher_actions"].shape == (2, 2)


def test_legacy_v1_loader_recovers_contract_without_changing_fingerprint(tmp_path):
    original = _legacy_dataset(tmp_path)
    expected_fingerprint = demonstration_manifest_fingerprint(original)

    loaded = load_demonstration_manifest(tmp_path)
    splits = load_demonstration_splits(tmp_path)

    assert loaded["schema_version"] == 1
    assert loaded["observation_encoder"] == _encoder_contract()
    assert loaded["prior_parameterization"] == _prior_contract()
    assert demonstration_manifest_fingerprint(loaded) == expected_fingerprint
    assert splits["train"]["observations"].shape == (2, 3)
    with (tmp_path / "manifest.json").open("r", encoding="utf-8") as handle:
        unchanged = json.load(handle)
    assert "observation_encoder" not in unchanged
    assert "prior_parameterization" not in unchanged


def test_legacy_v1_loader_requires_resolved_config_audit(tmp_path):
    _legacy_dataset(tmp_path, include_audit=False)

    with pytest.raises(ValueError, match="requires audit/resolved_configs"):
        load_demonstration_manifest(tmp_path)


@pytest.mark.parametrize("mismatch", ("encoder", "prior"))
def test_legacy_v1_loader_rejects_inconsistent_resolved_contracts(
    tmp_path, mismatch
):
    _legacy_dataset(tmp_path, mismatch=mismatch)

    with pytest.raises(ValueError, match="differs across resolved configs"):
        load_demonstration_manifest(tmp_path)


def test_new_manifest_writer_refuses_legacy_v1(tmp_path):
    legacy = _legacy_dataset(tmp_path)

    with pytest.raises(ValueError, match="unsupported.*version"):
        write_demonstration_manifest(tmp_path, legacy)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda arrays: arrays.update({"truth_pose": np.zeros((2, 3))}), "allowlist"),
        (lambda arrays: arrays.pop("teacher_action"), "allowlist"),
        (
            lambda arrays: arrays["observation"].__setitem__((0, 0), np.nan),
            "finite",
        ),
        (
            lambda arrays: arrays["teacher_action"].__setitem__((0, 0), 1.2),
            r"\[-1, 1\]",
        ),
        (
            lambda arrays: arrays["step"].__setitem__(1, 2),
            "contiguous",
        ),
    ],
)
def test_validation_rejects_privileged_missing_invalid_or_partial_fields(
    mutation, match
):
    arrays = _arrays(4)
    mutation(arrays)
    with pytest.raises(ValueError, match=match):
        validate_demonstration_arrays(arrays)


def test_loader_rejects_checksum_tampering(tmp_path):
    _dataset(tmp_path)
    with (tmp_path / "splits" / "train.npz").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_demonstration_split(tmp_path, "train")


def test_loader_rejects_manifest_byte_count_mismatch(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["splits"]["train"]["bytes"] += 1
    write_demonstration_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="byte count"):
        load_demonstration_split(tmp_path, "train")


def test_loader_rejects_extra_npz_field_even_with_updated_checksum(tmp_path):
    manifest = _dataset(tmp_path)
    path = tmp_path / "splits" / "train.npz"
    arrays = _arrays(1)
    np.savez_compressed(str(path), **arrays, truth_pose=np.zeros((2, 3)))
    manifest["splits"]["train"]["sha256"] = sha256_file(path)
    manifest["splits"]["train"]["bytes"] = path.stat().st_size
    with (tmp_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="allowlist"):
        load_demonstration_split(tmp_path, "train")


def test_loader_rejects_actual_episode_overlap_across_splits(tmp_path):
    manifest = _dataset(tmp_path)
    path = tmp_path / "splits" / "validation.npz"
    result = write_demonstration_shard(path, _arrays(1))
    manifest["splits"]["validation"].update({
        "sha256": result["sha256"],
        "bytes": result["bytes"],
        "samples": result["sample_count"],
        "episodes": result["episode_count"],
    })
    write_demonstration_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="episode_id leakage.*train.*validation"):
        load_demonstration_splits(tmp_path)


def test_manifest_rejects_absolute_pose_and_incomplete_semantic_contracts(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["observation_encoder"]["include_absolute_pose"] = True
    with pytest.raises(ValueError, match="cannot include absolute pose"):
        write_demonstration_manifest(tmp_path, manifest)

    manifest = _dataset(tmp_path)
    manifest["observation_encoder"].pop("lidar_max_range")
    with pytest.raises(ValueError, match="complete canonical"):
        write_demonstration_manifest(tmp_path, manifest)

    manifest = _dataset(tmp_path)
    manifest["prior_parameterization"]["kind"] = "control_knots"
    with pytest.raises(ValueError, match="prior kind does not match"):
        write_demonstration_manifest(tmp_path, manifest)


def test_manifest_accepts_direct_control_and_rejects_semantic_mismatch(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["action_mode"] = "direct_control"
    manifest["prior_parameterization"] = PriorParameterizationConfig(
        kind="control_knots", num_knots=6, learn_covariance=False
    ).to_dict()
    manifest["teacher"] = {
        "class": "ScriptedPolylineDirectControl",
        "action_space": "normalized_direct_control_v_omega",
        "student_observation_source": "DirectControlEnv.reset_and_step",
    }
    write_demonstration_manifest(tmp_path, manifest)
    assert load_demonstration_manifest(tmp_path)["action_mode"] == "direct_control"

    manifest["teacher"]["action_space"] = (
        "normalized_local_subgoal_distance_bearing"
    )
    with pytest.raises(ValueError, match="action contract"):
        write_demonstration_manifest(tmp_path, manifest)


def test_bc_anchor_fails_closed_on_action_mode_mismatch(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["action_mode"] = "direct_control"
    manifest["prior_parameterization"] = PriorParameterizationConfig(
        kind="control_knots", num_knots=6, learn_covariance=False
    ).to_dict()
    manifest["teacher"] = {
        "class": "ScriptedPolylineDirectControl",
        "action_space": "normalized_direct_control_v_omega",
        "student_observation_source": "DirectControlEnv.reset_and_step",
    }
    write_demonstration_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="action_mode does not match"):
        BehaviorCloningAnchor(
            {"enabled": True, "dataset_dir": str(tmp_path)},
            tmp_path,
            observation_dim=3,
            action_dim=2,
            seed=5,
            action_mode="mppi_prior",
        )


def test_manifest_rejects_seed_leakage_and_escaped_shard_path(tmp_path):
    manifest = _dataset(tmp_path)
    manifest["split_plan"]["validation"] = [11]
    with pytest.raises(ValueError, match="outside split_plan|leak"):
        write_demonstration_manifest(tmp_path, manifest)

    manifest = _dataset(tmp_path)
    manifest["splits"]["train"]["file"] = "../train.npz"
    with (tmp_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)
    with pytest.raises(ValueError, match="escapes"):
        load_demonstration_split(tmp_path, "train")


class _FakePolicy:
    def reset(self):
        pass

    def action(self, pose):
        action = np.asarray((0.25 * pose.x, -0.5), dtype=np.float32)
        return action, {
            "route_progress": pose.x,
            "target_progress": pose.x + 0.7,
            "route_length": 2.0,
            "cross_track_error": 0.0,
            "target_x": pose.x + 0.7,
            "target_y": 0.0,
            "target_distance": 0.7,
            "target_bearing": 0.0,
        }


class _FakeEnvironment:
    def __init__(self, success=True):
        self.config = {
            "scene": {"name": "fake"},
            "sensors": {"pose_source": "ground_truth", "twist_source": "ground_truth"},
        }
        self.success = success
        self.index = 0
        self.truth = None
        self.perceived = None

    @staticmethod
    def _pose(x):
        return SimpleNamespace(x=float(x), y=0.0, theta=0.0)

    def _state(self, x):
        pose = self._pose(x)
        self.truth = SimpleNamespace(pose=pose, timestamp=float(x))
        self.perceived = SimpleNamespace(
            observation=SimpleNamespace(pose=self._pose(x))
        )

    def reset(self, seed=None):
        self.index = 0
        self._state(0.0)
        return np.asarray((10.0, 11.0, 12.0), dtype=np.float32), {
            "goal_distance": 3.0,
        }

    def step(self, action):
        self.index += 1
        self._state(float(self.index))
        done = self.index == 2
        info = {
            "proposed_control": np.asarray((0.1, 0.2)),
            "executed_control": np.asarray((0.1, 0.2)),
            "goal_distance": float(3 - self.index),
            "minimum_clearance": 0.5,
            "safety_override": False,
            "collision": False,
            "success": bool(done and self.success),
        }
        observation = np.asarray(
            (10.0 + self.index, 11.0 + self.index, 12.0 + self.index),
            dtype=np.float32,
        )
        return observation, 1.0, done, False, info


def test_collection_uses_reset_step_observations_and_exact_teacher_actions():
    arrays, _, summary = _collect_episode(
        _FakeEnvironment(success=True), _FakePolicy(), 7, 101, "train"
    )

    np.testing.assert_allclose(
        arrays["observation"],
        [[10.0, 11.0, 12.0], [11.0, 12.0, 13.0]],
    )
    np.testing.assert_allclose(
        arrays["teacher_action"], [[0.0, -0.5], [0.25, -0.5]]
    )
    np.testing.assert_array_equal(arrays["step"], [0, 1])
    assert summary["included_in_training_shard"] is True


def test_failed_teacher_episode_is_marked_audit_only():
    _, audit, summary = _collect_episode(
        _FakeEnvironment(success=False), _FakePolicy(), 8, 102, "validation"
    )

    assert audit
    assert summary["success"] is False
    assert summary["included_in_training_shard"] is False
