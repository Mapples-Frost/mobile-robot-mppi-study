import csv
import hashlib
import json
from itertools import combinations

import numpy as np
import pytest

from src.learning.residual_dataset import (
    REQUIRED_FIELDS,
    ResidualDataset,
    contiguous_rollout_windows,
    split_by_episode,
    wrapped_finite_difference,
)
from mobile_robot_mppi.learning.trainer import dataset_provenance


EXPECTED_REQUIRED_FIELDS = (
    "episode_id",
    "seed",
    "step",
    "time",
    "dt",
    "state_t",
    "control_t",
    "applied_control_t",
    "state_t_plus_1",
    "nominal_derivative",
    "observed_derivative",
    "residual_target",
    "disturbance_type",
    "disturbance_parameters",
    "scene",
    "data_source",
    "model_version",
)


def _mapping_for_episode_steps(episode_steps):
    """Build a valid dataset mapping from ``(episode, steps, disturbance)`` rows."""

    episode_ids = []
    disturbance_types = []
    steps = []
    for episode_id, episode_step_values, disturbance_type in episode_steps:
        episode_ids.extend([episode_id] * len(episode_step_values))
        disturbance_types.extend([disturbance_type] * len(episode_step_values))
        steps.extend(episode_step_values)

    size = len(steps)
    step_array = np.asarray(steps, dtype=np.int64)
    row = np.arange(size, dtype=np.float64)
    dt = np.full(size, 0.1, dtype=np.float64)
    state_t = np.column_stack((row, -row, 0.05 * row))
    control_t = np.column_stack((1.0 + 0.1 * row, -0.2 + 0.01 * row))
    applied_control_t = control_t + np.asarray([0.05, -0.02])
    nominal_derivative = np.column_stack(
        (np.cos(state_t[:, 2]), np.sin(state_t[:, 2]), control_t[:, 1])
    )
    residual_target = np.column_stack(
        (0.01 * row, -0.02 * row, np.full(size, 0.1))
    )
    observed_derivative = nominal_derivative + residual_target
    state_t_plus_1 = state_t + dt[:, None] * observed_derivative

    return {
        "episode_id": np.asarray(episode_ids, dtype="U32"),
        "seed": np.asarray(
            [100 + int(episode_id.rsplit("-", 1)[-1]) for episode_id in episode_ids],
            dtype=np.int64,
        ),
        "step": step_array,
        "time": 0.1 * step_array.astype(np.float64),
        "dt": dt,
        "state_t": state_t,
        "control_t": control_t,
        "applied_control_t": applied_control_t,
        "state_t_plus_1": state_t_plus_1,
        "nominal_derivative": nominal_derivative,
        "observed_derivative": observed_derivative,
        "residual_target": residual_target,
        "disturbance_type": np.asarray(disturbance_types, dtype="U32"),
        "disturbance_parameters": np.asarray(
            [json.dumps({"gain": 1.0 + 0.1 * index}, sort_keys=True) for index in row],
            dtype="U96",
        ),
        "scene": np.asarray(["empty"] * size, dtype="U16"),
        "data_source": np.asarray(["pytest"] * size, dtype="U16"),
        "model_version": np.asarray(["model-v0"] * size, dtype="U16"),
    }


def _small_mapping():
    return _mapping_for_episode_steps(
        [
            ("episode-0", [0, 1, 2], "velocity_gain"),
            ("episode-1", [0, 1], "yaw_bias"),
        ]
    )


def _assert_same_dataset(actual, expected):
    assert len(actual) == len(expected)
    assert actual.state_dim == expected.state_dim
    assert actual.control_dim == expected.control_dim
    for field in REQUIRED_FIELDS:
        np.testing.assert_array_equal(
            getattr(actual, field),
            getattr(expected, field),
            err_msg=field,
        )


def test_required_fields_are_complete_and_each_missing_field_is_rejected():
    assert tuple(REQUIRED_FIELDS) == EXPECTED_REQUIRED_FIELDS
    mapping = _small_mapping()

    for missing_field in REQUIRED_FIELDS:
        incomplete = {key: value for key, value in mapping.items() if key != missing_field}
        with pytest.raises((KeyError, ValueError), match=missing_field):
            ResidualDataset.from_mapping(incomplete)


def test_dataset_rejects_inconsistent_transition_count():
    mapping = _small_mapping()
    mapping["control_t"] = mapping["control_t"][:-1]

    with pytest.raises(ValueError, match="control_t"):
        ResidualDataset.from_mapping(mapping)


def test_from_records_validates_required_fields_per_transition():
    mapping = _small_mapping()
    records = [
        {field: mapping[field][row] for field in REQUIRED_FIELDS}
        for row in range(len(mapping["episode_id"]))
    ]
    incomplete = dict(records[1])
    incomplete.pop("state_t_plus_1")
    records[1] = incomplete

    with pytest.raises(ValueError, match=r"record 1.*state_t_plus_1"):
        ResidualDataset.from_records(records)


def test_wrapped_finite_difference_uses_short_heading_delta_across_branch_cut():
    state_t = np.asarray([2.0, -1.0, np.pi - 0.05])
    state_t_plus_1 = np.asarray([2.2, -1.3, -np.pi + 0.05])

    observed = wrapped_finite_difference(
        state_t,
        state_t_plus_1,
        dt=0.1,
        angle_indices=(2,),
    )

    np.testing.assert_allclose(observed, [2.0, -3.0, 1.0], atol=1e-12)


def test_npz_json_csv_roundtrip_uses_no_pickle(tmp_path):
    original = ResidualDataset.from_mapping(
        _small_mapping(),
        metadata={"schema_version": 1, "purpose": "roundtrip-test"},
    )
    prefix = tmp_path / "residual-transitions"

    artifacts = original.save(prefix)

    npz_paths = list(tmp_path.rglob("*.npz"))
    json_paths = list(tmp_path.rglob("*.json"))
    csv_paths = list(tmp_path.rglob("*.csv"))
    assert len(npz_paths) == len(json_paths) == len(csv_paths) == 1

    # Merely opening an object array succeeds with allow_pickle=False; indexing it
    # is what proves every persisted field is actually pickle-free.
    with np.load(npz_paths[0], allow_pickle=False) as archive:
        for name in archive.files:
            persisted = archive[name]
            assert persisted.dtype != object, name

    assert artifacts["npz"] == npz_paths[0]
    assert artifacts["metadata"] == json_paths[0]
    assert artifacts["summary"] == csv_paths[0]

    metadata = json.loads(json_paths[0].read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["metadata"]["schema_version"] == 1
    assert metadata["metadata"]["purpose"] == "roundtrip-test"
    assert metadata["dataset"]["summary"]["transition_count"] == len(original)

    with csv_paths[0].open(newline="", encoding="utf-8") as csv_file:
        summary_rows = {row["metric"]: row["value"] for row in csv.DictReader(csv_file)}
    assert int(summary_rows["transition_count"]) == len(original)
    assert int(summary_rows["episode_count"]) == 2

    restored = ResidualDataset.load(npz_paths[0])
    _assert_same_dataset(restored, original)
    assert restored.metadata["schema_version"] == 1
    assert restored.metadata["purpose"] == "roundtrip-test"


def test_training_dataset_provenance_hashes_all_available_splits(tmp_path):
    dataset = ResidualDataset.from_mapping(_small_mapping())
    for name in ("train", "validation", "test", "unseen"):
        dataset.save(tmp_path / name)
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text('{"version": 1}\n', encoding="utf-8")

    provenance = dataset_provenance(tmp_path)

    assert set(provenance["splits"]) == {"train", "validation", "test", "unseen"}
    for name, record in provenance["splits"].items():
        path = tmp_path / (name + ".npz")
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["transition_count"] == len(dataset)
        assert record["episode_count"] == 2
    assert provenance["manifest"]["sha256"] == hashlib.sha256(
        manifest.read_bytes()
    ).hexdigest()


def test_episode_group_split_is_deterministic_leak_free_and_separates_unseen():
    episode_specs = []
    for episode_number in range(10):
        disturbance = "held-out" if episode_number in (8, 9) else (
            "velocity_gain" if episode_number % 2 == 0 else "yaw_bias"
        )
        episode_specs.append(
            (f"episode-{episode_number}", [0, 1, 2], disturbance)
        )
    dataset = ResidualDataset.from_mapping(_mapping_for_episode_steps(episode_specs))

    first = split_by_episode(
        dataset,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=2718,
        unseen_disturbance_types=("held-out",),
    )
    second = split_by_episode(
        dataset,
        validation_fraction=0.25,
        test_fraction=0.25,
        seed=2718,
        unseen_disturbance_types=("held-out",),
    )

    names = ("train", "validation", "test", "unseen")
    index_sets = {}
    for name in names:
        first_indices = np.asarray(getattr(first, f"{name}_indices"), dtype=np.int64)
        second_indices = np.asarray(getattr(second, f"{name}_indices"), dtype=np.int64)
        np.testing.assert_array_equal(first_indices, second_indices)
        index_sets[name] = set(first_indices.tolist())

    for left, right in combinations(names, 2):
        assert index_sets[left].isdisjoint(index_sets[right])
    assert set().union(*index_sets.values()) == set(range(len(dataset)))

    episode_to_split = {}
    for split_name, indices in index_sets.items():
        for episode_id in np.unique(dataset.episode_id[sorted(indices)]):
            assert episode_id not in episode_to_split
            episode_to_split[episode_id] = split_name

    unseen_indices = np.asarray(first.unseen_indices, dtype=np.int64)
    assert set(dataset.disturbance_type[unseen_indices]) == {"held-out"}
    for name in ("train", "validation", "test"):
        indices = np.asarray(getattr(first, f"{name}_indices"), dtype=np.int64)
        assert "held-out" not in set(dataset.disturbance_type[indices])


def test_contiguous_rollout_windows_never_cross_episode_or_step_gap():
    dataset = ResidualDataset.from_mapping(
        _mapping_for_episode_steps(
            [
                ("episode-0", [0, 1, 2, 3], "velocity_gain"),
                ("episode-1", [0, 1, 3, 4], "yaw_bias"),
                ("episode-2", [0, 1], "velocity_gain"),
            ]
        )
    )

    windows = contiguous_rollout_windows(dataset, horizon=3)

    np.testing.assert_array_equal(windows, [[0, 1, 2], [1, 2, 3]])
    np.testing.assert_array_equal(dataset.rollout_window_indices(3), windows)
    for window in windows:
        assert len(set(dataset.episode_id[window])) == 1
        np.testing.assert_array_equal(np.diff(dataset.step[window]), 1)
