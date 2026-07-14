"""Strict, non-privileged demonstration dataset contract for RL priors.

The student is allowed to see only observations produced by
``MppiPriorEnv.reset``/``step`` and the normalized local-subgoal action chosen
by the teacher.  Route geometry, simulator truth and teacher diagnostics live
in a physically separate ``audit/`` tree and are intentionally absent from
the shards accepted by this loader.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


DEMONSTRATION_SCHEMA = "mobile_robot_mppi.scripted_subgoal_demonstrations"
DEMONSTRATION_SCHEMA_VERSION = 1
DEMONSTRATION_SPLITS = ("train", "validation", "test")
SHARD_FIELDS = frozenset((
    "observation",
    "teacher_action",
    "episode_id",
    "step",
))


def sha256_file(path, chunk_size=1024 * 1024):
    """Return a lowercase SHA-256 digest without loading a shard into RAM."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def split_episode_seeds(
    seeds,
    validation_fraction=0.2,
    test_fraction=0.2,
    seed=0,
):
    """Deterministically split whole seeds; a seed can never cross splits.

    A positive validation/test fraction receives at least one seed.  The
    function fails when there are too few seeds to retain a training split,
    rather than silently leaking timesteps from one episode across splits.
    """

    values = [int(value) for value in seeds]
    if not values:
        raise ValueError("demonstration split requires at least one seed")
    if len(set(values)) != len(values):
        raise ValueError("demonstration seeds must be unique")
    if any(value < 0 or value > 2 ** 32 - 1 for value in values):
        raise ValueError("demonstration seeds must be uint32-compatible")
    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    fractions = np.asarray(
        (validation_fraction, test_fraction), dtype=np.float64
    )
    if (
        not np.isfinite(fractions).all()
        or np.any(fractions < 0.0)
        or float(np.sum(fractions)) >= 1.0
    ):
        raise ValueError(
            "validation/test fractions must be finite, non-negative and sum below one"
        )
    required = 1 + int(validation_fraction > 0.0) + int(test_fraction > 0.0)
    if len(values) < required:
        raise ValueError(
            "too few seeds for non-empty train/validation/test split"
        )

    rng = np.random.RandomState(int(seed))
    shuffled = [values[index] for index in rng.permutation(len(values))]
    validation_count = (
        max(1, int(round(len(values) * validation_fraction)))
        if validation_fraction > 0.0
        else 0
    )
    test_count = (
        max(1, int(round(len(values) * test_fraction)))
        if test_fraction > 0.0
        else 0
    )
    overflow = validation_count + test_count - (len(values) - 1)
    while overflow > 0:
        if test_count > int(test_fraction > 0.0) and test_count >= validation_count:
            test_count -= 1
        elif validation_count > int(validation_fraction > 0.0):
            validation_count -= 1
        else:
            raise ValueError("split fractions leave no seed for training")
        overflow -= 1

    test_values = shuffled[:test_count]
    validation_values = shuffled[test_count:test_count + validation_count]
    train_values = shuffled[test_count + validation_count:]
    result = {
        "train": tuple(sorted(train_values)),
        "validation": tuple(sorted(validation_values)),
        "test": tuple(sorted(test_values)),
    }
    flattened = [item for name in DEMONSTRATION_SPLITS for item in result[name]]
    if set(flattened) != set(values) or len(flattened) != len(values):
        raise AssertionError("episode seed split is not a partition")
    return result


def validate_demonstration_arrays(
    arrays,
    expected_observation_dim=None,
    expected_action_dim=2,
    allow_empty=False,
):
    """Validate the exact non-privileged shard allowlist and array contract."""

    fields = set(arrays)
    if fields != SHARD_FIELDS:
        missing = sorted(SHARD_FIELDS - fields)
        extra = sorted(fields - SHARD_FIELDS)
        raise ValueError(
            "demonstration shard fields do not match allowlist "
            "(missing=%s, extra=%s)" % (missing, extra)
        )
    observation = np.asarray(arrays["observation"])
    teacher_action = np.asarray(arrays["teacher_action"])
    episode_id = np.asarray(arrays["episode_id"])
    steps = np.asarray(arrays["step"])
    if observation.ndim != 2 or not np.issubdtype(observation.dtype, np.floating):
        raise ValueError("observation must be a floating [N, observation_dim] array")
    if teacher_action.ndim != 2 or not np.issubdtype(
        teacher_action.dtype, np.floating
    ):
        raise ValueError("teacher_action must be a floating [N, action_dim] array")
    if episode_id.ndim != 1 or not np.issubdtype(episode_id.dtype, np.integer):
        raise ValueError("episode_id must be an integer [N] array")
    if steps.ndim != 1 or not np.issubdtype(steps.dtype, np.integer):
        raise ValueError("step must be an integer [N] array")
    sample_count = int(observation.shape[0])
    if not allow_empty and sample_count == 0:
        raise ValueError("demonstration shard cannot be empty")
    if teacher_action.shape[0] != sample_count:
        raise ValueError("observation and teacher_action sample counts differ")
    if episode_id.shape[0] != sample_count or steps.shape[0] != sample_count:
        raise ValueError("demonstration identifiers do not match sample count")
    if expected_observation_dim is not None and observation.shape[1] != int(
        expected_observation_dim
    ):
        raise ValueError("demonstration observation dimension is inconsistent")
    if teacher_action.shape[1] != int(expected_action_dim):
        raise ValueError("teacher action dimension is inconsistent")
    if not np.isfinite(observation).all() or not np.isfinite(teacher_action).all():
        raise ValueError("demonstration observations/actions must be finite")
    if np.any(teacher_action < -1.000001) or np.any(teacher_action > 1.000001):
        raise ValueError("normalized teacher actions must lie in [-1, 1]")
    if np.any(episode_id < 0) or np.any(steps < 0):
        raise ValueError("episode_id and step must be non-negative")

    # Detect accidental timestep mixing, duplication or partial episode cuts.
    for identifier in np.unique(episode_id):
        episode_steps = np.sort(steps[episode_id == identifier].astype(np.int64))
        expected = np.arange(episode_steps.size, dtype=np.int64)
        if not np.array_equal(episode_steps, expected):
            raise ValueError(
                "steps for episode %d must be unique and contiguous from zero"
                % int(identifier)
            )
    return {
        "sample_count": sample_count,
        "observation_dim": int(observation.shape[1]),
        "action_dim": int(teacher_action.shape[1]),
        "episode_count": int(np.unique(episode_id).size),
    }


def write_demonstration_shard(path, arrays, allow_empty=False):
    """Write one compressed shard containing only the approved student data."""

    summary = validate_demonstration_arrays(arrays, allow_empty=allow_empty)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(target),
        observation=np.asarray(arrays["observation"], dtype=np.float32),
        teacher_action=np.asarray(arrays["teacher_action"], dtype=np.float32),
        episode_id=np.asarray(arrays["episode_id"], dtype=np.int64),
        step=np.asarray(arrays["step"], dtype=np.int64),
    )
    result = dict(summary)
    result.update({
        "sha256": sha256_file(target),
        "bytes": int(target.stat().st_size),
    })
    return result


def _safe_dataset_path(dataset_dir, relative_path):
    root = Path(dataset_dir).resolve()
    candidate = (root / str(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        raise ValueError("demonstration manifest path escapes dataset directory")
    return candidate


def _validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("demonstration manifest must be a JSON object")
    required = {
        "schema",
        "schema_version",
        "created_utc",
        "git_sha",
        "config",
        "observation_dim",
        "action_dim",
        "teacher",
        "split_plan",
        "splits",
        "counts",
        "audit",
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError("demonstration manifest is missing keys: %s" % missing)
    if manifest["schema"] != DEMONSTRATION_SCHEMA:
        raise ValueError("unsupported demonstration schema")
    if int(manifest["schema_version"]) != DEMONSTRATION_SCHEMA_VERSION:
        raise ValueError("unsupported demonstration schema version")
    observation_dim = int(manifest["observation_dim"])
    action_dim = int(manifest["action_dim"])
    if observation_dim <= 0 or action_dim != 2:
        raise ValueError("demonstration dimensions must be observation_dim>0/action_dim=2")
    teacher = manifest["teacher"]
    if not isinstance(teacher, dict):
        raise ValueError("demonstration teacher metadata must be an object")
    if teacher.get("class") != "ScriptedPolylineSubgoal":
        raise ValueError("demonstration teacher class is not approved")
    if teacher.get("action_space") != "normalized_local_subgoal_distance_bearing":
        raise ValueError("demonstration teacher action contract is not approved")
    if teacher.get("student_observation_source") != "MppiPriorEnv.reset_and_step":
        raise ValueError("demonstration observation provenance is not approved")
    if not isinstance(manifest["config"], dict) or not isinstance(
        manifest["counts"], dict
    ):
        raise ValueError("demonstration config/counts must be objects")
    split_plan = manifest["split_plan"]
    splits = manifest["splits"]
    if set(split_plan) != set(DEMONSTRATION_SPLITS):
        raise ValueError("demonstration split_plan must contain train/validation/test")
    if set(splits) != set(DEMONSTRATION_SPLITS):
        raise ValueError("demonstration splits must contain train/validation/test")
    planned_sets = []
    for name in DEMONSTRATION_SPLITS:
        planned = [int(value) for value in split_plan[name]]
        if len(planned) != len(set(planned)):
            raise ValueError("split_plan contains duplicate seeds")
        planned_sets.append(set(planned))
        descriptor = splits[name]
        if not isinstance(descriptor, dict):
            raise ValueError("demonstration split descriptor must be an object")
        descriptor_required = {
            "file", "sha256", "bytes", "samples", "episodes", "seeds"
        }
        if set(descriptor) != descriptor_required:
            raise ValueError("demonstration split descriptor fields are invalid")
        checksum = str(descriptor["sha256"])
        if len(checksum) != 64 or any(
            value not in "0123456789abcdef" for value in checksum
        ):
            raise ValueError("demonstration split checksum is invalid")
        if int(descriptor["samples"]) < 0 or int(descriptor["episodes"]) < 0:
            raise ValueError("demonstration split counts must be non-negative")
        successful_seeds = [int(value) for value in descriptor["seeds"]]
        if not set(successful_seeds).issubset(set(planned)):
            raise ValueError("successful split seeds are outside split_plan")
    for index, first in enumerate(planned_sets):
        for second in planned_sets[index + 1:]:
            if first.intersection(second):
                raise ValueError("episode seeds leak across demonstration splits")
    audit = manifest["audit"]
    if not isinstance(audit, dict) or audit.get("directory") != "audit":
        raise ValueError("privileged audit data must be physically separated in audit/")
    if audit.get("included_in_training_shards") is not False:
        raise ValueError("privileged audit data cannot enter training shards")
    return manifest


def write_demonstration_manifest(dataset_dir, manifest):
    """Validate and write ``manifest.json`` after all shards exist."""

    validated = _validate_manifest(dict(manifest))
    root = Path(dataset_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, sort_keys=True, allow_nan=False)
    return path


def load_demonstration_manifest(dataset_dir):
    """Load and validate the public manifest (never an audit-sidecar file)."""

    path = Path(dataset_dir).resolve() / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return _validate_manifest(manifest)


def load_demonstration_split(dataset_dir, split):
    """Load a verified split through an exact field allowlist.

    Returns pluralized, copied NumPy arrays so callers cannot retain an open
    ``NpzFile`` handle.  Checksums and manifest counts are verified before the
    arrays are made available to training code.
    """

    split = str(split)
    if split not in DEMONSTRATION_SPLITS:
        raise ValueError("demonstration split must be train, validation or test")
    manifest = load_demonstration_manifest(dataset_dir)
    descriptor = manifest["splits"][split]
    path = _safe_dataset_path(dataset_dir, descriptor["file"])
    if not path.is_file():
        raise FileNotFoundError("demonstration shard is missing: %s" % path)
    if sha256_file(path) != descriptor["sha256"]:
        raise ValueError("demonstration shard checksum mismatch: %s" % split)
    try:
        with np.load(str(path), allow_pickle=False) as loaded:
            if set(loaded.files) != SHARD_FIELDS:
                missing = sorted(SHARD_FIELDS - set(loaded.files))
                extra = sorted(set(loaded.files) - SHARD_FIELDS)
                raise ValueError(
                    "demonstration shard fields do not match allowlist "
                    "(missing=%s, extra=%s)" % (missing, extra)
                )
            arrays = {name: np.asarray(loaded[name]).copy() for name in SHARD_FIELDS}
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and "allowlist" in str(error):
            raise
        raise ValueError("failed to read safe demonstration shard: %s" % error)
    summary = validate_demonstration_arrays(
        arrays,
        expected_observation_dim=manifest["observation_dim"],
        expected_action_dim=manifest["action_dim"],
        allow_empty=True,
    )
    if summary["sample_count"] != int(descriptor["samples"]):
        raise ValueError("demonstration shard sample count disagrees with manifest")
    if summary["episode_count"] != int(descriptor["episodes"]):
        raise ValueError("demonstration shard episode count disagrees with manifest")
    return {
        "observations": arrays["observation"].astype(np.float32, copy=False),
        "teacher_actions": arrays["teacher_action"].astype(np.float32, copy=False),
        "episode_ids": arrays["episode_id"].astype(np.int64, copy=False),
        "steps": arrays["step"].astype(np.int64, copy=False),
    }
