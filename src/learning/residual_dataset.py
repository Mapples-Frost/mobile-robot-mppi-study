"""Validated, pickle-free transition datasets for residual dynamics learning.

The storage format deliberately separates numeric/string arrays (compressed NPZ)
from human-readable experiment metadata (JSON) and a compact summary (CSV).
No object arrays are written, so every dataset can be loaded with
``allow_pickle=False``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 1

REQUIRED_FIELDS: Tuple[str, ...] = (
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

_TEXT_FIELDS = (
    "disturbance_type",
    "scene",
    "data_source",
    "model_version",
)
_STATE_FIELDS = (
    "state_t",
    "state_t_plus_1",
    "nominal_derivative",
    "observed_derivative",
    "residual_target",
)
_CONTROL_FIELDS = ("control_t", "applied_control_t")


def _json_compatible(value: Any, path: str = "metadata") -> Any:
    """Return a lossless JSON-domain representation or raise a useful error."""

    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("{} contains NaN or Inf".format(path))
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist(), path)
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("{} has non-string key {!r}".format(path, key))
            result[key] = _json_compatible(item, "{}.{}".format(path, key))
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_compatible(item, "{}[{}]".format(path, index))
            for index, item in enumerate(value)
        ]
    raise TypeError(
        "{} contains non-JSON-serializable value of type {}".format(
            path, type(value).__name__
        )
    )


def _metadata_dict(metadata: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    converted = _json_compatible(metadata)
    # Round-tripping here also proves strict JSON serializability and detaches
    # the dataset from mutable caller-owned containers.
    encoded = json.dumps(converted, ensure_ascii=False, allow_nan=False)
    return json.loads(encoded)


def _one_dimensional(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("{} must have shape (N,), got {}".format(name, array.shape))
    return array


def _integer_vector(values: Any, name: str, nonnegative: bool = True) -> np.ndarray:
    array = _one_dimensional(values, name)
    if array.dtype.kind not in "iu":
        if array.dtype.kind == "O" and all(
            isinstance(item, (int, np.integer))
            and not isinstance(item, (bool, np.bool_))
            for item in array.tolist()
        ):
            pass
        else:
            raise TypeError("{} must contain integers".format(name))
    if array.dtype.kind == "b":
        raise TypeError("{} must contain integers, not booleans".format(name))
    try:
        result = np.asarray(array, dtype=np.int64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{} cannot be represented as int64".format(name)) from exc
    if nonnegative and np.any(result < 0):
        raise ValueError("{} must be nonnegative".format(name))
    return np.ascontiguousarray(result)


def _float_vector(values: Any, name: str) -> np.ndarray:
    array = _one_dimensional(values, name)
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric".format(name)) from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("{} contains NaN or Inf".format(name))
    return np.ascontiguousarray(result)


def _float_matrix(values: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric".format(name)) from exc
    if result.ndim != 2:
        raise ValueError("{} must have shape (N, D), got {}".format(name, result.shape))
    if result.shape[1] <= 0:
        raise ValueError("{} feature dimension must be positive".format(name))
    if not np.all(np.isfinite(result)):
        raise ValueError("{} contains NaN or Inf".format(name))
    return np.ascontiguousarray(result)


def _episode_vector(values: Any) -> np.ndarray:
    array = _one_dimensional(values, "episode_id")
    if array.dtype.kind in "iu":
        return np.ascontiguousarray(array.astype(np.int64, copy=False))
    if array.dtype.kind == "b":
        raise TypeError("episode_id must contain integer or string identifiers")
    items = array.tolist()
    if all(isinstance(item, str) for item in items):
        return np.ascontiguousarray(np.asarray(items, dtype=np.str_))
    if all(
        isinstance(item, (int, np.integer))
        and not isinstance(item, (bool, np.bool_))
        for item in items
    ):
        return np.ascontiguousarray(np.asarray(items, dtype=np.int64))
    if not items:
        # Preserve a safe, pickle-free representation for a standalone empty set.
        return np.asarray([], dtype=np.str_)
    raise TypeError("episode_id must contain only integers or only strings")


def _unicode_vector(values: Any, name: str) -> np.ndarray:
    array = _one_dimensional(values, name)
    items = array.tolist()
    if not all(isinstance(item, (str, np.str_)) for item in items):
        raise TypeError("{} must contain strings".format(name))
    return np.ascontiguousarray(np.asarray(items, dtype=np.str_))


def _json_unicode_vector(values: Any) -> np.ndarray:
    array = _one_dimensional(values, "disturbance_parameters")
    serialized: List[str] = []
    for index, item in enumerate(array.tolist()):
        if isinstance(item, (str, np.str_)):
            try:
                parsed = json.loads(str(item))
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "disturbance_parameters[{}] is not valid JSON".format(index)
                ) from exc
        else:
            parsed = item
        compatible = _json_compatible(
            parsed, "disturbance_parameters[{}]".format(index)
        )
        serialized.append(
            json.dumps(
                compatible,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return np.ascontiguousarray(np.asarray(serialized, dtype=np.str_))


def _validate_angle_indices(angle_indices: Sequence[int], state_dim: int) -> Tuple[int, ...]:
    if angle_indices is None:
        return ()
    try:
        values = tuple(angle_indices)
    except TypeError as exc:
        raise TypeError("angle_indices must be an iterable of integers") from exc
    result: List[int] = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError("angle indices must be integers")
        index = int(value)
        if index < 0 or index >= state_dim:
            raise ValueError(
                "angle index {} is outside state dimension {}".format(index, state_dim)
            )
        if index in result:
            raise ValueError("duplicate angle index {}".format(index))
        result.append(index)
    return tuple(result)


def wrapped_finite_difference(
    state_t: Any,
    state_t_plus_1: Any,
    dt: Any,
    angle_indices: Sequence[int] = (),
) -> np.ndarray:
    """Compute raw-state derivatives with shortest-arc angular differences.

    Inputs may be a single state ``(nx,)`` or a batch ``(N, nx)``. ``dt`` may
    be a positive scalar or one positive value per batch row. The return shape
    matches the state input shape.
    """

    try:
        current = np.asarray(state_t, dtype=np.float64)
        following = np.asarray(state_t_plus_1, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("states must be numeric") from exc
    if current.ndim not in (1, 2):
        raise ValueError("states must have shape (nx,) or (N, nx)")
    if following.shape != current.shape:
        raise ValueError(
            "state_t_plus_1 shape {} does not match state_t {}".format(
                following.shape, current.shape
            )
        )
    if current.shape[-1] <= 0:
        raise ValueError("state dimension must be positive")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(following)):
        raise ValueError("states contain NaN or Inf")

    indices = _validate_angle_indices(angle_indices, int(current.shape[-1]))
    delta = following - current
    if indices:
        angle_array = np.asarray(indices, dtype=np.int64)
        delta[..., angle_array] = np.arctan2(
            np.sin(delta[..., angle_array]), np.cos(delta[..., angle_array])
        )

    try:
        dt_array = np.asarray(dt, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("dt must be numeric") from exc
    if current.ndim == 1:
        if dt_array.ndim == 1 and dt_array.size == 1:
            dt_array = dt_array.reshape(())
        if dt_array.ndim != 0:
            raise ValueError("dt must be scalar for a single state")
        denominator = dt_array
    else:
        if dt_array.ndim == 0:
            denominator = dt_array
        elif dt_array.shape == (current.shape[0],):
            denominator = dt_array[:, None]
        elif dt_array.shape == (current.shape[0], 1):
            denominator = dt_array
        else:
            raise ValueError("dt must be scalar or have shape (N,) for batched states")
    if not np.all(np.isfinite(dt_array)):
        raise ValueError("dt contains NaN or Inf")
    if np.any(dt_array <= 0.0):
        raise ValueError("dt must be strictly positive")
    return np.ascontiguousarray(delta / denominator)


# A descriptive alias used by collectors.
compute_observed_derivative = wrapped_finite_difference


@dataclass(eq=False)
class ResidualDataset:
    """A validated columnar collection of residual-learning transitions."""

    episode_id: Any
    seed: Any
    step: Any
    time: Any
    dt: Any
    state_t: Any
    control_t: Any
    applied_control_t: Any
    state_t_plus_1: Any
    nominal_derivative: Any
    observed_derivative: Any
    residual_target: Any
    disturbance_type: Any
    disturbance_parameters: Any
    scene: Any
    data_source: Any
    model_version: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.episode_id = _episode_vector(self.episode_id)
        self.seed = _integer_vector(self.seed, "seed")
        self.step = _integer_vector(self.step, "step")
        self.time = _float_vector(self.time, "time")
        self.dt = _float_vector(self.dt, "dt")
        if np.any(self.dt <= 0.0):
            raise ValueError("dt must be strictly positive")

        for name in _STATE_FIELDS + _CONTROL_FIELDS:
            setattr(self, name, _float_matrix(getattr(self, name), name))
        for name in _TEXT_FIELDS:
            setattr(self, name, _unicode_vector(getattr(self, name), name))
        self.disturbance_parameters = _json_unicode_vector(
            self.disturbance_parameters
        )
        self.metadata = _metadata_dict(self.metadata)

        count = int(self.episode_id.shape[0])
        for name in REQUIRED_FIELDS:
            array = getattr(self, name)
            if array.shape[0] != count:
                raise ValueError(
                    "{} has {} rows; expected {}".format(name, array.shape[0], count)
                )

        state_dim = int(self.state_t.shape[1])
        for name in _STATE_FIELDS[1:]:
            if getattr(self, name).shape[1] != state_dim:
                raise ValueError(
                    "{} feature dimension {} does not match state dimension {}".format(
                        name, getattr(self, name).shape[1], state_dim
                    )
                )
        control_dim = int(self.control_t.shape[1])
        if self.applied_control_t.shape[1] != control_dim:
            raise ValueError(
                "applied_control_t feature dimension {} does not match control dimension {}".format(
                    self.applied_control_t.shape[1], control_dim
                )
            )
        for name in REQUIRED_FIELDS:
            if getattr(self, name).dtype.kind == "O":
                raise TypeError("{} cannot use object dtype".format(name))

    @classmethod
    def from_mapping(
        cls,
        arrays: Mapping[str, Any],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ResidualDataset":
        """Build a dataset from a mapping, rejecting missing or unknown columns."""

        if not isinstance(arrays, Mapping):
            raise TypeError("arrays must be a mapping")
        missing = [name for name in REQUIRED_FIELDS if name not in arrays]
        if missing:
            raise ValueError("missing required fields: {}".format(", ".join(missing)))
        unknown = sorted(
            str(name)
            for name in arrays.keys()
            if name not in REQUIRED_FIELDS and name != "metadata"
        )
        if unknown:
            raise ValueError("unknown dataset fields: {}".format(", ".join(unknown)))
        if metadata is None and "metadata" in arrays:
            metadata = arrays["metadata"]
        return cls(
            **{name: arrays[name] for name in REQUIRED_FIELDS},
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ResidualDataset":
        """Build from row mappings; useful for simulation data collectors."""

        rows = list(records)
        if not rows:
            raise ValueError("records must contain at least one transition")
        columns: Dict[str, List[Any]] = {name: [] for name in REQUIRED_FIELDS}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError("record {} must be a mapping".format(index))
            missing = [name for name in REQUIRED_FIELDS if name not in row]
            if missing:
                raise ValueError(
                    "record {} is missing fields: {}".format(index, ", ".join(missing))
                )
            for name in REQUIRED_FIELDS:
                columns[name].append(row[name])
        return cls.from_mapping(columns, metadata=metadata)

    @classmethod
    def load(
        cls,
        path: Any,
        metadata_path: Optional[Any] = None,
    ) -> "ResidualDataset":
        npz_path, default_metadata, _ = _artifact_paths(path)
        json_path = Path(metadata_path) if metadata_path is not None else default_metadata
        if not npz_path.is_file():
            raise FileNotFoundError(str(npz_path))
        try:
            with np.load(str(npz_path), allow_pickle=False) as archive:
                missing = [name for name in REQUIRED_FIELDS if name not in archive.files]
                if missing:
                    raise ValueError(
                        "dataset archive is missing fields: {}".format(", ".join(missing))
                    )
                arrays = {name: archive[name] for name in REQUIRED_FIELDS}
        except ValueError as exc:
            raise ValueError("failed to load safe NPZ dataset: {}".format(exc)) from exc

        metadata: Mapping[str, Any] = {}
        if json_path.is_file():
            with json_path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if not isinstance(payload, Mapping):
                raise ValueError("metadata sidecar must contain a JSON object")
            version = payload.get("schema_version", SCHEMA_VERSION)
            if version != SCHEMA_VERSION:
                raise ValueError("unsupported dataset schema version {}".format(version))
            stored = payload.get("metadata", {})
            if not isinstance(stored, Mapping):
                raise ValueError("metadata sidecar 'metadata' must be an object")
            metadata = stored
        return cls.from_mapping(arrays, metadata=metadata)

    @property
    def size(self) -> int:
        return int(self.episode_id.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.state_t.shape[1])

    @property
    def control_dim(self) -> int:
        return int(self.control_t.shape[1])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, name: str) -> np.ndarray:
        if name not in REQUIRED_FIELDS:
            raise KeyError(name)
        return getattr(self, name)

    def __iter__(self) -> Iterator[str]:
        return iter(REQUIRED_FIELDS)

    def keys(self) -> Tuple[str, ...]:
        return REQUIRED_FIELDS

    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        for name in REQUIRED_FIELDS:
            yield name, getattr(self, name)

    def as_dict(self, copy: bool = False) -> Dict[str, np.ndarray]:
        return {
            name: getattr(self, name).copy() if copy else getattr(self, name)
            for name in REQUIRED_FIELDS
        }

    # Explicit alias for callers that think in columnar mappings.
    as_mapping = as_dict

    def subset(
        self,
        indices: Any,
        metadata_updates: Optional[Mapping[str, Any]] = None,
    ) -> "ResidualDataset":
        selected = _normalize_indices(indices, self.size)
        metadata = dict(self.metadata)
        if metadata_updates:
            metadata.update(_metadata_dict(metadata_updates))
        return ResidualDataset.from_mapping(
            {name: getattr(self, name)[selected] for name in REQUIRED_FIELDS},
            metadata=metadata,
        )

    take = subset

    def summary(self) -> Dict[str, Any]:
        episode_values = {_stable_scalar(value) for value in self.episode_id.tolist()}
        disturbance_counts: Dict[str, int] = {}
        for value in self.disturbance_type.tolist():
            disturbance_counts[str(value)] = disturbance_counts.get(str(value), 0) + 1
        scene_counts: Dict[str, int] = {}
        for value in self.scene.tolist():
            scene_counts[str(value)] = scene_counts.get(str(value), 0) + 1
        result: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "transition_count": self.size,
            "episode_count": len(episode_values),
            "seed_count": len(set(int(value) for value in self.seed.tolist())),
            "state_dim": self.state_dim,
            "control_dim": self.control_dim,
            "disturbance_counts": disturbance_counts,
            "scene_counts": scene_counts,
        }
        if self.size:
            result.update(
                {
                    "time_min": float(np.min(self.time)),
                    "time_max": float(np.max(self.time)),
                    "dt_min": float(np.min(self.dt)),
                    "dt_max": float(np.max(self.dt)),
                    "dt_mean": float(np.mean(self.dt)),
                }
            )
        else:
            result.update(
                {
                    "time_min": None,
                    "time_max": None,
                    "dt_min": None,
                    "dt_max": None,
                    "dt_mean": None,
                }
            )
        return result

    def save(
        self,
        path: Any,
        metadata_path: Optional[Any] = None,
        summary_path: Optional[Any] = None,
    ) -> Dict[str, Path]:
        """Atomically publish NPZ last, after its JSON and CSV sidecars."""

        npz_path, default_metadata, default_summary = _artifact_paths(path)
        json_path = Path(metadata_path) if metadata_path is not None else default_metadata
        csv_path = Path(summary_path) if summary_path is not None else default_summary
        for destination in (npz_path, json_path, csv_path):
            destination.parent.mkdir(parents=True, exist_ok=True)

        token = ".tmp-{}".format(uuid.uuid4().hex)
        tmp_npz = npz_path.with_name(npz_path.name + token)
        tmp_json = json_path.with_name(json_path.name + token)
        tmp_csv = csv_path.with_name(csv_path.name + token)
        temporary = (tmp_npz, tmp_json, tmp_csv)
        try:
            with tmp_npz.open("wb") as stream:
                np.savez_compressed(stream, **self.as_dict(copy=False))
                stream.flush()
                os.fsync(stream.fileno())

            payload = {
                "schema_version": SCHEMA_VERSION,
                "metadata": self.metadata,
                "dataset": {
                    "required_fields": list(REQUIRED_FIELDS),
                    "shapes": {
                        name: list(getattr(self, name).shape) for name in REQUIRED_FIELDS
                    },
                    "dtypes": {
                        name: str(getattr(self, name).dtype) for name in REQUIRED_FIELDS
                    },
                    "summary": self.summary(),
                },
            }
            _write_json_file(tmp_json, payload)
            _write_summary_csv(tmp_csv, self.summary())

            os.replace(str(tmp_json), str(json_path))
            os.replace(str(tmp_csv), str(csv_path))
            # Publishing the NPZ last makes it the commit marker for a new artifact.
            os.replace(str(tmp_npz), str(npz_path))
        finally:
            for candidate in temporary:
                try:
                    candidate.unlink()
                except FileNotFoundError:
                    pass
        return {"npz": npz_path, "metadata": json_path, "summary": csv_path}

    def split(
        self,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        unseen_disturbance_types: Sequence[str] = (),
        seed: int = 0,
        train_fraction: Optional[float] = None,
    ) -> "DatasetSplits":
        return split_by_episode(
            self,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            unseen_disturbance_types=unseen_disturbance_types,
            seed=seed,
            train_fraction=train_fraction,
        )

    def rollout_window_indices(
        self,
        horizon: int,
        stride: int = 1,
        require_time_continuity: bool = True,
        require_constant_dt: bool = False,
        rtol: float = 1.0e-7,
        atol: float = 1.0e-9,
    ) -> np.ndarray:
        return contiguous_rollout_windows(
            self,
            horizon=horizon,
            stride=stride,
            require_time_continuity=require_time_continuity,
            require_constant_dt=require_constant_dt,
            rtol=rtol,
            atol=atol,
        )


def _artifact_paths(path: Any) -> Tuple[Path, Path, Path]:
    candidate = Path(path)
    if candidate.exists() and candidate.is_dir():
        npz_path = candidate / "residual_dataset.npz"
    elif candidate.suffix.lower() == ".npz":
        npz_path = candidate
    elif not candidate.suffix:
        npz_path = candidate.with_suffix(".npz")
    else:
        raise ValueError("dataset path must be a directory, prefix, or .npz file")
    return (
        npz_path,
        npz_path.with_suffix(".metadata.json"),
        npz_path.with_suffix(".summary.csv"),
    )


def _write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            payload,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_summary_csv(path: Path, summary: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "value"))
        for key in sorted(summary):
            value = summary[key]
            if isinstance(value, (dict, list)):
                rendered = json.dumps(
                    value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
            elif value is None:
                rendered = ""
            else:
                rendered = str(value)
            writer.writerow((key, rendered))
        stream.flush()
        os.fsync(stream.fileno())


def _normalize_indices(indices: Any, size: int) -> np.ndarray:
    if isinstance(indices, slice):
        return np.arange(size, dtype=np.int64)[indices]
    array = np.asarray(indices)
    if array.ndim == 0:
        if isinstance(array.item(), (bool, np.bool_)):
            raise TypeError("a scalar boolean is not a valid index")
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if array.size == 0:
        return np.empty((0,), dtype=np.int64)
    if array.dtype.kind == "b":
        if array.shape != (size,):
            raise ValueError("boolean mask must have shape ({},)".format(size))
        return np.flatnonzero(array).astype(np.int64, copy=False)
    if array.dtype.kind not in "iu":
        raise TypeError("indices must contain integers or booleans")
    result = array.astype(np.int64, copy=False)
    result = np.where(result < 0, result + size, result)
    if np.any(result < 0) or np.any(result >= size):
        raise IndexError("dataset index out of range")
    return np.ascontiguousarray(result)


def _stable_scalar(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return "int:{}".format(int(value))
    return "str:{}".format(str(value))


def _validate_fraction(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("{} must be a finite scalar".format(name))
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("{} must be a finite scalar".format(name)) from exc
    if not np.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("{} must be in [0, 1]".format(name))
    return result


def _group_counts(group_count: int, fractions: Sequence[float]) -> List[int]:
    raw = np.asarray(fractions, dtype=np.float64) * float(group_count)
    counts = np.floor(raw).astype(np.int64)
    remaining = int(group_count - int(np.sum(counts)))
    if remaining:
        remainders = raw - counts
        order = sorted(range(len(fractions)), key=lambda i: (-remainders[i], i))
        for index in order[:remaining]:
            counts[index] += 1
    return [int(value) for value in counts.tolist()]


@dataclass(eq=False)
class DatasetSplits:
    """Episode-disjoint dataset subsets and their source-row indices."""

    train: ResidualDataset
    validation: ResidualDataset
    test: ResidualDataset
    unseen: ResidualDataset
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    unseen_indices: np.ndarray
    seed: int = 0
    unseen_disturbance_types: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        seen: set = set()
        for name in ("train", "validation", "test", "unseen"):
            indices_name = "{}_indices".format(name)
            indices = _integer_vector(getattr(self, indices_name), indices_name)
            if len(indices) != len(getattr(self, name)):
                raise ValueError("{} length does not match its dataset".format(indices_name))
            overlap = seen.intersection(int(value) for value in indices.tolist())
            if overlap:
                raise ValueError("split indices overlap: {}".format(sorted(overlap)))
            seen.update(int(value) for value in indices.tolist())
            setattr(self, indices_name, indices)
        self.seed = int(self.seed)
        self.unseen_disturbance_types = tuple(
            sorted(str(value) for value in self.unseen_disturbance_types)
        )

        episode_sets = []
        for dataset in (self.train, self.validation, self.test, self.unseen):
            episode_sets.append({_stable_scalar(value) for value in dataset.episode_id.tolist()})
        for left in range(len(episode_sets)):
            for right in range(left + 1, len(episode_sets)):
                if episode_sets[left].intersection(episode_sets[right]):
                    raise ValueError("episode leakage detected between dataset splits")

    @property
    def val(self) -> ResidualDataset:
        return self.validation

    @property
    def val_indices(self) -> np.ndarray:
        return self.validation_indices

    def __getitem__(self, name: str) -> ResidualDataset:
        if name == "val":
            name = "validation"
        if name not in ("train", "validation", "test", "unseen"):
            raise KeyError(name)
        return getattr(self, name)

    def save(self, directory: Any, prefix: str = "dataset") -> Dict[str, Any]:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        if not isinstance(prefix, str) or not prefix:
            raise ValueError("prefix must be a non-empty string")
        artifacts: Dict[str, Any] = {}
        for name in ("train", "validation", "test", "unseen"):
            artifacts[name] = getattr(self, name).save(
                destination / "{}_{}.npz".format(prefix, name)
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "seed": self.seed,
            "unseen_disturbance_types": list(self.unseen_disturbance_types),
            "indices": {
                name: getattr(self, "{}_indices".format(name)).tolist()
                for name in ("train", "validation", "test", "unseen")
            },
        }
        manifest_path = destination / "{}_splits.json".format(prefix)
        tmp_path = manifest_path.with_name(
            manifest_path.name + ".tmp-{}".format(uuid.uuid4().hex)
        )
        try:
            _write_json_file(tmp_path, manifest)
            os.replace(str(tmp_path), str(manifest_path))
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
        artifacts["manifest"] = manifest_path
        return artifacts


def split_by_episode(
    dataset: ResidualDataset,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    unseen_disturbance_types: Sequence[str] = (),
    seed: int = 0,
    train_fraction: Optional[float] = None,
) -> DatasetSplits:
    """Deterministically split whole episodes, optionally holding out types."""

    if not isinstance(dataset, ResidualDataset):
        raise TypeError("dataset must be a ResidualDataset")
    validation = _validate_fraction(validation_fraction, "validation_fraction")
    test = _validate_fraction(test_fraction, "test_fraction")
    if train_fraction is None:
        train = 1.0 - validation - test
        if train < -1.0e-12:
            raise ValueError("validation_fraction + test_fraction cannot exceed 1")
        train = max(0.0, train)
    else:
        train = _validate_fraction(train_fraction, "train_fraction")
        if not np.isclose(train + validation + test, 1.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("train, validation, and test fractions must sum to 1")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if unseen_disturbance_types is None:
        unseen_disturbance_types = ()
    try:
        raw_unseen_names = tuple(unseen_disturbance_types)
    except TypeError as exc:
        raise TypeError("unseen_disturbance_types must be an iterable of strings") from exc
    if any(not isinstance(value, (str, np.str_)) for value in raw_unseen_names):
        raise TypeError("unseen_disturbance_types must contain strings")
    unseen_names = tuple(sorted({str(value) for value in raw_unseen_names}))
    unseen_set = set(unseen_names)

    episode_rows: Dict[str, List[int]] = {}
    episode_labels: Dict[str, Any] = {}
    for row, value in enumerate(dataset.episode_id.tolist()):
        key = _stable_scalar(value)
        episode_rows.setdefault(key, []).append(row)
        episode_labels[key] = value

    unseen_keys = set()
    for key, rows in episode_rows.items():
        if any(str(dataset.disturbance_type[row]) in unseen_set for row in rows):
            unseen_keys.add(key)
    seen_keys = [key for key in episode_rows if key not in unseen_keys]

    def order_key(key: str) -> str:
        return hashlib.sha256(
            "{}\0{}".format(seed, key).encode("utf-8")
        ).hexdigest()

    seen_keys.sort(key=lambda key: (order_key(key), key))
    counts = _group_counts(len(seen_keys), (train, validation, test))
    train_end = counts[0]
    validation_end = train_end + counts[1]
    key_groups = {
        "train": seen_keys[:train_end],
        "validation": seen_keys[train_end:validation_end],
        "test": seen_keys[validation_end:],
        "unseen": sorted(unseen_keys, key=lambda key: (order_key(key), key)),
    }

    index_groups: Dict[str, np.ndarray] = {}
    subsets: Dict[str, ResidualDataset] = {}
    for name, keys in key_groups.items():
        selected_set = set(keys)
        indices = np.asarray(
            [
                row
                for row, value in enumerate(dataset.episode_id.tolist())
                if _stable_scalar(value) in selected_set
            ],
            dtype=np.int64,
        )
        index_groups[name] = indices
        subsets[name] = dataset.subset(indices, metadata_updates={"split": name})

    return DatasetSplits(
        train=subsets["train"],
        validation=subsets["validation"],
        test=subsets["test"],
        unseen=subsets["unseen"],
        train_indices=index_groups["train"],
        validation_indices=index_groups["validation"],
        test_indices=index_groups["test"],
        unseen_indices=index_groups["unseen"],
        seed=seed,
        unseen_disturbance_types=unseen_names,
    )


def contiguous_rollout_windows(
    dataset: ResidualDataset,
    horizon: int,
    stride: int = 1,
    require_time_continuity: bool = True,
    require_constant_dt: bool = False,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
) -> np.ndarray:
    """Return ``(window_count, horizon)`` source-row index windows.

    Rows are ordered logically by step inside each episode, not by file order.
    Every adjacent pair must increment ``step`` by one. By default the next
    timestamp must equal ``time + dt``; optional constant-dt checking is useful
    for fixed-step rollout losses.
    """

    if not isinstance(dataset, ResidualDataset):
        raise TypeError("dataset must be a ResidualDataset")
    for value, name in ((horizon, "horizon"), (stride, "stride")):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise TypeError("{} must be a positive integer".format(name))
        if int(value) <= 0:
            raise ValueError("{} must be positive".format(name))
    horizon = int(horizon)
    stride = int(stride)
    for value, name in ((rtol, "rtol"), (atol, "atol")):
        if not np.isscalar(value) or not np.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("{} must be a finite nonnegative scalar".format(name))
    rtol = float(rtol)
    atol = float(atol)

    episodes: Dict[str, List[int]] = {}
    for index, episode in enumerate(dataset.episode_id.tolist()):
        episodes.setdefault(_stable_scalar(episode), []).append(index)

    windows: List[List[int]] = []
    for key in sorted(episodes):
        ordered = sorted(
            episodes[key],
            key=lambda index: (
                int(dataset.step[index]),
                float(dataset.time[index]),
                index,
            ),
        )
        runs: List[List[int]] = []
        current_run: List[int] = []
        for index in ordered:
            if not current_run:
                current_run = [index]
                continue
            previous = current_run[-1]
            adjacent = int(dataset.step[index]) == int(dataset.step[previous]) + 1
            if adjacent and require_time_continuity:
                expected_time = float(dataset.time[previous] + dataset.dt[previous])
                adjacent = bool(
                    np.isclose(dataset.time[index], expected_time, rtol=rtol, atol=atol)
                )
            if adjacent and require_constant_dt:
                adjacent = bool(
                    np.isclose(
                        dataset.dt[index], dataset.dt[previous], rtol=rtol, atol=atol
                    )
                )
            if adjacent:
                current_run.append(index)
            else:
                runs.append(current_run)
                current_run = [index]
        if current_run:
            runs.append(current_run)

        for run in runs:
            if len(run) < horizon:
                continue
            for start in range(0, len(run) - horizon + 1, stride):
                windows.append(run[start : start + horizon])

    if not windows:
        return np.empty((0, horizon), dtype=np.int64)
    return np.ascontiguousarray(np.asarray(windows, dtype=np.int64))


# Concise discoverable aliases.
group_split = split_by_episode
rollout_window_indices = contiguous_rollout_windows


__all__ = [
    "SCHEMA_VERSION",
    "REQUIRED_FIELDS",
    "DatasetSplits",
    "ResidualDataset",
    "compute_observed_derivative",
    "contiguous_rollout_windows",
    "group_split",
    "rollout_window_indices",
    "split_by_episode",
    "wrapped_finite_difference",
]
