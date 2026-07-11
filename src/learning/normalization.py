"""NumPy-only, train-split normalization for learned residual dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np


def _positive_epsilon(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError("epsilon must be a finite positive scalar")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("epsilon must be a finite positive scalar") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return result


def _fit_matrix(values: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric".format(name)) from exc
    if array.ndim != 2:
        raise ValueError("{} must have shape (N, D), got {}".format(name, array.shape))
    if array.shape[0] <= 0:
        raise ValueError("{} must contain at least one training row".format(name))
    if array.shape[1] <= 0:
        raise ValueError("{} feature dimension must be positive".format(name))
    if not np.all(np.isfinite(array)):
        raise ValueError("{} contains NaN or Inf".format(name))
    return array


class StandardNormalizer:
    """Feature-wise population standardization with safe constant features.

    Statistics are always fitted over axis zero of a two-dimensional training
    matrix. Features whose standard deviation is below ``epsilon`` receive a
    scale of one, avoiding division by zero while still centering them.
    """

    def __init__(self, epsilon: float = 1.0e-12) -> None:
        self.epsilon = _positive_epsilon(epsilon)
        self.mean_: Optional[np.ndarray] = None
        self.variance_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.constant_mask_: Optional[np.ndarray] = None
        self.sample_count_: int = 0

    @property
    def is_fitted(self) -> bool:
        return self.mean_ is not None

    @property
    def fitted(self) -> bool:
        return self.is_fitted

    @property
    def feature_dim(self) -> int:
        self._require_fitted()
        assert self.mean_ is not None
        return int(self.mean_.shape[0])

    @property
    def mean(self) -> np.ndarray:
        self._require_fitted()
        assert self.mean_ is not None
        return self.mean_

    @property
    def variance(self) -> np.ndarray:
        self._require_fitted()
        assert self.variance_ is not None
        return self.variance_

    @property
    def scale(self) -> np.ndarray:
        self._require_fitted()
        assert self.scale_ is not None
        return self.scale_

    @property
    def std(self) -> np.ndarray:
        return self.scale

    @property
    def std_(self) -> np.ndarray:
        """Scikit-learn-style alias for the effective feature scale."""

        return self.scale

    @property
    def constant_mask(self) -> np.ndarray:
        self._require_fitted()
        assert self.constant_mask_ is not None
        return self.constant_mask_

    @property
    def sample_count(self) -> int:
        return self.sample_count_

    def fit(self, values: Any) -> "StandardNormalizer":
        array = _fit_matrix(values, "values")
        mean = np.mean(array, axis=0, dtype=np.float64)
        variance = np.mean(np.square(array - mean), axis=0, dtype=np.float64)
        # Round-off can create a tiny negative only through unusual NumPy
        # implementations; clipping keeps the serialized state well-defined.
        variance = np.maximum(variance, 0.0)
        raw_scale = np.sqrt(variance)
        constant = raw_scale <= self.epsilon
        scale = np.where(constant, 1.0, raw_scale)
        self.mean_ = np.ascontiguousarray(mean)
        self.variance_ = np.ascontiguousarray(variance)
        self.scale_ = np.ascontiguousarray(scale)
        self.constant_mask_ = np.ascontiguousarray(constant, dtype=np.bool_)
        self.sample_count_ = int(array.shape[0])
        return self

    @classmethod
    def fitted_from(
        cls, values: Any, epsilon: float = 1.0e-12
    ) -> "StandardNormalizer":
        return cls(epsilon=epsilon).fit(values)

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("normalizer has not been fitted")

    def _values(self, values: Any, name: str) -> np.ndarray:
        self._require_fitted()
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("{} must be numeric".format(name)) from exc
        if array.ndim < 1:
            raise ValueError("{} must have a feature dimension".format(name))
        if array.shape[-1] != self.feature_dim:
            raise ValueError(
                "{} last dimension must be {}, got {}".format(
                    name, self.feature_dim, array.shape[-1]
                )
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("{} contains NaN or Inf".format(name))
        return array

    def transform(self, values: Any) -> np.ndarray:
        array = self._values(values, "values")
        assert self.mean_ is not None and self.scale_ is not None
        return np.ascontiguousarray((array - self.mean_) / self.scale_)

    def inverse_transform(self, values: Any) -> np.ndarray:
        array = self._values(values, "normalized values")
        assert self.mean_ is not None and self.scale_ is not None
        return np.ascontiguousarray(array * self.scale_ + self.mean_)

    inverse = inverse_transform

    def state_dict(self) -> Dict[str, Any]:
        self._require_fitted()
        assert self.mean_ is not None
        assert self.variance_ is not None
        assert self.scale_ is not None
        assert self.constant_mask_ is not None
        return {
            "epsilon": self.epsilon,
            "sample_count": self.sample_count_,
            "mean": self.mean_.tolist(),
            "variance": self.variance_.tolist(),
            "scale": self.scale_.tolist(),
            "constant_mask": self.constant_mask_.tolist(),
        }

    to_dict = state_dict

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "StandardNormalizer":
        if not isinstance(state, Mapping):
            raise TypeError("normalizer state must be a mapping")
        required = ("epsilon", "sample_count", "mean", "variance", "scale")
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError(
                "normalizer state is missing: {}".format(", ".join(missing))
            )
        instance = cls(epsilon=state["epsilon"])
        mean = cls._state_vector(state["mean"], "mean")
        variance = cls._state_vector(state["variance"], "variance")
        scale = cls._state_vector(state["scale"], "scale")
        if mean.size <= 0:
            raise ValueError("normalizer state must have at least one feature")
        if variance.shape != mean.shape or scale.shape != mean.shape:
            raise ValueError("normalizer state vectors must have matching shapes")
        if np.any(variance < 0.0) or np.any(scale <= 0.0):
            raise ValueError("normalizer variance/scale values are invalid")
        if "constant_mask" in state:
            constant = np.asarray(state["constant_mask"])
            if constant.shape != mean.shape or constant.dtype.kind != "b":
                raise ValueError("constant_mask must be a boolean feature vector")
            constant = constant.astype(np.bool_, copy=False)
        else:
            constant = np.sqrt(variance) <= instance.epsilon
        count = state["sample_count"]
        if isinstance(count, (bool, np.bool_)) or not isinstance(
            count, (int, np.integer)
        ):
            raise TypeError("sample_count must be a positive integer")
        count = int(count)
        if count <= 0:
            raise ValueError("sample_count must be positive")
        instance.mean_ = np.ascontiguousarray(mean)
        instance.variance_ = np.ascontiguousarray(variance)
        instance.scale_ = np.ascontiguousarray(scale)
        instance.constant_mask_ = np.ascontiguousarray(constant)
        instance.sample_count_ = count
        return instance

    @staticmethod
    def _state_vector(values: Any, name: str) -> np.ndarray:
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("{} must be numeric".format(name)) from exc
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("{} must be a finite feature vector".format(name))
        return array

    from_dict = from_state_dict


def _encode(encoder: Any, states: Any) -> np.ndarray:
    operation = getattr(encoder, "encode", None)
    if operation is None:
        operation = encoder if callable(encoder) else None
    if operation is None:
        raise TypeError("state_encoder must be callable or provide encode()")
    encoded = operation(states)
    try:
        array = np.asarray(encoded, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("state encoder output must be numeric") from exc
    if array.ndim not in (1, 2):
        raise ValueError("state encoder output must have shape (D,) or (N, D)")
    if array.shape[-1] <= 0 or not np.all(np.isfinite(array)):
        raise ValueError("state encoder output must be finite and nonempty")
    return array


def _encoder_config(encoder: Any) -> Dict[str, Any]:
    if encoder is None:
        return {"mode": "pre_encoded"}
    method = getattr(encoder, "to_config", None)
    if callable(method):
        config = method()
        if not isinstance(config, Mapping):
            raise TypeError("state_encoder.to_config() must return a mapping")
        return _json_state(dict(config), "encoder_config")
    return {
        "mode": "external",
        "class": "{}.{}".format(
            type(encoder).__module__, type(encoder).__qualname__
        ),
    }


def _json_state(value: Any, name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        result = float(value)
        if not np.isfinite(result):
            raise ValueError("{} contains NaN or Inf".format(name))
        return result
    if isinstance(value, np.ndarray):
        return _json_state(value.tolist(), name)
    if isinstance(value, (list, tuple)):
        return [_json_state(item, name) for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("{} keys must be strings".format(name))
            result[key] = _json_state(item, name)
        return result
    raise TypeError("{} is not JSON serializable".format(name))


def _column(source: Any, name: str) -> Any:
    if isinstance(source, Mapping):
        if name not in source:
            raise ValueError("training data is missing '{}'".format(name))
        return source[name]
    if not hasattr(source, name):
        raise TypeError("training data must be a dataset or mapping")
    return getattr(source, name)


@dataclass(eq=False)
class NormalizerBundle:
    """Normalizer triplet fitted exclusively from the supplied training rows."""

    state: StandardNormalizer
    control: StandardNormalizer
    residual: StandardNormalizer
    encoder_config: Mapping[str, Any] = field(default_factory=dict)
    state_encoder: Optional[Any] = field(default=None, repr=False, compare=False)
    fit_sample_count: int = 0

    def __post_init__(self) -> None:
        for value, name in (
            (self.state, "state"),
            (self.control, "control"),
            (self.residual, "residual"),
        ):
            if not isinstance(value, StandardNormalizer) or not value.is_fitted:
                raise TypeError("{} must be a fitted StandardNormalizer".format(name))
        self.encoder_config = _json_state(dict(self.encoder_config), "encoder_config")
        if self.fit_sample_count == 0:
            self.fit_sample_count = self.state.sample_count
        if isinstance(self.fit_sample_count, (bool, np.bool_)) or not isinstance(
            self.fit_sample_count, (int, np.integer)
        ):
            raise TypeError("fit_sample_count must be a positive integer")
        self.fit_sample_count = int(self.fit_sample_count)
        if self.fit_sample_count <= 0:
            raise ValueError("fit_sample_count must be positive")
        counts = {
            self.state.sample_count,
            self.control.sample_count,
            self.residual.sample_count,
            self.fit_sample_count,
        }
        if len(counts) != 1:
            raise ValueError("all bundle normalizers must use the same training rows")

    @classmethod
    def fit(
        cls,
        train_data: Optional[Any] = None,
        state_encoder: Optional[Any] = None,
        encoded_state: Optional[Any] = None,
        control: Optional[Any] = None,
        residual: Optional[Any] = None,
        epsilon: float = 1.0e-12,
    ) -> "NormalizerBundle":
        """Fit from a training dataset/mapping or three explicit matrices.

        If ``state_encoder`` is supplied, it is applied to ``state_t``. Pass
        ``encoded_state`` instead when features were encoded upstream. Control
        defaults to ``control_t`` and residual defaults to ``residual_target``.
        """

        epsilon = _positive_epsilon(epsilon)
        if train_data is None:
            if encoded_state is None or control is None or residual is None:
                raise ValueError(
                    "explicit fitting requires encoded_state, control, and residual"
                )
            if state_encoder is not None:
                raise ValueError(
                    "state_encoder cannot be combined with pre-encoded explicit state"
                )
            state_values = encoded_state
        else:
            if encoded_state is not None and state_encoder is not None:
                raise ValueError("provide state_encoder or encoded_state, not both")
            raw_state = _column(train_data, "state_t")
            if encoded_state is not None:
                state_values = encoded_state
            elif state_encoder is not None:
                state_values = _encode(state_encoder, raw_state)
            else:
                state_values = raw_state
            if control is None:
                # The learned residual is queried by the planner with its
                # commanded control.  Delay-induced differences are therefore
                # part of the target (and are explicitly marked non-Markov),
                # while applied_control_t remains diagnostic plant metadata.
                control = _column(train_data, "control_t")
            if residual is None:
                residual = _column(train_data, "residual_target")

        state_matrix = _fit_matrix(state_values, "encoded_state")
        control_matrix = _fit_matrix(control, "control")
        residual_matrix = _fit_matrix(residual, "residual")
        row_count = int(state_matrix.shape[0])
        if control_matrix.shape[0] != row_count or residual_matrix.shape[0] != row_count:
            raise ValueError("state, control, and residual training row counts must match")
        config = _encoder_config(state_encoder)
        if state_encoder is None and encoded_state is None and train_data is not None:
            config = {"mode": "raw", "output_dim": int(state_matrix.shape[1])}
        elif state_encoder is None:
            config = {"mode": "pre_encoded", "output_dim": int(state_matrix.shape[1])}
        return cls(
            state=StandardNormalizer(epsilon).fit(state_matrix),
            control=StandardNormalizer(epsilon).fit(control_matrix),
            residual=StandardNormalizer(epsilon).fit(residual_matrix),
            encoder_config=config,
            state_encoder=state_encoder,
            fit_sample_count=row_count,
        )

    @classmethod
    def from_training_data(
        cls,
        train_data: Any,
        state_encoder: Optional[Any] = None,
        epsilon: float = 1.0e-12,
    ) -> "NormalizerBundle":
        return cls.fit(
            train_data=train_data, state_encoder=state_encoder, epsilon=epsilon
        )

    @property
    def state_normalizer(self) -> StandardNormalizer:
        return self.state

    @property
    def control_normalizer(self) -> StandardNormalizer:
        return self.control

    @property
    def residual_normalizer(self) -> StandardNormalizer:
        return self.residual

    def encode_state(self, values: Any) -> np.ndarray:
        if self.state_encoder is None:
            try:
                array = np.asarray(values, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError("state values must be numeric") from exc
            return array
        return _encode(self.state_encoder, values)

    def transform_state(self, values: Any, pre_encoded: bool = False) -> np.ndarray:
        encoded = values if pre_encoded else self.encode_state(values)
        return self.state.transform(encoded)

    def inverse_state(self, values: Any) -> np.ndarray:
        """Undo normalization, returning encoded (not raw-angle) state features."""

        return self.state.inverse_transform(values)

    inverse_transform_state = inverse_state

    def transform_control(self, values: Any) -> np.ndarray:
        return self.control.transform(values)

    def inverse_control(self, values: Any) -> np.ndarray:
        return self.control.inverse_transform(values)

    inverse_transform_control = inverse_control

    def transform_residual(self, values: Any) -> np.ndarray:
        return self.residual.transform(values)

    def inverse_residual(self, values: Any) -> np.ndarray:
        return self.residual.inverse_transform(values)

    inverse_transform_residual = inverse_residual

    def transform_inputs(self, states: Any, controls: Any) -> np.ndarray:
        normalized_state = self.transform_state(states)
        normalized_control = self.transform_control(controls)
        if normalized_state.ndim != normalized_control.ndim:
            raise ValueError("state and control batches must have matching ranks")
        if normalized_state.shape[:-1] != normalized_control.shape[:-1]:
            raise ValueError("state and control batch shapes must match")
        return np.ascontiguousarray(
            np.concatenate((normalized_state, normalized_control), axis=-1)
        )

    def state_dict(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "fit_sample_count": self.fit_sample_count,
            "encoder_config": _json_state(self.encoder_config, "encoder_config"),
            "state": self.state.state_dict(),
            "control": self.control.state_dict(),
            "residual": self.residual.state_dict(),
        }

    to_dict = state_dict

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        state_encoder: Optional[Any] = None,
    ) -> "NormalizerBundle":
        if not isinstance(state, Mapping):
            raise TypeError("bundle state must be a mapping")
        required = ("state", "control", "residual", "encoder_config")
        missing = [name for name in required if name not in state]
        if missing:
            raise ValueError("bundle state is missing: {}".format(", ".join(missing)))
        config = state["encoder_config"]
        if not isinstance(config, Mapping):
            raise TypeError("encoder_config must be a mapping")
        if state_encoder is None:
            state_encoder = _rebuild_known_encoder(config)
        state_normalizer = StandardNormalizer.from_state_dict(state["state"])
        fit_count = state.get("fit_sample_count", state_normalizer.sample_count)
        return cls(
            state=state_normalizer,
            control=StandardNormalizer.from_state_dict(state["control"]),
            residual=StandardNormalizer.from_state_dict(state["residual"]),
            encoder_config=dict(config),
            state_encoder=state_encoder,
            fit_sample_count=fit_count,
        )

    from_dict = from_state_dict


def _rebuild_known_encoder(config: Mapping[str, Any]) -> Optional[Any]:
    mode = config.get("mode")
    if mode not in ("raw", "sincos") or "state_dim" not in config:
        return None
    try:
        from ..dynamics.state_encoding import StateEncoder
    except (ImportError, ValueError):
        try:
            from dynamics.state_encoding import StateEncoder  # type: ignore
        except ImportError:
            return None
    return StateEncoder(
        mode=str(mode),
        state_dim=int(config["state_dim"]),
        angle_indices=tuple(config.get("angle_indices", ())),
    )


__all__ = ["NormalizerBundle", "StandardNormalizer"]
