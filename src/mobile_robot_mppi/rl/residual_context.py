"""Causal ICODE context exposed to a residual-conditioned RL policy.

Only quantities available before the current control decision are used.  In
particular, the innovation channels summarize the most recently *completed*
prediction/execution transition; simulator domain labels and future plant
states are never policy inputs.
"""

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ResidualContextConfig:
    enabled: bool = False
    state_names: Tuple[str, ...] = ("v", "omega")
    residual_scales: Tuple[float, ...] = (0.25, 0.60)
    innovation_scales: Tuple[float, ...] = (0.25, 0.60)
    signed_clip: float = 5.0
    disagreement_clip: float = 5.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            state_names=tuple(str(v) for v in values.get(
                "state_names", ("v", "omega")
            )),
            residual_scales=tuple(float(v) for v in values.get(
                "residual_scales", (0.25, 0.60)
            )),
            innovation_scales=tuple(float(v) for v in values.get(
                "innovation_scales", (0.25, 0.60)
            )),
            signed_clip=float(values.get("signed_clip", 5.0)),
            disagreement_clip=float(values.get("disagreement_clip", 5.0)),
        )

    @property
    def dimension(self):
        # residual, signed completed-transition innovation, disagreement,
        # support confidence, and innovation-valid flag.
        return 2 * len(self.state_names) + 3

    def validate(self):
        count = len(self.state_names)
        if count <= 0 or len(set(self.state_names)) != count:
            raise ValueError("residual context state_names must be unique and nonempty")
        for name, values in (
            ("residual_scales", self.residual_scales),
            ("innovation_scales", self.innovation_scales),
        ):
            array = np.asarray(values, dtype=np.float64)
            if (
                array.shape != (count,)
                or not np.isfinite(array).all()
                or np.any(array <= 0.0)
            ):
                raise ValueError(
                    "residual context %s must be finite, positive, and align "
                    "with state_names" % name
                )
        if (
            not np.isfinite(self.signed_clip)
            or not np.isfinite(self.disagreement_clip)
            or self.signed_clip <= 0.0
            or self.disagreement_clip <= 0.0
        ):
            raise ValueError("residual context clips must be finite and positive")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ResidualCorrectionAuthorityConfig:
    """Causal authority assigned to a residual-conditioned Actor correction."""

    enabled: bool = False
    innovation_onset: float = 0.03
    innovation_full: float = 0.08
    support_power: float = 1.0
    minimum_authority: float = 0.0
    maximum_authority: float = 1.0

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, Any]] = None):
        values = dict(values or {})
        return cls(
            enabled=bool(values.get("enabled", False)),
            innovation_onset=float(values.get("innovation_onset", 0.03)),
            innovation_full=float(values.get("innovation_full", 0.08)),
            support_power=float(values.get("support_power", 1.0)),
            minimum_authority=float(values.get("minimum_authority", 0.0)),
            maximum_authority=float(values.get("maximum_authority", 1.0)),
        )

    def validate(self):
        values = np.asarray((
            self.innovation_onset,
            self.innovation_full,
            self.support_power,
            self.minimum_authority,
            self.maximum_authority,
        ), dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("residual correction authority must be finite")
        if self.innovation_onset < 0.0 or self.innovation_full <= self.innovation_onset:
            raise ValueError("authority innovation_full must exceed nonnegative onset")
        if self.support_power < 0.0:
            raise ValueError("authority support_power must be nonnegative")
        if not (0.0 <= self.minimum_authority <= self.maximum_authority <= 1.0):
            raise ValueError("authority bounds must satisfy 0 <= min <= max <= 1")


class ResidualCorrectionAuthority:
    """Map prior completed-transition innovation to safe correction authority."""

    def __init__(self, context_dimension, config=None):
        self.config = (
            config
            if isinstance(config, ResidualCorrectionAuthorityConfig)
            else ResidualCorrectionAuthorityConfig.from_mapping(config)
        )
        self.config.validate()
        self.context_dimension = int(context_dimension)
        self.state_count = (self.context_dimension - 3) // 2
        if self.state_count <= 0 or 2 * self.state_count + 3 != self.context_dimension:
            raise ValueError("residual authority context dimension is invalid")

    def evaluate(self, context_features):
        features = np.asarray(context_features, dtype=np.float64)
        if (
            features.ndim != 2
            or features.shape[1] != self.context_dimension
            or not np.isfinite(features).all()
        ):
            raise ValueError("residual authority features must be finite [B,C]")
        innovation = features[:, self.state_count : 2 * self.state_count]
        magnitude = np.mean(np.abs(innovation), axis=1)
        support = np.clip(features[:, -2], 0.0, 1.0)
        valid = features[:, -1] > 0.5
        scaled = np.clip(
            (magnitude - self.config.innovation_onset)
            / (self.config.innovation_full - self.config.innovation_onset),
            0.0,
            1.0,
        )
        scaled *= support ** self.config.support_power
        authority = (
            self.config.minimum_authority
            + (self.config.maximum_authority - self.config.minimum_authority)
            * scaled
        )
        authority = np.where(valid, authority, 0.0)
        if not np.isfinite(authority).all():
            raise FloatingPointError("residual correction authority is invalid")
        return authority.astype(np.float64, copy=False)


def _residual_chain(residual):
    """Yield transparent residual wrappers without assuming their concrete type."""

    seen = set()
    current = residual
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        child = getattr(current, "residual", None)
        if child is current:
            break
        current = child


class ResidualContextEncoder:
    """Build the seven-dimensional causal context used by the new Actor."""

    def __init__(self, dynamics, state_spec, config=None):
        self.config = (
            config
            if isinstance(config, ResidualContextConfig)
            else ResidualContextConfig.from_mapping(config)
        )
        self.config.validate()
        if not self.config.enabled:
            raise ValueError("ResidualContextEncoder requires enabled=true")
        self.dynamics = dynamics
        self.state_spec = state_spec
        self.residual = getattr(dynamics, "residual", None)
        if self.residual is None:
            raise ValueError("residual-conditioned policy requires residual dynamics")
        unknown = [
            name for name in self.config.state_names
            if name not in self.state_spec.names
        ]
        if unknown:
            raise ValueError("residual context states are unavailable: %s" % unknown)
        self.state_indices = np.asarray(
            [self.state_spec.index(name) for name in self.config.state_names],
            dtype=np.int64,
        )
        self.residual_scales = np.asarray(
            self.config.residual_scales, dtype=np.float64
        )
        self.innovation_scales = np.asarray(
            self.config.innovation_scales, dtype=np.float64
        )
        self.diagnostic_residual = None
        for candidate in _residual_chain(self.residual):
            if (
                callable(getattr(candidate, "disagreement", None))
                and callable(getattr(candidate, "support_confidence", None))
            ):
                self.diagnostic_residual = candidate
                break
        if self.diagnostic_residual is None:
            raise ValueError(
                "residual-conditioned policy requires an ensemble with "
                "disagreement and support confidence"
            )

    @property
    def dimension(self):
        return self.config.dimension

    def reset(self):
        reset = getattr(self.diagnostic_residual, "reset", None)
        if callable(reset):
            reset()

    def features(self, states, controls):
        states = np.asarray(states, dtype=np.float64)
        controls = np.asarray(controls, dtype=np.float64)
        unbatched = states.ndim == 1
        if unbatched:
            states = states[None, :]
            controls = controls[None, :]
        expected_controls = states.shape[:-1] + (int(self.dynamics.control_dim),)
        if (
            states.ndim != 2
            or states.shape[1] != int(self.dynamics.state_dim)
            or controls.shape != expected_controls
            or not np.isfinite(states).all()
            or not np.isfinite(controls).all()
        ):
            raise ValueError("residual context inputs must be aligned finite batches")

        residual = np.asarray(
            self.residual.derivative(states, controls), dtype=np.float64
        )[:, self.state_indices]
        residual = np.clip(
            residual / self.residual_scales,
            -self.config.signed_clip,
            self.config.signed_clip,
        )
        signed_ema = np.asarray(
            getattr(
                self.diagnostic_residual,
                "innovation_error_vector_ema",
                np.zeros(int(self.dynamics.state_dim), dtype=np.float64),
            ),
            dtype=np.float64,
        ).reshape(-1)
        samples = int(getattr(
            self.diagnostic_residual, "innovation_samples", 0
        ))
        if signed_ema.shape != (int(self.dynamics.state_dim),):
            raise ValueError("ensemble signed innovation has an invalid dimension")
        innovation = np.clip(
            signed_ema[self.state_indices] / self.innovation_scales,
            -self.config.signed_clip,
            self.config.signed_clip,
        )
        innovation = np.broadcast_to(innovation, residual.shape)
        disagreement = np.asarray(
            self.diagnostic_residual.disagreement(states, controls),
            dtype=np.float64,
        ).reshape(-1, 1)
        disagreement = np.clip(
            disagreement, 0.0, self.config.disagreement_clip
        )
        support = np.asarray(
            self.diagnostic_residual.support_confidence(states, controls),
            dtype=np.float64,
        ).reshape(-1, 1)
        support = np.clip(support, 0.0, 1.0)
        valid = np.full((states.shape[0], 1), float(samples > 0))
        result = np.concatenate(
            (residual, innovation, disagreement, support, valid), axis=1
        ).astype(np.float32, copy=False)
        if (
            result.shape != (states.shape[0], self.dimension)
            or not np.isfinite(result).all()
        ):
            raise FloatingPointError("residual policy context is invalid")
        return result[0] if unbatched else result
