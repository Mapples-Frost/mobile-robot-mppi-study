"""Periodic-aware state feature encoding for residual models."""

from __future__ import annotations

from typing import Any, Sequence, Tuple

import numpy as np

from .interfaces import validate_angle_indices


class StateEncoder:
    """Encode raw states while representing periodic angles continuously.

    ``raw`` preserves all state components.  ``sincos`` replaces every angle
    component with ``sin(angle), cos(angle)`` at the same logical position.
    For the default unicycle state this maps ``[x, y, theta]`` to
    ``[x, y, sin(theta), cos(theta)]``.  Residual outputs remain derivatives in
    raw state coordinates; this class only transforms model inputs.
    """

    SUPPORTED_MODES = ("raw", "sincos")

    def __init__(
        self,
        mode: str = "raw",
        state_dim: int = 3,
        angle_indices: Sequence[int] = (2,),
    ) -> None:
        if isinstance(state_dim, (bool, np.bool_)) or not isinstance(
            state_dim, (int, np.integer)
        ):
            raise TypeError("state_dim must be a positive integer")
        self.state_dim = int(state_dim)
        if self.state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if not isinstance(mode, str):
            raise TypeError("mode must be a string")
        self.mode = mode.strip().lower()
        if self.mode not in self.SUPPORTED_MODES:
            raise ValueError(
                "unknown state encoding {!r}; expected one of {}".format(
                    mode, self.SUPPORTED_MODES
                )
            )
        self.angle_indices: Tuple[int, ...] = validate_angle_indices(
            angle_indices, self.state_dim
        )
        self.output_dim = self.state_dim + (
            len(self.angle_indices) if self.mode == "sincos" else 0
        )

    @property
    def feature_dim(self) -> int:
        """Alias used by learned residual model builders."""

        return self.output_dim

    def encode(self, states: Any) -> np.ndarray:
        """Encode a single ``(nx,)`` state or a batch ``(batch, nx)``."""

        try:
            array = np.asarray(states, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("states must be numeric") from exc
        if array.ndim not in (1, 2):
            raise ValueError(
                "states must have shape (state_dim,) or (batch, state_dim), got {}".format(
                    array.shape
                )
            )
        if array.shape[-1] != self.state_dim:
            raise ValueError(
                "states last dimension must be {}, got {}".format(
                    self.state_dim, array.shape[-1]
                )
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("states contain NaN or Inf")
        if self.mode == "raw":
            return array.copy()

        angle_set = set(self.angle_indices)
        features = []
        for index in range(self.state_dim):
            component = array[..., index : index + 1]
            if index in angle_set:
                features.extend((np.sin(component), np.cos(component)))
            else:
                features.append(component)
        return np.concatenate(features, axis=-1)

    def __call__(self, states: Any) -> np.ndarray:
        return self.encode(states)

    def to_config(self) -> dict:
        """Return JSON-serializable encoder metadata for checkpoints."""

        return {
            "mode": self.mode,
            "state_dim": self.state_dim,
            "angle_indices": list(self.angle_indices),
            "output_dim": self.output_dim,
        }

