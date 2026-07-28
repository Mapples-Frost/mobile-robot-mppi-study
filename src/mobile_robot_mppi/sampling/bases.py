"""Noise bases for MPPI control-sequence sampling.

Every basis here is *marginal-variance preserving*: for a requested per-dimension
sigma, each generated perturbation eps[k, h, d] has marginal variance sigma[d]**2
at every horizon step h. Bases therefore differ only in their temporal
correlation structure, never in per-step magnitude.

That property is what makes basis comparisons honest. A correlated basis that
also inflated the marginal would trivially reach further than i.i.d. noise, and
the comparison would say nothing about temporal structure.

The i.i.d. basis reproduces the current production sampler at
``planning/mppi.py:1653`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "NoiseBasis",
    "IidBasis",
    "PiecewiseConstantBasis",
    "Ar1Basis",
    "MixtureBasis",
    "build_basis",
    "verify_marginal_variance",
]


@dataclass(frozen=True)
class NoiseBasis:
    """Base class. Subclasses implement ``sample``."""

    name: str

    def sample(
        self,
        rng: np.random.Generator,
        count: int,
        horizon: int,
        sigma: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class IidBasis(NoiseBasis):
    """White noise, independent at every horizon step.

    This is the current production behaviour.
    """

    name: str = "iid"

    def sample(self, rng, count, horizon, sigma):
        sigma = np.asarray(sigma, dtype=np.float64)
        return rng.normal(size=(count, horizon, sigma.shape[0])) * sigma[None, None, :]


@dataclass(frozen=True)
class PiecewiseConstantBasis(NoiseBasis):
    """Constant perturbation within each of ``segments`` contiguous blocks.

    Mirrors the multi-stage ``(v, omega)`` parameterisation used by the offline
    oracle in ``complex_supervised_actor_oracle_gate_v4``, which is known to
    find feasible maneuvers the online sampler misses.

    Marginal variance is exactly sigma**2 because each step takes the value of a
    single N(0, sigma**2) draw.
    """

    name: str = "piecewise"
    segments: int = 4

    def sample(self, rng, count, horizon, sigma):
        sigma = np.asarray(sigma, dtype=np.float64)
        segments = max(1, min(int(self.segments), int(horizon)))
        knots = rng.normal(size=(count, segments, sigma.shape[0])) * sigma[None, None, :]
        edges = np.linspace(0, horizon, segments + 1).astype(int)
        out = np.empty((count, horizon, sigma.shape[0]), dtype=np.float64)
        for index in range(segments):
            start, stop = edges[index], edges[index + 1]
            if stop > start:
                out[:, start:stop, :] = knots[:, index : index + 1, :]
        return out


@dataclass(frozen=True)
class Ar1Basis(NoiseBasis):
    """Stationary AR(1) / discretised Ornstein-Uhlenbeck perturbation.

    eps[h] = rho * eps[h-1] + sqrt(1 - rho**2) * sigma * eta[h]

    with eps[0] ~ N(0, sigma**2). The sqrt(1 - rho**2) factor is what keeps the
    stationary marginal variance at exactly sigma**2 for any rho, so increasing
    correlation never smuggles in extra magnitude.

    ``correlation_time_s`` is converted to rho via rho = exp(-dt / tau).
    """

    name: str = "ar1"
    correlation_time_s: float = 1.0
    dt: float = 0.1

    @property
    def rho(self) -> float:
        tau = max(float(self.correlation_time_s), 1e-9)
        return float(np.exp(-float(self.dt) / tau))

    def sample(self, rng, count, horizon, sigma):
        sigma = np.asarray(sigma, dtype=np.float64)
        rho = self.rho
        innovation = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
        eta = rng.normal(size=(count, horizon, sigma.shape[0]))
        out = np.empty_like(eta)
        out[:, 0, :] = eta[:, 0, :]
        for step in range(1, horizon):
            out[:, step, :] = rho * out[:, step - 1, :] + innovation * eta[:, step, :]
        return out * sigma[None, None, :]


@dataclass(frozen=True)
class MixtureBasis(NoiseBasis):
    """Allocate the rollout budget across several bases.

    Preserves total sample count. Each component keeps its own marginal
    variance, so the mixture marginal is also sigma**2.
    """

    name: str = "mixture"
    components: Sequence[NoiseBasis] = ()
    weights: Sequence[float] = ()

    def sample(self, rng, count, horizon, sigma):
        sigma = np.asarray(sigma, dtype=np.float64)
        if not self.components:
            raise ValueError("mixture basis requires at least one component")
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.shape[0] != len(self.components) or np.any(weights < 0):
            raise ValueError("weights must be non-negative and match components")
        weights = weights / weights.sum()
        counts = np.floor(weights * count).astype(int)
        counts[-1] = count - counts[:-1].sum()
        blocks = [
            component.sample(rng, int(size), horizon, sigma)
            for component, size in zip(self.components, counts)
            if size > 0
        ]
        return np.concatenate(blocks, axis=0)


def build_basis(spec: str, dt: float = 0.1) -> NoiseBasis:
    """Construct a basis from a string spec.

    Accepted forms:

    ``iid``               white noise; the production sampler
    ``piecewise:<k>``     constant within each of k contiguous segments
    ``ar1:<tau_seconds>`` stationary AR(1), correlation time in seconds

    Raises ValueError on anything else, so a typo in a config fails loudly at
    validation time rather than silently falling back to a different sampler.
    """

    text = str(spec).strip().lower()
    if text == "iid":
        return IidBasis()
    if ":" in text:
        head, _, tail = text.partition(":")
        if head == "piecewise":
            segments = int(tail)
            if segments < 1:
                raise ValueError("piecewise basis needs at least one segment")
            return PiecewiseConstantBasis(segments=segments)
        if head == "ar1":
            tau = float(tail)
            if not np.isfinite(tau) or tau <= 0.0:
                raise ValueError("ar1 correlation time must be positive and finite")
            return Ar1Basis(correlation_time_s=tau, dt=float(dt))
    raise ValueError(
        "unknown noise basis spec %r; expected 'iid', 'piecewise:<k>', "
        "or 'ar1:<tau_seconds>'" % (spec,)
    )


def verify_marginal_variance(
    basis: NoiseBasis,
    sigma: np.ndarray,
    horizon: int = 36,
    count: int = 200_000,
    seed: int = 0,
    tolerance: float = 0.03,
) -> dict:
    """Empirically confirm the basis preserves per-step marginal variance.

    Returns the worst relative deviation across all horizon steps and dims.
    Any basis used in a comparison must pass this, otherwise differences in
    reach are confounded with differences in magnitude.
    """

    sigma = np.asarray(sigma, dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = basis.sample(rng, count, horizon, sigma)
    empirical = samples.std(axis=0)
    relative = np.abs(empirical - sigma[None, :]) / sigma[None, :]
    return {
        "basis": basis.name,
        "max_relative_deviation": float(relative.max()),
        "passes": bool(relative.max() <= tolerance),
        "tolerance": float(tolerance),
    }
