"""Low-dimensional constrained contextual bandit for MPPI rollout budgets."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BudgetDecision:
    add_samples: bool
    predicted_advantage: float
    confidence_width: float
    dual_price: float
    score: float
    exploratory: bool


class PrimalDualBudgetBandit:
    """Choose STOP/ADD with a learned utility model and compute constraint.

    The contextual reward of ADD is the improvement over STOP. STOP has zero
    advantage by definition. During online training the advantage model is
    updated only when ADD is selected; the dual variable is updated after every
    decision to enforce a target long-run ADD rate. Deployment is deterministic.
    """

    def __init__(
        self,
        feature_names: Sequence[str],
        feature_mean,
        feature_scale,
        ridge=1.0,
        exploration_alpha=0.25,
        epsilon=0.10,
        target_add_fraction=0.50,
        dual_learning_rate=0.02,
        maximum_dual_price=0.25,
        seed=0,
    ):
        self.feature_names = tuple(str(value) for value in feature_names)
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("budget-bandit feature names must be unique and non-empty")
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64).reshape(-1)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64).reshape(-1)
        if (
            self.feature_mean.shape != (len(self.feature_names),)
            or self.feature_scale.shape != self.feature_mean.shape
            or not np.isfinite(self.feature_mean).all()
            or not np.isfinite(self.feature_scale).all()
            or np.any(self.feature_scale <= 0.0)
        ):
            raise ValueError("budget-bandit normalization is invalid")
        self.ridge = float(ridge)
        self.exploration_alpha = float(exploration_alpha)
        self.epsilon = float(epsilon)
        self.target_add_fraction = float(target_add_fraction)
        self.dual_learning_rate = float(dual_learning_rate)
        self.maximum_dual_price = float(maximum_dual_price)
        if not np.isfinite((
            self.ridge, self.exploration_alpha, self.epsilon,
            self.target_add_fraction, self.dual_learning_rate,
            self.maximum_dual_price,
        )).all():
            raise ValueError("budget-bandit parameters must be finite")
        if self.ridge <= 0.0 or self.exploration_alpha < 0.0:
            raise ValueError("ridge must be positive and exploration non-negative")
        if not 0.0 <= self.epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        if not 0.0 < self.target_add_fraction < 1.0:
            raise ValueError("target ADD fraction must be in (0, 1)")
        if self.dual_learning_rate <= 0.0 or self.maximum_dual_price <= 0.0:
            raise ValueError("dual parameters must be positive")
        dimension = len(self.feature_names) + 1
        self._a = self.ridge * np.eye(dimension, dtype=np.float64)
        self._a[0, 0] = max(self.ridge, 1e-6)
        self._b = np.zeros(dimension, dtype=np.float64)
        # Frozen deployment predicts at every control step.  Cache the exact
        # ridge inverse and coefficients; updates explicitly invalidate them.
        # Besides removing redundant work, this avoids repeatedly activating
        # a BLAS thread pool between two small ICODE rollout batches.
        self._inverse_cache = None
        self._theta_cache = None
        self.dual_price = 0.0
        self.decisions = 0
        self.add_decisions = 0
        self.observed_add_rewards = 0
        self.rng = np.random.RandomState(int(seed))

    @classmethod
    def from_rows(cls, rows, feature_names, **kwargs):
        matrix = np.asarray([
            [float(row[name]) for name in feature_names] for row in rows
        ], dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
            raise ValueError("budget-bandit normalization rows are invalid")
        scale = matrix.std(axis=0)
        scale[scale < 1e-9] = 1.0
        return cls(feature_names, matrix.mean(axis=0), scale, **kwargs)

    def _vector(self, features):
        if isinstance(features, Mapping):
            raw = np.asarray(
                [float(features[name]) for name in self.feature_names],
                dtype=np.float64,
            )
        else:
            raw = np.asarray(features, dtype=np.float64).reshape(-1)
        if raw.shape != self.feature_mean.shape or not np.isfinite(raw).all():
            raise ValueError("budget-bandit feature vector is invalid")
        standardized = (raw - self.feature_mean) / self.feature_scale
        return np.concatenate(([1.0], standardized))

    def predict(self, features):
        vector = self._vector(features)
        inverse = self._inverse_cache
        theta = self._theta_cache
        if inverse is None or theta is None:
            inverse = np.linalg.inv(self._a)
            theta = inverse @ self._b
            self._inverse_cache = inverse
            self._theta_cache = theta
        mean = float(vector @ theta)
        width = float(np.sqrt(max(0.0, vector @ inverse @ vector)))
        return mean, width

    def decide(self, features, explore=False):
        mean, width = self.predict(features)
        score = mean - self.dual_price
        exploratory = False
        if explore:
            score += self.exploration_alpha * width
            if self.rng.uniform() < self.epsilon:
                exploratory = True
                add = bool(self.rng.randint(0, 2))
            else:
                add = bool(score > 0.0)
        else:
            add = bool(score > 0.0)
        return BudgetDecision(
            add, mean, width, float(self.dual_price), float(score), exploratory
        )

    def update(self, features, decision, observed_advantage=None):
        add = bool(
            decision.add_samples
            if isinstance(decision, BudgetDecision) else decision
        )
        if add:
            if observed_advantage is None or not np.isfinite(observed_advantage):
                raise ValueError("ADD decision requires one finite observed advantage")
            vector = self._vector(features)
            self._a += np.outer(vector, vector)
            self._b += float(observed_advantage) * vector
            self._inverse_cache = None
            self._theta_cache = None
            self.observed_add_rewards += 1
        self.decisions += 1
        self.add_decisions += int(add)
        self.dual_price = float(np.clip(
            self.dual_price + self.dual_learning_rate * (
                float(add) - self.target_add_fraction
            ),
            0.0,
            self.maximum_dual_price,
        ))

    def state_dict(self):
        return {
            "schema_version": 1,
            "model_class": type(self).__name__,
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "ridge": self.ridge,
            "exploration_alpha": self.exploration_alpha,
            "epsilon": self.epsilon,
            "target_add_fraction": self.target_add_fraction,
            "dual_learning_rate": self.dual_learning_rate,
            "maximum_dual_price": self.maximum_dual_price,
            "dual_price": self.dual_price,
            "a": self._a.tolist(),
            "b": self._b.tolist(),
            "decisions": self.decisions,
            "add_decisions": self.add_decisions,
            "observed_add_rewards": self.observed_add_rewards,
        }

    @classmethod
    def from_state_dict(cls, state):
        if int(state.get("schema_version", 0)) != 1:
            raise ValueError("unsupported budget-bandit schema")
        result = cls(
            state["feature_names"], state["feature_mean"], state["feature_scale"],
            ridge=state["ridge"],
            exploration_alpha=state["exploration_alpha"],
            epsilon=state["epsilon"],
            target_add_fraction=state["target_add_fraction"],
            dual_learning_rate=state["dual_learning_rate"],
            maximum_dual_price=state["maximum_dual_price"],
        )
        result._a = np.asarray(state["a"], dtype=np.float64)
        result._b = np.asarray(state["b"], dtype=np.float64)
        dimension = len(result.feature_names) + 1
        if result._a.shape != (dimension, dimension) or result._b.shape != (dimension,):
            raise ValueError("budget-bandit checkpoint matrix shape is invalid")
        if not np.isfinite(result._a).all() or not np.isfinite(result._b).all():
            raise ValueError("budget-bandit checkpoint contains NaN or Inf")
        result.dual_price = float(state["dual_price"])
        result.decisions = int(state.get("decisions", 0))
        result.add_decisions = int(state.get("add_decisions", 0))
        result.observed_add_rewards = int(state.get("observed_add_rewards", 0))
        return result


__all__ = ["BudgetDecision", "PrimalDualBudgetBandit"]
