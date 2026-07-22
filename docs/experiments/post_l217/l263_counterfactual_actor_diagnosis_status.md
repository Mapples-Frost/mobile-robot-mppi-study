# L263 Counterfactual Actor Diagnosis — Causal Status

Status: diagnostic contract complete; no training or controller parameter was changed.

## Integrity

- Frozen code: `6b7c5c9c3f8694e0980b4c1e4caf0e82c0927d33`.
- Frozen L262 checkpoint SHA256: `48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`.
- 26 accepted states, 702 unique state-action pairs, and 5,616 rollout rows.
- Every accepted state has the complete 5x5 grid plus Actor and recovery actions.
- Every state-action has both continuations and horizons 1/10/20/40.
- One preregistered state was retained as rejected because both deterministic
  offset signs collided at reset; it was not replaced or deleted.
- Reset observation repeat error is 0; all rollout metrics and 25-quantile
  outputs are finite; stderr is empty.

## Causal status

Primary supported cause: **the learned Critic does not reliably rank actions
by their realized MuJoCo return at the decision-relevant horizon**.

- At horizon 40 with Actor continuation, mean per-state Spearman correlation
  between minimum twin-Q and realized return is -0.020, with 15.4% top-1
  agreement.
- For recovery versus fast-forward at horizon 40 under constant continuation,
  realized return prefers recovery in 76.9% of all states and 75.0% of offset
  states, while Critic pairwise accuracy is only 34.6% and 37.5%, respectively.
- The Critic's grid optimum is `(v_norm=1, omega_norm=-1)` in 12/26 states,
  whereas the realized Actor-follow grid optima at horizon 40 all use
  `v_norm` of -1 or -0.5.

Therefore the current evidence does **not** support the earlier global claim
that the reward simply prefers high-speed deviation. The frozen CTE reward
still loses tail resolution above 1.5 m, and signed CTE observation saturates
at 0.75 m, so reward/observation resolution and replay coverage remain
plausible contributors to the Critic error. They are not yet isolated causes.

Actor optimization and multi-scene gradient interference are not adjudicated:
Actor gradients inherit the unreliable Critic ordering, so the preregistered
gradient audit stopping condition is not met. No further training is allowed
from L263. The next separately preregistered step should audit Critic target,
Bellman ranking, extrapolation, event coverage, and observation aliasing before
choosing any reward, observation, or optimization change.
