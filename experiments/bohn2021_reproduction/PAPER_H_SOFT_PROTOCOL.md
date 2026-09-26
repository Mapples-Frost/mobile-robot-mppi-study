# Frozen Paper-Configuration Soft-Return Probe

This is a post-hoc validation diagnosis following `paper_h_credit_probe.py`.
It is not a newly trained policy, a confirmation experiment, or strict numerical
reproduction of the original paper. The machine-readable protocol and source,
model, configuration and scenario hashes were written before the first rollout
under `results/paper_h_soft_2026-09-24`.

## Design

Retain all three final training seeds of author RL and min-Q on both reconstructed
paper tasks. Reuse only validation scenes 0 and 3, after 20 and 60 saved actor
decisions. These 48 anchors are correlated observations of four scenes, not
48 independent experimental replicates. No test scene is read by the probe.

Replay the saved prefix with the model's own frozen terminal polynomial,
including its original reset warmup. At that state execute one of the deterministic
actor H, Q1-grid maximum H, min-Q-grid maximum H or the previously selected fixed
H25/H30. The fixed-H arm is a single-decision intervention under the adaptive
model's terminal and continuation; it is NOT a full fixed-H policy evaluation.
Merge identical first H choices within each anchor and noise repeat.

After the first action, follow the same frozen stochastic actor until environment
termination. Use four predeclared Gaussian sequences per anchor, shared across
all first-H arms. Standardized noise is paired; actions need not match because
state-dependent means and standard deviations change along each trajectory.
The simulation is deterministic conditional on the saved scenario and noise.
Execution uses at most three workers; timing is not an experimental outcome.

## Return Definitions

Let c[k] be physical cost + linear H proxy + constraint penalty. Let r[k] be
-c[k]/scale with scale 0.3 for vehicle and 0.6 for pendulum, gamma 0.97 and
alpha 1.0, matching these models' training settings. The author checkpoint does
not serialize reward_scale, so use the registered scale explicitly rather than
the reloaded default 1.0. Frozen inference does not use that default attribute.

Finite soft Q sample:

    G = sum(gamma**k * r[k], k=0..L-1)
        - alpha * sum(gamma**k * log_pi[k], k=1..L-1)

The forced first action has NO entropy term. Later actions and log probabilities
use the author's float32 Gaussian reparameterization and tanh correction, with
its epsilon conventions. There is no inverse-tanh clipping or substitution of
categorical probability mass for continuous density. Raw noise, latent variable,
mean, log standard deviation, action and observation are saved for auditing.

At a time-limit termination only, also report

    G_bootstrap = G + gamma**L * target_V(final_observation).

At a physical constraint or vehicle goal, the tail is zero. Report finite return,
entropy contribution, discounted physical/H/constraint cost, tail and termination
separately. A learned tail is not an independent ground-truth value estimate.

## Audit and Interpretation

Recompute all saved costs from state, input and scenario reference. Verify
identical pre-action states, noise pairing, float32 H mapping, log probabilities,
terminal semantics, suffix totals, hashes and frozen weights. Re-infer saved
actor distribution parameters, initial Q grids and timeout target values.
Record explicit prefix, suffix and reset-warmup work; H is a computation proxy
and concurrent wall time cannot establish a latency advantage.

Each comparison retains all four paired differences, their mean and solver or
physical failures. A consistently adverse sign in both finite and bootstrapped
samples addresses the deterministic/entropy/timeout explanation only within
this small sample. It does not prove the exact expected soft-Q ordering or a
population failure rate. Online terminal learning during the original training
is another difference from the frozen deployment continuation.

An additional post-hoc read-only actor-objective scan evaluates the actual
Q1 objective for author RL and min-Q objective for min-Q training, holding each
state's standard deviation fixed while varying the Gaussian mean. Compare
64- and 128-node Gauss-Hermite integration. A numerically positive objective
gap is not proof of a better controller, and independent per-state choices have
more freedom than a shared actor network. This scan consumes no physical rollout.

## Execution

Run from the repository root under the pinned Python 3.7 environment:

```bash
/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/paper_h_soft_probe.py --all
/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/paper_h_soft_report.py
/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/paper_h_actor_objective.py
```

The suite and each model directory have process locks. Completed model records
are reused; no training job is launched. The original deterministic probe remains
unchanged. Any subsequent training intervention requires a separate registration
and fresh validation/test banks; this diagnostic is not authorization to tune on
the exposed min-Q test or open failed probes' reserved tests.
