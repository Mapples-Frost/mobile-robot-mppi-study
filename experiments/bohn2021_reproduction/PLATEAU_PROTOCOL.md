# Pendulum Plateau and Preview Screening

Registered pilot on 2026-09-18. The user requested a few experiments of long
stationary-reference intervals interrupted by previewed cart-position moves.
This is a task extension, not a recreation of the missing paper scenario bank.
No RL training is performed.

## Scenarios

- 600 evaluated steps at 0.04 seconds per step (24 seconds), after the common
  zero-terminal H50 reset warmup.
- Cart references alternate between -A and +A, with A in {0.4, 0.6, 0.8} m.
- Two reversal clocks are drawn from inclusive intervals [180, 220] and
  [380, 420]. The first direction and small initial perturbations are randomized.
- The stored reference extends beyond the episode to cover the entire preview.
- Validation: three scenes, one per amplitude, seed 26091830.
- Holdout: six new scenes, two per amplitude, seed 26091831.
- All methods receive the same reference information and matched reset state.
  Plant equations, input/state limits, stage objective, H coefficient, and the
  float32 analytic Riccati terminal are shared.

## Arms and Selection

Coarse fixed horizons are 1, 5, 10, ..., 50. After the coarse comparison, all
integers within +/-4 of the best coarse horizon are added, clipped to [1, 50].
The result is the best tested fixed H, not a proof of the best of all 50 integers.

Four switch rules use short H in {5, 10} and long H in {30, 40}. A rule chooses
long H if a reference change appears within its next long-H preview steps, or
if any of these conditions holds:

- absolute position tracking error > 0.03 m;
- absolute velocity > 0.08 m/s;
- absolute pole angle > 0.04 rad;
- absolute angular velocity > 0.12 rad/s.

Otherwise it chooses short H. There is no learned policy, future physical-state
oracle, or access to the complete event schedule in action selection.

One fixed H and one switch rule are selected globally by mean validation total
cost, with deterministic arm-name tie breaking. They and the prior H30 reference
are evaluated on all six holdout scenes. Failed episodes remain in the results.
No thresholds or scene parameters are changed after seeing results.

## Outcomes and Checks

Report original total cost and its physical/H/constraint components. Also report
adjusted total = original total + 0.4905 * 600. The common additive constant
preserves policy ordering; for full episodes it shifts the upright equilibrium
stage cost to zero. Percentages, if given, are explicitly relative to this
normalization, not percentages of a negative raw cost.

Report tracking RMSE, mean H, physical failures, solver failures, and costs for
two common exogenous phases. Transition windows extend 50 steps before and 100
steps after each reference reversal; remaining steps are labelled plateau.
The existing failure penalty of 10 times remaining steps is retained and shown
separately. This penalty is larger early in a longer episode.

Each rollout has a fresh MPC instance. Runtime assertions verify terminal
weights after every step. An independent trajectory reader reconstructs physical
costs, reference clocks, rule decisions, constraints, summary values and matched
starts. Registered source files and scenario banks are hashed before rollout.

The first validation H30 worker is also the integration check and its three
episodes are reused, not counted as extra samples. At most two MPC workers run
concurrently. Maximum planned evaluation count is 87 episodes / 52,200 steps,
less if a selected arm duplicates H30, refinement reaches a boundary, or an
episode terminates early. No training transitions are collected.

## Interpretation

This small pilot asks whether phase-dependent horizon selection has useful
headroom. It cannot establish broad statistical generalization or that an RL
agent can learn the switching rule. The H penalty is a computation proxy,
not a measurement of CPU speedup. Amplitude-specific rows have very few scenes;
they are descriptive, not an isolated causal amplitude-effect estimate.

Source inspection also clarifies an earlier description: the original TVP
generator redraws reference values at deterministic 25-step intervals when its
redraw_probability setting is 0.04. It does not draw geometrically distributed
event times. This pilot explicitly randomizes its two event times.

## Commands

```bash
/home/mapples/.local/share/bohn2021-python37/bin/python experiments/bohn2021_reproduction/plateau_screen.py
.venv/bin/python experiments/bohn2021_reproduction/plateau_report.py
```

The immutable machine registration and raw rollouts live under
`research_artifacts/bohn2021_reproduction_2026-09-17/results/plateau_screen`.
