# Residual Dataset Pipeline

## Sources

The collector supports two honest source labels:

- `random_exploration`: reproducible piecewise-constant excitation over the
  configured operational envelope;
- `task_specific_goal_tracking`: a proportional goal-tracking controller for
  the clean `(3,3)` task.

The second source is task-specific but is not mislabeled as learned-MPPI
on-policy data.  `collect_dataset(policy=..., model_version=...)` is the
extension point for later iterative aggregation.

## Transition schema

Every sample stores episode/seed/step/time/dt, `state_t`, commanded
`control_t`, diagnostic `applied_control_t`, `state_t_plus_1`, nominal and
observed derivatives, residual target, disturbance type and JSON parameters,
scene, source, and model version.

```text
observed_derivative ~= wrapped(state_t_plus_1 - state_t) / dt
residual_target      = observed_derivative - f_nom(state_t, commanded_control)
```

Using the commanded control is intentional: that is the value available to
the planner and learned model.  Applied control remains recorded so delay and
actuator effects are auditable.

## Artifacts and provenance

`ResidualDataset.save` writes compressed pickle-free NPZ arrays, a structured
JSON sidecar, and a human-readable metric/value summary CSV.  The collector
also writes an episode summary, config snapshot, run manifest, SHA-256 hashes,
UTC creation time, Git SHA/dirty state, Python/NumPy versions, seeds, sample
counts, dimensions, and data sources.

Publishing uses temporary files and replacement so the NPZ is the final
commit marker.  Results live under ignored `results/icode/` directories.

## Split rules

Train/validation/test splitting is deterministic and group-based by episode;
no timestep from one episode can leak into another split.  Complete held-out
disturbance regimes become the unseen split.  The default seen training
disturbances contain no delay; delay is retained in the unseen stress regime.

Normalization statistics are fitted only from the supplied training subset.
`NormalizerBundle` stores state-feature, commanded-control, and residual
statistics and is serialized inside every checkpoint.

## Rollout windows

Contiguous windows require the same episode, consecutive step numbers, time
continuity, and optionally constant `dt`.  They never cross episode boundaries
or gaps.  A horizon `H` window contains `H` controls and `H` target next states.

## Commands

```bash
.venv/bin/python experiments/icode/collect_synthetic_residual_data.py --smoke
.venv/bin/python experiments/icode/collect_synthetic_residual_data.py \
  --output-dir results/icode/datasets/research_protocol --overwrite
```
