# Residual Dynamics Stage 2: Three-Seed Development Result

Date: 2026-07-23  
Status: **development qualification passed; deployment timing and sealed
qualification remain open.**

## Frozen question

The experiment tested whether the task-aware residual family could participate
in the dynamic-obstacle closed loop without reducing safety or task
completion. The independent unit was one complete MuJoCo episode. The frozen
matrix was

`{730100006, 730100008, 730100010} x {nominal, R01, R02, R03}`.

The three residual checkpoints were training-seed replicates. Common random
numbers paired every residual run with the nominal run for the same obstacle
seed. RL was disabled and sealed seeds were not opened.

Seed `730100006` was carried forward from the preregistered pilot. Its resolved
per-episode configs and copied files were byte-identical to the pilot source.
Seeds `730100008` and `730100010` were then run under the same frozen protocol.

## Closed-loop outcomes

All 12 episodes reached the goal and none collided. There were no safety-layer
interventions.

| Obstacle seed | Nominal steps | R01 steps (delta) | R02 steps (delta) | R03 steps (delta) |
|---:|---:|---:|---:|---:|
| 730100006 | 327 | 327 (0) | 324 (-3) | 332 (+5) |
| 730100008 | 331 | 317 (-14) | 329 (-2) | 321 (-10) |
| 730100010 | 330 | 337 (+7) | 335 (+5) | 339 (+9) |

Across the nine paired residual comparisons, the step delta had median `0`,
mean `-0.33`, and range `[-14, +9]`. By model block, the median deltas were
`0`, `-2`, and `+5` steps for R01, R02, and R03. This is not a consistent
completion-time benefit, and it does not justify selecting one checkpoint.

The automatic frozen gate passed:

- maximum collision-count increase in every model block: `0`;
- median completion delta: `-0.000142`, above the frozen `-0.02` limit;
- all three blocks retained positive offline H36 position-prediction gains;
- artifact integrity, checkpoint loading, forecast contracts, and finite
  diagnostics passed;
- no sealed seed was opened.

## Did the residual actually affect control?

Yes, but within the shield's bounds.

| Block | Median shield acceptance | Median effective acceptance | Median executed-control difference > 0.01 | Maximum paired position difference |
|---|---:|---:|---:|---:|
| R01 / 20261201 | 67.6% | 29.7% | 59.0% | 0.110 m |
| R02 / 20261202 | 71.4% | 22.1% | 48.8% | 0.053 m |
| R03 / 20261203 | 70.4% | 31.2% | 52.0% | 0.080 m |

Here, shield acceptance means the residual proposal passed the safety checks
in that run. Effective acceptance is stricter: the accepted residual also had
to produce an executed action measurably different from the paired nominal
run. Its median over all nine pairs was `29.7%`.

These paired traces are realized closed-loop comparisons, not exact
same-state, per-step counterfactuals: once two controllers choose different
actions, their later states differ. They are sufficient to reject the
degenerate explanation that the shield always fell back to nominal, but they
do not isolate a causal per-step residual effect.

Fallback reasons overlap. Their median fractions across paired runs were:

- nominal-view risk increase: `75.8%`;
- model-position tube violation: `41.2%`;
- nominal progress regression: `17.5%`;
- candidate probability above the hard limit: `5.1%`;
- unexplained fallback: `0%`.

Thus the shield mainly rejected proposals because they increased risk under
the nominal view or moved too far from the nominal rollout, rather than
because residual inference silently failed.

## Clearance interpretation

The identical reported minimum clearance (`about 0.39754 m`) is not evidence
that all controllers followed the same near-obstacle path. In every run, that
minimum occurred at the first recorded step near the fixed starting pose.
Consequently, the global minimum is dominated by initialization and is a weak
crossing-safety statistic.

The fifth-percentile clearance is more informative. Across the nine pairs,
its residual-minus-nominal delta had median approximately `0 m` and range
`[-0.0357, +0.0158] m`. The fraction of time below `0.50 m` remained about
`1.5%`. These descriptive distributions show no material clearance shift in
this small development sample.

## Statistical scope

The obstacle episode seed, not the controller time step, is the independent
experimental unit. There are only three development seeds per model block.
Formal p-values or narrow confidence intervals would therefore imply more
independent evidence than the experiment contains. The appropriate conclusion
is a preregistered development-gate decision plus seed-level descriptive
effect sizes.

This experiment supports:

> The task-aware residual family passed the frozen three-seed development
> safety and non-inferiority gate while materially participating in control.

It does not support:

- a general safety guarantee;
- a statistically established speed advantage;
- selection of a single best checkpoint;
- a claim that lower offline H36 error necessarily improves MPPI control.

## Runtime and next stage

Residual-planner P95 compute time ranged from `221.84` to `422.08 ms`
(`247.05 ms` median across residual episodes). It passed the development
artifact limit of `600 ms`, but it does not meet the `100 ms` control-period
deployment budget.

The subsequent behavior-preserving runtime stage has now passed on the
development machine. CUDA Graph rollout plus parallel nominal/residual shield
planning achieved maximum P95 `97.19 ms` while exactly matching the sequential
optimized closed-loop traces. See `RESIDUAL_RUNTIME_STAGE3_RESULT.md`. RL
remained disabled during runtime qualification.

## Artifacts

- Frozen protocol:
  `configs/research/dynamic_uncertainty_residual_stage2_task_aware_development.yaml`
- Automatic gate:
  `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/gate.json`
- Episode summary:
  `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/episode_summary.csv`
- Contribution audit:
  `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/contribution_audit.json`
- Provenance:
  `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/provenance.json`
