# Single Dynamic Obstacle Publication-Confirmatory Preregistration

Frozen: 2026-07-25 09:35 CST, before opening any held-out closed-loop trajectory.

## Research question and independent unit

The confirmatory question is whether the retained V5 Amendment 6 update-250 Actor, under the frozen causal LaserScan/probability-forecast MPPI stack, preserves safety and task completion while remaining compute-feasible relative to the V3 roll-in Source Actor. The independent unit is one complete paired MuJoCo episode seed. Controller cycles within an episode are repeated measurements, not independent replicates.

The primary comparison uses common random numbers: Source and Candidate receive the same seeded obstacle trajectory and stochastic streams. The only primary-arm difference is the frozen Actor checkpoint. Exactly one dynamic obstacle is present. Control may use only the Polyline reference, online LaserScan tracking, and online probability forecasts; MuJoCo obstacle identity, trajectory, modes, events, and future truth are forbidden as control inputs.

## Confirmatory sample and randomization

- 50 untouched held-out ID seeds: 730200001--730200050.
- 50 untouched held-out OOD seeds: 730300001--730300050.
- The OOD arm uses the registered faster, more frequently changing process shift. The offline `sensor_shift` profile is not claimed in closed-loop because its synthetic observation field does not enter the runtime LaserScan path.
- A seeded schedule (`730299901`) randomizes 100 seed blocks. Primary arm order is balanced. The 25 ablation blocks per split use a five-arm Latin position balance.
- No tuning, seed replacement, automatic retry, artifact overwrite, or healthy-process restart is authorized after execution begins.

## Power and precision fixed before execution

Development data are used only to estimate paired variance, not the held-out effect. The smallest effect of interest is 10 outcome-aware control steps (1.0 s at the 0.1 s control period). The development paired SD was 29.7252 steps, giving `dz=0.3364`. A two-sided paired t approximation at alpha=.05 gives 91.5% power with 100 pairs. Sensitivity values are 119 pairs for `dz=0.30`, 88 for `dz=0.35`, and 68 for `dz=0.40` at 90% power.

For a harm event absent in 100 pairs, the exact one-sided 95% upper probability bound is 2.95%; within either 50-seed split it is 5.82%. The frozen noninferiority limits are 5% pooled and 6.5% per split, together with a stricter zero observed Candidate-only collision and zero lost Source success rule.

## Arms and ablations

Every controller decision has exactly 600 MPPI rollouts and horizon 36.

1. `source_actor`: V3 roll-in checkpoint, full safety stack.
2. `dynamic_actor`: V5 Amendment 6 update-250 checkpoint, full safety stack.
3. `ablation_no_same_cycle_filter`: Candidate without same-cycle guided-cost filtering.
4. `ablation_no_pareto_commit`: Candidate without the Pareto forward-commit permission.
5. `ablation_no_temporal_escape_package`: Candidate without the added vetted temporal lattice/hold/commit package; the pre-existing base scan guard remains.

The three ablations are mechanism-estimation contrasts against the full Candidate on the same 50 seeds. Their outcome-aware efficiency p-values are Holm-adjusted. They do not replace or weaken the 100-pair primary comparison.

## Outcomes and analysis

The primary continuous estimand is Candidate minus Source outcome-aware efficiency steps: successful episodes use observed steps; unsuccessful episodes use `max(observed steps, 400)`. Noninferiority requires the upper paired-t 95% confidence limit to be no greater than 1% of Source mean efficiency. A two-sided paired t-test and 200,000-replicate whole-seed bootstrap CI are reported; superiority is a separate claim requiring a negative mean, CI upper bound below zero, and `p<.05`.

Paired success and collision use exact McNemar tests. Candidate-only collisions and lost Source successes use exact one-sided binomial upper bounds. Final distance, minimum clearance, success, collision, raw steps, and mechanism counters are reported by pooled, ID, and OOD strata. No missing episode is imputed and no replacement seed is allowed.

Real-time qualification is distribution-level: the pooled, ID, and OOD decision-level planner P95 must each be below 100 ms. Every valid decision in every arm must record exactly 600 rollouts. Tracker enablement must be complete and Candidate forecast availability must be at least 95%.

All raw trajectories, resolved configs, metrics, provenance files, protocol/code/checkpoint hashes, schedule, seed split, and arm order are integrity audited. Negative, null, timing-violation, collision, boundary, incomplete, and ablation results are retained without relabeling.
