# Table 25. Pareto active-traversal repair and Amendment 1 evidence

## Bounded failure repair

The original expanded development matrix identified seed 730100208 as a candidate-only collision and seed 730100206 as a lost source success. Observation-only forensics showed that reactive reverse escape could discover a lower-risk forward candidate but then abandon it, repeatedly turn in the robot frame, or be overridden by dynamic recovery. The final bounded repair therefore:

- permits reverse-to-forward commitment only when one candidate is simultaneously the lowest-risk and lowest-cost option and has strictly lower predicted risk;
- converts the selected finite turn to a fixed world-frame heading;
- prevents dynamic recovery from overriding an already selected, forecast-vetted planner escape; and
- records causal emergency intent, rearm and Pareto-commit diagnostics without using obstacle future truth.

Failed approaches were retained: longer TTC rearm/clear hold, a broad lowest-risk veto, uncommitted Pareto forward selection, forward commitment without heading stabilization, and stabilized heading while recovery could still override the planner.

## Targeted replication

| Episode | Outcome before repair | Final Candidate outcome | Steps | Final distance | Minimum clearance | Planner P95 | Pareto commits |
|---|---|---|---:|---:|---:|---:|---:|
| seed 730100208 | candidate-only collision | success, no collision | 367 | 0.28206 m | 0.16813 m | 79.657 ms | 1 |
| seed 730100206 | lost source success | success, no collision | 295 | 0.28894 m | 0.39570 m | 79.315 ms | 0 |

The final seed-730100208 result was reproduced in a separate replication artifact. Every planning cycle retained exactly 600 rollouts. Seed 730100206 did not regress, so the repair was not merely a swap between the two diagnosed failures.

## Frozen 24-pair Amendment 1 result

| Quantity | Source Actor | Repaired Candidate | Paired change |
|---|---:|---:|---:|
| Development pairs | 24 | 24 | seeds 730100204--227 |
| Goal success | 20/24 | 21/24 | +1 episode |
| Collision | 4/24 | 3/24 | -1 collision |
| New Candidate collisions | -- | 0 | none |
| Lost Source successes | -- | 0 | none |
| Rescued failure / prevented collision | -- | seed 730100208 | one paired rescue |
| Mean final goal distance | 0.94236 m | 0.85880 m | -0.08356 m (-8.87%) |
| Total steps | 7097 | 7292 | +195 (+2.75%) |
| Maximum planner P95 | 93.666 ms | 98.378 ms | both below 100 ms |

Shared Candidate collisions remained on seeds 730100211, 730100218 and 730100222. All 24 pairs used proposal-only Actor integration, same-cycle filtering, the fixed 600-rollout budget, frozen checkpoints and frozen execution order. Sealed seeds remained unopened.

The formal Gate remains failed because total_steps_noninferior and at_least_two_nonregressive_batches are false. This is not rewritten as a pass: preventing the seed-730100208 collision allowed the repaired episode to run 101 more steps and reach the goal instead of terminating early, so the uncensored total-step sum penalizes successful continuation. All completion, collision, distance, runtime, filter and protocol-integrity checks passed.

The integrity audit passed with zero issues, 220 bound raw files and raw-manifest SHA-256 3093be72876037a414ac5f46f0cc5bd6d261b9afe477ef0a45b2d90f258f73c5. Targeted regression testing passed 182 tests, and git diff --check reported no whitespace errors.

Primary evidence:

- research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_replication_seed730100208/result.json
- research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment1/summary.json
- research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment1/gate.json
- research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment1/paired_analysis.json
- research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment1/integrity_audit.json
- configs/research/dynamic_actor_v5a6_samecycle_expanded_development_amendment1.yaml
