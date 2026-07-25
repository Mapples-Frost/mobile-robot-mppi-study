# Table 26. Final bounded single-obstacle development validation

## Frozen 24-pair Amendment 2 matrix

| Quantity | Source Actor | Candidate | Paired interpretation |
|---|---:|---:|---|
| Development pairs | 24 | 24 | seeds 730100228--251 |
| Goal success | 20/24 | 22/24 | +2; exact McNemar p=0.500 |
| Collision | 3/24 | 1/24 | -2; exact McNemar p=0.500 |
| New Candidate collisions | -- | 0 | registered safety gate passed |
| Lost Source successes | -- | 0 | registered completion gate passed |
| Outcome-aware efficiency | 7741 | 7615 | -126 penalty-adjusted steps |
| Mean final distance | 0.773262 m | 0.523574 m | mean paired delta -0.249688 m |
| Raw total steps | 7151 | 7397 | Candidate usually slower; Wilcoxon p=0.024 |

All 11 registered gates passed. Both arms used proposal-only same-cycle filtering and exactly 600 rollouts per controller decision. The 220-file integrity manifest has zero missing or mismatched files. Sealed seeds remain unopened.

Formal paired tests do not establish broad superiority: success/collision discordances number only two each, the final-distance bootstrap interval crosses zero, and outcome-aware efficiency Wilcoxon p=0.182. The evidence supports the frozen engineering gate within development scope.

## Fresh real-time mechanism confirmation

| Quantity | Source Actor | Candidate |
|---|---:|---:|
| Seed | 730100255 | 730100255 |
| Goal / collision | success / none | success / none |
| Steps | 316 | 300 |
| Planner P95 | 78.644 ms | 83.537 ms |
| Valid decisions at 600 rollouts | 316/316 | 300/300 |
| Temporal emergency triggered/vetted | 17/17 | 17/17 |
| Online tracker and forecast valid | every decision | every decision |

The fresh confirmation's nine raw files and frozen checkpoint/config/code bindings pass integrity audit. Targeted validation passed 156 tests, Python compilation and `git diff --check`. Timing remains environment-sensitive: the older matrix Candidate P95 maximum was 115.15 ms, so the fresh pass is not a distribution-wide hard-real-time guarantee.

Primary evidence:

- `research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/`
- `research_artifacts/dynamic_actor_v5a6_single_obstacle_final_confirmation_seed730100255/`
- `docs/experiments/dynamic_uncertainty/SINGLE_OBSTACLE_FINAL_VALIDATION_2026-07-25.md`
