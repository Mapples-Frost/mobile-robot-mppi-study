# Table 12: Gate 2 critic-aligned ICODE evidence chain

Date: 2026-07-18

All closed-loop effects compare a task-only ICODE control checkpoint with an
ICODE checkpoint fine-tuned from the same data and initialization family.
Positive percentages below are favorable unless marked otherwise.

| Run | Alignment mechanism | Independent seed clusters | Final-distance change | Success change | Stuck-step change | Jerk change | Collision change | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---|
| L188 | Unconditional critic value consistency | 3 | 11.29% worse | -5.56 pp | not primary | 1.72% better | equal at zero | Failed; negative transfer retained |
| L190 | Support × critic agreement × task competence | 3 | 25.22% better | +5.56 pp | 24.62% better | 0.38% better | equal at zero | Development Gate passed |
| L191 | Same frozen competence-gated mechanism | 5 | 19.86% better | +10.00 pp | 26.70% better | 0.066% worse | equal at zero | Primary progress confirmed; exact-zero jerk clause failed |

L191 seed-cluster bootstrap intervals:

| Endpoint | Favorable paired effect | 95% interval | Paired dz |
|---|---:|---:|---:|
| Final goal distance | 0.12686 m | [0.03246, 0.22010] m | 1.045 |
| Success | 10.00 pp | [3.33, 16.67] pp | 1.095 |
| Stuck steps | 1.5667 steps | [0.0667, 3.1000] | 0.804 |
| Control jerk | -0.0000645 | [-0.00208, 0.00290] | -0.019 |
| Minimum clearance | -0.00328 m | [-0.00923, 0.00087] m | -0.509 |

Scope: the confirmed effect is concentrated in the solvable clean task and
bounded dynamics shifts. It does not resolve the narrow-corridor planning
floor. Concurrent L191 execution shards are valid for deterministic control
outcomes but not for wall-time efficiency comparisons.

Source artifacts:

- `results/research_platform/rl/gate2_value_alignment_paired_l188/paired_comparison.json`
- `results/research_platform/rl/gate2_competence_gated_paired_l190/paired_comparison.json`
- `results/research_platform/rl/gate2_competence_confirmation_l191/gate2_full_analysis.json`
- `docs/rl/173_gate2_competence_gated_value_alignment_results_2026-07-18.md`
