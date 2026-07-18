# Table 05: L37 bounded-RL by ICODE paired factorial

Source: `results/research_platform/rl/l37_bounded_rl_icode_factorial_20260716_v1/factorial_summary.json`

| Condition | Success | Collision | Mean final distance | Mean planner time |
|---|---:|---:|---:|---:|
| Traditional + nominal | 15/45 (33.3%) | 21/45 (46.7%) | 1.235 m | 4.86 ms |
| Traditional + ICODE | 3/45 (6.7%) | 28/45 (62.2%) | 1.567 m | 34.27 ms |
| Bounded RL + nominal | 10/45 (22.2%) | 32/45 (71.1%) | 1.669 m | 6.28 ms |
| Bounded RL + ICODE | 8/45 (17.8%) | 34/45 (75.6%) | 1.777 m | 34.86 ms |

The combined controller lost seven net successes, added thirteen collisions and worsened mean final distance by 0.542 m relative to traditional nominal. The success interaction was positive (+0.222, 95% hierarchical-bootstrap CI 0.022 to 0.422), but this does not offset the unfavorable absolute performance or negative main effects. The development gate failed in every model block.
