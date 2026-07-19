# Table 14: Full Proposed success-resolved factorial confirmation

Source revision: `cba00cbf3ab65f2350159277da6bb0f4cdb583de`

## Sealed clean-task confirmation

| Arm | Success | Collision | Final distance (m) | Jerk | Planner (ms) |
|---|---:|---:|---:|---:|---:|
| ordinary fixed | 9/15 | 0/15 | 0.47896 | 0.09955 | 186.28 |
| value fixed | 14/15 | 0/15 | 0.30794 | 0.10054 | 186.78 |
| ordinary adaptive | 13/15 | 0/15 | 0.31899 | 0.09833 | 183.44 |
| full proposed | 14/15 | 0/15 | 0.29553 | 0.10068 | 187.73 |

Full Proposed versus ordinary fixed, expressed in favorable direction:

| Outcome | Effect | Seed-cluster 95% CI |
|---|---:|---:|
| success | +0.3333 | [+0.1333, +0.5333] |
| final distance | +0.18343 m | [+0.14505, +0.22166] |
| collision | 0 | [0, 0] |
| jerk | -0.00113 | [-0.00426, +0.00201] |
| planner time | -1.45 ms | [-4.75, +2.53] |

The success interaction was -0.2667 with interval [-0.4667, -0.0667].
This is adverse/sub-additive, not evidence of synergy.

## Lab-complex supplement

All four arms had 0/15 successes and 0/15 collisions at 180 steps. Full
Proposed versus ordinary fixed saved 9.94 ms per planning step with interval
[8.53, 10.88] ms. This supports compute robustness but not complex-navigation
success.

## Forensic bindings

- `results/research_platform/rl/full_proposed_confirmation_l211/analysis_summary.json`
- `results/research_platform/rl/full_proposed_confirmation_l211/paired_comparisons.json`
- `results/research_platform/rl/full_proposed_confirmation_l211/factorial_contrasts.json`
- `results/research_platform/rl/full_proposed_lab_supplement_l213/analysis_summary.json`
- `results/research_platform/rl/full_proposed_lab_supplement_l213/paired_comparisons.json`
- `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`
