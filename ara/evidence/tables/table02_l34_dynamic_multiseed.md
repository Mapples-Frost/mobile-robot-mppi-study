# Table 02: L34 bounded dynamic correction across training seeds

Source: `results/research_platform/rl/l34_bounded_dynamic_multiseed_development_20260716_v1/training_seed_summary.csv`

| Training seed | Best step | Success step 0 -> best | Collision step 0 -> best | Mean final distance improvement |
|---:|---:|---:|---:|---:|
| 20260731 | 30000 | 1/6 -> 4/6 | 4/6 -> 2/6 | 0.701 m |
| 20260732 | 30000 | 2/6 -> 3/6 | 3/6 -> 2/6 | 0.335 m |
| 20260733 | 15000 | 1/6 -> 3/6 | 5/6 -> 3/6 | 0.604 m |

Across 18 paired held-out episodes: six success gains, zero success losses, five collision improvements and zero collision regressions. Mean seed-level final-distance improvement was 0.547 m. These are development/model-selection cells, not sealed final-test results.

