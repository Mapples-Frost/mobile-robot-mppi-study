# Geometry screen: completed data, preliminary reading

2026-09-09 11:44 Asia/Shanghai. Source:
research_artifacts/icode_sac_compute_2026-09-08/geometry_validation_2026-09-09/summary.json.
Completion marker reports all 480 scheduled episodes. Below is an arithmetic
grouping of that summary, not yet the queued raw-trace audit or final inference.

| Method | Episodes | Success | Collision | Mean Q | Compute s/episode |
|---|---:|---:|---:|---:|---:|
| Fixed ICODE 128/27 | 48 | 40 | 6 | 2.11 | 3.63 |
| Fixed ICODE 160/29 | 48 | 39 | 5 | 2.03 | 4.37 |
| Fixed ICODE 256/24 | 48 | 39 | 5 | 2.10 | 4.73 |
| Fixed nominal 128/27 | 48 | 38 | 8 | 2.56 | 0.69 |
| Fixed nominal 160/29 | 48 | 38 | 6 | 2.25 | 0.79 |
| Fixed nominal 256/24 | 48 | 39 | 5 | 2.07 | 0.81 |
| Joint full 9091301 | 48 | 42 | 5 | 1.80 | 3.54 |
| Joint full 9091302 | 48 | 44 | 4 | 1.51 | 4.48 |
| Joint masked 9091301 | 48 | 42 | 5 | 1.79 | 3.52 |
| Joint masked 9091302 | 48 | 43 | 5 | 1.72 | 2.81 |

The clearest repeated task-performance signal is door_offset: all four joint
policies succeed in 12/12 cases with zero collisions; each of the three fixed
ICODE budgets succeeds in 10/12, zero collisions. The nominal fixed budgets
succeed in 10/12, 10/12 and 9/12. Full-policy Q is 0.64/0.65 versus fixed ICODE
1.32/1.27/1.46. Masked Q is 0.67/0.67, so these observations do not establish
reliability features as the mechanism.

Counterevidence remains visible: sparse_slalom full initialization 1 succeeds
11/12 while fixed ICODE128/27 succeeds 12/12. Crossing remains collision prone
(full initializations 5/12 and 4/12 collisions). All methods succeed in all
open cases, whose geometry-scale repeats are not new independent geometries.
Nominal is substantially cheaper; ICODE efficiency superiority is not shown.
Two environment seeds and targeted scene selection constrain any inference.

Next: finish the already running 288-episode compound-scene pilot, audit both
datasets, inspect trajectories and plots, then choose a narrowly scoped
adaptation mechanism. Broader fixed-budget tuning and single-axis controls
remain required before independent validation. No best-initialization choice
or reliability/ICRA-readiness conclusion is made from this screen.
