# V6 A1 baseline-identity audit

The A1 engineering gate passed (safe success 1/8 to 2/8; collisions 5/8 to
4/8; zero new paired collisions), but the archived-v5 source-arm audit found a
protocol identity defect.

Four A1 blocks used a different ICODE `model_block` than the same seed used in
the v5 qualification:

| seed | A1 block | archived v5 block |
|---:|---:|---:|
| 750100005 | 0 | 1 |
| 750200042 | 0 | 1 |
| 750200036 | 2 | 0 |
| 750200020 | 1 | 2 |

The remaining four blocks matched exactly.  For each matched block, source-arm
success/collision/steps/minimum-clearance/final-distance reproduced exactly.
The apparent `750200020` source regression was therefore a model-block change,
not an implementation drift.

A1 remains valid as paired, outcome-informed mechanism development under its
resolved configurations, but its strata and source-outcome labels are not a
faithful replay of the archived v5 cohort.  It cannot authorize replication or
held-out qualification.  A1b will change only these four model-block
assignments; all mechanism parameters, seeds, arm orders, budget, and gates
remain frozen.
