# Geometry operating-region screen (development exploration)

Before execution, after all-family retrospective diagnosis of the completed
mechanism experiment. Purpose: search for interpretable conditions where
dynamic joint K/H helps, and conditions where it fails. This is deliberately
targeted development, not a representative robot benchmark or confirmation.

Door-offset showed Q improvement versus fixed128/27 in both initializations;
crossing showed opposite safety outcomes; sparse-slalom showed instability;
open was almost unchanged. Retain all four families to cover opportunity,
failure and a simple control. The full earlier 12-family results stay visible.

For every family use lateral geometry scales0.8,1.0,1.2, both existing speed
caps, and fresh environment seeds9590101/9590102. Apply the same transformation
to all methods: lateral obstacle coordinates, box half-widths, and moving
obstacle path endpoints. Cylinder radius, longitudinal coordinates, robot,
goals and planner stay unchanged. The crossing period remains fixed, so lateral
scaling also changes obstacle velocity; do not isolate width from speed there.
The open scene repeats unchanged as a timing/variation control, not independent
geometry replication. Known13kg dynamics stay fixed in this screen.

Use BOTH final independent ICODE full and masked joint initializations9091301/
9091302. Fixed references: both nominal/ICODE with128/27,160/29,256/24. Ten
methods x4families x3scales x2speed caps x2seeds =480 episodes. Schedule seed
9195300 (inherited serial evaluator). No retraining, policy selection or tuning from screen outcomes.
Serial isolated workers; raw/config/source/seed/checkpoint records preserved.
All10methods see identical scenes and limits. No arbitrary extra latency on
baselines. Final909030x seeds remain sealed.

Output every family/scale/speed cell with successes, collisions, Q, measured
compute, stage cost, deadlines and initialization-specific fixed/masked
contrasts. Never report only the winning scale or replace earlier failures.
Any promising operating region is a hypothesis requiring new fresh-seed
validation and matched nominal-SAC support. A negative screen also completes
its scientific purpose. This screen runs before the matched training factorial;
its outcomes do not change that factorial's already specified design.
