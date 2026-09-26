# SIS / dynamic-horizon fidelity report

Kim et al., [IEEE T-RO 2025](https://doi.org/10.1109/TRO.2025.3626660).
The [author project](https://euncheolim.github.io/Single-Instance-Sampling-for-Real-Time-Task-Space-MPPI-Control/)
describes one trajectory-wise perturbation reused throughout prediction, plus
distance-dependent foresight with a short fixed-step and a variable-step segment.
The task is 7-DoF Franka manipulation; examples use 64 prediction steps and a
1ms near segment. A deployable author implementation was not found.

Implemented verified constant-perturbation sampling on the common mobile-robot
backbone. Kept the repository's exact-prior candidate and ordinary weighting;
these are adaptation choices. Dynamic foresight is an explicitly declared port:
`a_H=clip(goal_distance/3m,0,1)`, mapped to the frozen integer H grid at dt=.1s.
K is fixed256. The 3m scale is a development default, not a published equation.
It neither recreates the binary variable-dt algorithm nor its analytical/task-space
parallel rollout speedups. No original manipulation cost/collision network is used.

Label: **SIS sampling + distance-H principle port**, never exact SIS/DTH replication.
Tests verify constant pre-clipping perturbations, distance endpoints, action bounds,
warm-start and two real MuJoCo cycles. Full-text/time-grid fidelity remains incomplete,
so later results cannot establish superiority over the original system.
