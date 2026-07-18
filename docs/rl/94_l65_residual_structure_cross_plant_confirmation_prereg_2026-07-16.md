# L65 residual-structure cross-plant sealed confirmation preregistration

Date: 2026-07-16

L64 passed every preregistered artifact, learned-model, safety, compute, and
cross-plant structure check. Before running any confirmation episode, L65
freezes the identical checkpoints, paths, physical domains, conditions,
statistics, and thresholds. The only intended changes are:

- `confirmation_mode: true`;
- execution seeds become the five untouched values 21860831--21860835;
- schedule and bootstrap random seeds are fresh;
- L64 development seeds become protected.

The confirmation contains 540 episodes: three matched model blocks, two paths,
six physical domains, five execution seeds, and three conditions. The L64
decision rule is reused without relaxation. Any failure is reported as a failed
confirmation; no additional seed set will be opened in this experiment family.

Passing supports the bounded empirical statement that, for these frozen MuJoCo
parameter shifts, the parameter-matched control-affine ICODE residual improves
closed-loop MPPI path tracking beyond both nominal dynamics and an unstructured
MLP residual. It does not prove the causal mechanism, arbitrary OOD robustness,
real-robot transfer, or RL benefit.
