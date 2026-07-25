# Residual-Dynamics Stage 1 Amendment 3 Result

Status: **failed at the preregistered first residual block**  
Date: 2026-07-23  
RL status: **disabled**

The innovation reliability gate became active but did not prevent collision in
checkpoint block 1, so the probe stopped before opening the other conditions.

- collision: yes;
- mean / maximum reliability authority: 0.551 / 1.000;
- stall fail-safe latched: yes;
- final goal distance: 6.129 m;
- planner p95: 425.46 ms.

This rules out the narrow explanation that cold-start residual authority alone
caused the failure. The residual was still permitted to rewrite known pose
kinematics once prediction evidence became favorable. Amendment 4 tests the
repository's previously established structure-preserving residual interface.

Artifact:
`research_artifacts/dynamic_uncertainty_residual_stage1_amendment3_probe/`.

