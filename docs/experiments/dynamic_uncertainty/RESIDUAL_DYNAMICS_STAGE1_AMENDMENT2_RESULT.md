# Residual-Dynamics Stage 1 Amendment 2 Result

Status: **failed at the preregistered first residual block**  
Date: 2026-07-23  
RL status: **disabled**

The causal stall guard behaved as implemented but did not prevent collision.
Checkpoint block 2 latched residual authority off at step 171, then collided at
step 224. The run stopped immediately; the other two residual blocks and shared
nominal were not opened under this amendment.

Key observations:

- residual stall guard latched: yes, step 171;
- minimum residual authority: 0;
- collision: yes;
- minimum clearance: -0.0071 m;
- final goal distance: 6.0921 m;
- planner p95: 359.75 ms.

The intervention was too late. At step 170 the residual trajectory was already
near `(-1.674, -2.416)`, while the saved shared nominal trajectory was near
`(-1.694, -2.153)`: a roughly 0.26 m lateral separation into the obstacle's
return corridor. Disabling residual dynamics then permitted brief progress but
could not undo the accumulated closed-loop state displacement.

Artifact:
`research_artifacts/dynamic_uncertainty_residual_stage1_amendment2_probe/`.

