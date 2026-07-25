# Residual-Dynamics Stage 1 Amendment 5 Result

Status: **failed at the preregistered first residual block**  
Date: 2026-07-23  
RL status: **disabled**

The concurrent nominal-noninterference shield accepted residual plans on 50% of
steps in checkpoint block 0 but still collided, so the probe stopped.

- collision: yes;
- shield acceptance / fallback fractions: 0.50 / 0.50;
- maximum reliability authority: 1.0;
- final goal distance: 6.013 m;
- total planner p95: 472.86 ms.

The result refutes step-local dual-model risk nonincrease as a sufficient
closed-loop safety certificate. Repeated locally admissible steering changes
can accumulate into a different route. Amendment 6 removes residual authority
over `omega_cmd` and retains only speed-command authority inside the same
shield.

Artifact:
`research_artifacts/dynamic_uncertainty_residual_stage1_amendment5_probe/`.

