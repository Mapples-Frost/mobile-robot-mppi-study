# Residual Runtime Stage 3 Closed-Loop Preregistration

Date: 2026-07-23  
Status: preregistered before optimized MuJoCo episodes.

## Authorized implementation

The selected runtime arm uses the unchanged Stage 2 checkpoints with:

- CUDA inference;
- device-resident float64 nominal/RK4 integration;
- float32 residual-network inputs and outputs promoted to float64;
- CUDA Graph replay for the fixed H36, K600 rollout;
- one completed-trajectory host copy.

The Amendment 2 screen passed all three checkpoint blocks with worst-block P95
`15.19 ms`, 83.8% below its concurrent legacy-CUDA baseline, and maximum
trajectory difference below `8e-16`.

## Complete-episode design

All 12 cells in

`{730100006,730100008,730100010} x {nominal,R01,R02,R03}`

will be rerun. No prior episode is carried forward. The complete episode is the
independent unit; obstacle seed and checkpoint seed are blocks. RL is disabled
and sealed seeds remain unauthorized.

Everything except runtime backend remains frozen: Amendment 17 environment,
Change-Aware IMM, collision-risk calculation, H36, K600, RK4, checkpoints,
residual mask and shield thresholds.

## Gates

The optimized matrix passes only if:

- every episode reaches the goal;
- no residual block increases collision count;
- median completion and clearance deltas retain the Stage 2 noninferiority
  limits;
- residual participation remains nondegenerate;
- all checkpoint, forecast and finite-diagnostic contracts pass;
- maximum residual-controller P95 is at most `100 ms`.

After the automatic gate, a paired audit will compare the optimized and Stage 2
development traces. Tiny numerical equivalence in an isolated rollout does not
guarantee identical receding-horizon episodes, so control, position,
completion, risk and shield-acceptance differences will be reported rather
than assumed away.
