# Residual-Dynamics Stage 1 Amendment 6 Preregistration

Status: **frozen before Amendment 6 closed-loop outcomes**  
Date: 2026-07-23  
Scope: route-preserving speed-only residual authority  
RL status: **disabled**

## Single intervention

Amendment 6 retains the complete Amendment 5 shield but restricts residual
control authority to `v_cmd`. Every candidate starts from the concurrent
nominal MPPI sequence; only its speed channel is replaced by the residual
planner's speed sequence. `omega_cmd` remains exactly nominal for the complete
36-step horizon.

The resulting hybrid sequence is rerun through both nominal and residual
dynamics and must satisfy the unchanged hard-risk, zero nominal-risk increase,
`0.10 m` model-view position tube and `0.02 m` progress-noninferiority checks.
Failure returns the exact nominal plan.

This targets the observed cumulative route displacement without changing
thresholds. It preserves a possible residual benefit in longitudinal dynamics
and obstacle-crossing timing while removing learned steering-route authority.

## Development screen

- observed development seed `730100003`;
- three frozen checkpoint blocks plus one shared nominal;
- randomized schedule seed `730199995`;
- collision stops the screen;
- no sealed seed is authorized.

The Amendment 5 gates remain unchanged, including nonzero shield use and the
150 ms compute gate. If this safety screen fails, no further wrapper or
threshold amendment is authorized for the current checkpoints; the next
residual work must collect obstacle-interaction data and retrain.

