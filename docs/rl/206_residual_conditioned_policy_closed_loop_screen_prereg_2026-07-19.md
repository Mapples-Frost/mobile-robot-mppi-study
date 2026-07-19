# L206 Residual-Conditioned Policy Closed-Loop Screen

L205 produced a competence-preserving but small direct-Actor validation gain.
Before further training changes, L206 tests whether that correction improves the
actual downstream RL-Driven MPPI proposal distribution.

## Frozen comparison

- Base: ordinary ICODE ensemble + frozen L185 Actor prior.
- Proposed: the same ICODE ensemble + L205 residual-conditioned bounded Actor.

Both use identical total rollout budget, horizon, proposal fractions, MPPI
costs, LaserScan perception, scan_guard and safety arbitration. Terminal value
is disabled so the policy-prior coupling is the only changed factor.

## Development blocks

- route: high-dynamic reverse-S terminal development route;
- physics: nominal-seen and combined-unseen;
- new seeds: 575, 576, 577;
- paired metric priority: success/collision, cross-track RMSE, final distance,
  jerk and planner compute time.

This is a development screen, not sealed evidence. The correction advances only
if success and collision are noninferior and pooled cross-track RMSE improves
without a greater than 5% jerk regression. Otherwise a new correction objective
must be preregistered; these seeds may not be reused for confirmation.
