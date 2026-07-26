# Single dynamic obstacle v13 mechanism contract

This is an outcome-informed engineering development study, not confirmatory
effect estimation. It tests one mechanism family:
`predictive_counterflow_active_mobility`.

The controller turns the already available causal obstacle velocity estimate
into a preferred escape direction containing two components:

1. lateral motion away from the predicted swept line;
2. motion opposite the obstacle velocity.

When associated-obstacle surface range is below the frozen trigger distance,
the same six-slot, 600-rollout candidate lattice is activated even if scan-flow
TTC temporarily drops out. At the critical distance, straight/left/right
reverse templates are reserved so severe danger can choose a risk-vetted
reverse maneuver instead of being forced to stop.

The candidate forward speed limit is raised from 0.35 to 0.45 m/s. The
probabilistic collision evaluator, hard threshold, safety margin, IMM, Actor,
ICODE, HSS, rollout count, horizon, obstacle generator and episode limit remain
frozen. V4 remains behaviorally unchanged because all new settings default off.

The first gate contains four paired, previously opened seeds. It passes only if
there is at least one additional safe success, at least one prevented collision,
a net collision reduction, no new paired collision, no lost V4 success, all
decisions retain 600 rollouts, and both the counterflow and distance-trigger
paths are exercised. Failure evidence is retained and the family is stopped
after the frozen attempt limit.
