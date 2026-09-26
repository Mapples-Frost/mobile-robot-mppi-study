# L90 local covariance-context oracle preregistration

Date: 2026-07-18

## Question

Does a route contain reproducible local segments whose best MPPI sampling
covariance differs, or is L89's one-action-per-route policy already sufficient?
No segment-level RL policy will be trained unless this local oracle gate passes.

## Experimental unit and blocking

The independent unit is one 3.0 s MuJoCo branch initialized at a frozen metric
progress anchor.  Four support routes each provide anchors at 10%, 35% and 60%
of route length.  Every anchor is paired across `narrow` and `speed`
covariances, with physics domain and seed blocked.  Run order is seeded and
randomized.

The context key contains route and anchor only.  The two physics domains are a
nuisance block and are forbidden inputs, preventing a simulator-domain label
from becoming a deployable feature.  Initial position and tangent come from
the route, initial forward speed is 0.35 m/s, and terminal anchors are excluded.

Selection seeds `20269801--20269802` and evaluation seeds
`20269901--20269903` are disjoint.  Total schedule: 240 branches.

## Selection and primary gate

For each local context, discard colliding candidates.  Among candidates whose
selection RMSE is within 2 mm of the best local RMSE, choose maximum metric
route progress.  Select the strongest global fixed candidate by the same
pooled rule.

On evaluation seeds, the local oracle passes only if:

1. at least two candidates are selected and at least 25% of local contexts
   differ from the strongest global candidate;
2. collision rate does not increase;
3. the hierarchical-bootstrap lower 95% limit of paired local-progress change
   is greater than zero;
4. the upper 95% limit of paired cross-track RMSE change is at most +2 mm.

Bootstrap hierarchy is local route/anchor context first, then physics/seed
branches within context.  This is a non-deployable learnability oracle, not an
RL result.

