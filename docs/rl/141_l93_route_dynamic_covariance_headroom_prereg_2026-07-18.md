# L93 route × dynamic-crossing covariance headroom preregistration

Date: 2026-07-18

## Question

L89 proves that route geometry can select useful MPPI exploration covariance
in clean path tracking.  L92 rejects dynamic scene classification.  L93 asks a
narrower causal question before implementing a gate: on the *same straight
route*, does a moving obstacle crossing the route create reproducible headroom
for switching away from the route policy's `speed` covariance toward a more
turn-oriented covariance?

## Design

Five contexts share the same 5.5 m straight polyline and frozen high-dynamic
plant:

1. clean route;
2. slow perpendicular crossing;
3. fast perpendicular crossing;
4. larger reverse crossing;
5. diagonal crossing.

The moving cylinders have seeded phase, period scaling and endpoint jitter.
The planner receives them only through synthetic LaserScan and the existing
local obstacle/safety chain.

Candidates are `baseline=[1,1]`, `turn=[0.75,1.75]` and
`speed=[1.75,0.75]`.  ICODE, MPPI horizon/sample count, cost and safety remain
frozen.

- selection seeds: `20270401--20270403`;
- evaluation seeds: `20270501--20270505`;
- total: `5 * 3 * 8 = 120` closed-loop episodes.

## Selection and gate

Candidate selection remains lexicographic: success, collision, elapsed time,
final distance.  The held-out primary gate requires:

1. at least two selected candidates and at least one context different from
   the strongest global fixed candidate;
2. paired success-difference 95% CI lower bound above zero;
3. paired collision-difference 95% CI upper bound at or below zero.

Elapsed time, final distance, jerk, clearance and safety interventions are
secondary.  Context and then seed are resampled in the hierarchical bootstrap.

## Stopping rule

Failure stops the proposed scan-risk covariance fallback; it would mean that
the fixed action change itself lacks reproducible headroom.  Passing only
authorizes implementing and independently evaluating a LaserScan/scan-flow
gate.  Scene labels and obstacle truth can never enter the deployed gate.
