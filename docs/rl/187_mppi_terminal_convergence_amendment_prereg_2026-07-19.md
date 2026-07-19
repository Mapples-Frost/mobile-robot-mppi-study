# L187 MPPI terminal-convergence amendment (pre-registered)

Date: 2026-07-19  
Status: frozen before running development seed 553  
Scope: development-only; sealed seeds 561--565 remain unopened

## Motivation

The first integrated L185 screen used development seeds 551--552 on the
reverse-S path.  The path-conditioned Actor itself completed every direct
control episode, whereas most ICODE--RL--MPPI arms stopped 0.28--0.35 m from
the endpoint under a fixed 0.25 m success tolerance.  Cross-track error was
already low (approximately 5 cm), so this is a terminal-convergence failure,
not a route-following failure.

No success tolerance, episode length, collision definition, or path geometry
will be changed in this amendment.

## Single admissible controller change

The only candidate change is applied identically to all four factorial arms:

```text
planner.goal_terminal_weight: 25.0 -> 50.0
```

All other planner, plant, ICODE, Actor, sampling-budget, safety, and evaluation
parameters remain fixed.  In particular:

- position tolerance remains 0.25 m;
- the maximum episode length remains 360 control steps;
- terminal velocity and yaw-rate penalties remain unchanged;
- the Actor checkpoint remains L185 seed 20261901, selected at 50,000 steps;
- the rollout budget remains K=100 over two paper iterations;
- scan guard and safety arbitration remain enabled.

This change increases the cost of stopping just outside the target without
weakening the requirement that the robot decelerate safely at the endpoint.

## Development design

### Diagnostic block

Development seed 553 is evaluated twice:

1. T0: frozen original terminal weight 25;
2. T1: candidate terminal weight 50.

Each condition contains all four randomized factorial arms.  T1 is selected
only if:

1. it increases the number of successful arms relative to T0;
2. it causes no collision;
3. mean cross-track RMSE does not regress by more than 10%;
4. mean control jerk does not regress by more than 10%.

If the criteria fail, no additional terminal-cost candidates will be tried in
this experiment family without a new written amendment.

### Development confirmation

If T1 passes seed 553, it is frozen and evaluated on unused development seeds
554--555.  The MPPI integration Gate passes only if, across these two seeds:

1. Full Proposed succeeds in both episodes;
2. Full Proposed has zero collisions;
3. Full Proposed cross-track RMSE is no worse than ordinary fixed by more than
   5% on average;
4. Full Proposed control jerk is no worse than ordinary fixed by more than 10%
   on average.

Only after this Gate passes may the pre-registered sealed seeds 561--565 and
the two frozen L186 holdout geometries be opened.

## Interpretation boundary

Passing this amendment would establish that the path-conditioned Actor can be
integrated into the current MPPI controller without the previously observed
premature terminal stop.  It would not yet establish the paper's final
cross-layer interaction claim.  That claim still requires the sealed
factorial comparison and interaction-effect analysis.
