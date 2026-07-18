# L91 dynamic-obstacle covariance-context oracle preregistration

Date: 2026-07-18

## Motivation

L90 found only one reliable route-local covariance change among twelve
contexts, so segment-level path switching is not justified.  L91 instead
tests the advisor-motivated hypothesis that obstacle motion changes the useful
MPPI exploration regime.

## Design

Frozen ICODE-MPPI is evaluated in one clean point-goal scene and three
previously method-blind calibrated dynamic-obstacle scenes.  Five fixed
covariance candidates are paired by scene and seed.  Dynamic obstacle phase,
period scale and endpoint jitter remain seed-controlled plant variables;
scene is the oracle context only and will not be an input to any deployable
policy.

Selection seeds `20270001--20270003` and evaluation seeds
`20270101--20270105` are disjoint.  The 160-episode schedule is randomized
with seed `2026071831` and is resumable.

## Selection rule

For each scene and for the pooled global comparator:

1. maximize success rate;
2. among ties, minimize collision rate;
3. among safety ties, minimize elapsed episode time;
4. break remaining ties by final goal distance and candidate name.

Failures retain their full elapsed duration rather than being silently dropped.

## Primary gate

On held-out evaluation seeds, context oracle minus strongest global fixed must:

1. use at least two candidates and differ from global fixed in at least one of
   four scenes;
2. have nonnegative mean success delta and nonpositive collision delta;
3. have a hierarchical-bootstrap upper 95% bound below zero for elapsed time.

Scene is resampled first and seed second.  Clearance, final distance, jerk and
safety interventions are reported as secondary outcomes.  Passing L91 only
authorizes development of a LaserScan-observable obstacle policy; it is not a
deployable RL result.

