# L92 dynamic covariance safety confirmation preregistration

Date: 2026-07-18

## Motivation and separation from L91

L91's original elapsed-time gate failed, so it is not re-labelled as a
positive primary result.  It nevertheless produced a large held-out safety
signal: +0.30 success and -0.15 collision for a scene-context oracle.  L92 is
a new, explicitly safety-focused confirmation on six motion geometries that
already existed before L91 and were not selected after viewing L91 outcomes.

## Frozen system

The following remain fixed across every condition:

- the L57 ICODE checkpoint and TorchScript inference path;
- MPPI horizon, sample count, cost, actuation limits and command delay;
- MuJoCo robot model and physical mismatch;
- synthetic LaserScan, local obstacle layer, scan_guard and safety arbiter;
- five covariance candidates introduced before this confirmation.

No residual retraining, RL policy fitting or candidate tuning is permitted in
this gate.

## Context block

Six pre-existing moving-obstacle geometries are used: anchor, fast, large,
reverse, diagonal and offset.  Each motion configuration retains seeded phase,
period and endpoint variation where defined.

- selection seeds: `20270201--20270204`;
- held-out evaluation seeds: `20270301--20270306`;
- 5 candidates;
- total: `6 * (4 + 6) * 5 = 300` closed-loop episodes.

The schedule is randomized with seed `2026071841`, resumable and shardable.
Failures remain in the analysis with their full elapsed duration.

## Selection rule

For each motion geometry and for the pooled global comparator:

1. maximize success rate;
2. among ties, minimize collision rate;
3. among safety ties, minimize elapsed episode time;
4. break residual ties by final goal distance and candidate name.

## Primary gate

On held-out evaluation seeds, context oracle minus strongest global fixed must:

1. select at least two candidates and differ from global in at least one
   context;
2. have hierarchical-bootstrap 95% CI lower bound strictly above zero for
   paired success difference;
3. have hierarchical-bootstrap 95% CI upper bound at or below zero for paired
   collision difference.

Motion geometry is resampled first and seed second.  The bootstrap seed is
`2026071842`.  Elapsed time, final distance, jerk, minimum clearance and safety
interventions are secondary and cannot rescue a failed primary gate.

## Stopping rule and claim boundary

- If the gate fails, no obstacle-context deployment policy is trained from
  this branch; the strong L91 signal remains exploratory.
- If it passes, development may proceed to a policy whose inputs are limited
  to real-robot-compatible LaserScan geometry and temporal scan-flow features.

Even a pass establishes oracle headroom only.  It does not establish a
deployable RL improvement until a learned policy is evaluated on new seeds
without scene labels or simulator obstacle truth.
