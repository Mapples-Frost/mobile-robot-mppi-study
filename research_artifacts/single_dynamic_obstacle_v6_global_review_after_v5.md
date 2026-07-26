# Single-dynamic-obstacle global review after v5

## Evidential boundary

The v5 32-pair held-out cohort is now opened and may be used only for diagnosis
and v6 development.  It cannot be reused for v6 qualification or formal effect
estimation.  V5 remains frozen as a failed result at tag
`checkpoint/v5-held-out-qualification-fail-20260726`.

## Paired outcome inventory

- V4 Full: 19 safe successes, 10 collisions, 3 safe non-completions.
- V5: 19 safe successes, 11 collisions, 2 safe non-completions.
- V5 produced no safe-success conversion and introduced one OOD collision.

## Closed-loop failure mechanism

All ten V4 collision traces reached probabilistic hard risk.  At those cycles,
the ordinary MPPI candidate feasible fraction collapsed to zero and the active
fallback was the minimum-risk or minimum-accumulated-risk candidate.  The stop
candidate itself commonly had predicted risk 1.0.  The scan-guard reactive
escape then began only after clearance had already collapsed, leaving too little
time for the actuator-limited unicycle to create separation.  This is an
upstream candidate-availability and intervention-timing deficit, not merely a
post-conflict release delay.

The three V4 safe non-completions showed a second symptom: 69--72 near-zero
velocity cycles and 26--59 reverse cycles, with final goal distance still
1.19--2.07 m.  A deadline-only forward floor did not generalize because it did
not establish that the current hazard had ended.

## v6-A1 hypothesis

V6-A1 enables the already implemented temporal emergency candidate path at a
current online TTC of 1.5 s, holds its directional intent for 16 cycles, and
requires the escape command to be vetted by the same-cycle probabilistic cost.
It retains the stop candidate and allows a forward Pareto choice only when risk
is non-worse.  It does not use future truth, the offline conflict window, or an
episode deadline.

The first development panel contains four collision modes, two safe
non-completion modes, and two success-preservation sentinels from the now-opened
v5 cohort.  Passing this panel is only a mechanism screen.  A disjoint opened
replication panel and then a newly sealed, never-opened qualification are both
required before any formal server run can be proposed.
