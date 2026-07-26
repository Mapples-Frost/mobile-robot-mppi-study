# V13 standard-geometry held-out review

## Frozen outcome

The four-pair outcome-blind held-out gate completed without execution
failures. The frozen decision is `development_gate_fail`.

- V4 Full: 4/4 successes, 0/4 collisions.
- V13 counterflow: 4/4 successes, 0/4 collisions.
- Safe-success gain: 0.
- Prevented collisions: 0.
- New paired collisions: 0.
- Lost V4 successes: 0.
- Mean minimum-clearance delta: -0.0353 m.
- Every controller decision retained exactly 600 rollouts.

The counterflow mechanism was exercised for 143 steps, including 43
near-distance-trigger steps. It therefore executed as intended, but the
held-out sample exhibited a baseline ceiling: every V4 episode was already a
safe success. The sample cannot demonstrate a success-rate or collision-rate
improvement.

V13 remained collision-free and successful on all four pairs. On the two OOD
pairs it completed 26 and 11 steps earlier than V4, respectively. This is
useful engineering evidence but does not satisfy the frozen superiority gate.

## Scope decision

Seed 750300015 remains preserved in the prior R1 evidence as `stress_only`; it
was not used in this held-out gate. No held-out seed was removed or replaced
after outcomes were opened.

This outcome is archived as a failed superiority gate. It must not be relabeled
as a pass. Any subsequent test needs a separately preregistered,
controller-independent difficulty band that avoids both near-center late
interception stress cases and the all-success baseline ceiling observed here.
