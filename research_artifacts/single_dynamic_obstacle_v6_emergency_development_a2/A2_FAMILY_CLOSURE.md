# A2 family closure

## Result

A2 was a pre-registered development correction on the same eight paired,
already-opened diagnostic seeds. It changed only the scan-arbiter contract:
when reactive geometry required reverse motion, a positive vetted planner
command was vetoed in favor of the existing reactive reverse command.

The engineering gate failed:

- safe success: 2/8 versus 2/8 for V4;
- collisions: 4/8 versus 4/8 for V4;
- prevented collisions: 1;
- new paired collisions: 1;
- lost V4 successes: 1;
- mean minimum-clearance delta: +0.04498 m;
- every decision used exactly 600 rollouts.

The unfavorable transitions were unchanged at the outcome level:

- `id/750100038`: V4 collision to A2 safe non-completion;
- `ood/750200036`: V4 safe non-completion to A2 safe success;
- `ood/750200020`: V4 safe success to A2 collision.

## Stopping decision

A1b produced the first safety regression in the
`causal_temporal_emergency_candidates` family. A2 produced a second safety
regression despite the direction-veto correction. The frozen stop rule therefore
closes this mechanism family. No further threshold, hold, veto, or recovery
parameter tuning in this family is authorized.

The A2 result is retained as a failed development result, not as confirmatory
evidence. A fresh held-out qualification is not permitted for this family.

## Global review required

The repeated regression shows that the failure is not isolated to one
forward-command arbitration branch. The broader issue is that emergency
candidate selection and safety arbitration are not jointly certifying a
complete, collision-free escape trajectory. Future work, if any, must begin
with a global audit of candidate feasibility, state handoff, and safety-layer
authority. It must not continue local tuning of the A1/A1b/A2 controls.
