# Dynamic Actor expanded development Amendment 1 preregistration

## Scope

This amendment repeats the already-opened 24-pair single-dynamic-obstacle
development matrix. It does not open sealed seeds and does not overwrite the
original failed expanded-development evidence.

## Reason for amendment

The original matrix exposed one new Candidate collision and one lost Source
success. Causal replay of development seed `730100208` showed that a static
reverse escape intent remained authoritative after a forward candidate became
simultaneously lower-risk and lower-cost. Repeating a finite yaw command also
turned the intended passing arc into an over-rotation, while the local recovery
layer could override a currently vetted planner escape.

Amendment 1 freezes three changes before execution:

1. A reverse intent may commit to forward traversal only when the same
   budgeted lattice candidate is both the minimum-risk feasible candidate and
   the minimum-cost feasible candidate, with strictly lower forecast risk.
2. The committed finite arc is converted into a fixed world-frame heading, so
   the robot straightens after acquiring the passing direction.
3. Dynamic recovery cannot override a planner candidate that remains selected
   and forecast-vetted in the current cycle.

## Frozen treatment

- Candidate checkpoint: Dynamic Actor V5 Amendment 6 update 250.
- Source checkpoint: frozen V3 source Actor.
- Same-cycle guided-cost filter and proposal-only integration on both arms.
- Completion handover distances: `0.4 m` and `0.8 m`.
- Six emergency candidates remain inside the fixed `600`-rollout budget.
- Temporal trigger: `TTC <= 1.5 s`.
- Intent duration: `16` control steps, re-evaluated against the current causal
  forecast every cycle.
- Pareto forward commit enabled.
- No obstacle future truth is available to tracking, planning, or arbitration.

## Frozen design and gate

The seed schedule, arm-order balance, batch structure, and all gate checks are
identical to revision 1. No tuning, seed replacement, early stopping, or
artifact deletion is authorized during the matrix. The amendment passes only
if all 24 pairs complete, there are no new Candidate collisions or lost Source
successes, Candidate completion/steps/final distance are non-inferior, at least
one pooled endpoint improves strictly, at least two batches are non-regressive,
the same-cycle filter is exercised on every arm, and all bound hashes verify.
