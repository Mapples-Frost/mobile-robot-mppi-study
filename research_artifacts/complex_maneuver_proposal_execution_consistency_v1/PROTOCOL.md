# Proposal-to-Execution Consistency v1

## Frozen purpose

This is a development-only, falsifiable test of one shared mechanism. It does
not retrain the Actor and does not modify Risk, ICODE, Safety, maps, MPPI cost
components, or the fixed 600-rollout budget.

The bootstrap Gate D established that supervised proposals can be risk-feasible
and enter the elite set, yet they rarely survive into the executed control and
produce no success gain. The next test therefore targets arbitration, not
candidate generation.

## Code audit

The supervised heads are inserted by replacing fixed-budget guided rows
(`rl_driven_mppi.py:2020-2040`). Candidate feasibility is evaluated before
optimization (`rl_driven_mppi.py:2432-2522`).

The existing same-cycle filter compares the best feasible guided cost with the
best feasible Gaussian cost. With the current relative margin of zero, any
positive disadvantage removes the whole guided population
(`rl_driven_mppi.py:2602-2621`). This can discard all three supervised heads
even when an individual head is physically and probabilistically feasible.

The optimizer then produces a weighted mean of elites
(`rl_driven_mppi.py:2637-2669`), not a discrete selected candidate. The existing
`supervised_selected_count` records the label of the maximum-weight elite
(`rl_driven_mppi.py:2684-2697`), so it is not a complete measure of Actor
influence. Geometry and probabilistic guards can subsequently replace the
action (`rl_driven_mppi.py:2948-2965`).

## Phase Zero

Add diagnostics only. Measure the supervised elite weight mass, the difference
between the weighted update with and without supervised elites, and the change
between pre-guard and post-guard actions. Phase Zero must be behaviorally exact
for Actor-off and Actor-on executions; only new diagnostic fields may differ.

No mechanism result is interpreted until this equivalence test passes.

## Single intervention

Retain the same-cycle filter and Gaussian isolation. Test only the existing
relative cost margin in frozen ascending order:

1. 0.02
2. 0.05
3. 0.10

Each value is shared across all three maps. A later value is run only when the
earlier value causes no collision regression but fails the shared improvement
gate. Values cannot be selected per map.

The mechanism is intended to preserve a bounded amount of influence from safe
multi-stage maneuvers when a short-horizon Gaussian proposal is only marginally
cheaper. It does not force an unsafe candidate, bypass a hard filter, or add
rollouts.

## Development gate

For every opened map/seed block, treatment must:

- introduce no collision regression;
- improve success or terminal goal distance;
- increase counterfactual first-action influence from supervised heads;
- increase the fraction of supervised influence that survives the final guard;
- retain only 0 or 600 rollouts per control cycle.

Pooled Safety intervention fraction may not increase. A larger selected count
alone is not a pass.

Stop after three non-improving rounds or two consecutive safety regressions.
Any development pass requires a new, disjoint, unopened paired validation.
Paid-server or formal execution remains unauthorized.
