# Compound navigation pilot — September 9

The user requests fresh experimental design aimed at a strong ICRA submission,
without treating previous geometry or algorithm choices as binding. This pilot
is new development, not a final benchmark or a claim of novelty.

Hypothesis: sequential changes in navigation geometry create changing planning
requirements within a single episode. Test whether online budget adaptation
helps beyond fixed budgets. Neither a need for higher K nor longer H is assumed.
The labels gate_then_turn and turn_then_gate describe gate-center order, not a
guarantee that the policy must pass through the gaps. Finite walls permit bypass;
inspect recorded trajectories and report bypass behavior.

Three new layouts: two offset gates with center order -0.15/+0.45 m; the reverse
order; and three alternating gates at centers -0.35/+0.35/-0.35 m. Two opening
widths 0.9/0.7 m. Shared mirrored geometry and wall-pair translations for each
seed; initial state, 4.5 m goal, speed caps, task timeout, plant and cost unchanged.
Two speeds x two fresh environment seeds 9690101/9690102 x six layouts = 24
paired environments. Twelve methods = 288 episodes:

- nominal and ICODE fixed at K/H=128/27,160/29,256/24 (six controls);
- ICODE full KH and masked KH at both existing final initializations (four);
- ICODE H-only at both existing final initializations (two).

All learned policies are transferred without retraining. This cheaply tests
operating-region feasibility; it cannot establish the performance of a newly
trained method or reproduce a published baseline. No best-seed selection.
The fixed bank is only a pilot bank: final comparisons require development
budget tuning including cheaper budgets and single-axis K control if claiming
joint K/H superiority. Retain all cells, outcomes, cost and measured compute.
Serial run order randomized inside paired environments by the reused evaluator.
Training-seed separation, checkpoint completion and source hashes checked.

Run after the already active geometry batch to avoid overlapping timing loads.
One finite queue waits at most four hours and fails if the dependency is absent.
Do not change this experiment after it starts. Any changes require a new round.

Select at most one follow-up by the September 10 decision deadline, using both
initializations, task quality, collisions, compute and adjacent-width robustness.
If this pilot fails, do not grow the scene grid indefinitely. Independent
validation with separately frozen choices and fresh data remains necessary.

## Broader experiment package, not yet implemented

1. Within-route geometry transitions (this pilot).
2. Changing dynamics / model mismatch, only if the first decision supports a
   mechanism worth testing and implementation fits the remaining calendar.
3. Performance versus measured compute; explicit simulated budget restrictions
   must be labeled separately from measured hardware timing.

Desired outputs: success/collision/completion-time comparison; complete
quality-compute tradeoff; K/H time traces aligned to geometry transitions;
essential ablations; independent generalization. Never manufacture a desired
effect size or hide failures to improve a plot.

## Bounded literature reconnaissance

Accessed 2026-09-09 via web search: queries `adaptive horizon sample size MPPI
reinforcement learning computation Bohn` and `MPPI adaptive sampling horizon
computational budget robot navigation 2025 2026`. Targeted, not exhaustive.

- Bohn et al., Reinforcement Learning of the Prediction Horizon in Model
  Predictive Control, arXiv:2102.11122: https://arxiv.org/abs/2102.11122 .
  Abstract verified: state-dependent RL horizon adaptation already exists.
  Full reproduction protocol still needs full-text inspection; an H-only
  local policy must not be called a reproduction of this paper.
- Single-Instance Sampling MPPI author project:
  https://euncheolim.github.io/Single-Instance-Sampling-for-Real-Time-Task-Space-MPPI-Control/
  Search result describes distance-adaptive horizon on manipulators. Candidate
  related work, not yet a validated implementation comparison.
- Adaptive Dynamics Planning for Robot Navigation: search returned a candidate
  author PDF at https://cs.gmu.edu/~xiao/papers/adp.pdf but opening redirected
  to the department homepage. Full text unverified; do not claim reproduction.

Novelty is unresolved. A successful pilot alone is not evidence of publishability.
