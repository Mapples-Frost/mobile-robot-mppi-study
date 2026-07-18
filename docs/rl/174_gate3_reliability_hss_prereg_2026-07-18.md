# Gate 3 preregistration: reliability-calibrated hybrid sampling

Date: 2026-07-18

This document freezes Gate 3 before training the ensemble or running any
adaptive-sampling closed-loop comparison. Gate 3 follows the final research
plan and does not introduce a residual-conditioned recurrent policy,
Adaptive-K, online learning, or a learned black-box gate.

## Question

At the same MPPI rollout budget, can online signals available to the robot
identify when persistent SAC Actor samples deserve more or less sampling
authority than a fixed 30% guided ratio?

## Frozen mechanism

Three value-aligned ICODE members are independently initialized from the three
qualified L57 ICODE blocks and fine-tuned on an episode-bootstrap sample of
the L183 **training split only**. Actor, critic and base checkpoints are
frozen.

The MPPI prediction model is the unscaled ensemble mean:

$$
\dot x =
f_{\mathrm{nom}}(x,u)
+
\frac{1}{3}\sum_{i=1}^{3}f_{\mathrm{ICODE}}^{(i)}(x,u).
$$

Reliability does not attenuate this residual. It controls only the number of
persistent Actor-guided candidates:

$$
\rho_{\mathrm{RL}}\in\{0,\;0.3,\;0.6\}.
$$

The dynamics confidence is the conservative minimum of:

1. normalized ensemble-disagreement confidence along the Actor mean rollout;
2. minimum ICODE training-support confidence along that rollout;
3. completed-transition innovation confidence after its causal warm-up.

Actor confidence is obtained from its frozen observation-normalizer support
distance. The authority is:

$$
c_{\mathrm{authority}}=
c_{\mathrm{dyn}}c_{\pi}.
$$

Thresholds map authority to low, medium or high sampling ratios. The current
Actor mean rollout estimates authority for the **next** control cycle. This
one-cycle causal lag preserves the joint Actor batch and avoids adding a
second Actor pass. Only past observations affect the current HSS allocation.

## Calibration separation

- Training: L183 train split only.
- Threshold selection: L183 validation split only.
- Offline mechanism evaluation: L183 test and unseen splits, unopened during
  threshold selection.
- Closed-loop development: seeds 32--34.
- Closed-loop independent confirmation, opened only if development passes:
  seeds 35--39.

Episode is the independent unit. Overlapping rollout windows are never counted
as independent experimental replications.

## Equal-budget comparator

Both methods use:

- the same three-member ensemble mean;
- the same frozen L175 Actor and critic;
- the same terminal critic weight;
- 100 model rollouts per control decision;
- two MPPI iterations;
- the same scenes, physics domains, seeds and safety chain.

Arms:

1. fixed HSS: 30% persistent Actor samples;
2. reliability HSS: 0%, 30% or 60% persistent Actor samples.

Treatment order is randomized inside every
scene-by-physics-domain-by-seed block. No method may inspect the physics-domain
label at runtime.

## Gate 3A: confidence calibration

On both untouched offline splits:

- mean rollout error must be ordered
  `high confidence <= medium confidence <= low confidence`;
- all occupied bins must contain at least two independent episodes;
- the confidence/error episode-level rank association must be non-positive;
- low-confidence examples must occur, so the gate is not a constant 30% rule.

Failure stops closed-loop adaptive HSS work and retains the ensemble only as a
prediction ablation.

## Gate 3B: development

Reliability HSS passes development only if:

- at least one primary endpoint improves: success, final goal distance or
  stuck steps;
- success does not decrease and collision count does not increase;
- final goal distance does not worsen by more than 2%;
- control jerk does not worsen by more than the prospectively frozen 1%
  engineering-equivalence margin;
- average total model rollout count is identical;
- both low and high/medium authority are exercised;
- mean serial planner time does not increase by more than 10%.

The 1% jerk margin is fixed before Gate 3 data and is not applied
retroactively to L191.

## Gate 3C: independent confirmation

The five confirmation seeds remain sealed until Gate 3B passes. Confirmation
requires the same directions for primary outcomes, zero collision regression,
the 1% jerk margin, and a seed-cluster bootstrap interval favorable for at
least one of final distance, success or stuck steps.

The result remains bounded to the tested, solvable task families. Narrow
corridor is retained as a diagnostic planning floor and is not allowed to
dominate the dynamics-reliability calibration.
