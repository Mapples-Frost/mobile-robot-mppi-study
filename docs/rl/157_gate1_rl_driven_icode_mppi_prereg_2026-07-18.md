# Gate 1 preregistration: RL-Driven ICODE-MPPI simple combination

Date frozen: 2026-07-18

## Scientific question

Does a faithful RL-Driven MPPI integration rescue a learned policy that was
unsafe or ineffective when it replaced the whole MPPI proposal mean? The
mechanistic hypothesis is that the learned policy should contribute a bounded
subset of candidate sequences, while conventional warm starts and the shifted
previous solution remain available. A learned critic may then add long-horizon
information at the rollout terminal state. ICODE is used for every candidate
rollout and remains separated from the MuJoCo true plant.

This is the deliberately simple-combination baseline. It is not the proposed
value-aligned method and will not be described as a new algorithm.

## Frozen implementation factors

- Prediction model: ICODE continuous-time control-affine residual model.
- Memory cost: disabled.
- Perception and safety: existing LaserScan -> local obstacle layer -> MPPI ->
  scan_guard chain, unchanged.
- Candidate sources: learned RL proposal, shifted previous MPPI solution, and
  conventional goal warm start.
- Fixed source fractions: 0.30 RL, 0.40 shifted solution, 0.30 conventional.
- Equal total sample budget for every comparator.
- Elite update: best 20% of the pooled candidates.
- Online refinement: two iterations, with simultaneous mean and diagonal
  time-indexed covariance updates.
- Critic statistic: expected twin-critic value, not CVaR. Distributional CVaR
  is reserved for a later risk-sensitive ablation.
- Terminal term: one fixed, configured coefficient. The coefficient is chosen
  on development seeds only and then frozen before confirmation.
- The learned proposal is never sent directly to the robot. MPPI optimization
  and safety arbitration remain authoritative.

## Frozen model blocks

Three independently trained blocks are paired by index:

1. RL seed 20260784, checkpoint `l81_static_quantile_cvar_seed20260784_30k_20260717_v1/checkpoints/step_000030000.pt`; ICODE seed 20261201.
2. RL seed 20260785, checkpoint `l81_static_quantile_cvar_seed20260785_30k_20260717_v1/checkpoints/step_000030000.pt`; ICODE seed 20261202.
3. RL seed 20260786, checkpoint `l81_static_quantile_cvar_seed20260786_30k_20260717_v1/checkpoints/step_000030000.pt`; ICODE seed 20261203.

The 30k checkpoints are used rather than validation-selected initial
checkpoints because Gate 1 explicitly evaluates a trained terminal critic.
This choice is frozen before outcome evaluation.

## Comparators and ablations

1. ICODE-MPPI: conventional MPPI with ICODE rollout.
2. Always-on RL prior + ICODE: historical direct-mean integration.
3. Hybrid-policy ICODE-MPPI: pooled candidate sources, terminal critic off.
4. RL-Driven ICODE-MPPI: pooled candidate sources and expected-Q terminal term.

Stricter mechanism control added before multi-block evaluation (2026-07-18):
Hybrid-no-RL ICODE-MPPI uses the same rollout budget, elite update, iterations,
and covariance adaptation as (3), but duplicates the conventional proposal in
the RL-labelled source. Block 1 exposed a potentially large optimizer
confound; adding this control makes the test harder and does not relax any
passing threshold. Block-1 outcomes predate this amendment and are treated as
exploratory until the added control and remaining blocks are complete.

The primary comparison is (4) versus (1). Comparisons (2) and (3) test whether
any gain comes from limiting policy authority or from the critic.

## Experimental units and schedule

- Independent training seed pair is a model block.
- Episode seed is a repeated measure within a model block.
- Controller steps are technical measurements, never independent replicates.
- All variants use paired episode seeds and common MPPI sample seeds.
- Run order is shuffled within each model block.

Development episode seeds (open): 22410101, 22410102, 22410103.

Confirmation episode seeds (sealed until the development gate passes):
22410111, 22410112, 22410113, 22410114, 22410115.

The initial mechanism check uses the static blocking scene and the fixed unseen
physics domain from the existing cross-layer protocol. A passing method is then
checked on clean and dynamic-crossing scenes without retuning.

## Outcomes

Primary outcome: paired episode task cost, defined before aggregation from goal
progress/final distance, collision, and control regularity using the existing
experiment metrics.

Safety outcomes: success rate, collision rate, minimum clearance, scan_guard
override count, stuck steps, and spin steps.

Efficiency outcomes: mean and maximum planner time, plus success-conditioned
time to goal.

Mechanism diagnostics: candidate count per source, elite-source composition,
critic contribution, expected twin-Q disagreement, covariance scale, effective
sample size, and the fraction of RL candidates surviving the elite set.

## Development gate

Gate 1 passes only if all integrity/safety checks and at least two of the three
efficacy checks pass:

Integrity and safety (all required):

- no missing or duplicated episode keys;
- standard MPPI regression tests are bitwise/numerically unchanged when the
  new optimizer is disabled;
- no collision-rate regression versus ICODE-MPPI on development episodes;
- no bypass of LaserScan-derived obstacles or scan_guard.

Efficacy (at least two required):

- RL-Driven ICODE-MPPI improves paired mean task cost in at least two of three
  independent model blocks;
- aggregate paired task-cost improvement is at least 5%;
- success is improved or tied and median final goal distance is lower.

Failure of the gate is a valid negative result. In that event, sealed seeds are
not opened and the next method change must be separately preregistered.

## Confirmation rule

After passing development, all parameters are frozen. Confirmation uses the
five sealed seeds, all three model blocks, and bootstrap confidence intervals
over paired block-by-episode effects. Claims are limited to tested scenes,
physics domains, and checkpoints. No formal stability or convergence guarantee
is claimed.
