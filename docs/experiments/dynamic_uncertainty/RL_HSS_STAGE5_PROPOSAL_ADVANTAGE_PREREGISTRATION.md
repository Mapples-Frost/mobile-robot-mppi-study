# Stage 5 proposal-advantage HSS mechanism-probe preregistration

## Research question

Can HSS prevent the frozen L217 Actor from degrading the dynamic-obstacle
baseline when the Actor's own best proposals are repeatedly worse than current
Gaussian proposals under the shared MPPI objective?

Stage 4 established failure of the joint Actor/HSS treatment but did not
separate Actor quality from HSS authority.  Stage 5 is a bounded development
mechanism probe.  It changes only the online proposal-advantage veto and does
not train an Actor, modify obstacle prediction, change the safety stack, or
touch residual-dynamics runtime.

## Prior evidence and non-confirmatory status

The rule was selected from the completed Stage 4 development trajectories, as
reported in `RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_EDA.md`.  Stage 4 is therefore
threshold-selection evidence.  Stage 5 prospectively uses fresh development
seeds `730100064`, `730100066`, and `730100068`.  No held-out ID, held-out OOD,
or sealed seed is authorized.

## Experimental unit, blocking, and treatments

The independent unit is one complete MuJoCo episode.  Obstacle-process seed is
the block.  All three arms share the same task, nominal dynamics, obstacle
trajectory, observation process, 40 s horizon, MPPI cost, risk interface, scan
guard, and external safety arbitration.

Within every seed block, a frozen seeded permutation assigns run order to:

1. `rl_hss_off`: qualified Stage 3 nominal baseline, standard MPPI, 600
   Gaussian candidates in one iteration;
2. `rl_hss_shadow`: frozen Stage 4 Actor/HSS treatment, 300 candidates by two
   iterations, with the proposal-advantage rule logged but not applied;
3. `rl_hss_advantage_veto`: identical to the shadow arm until three consecutive
   disadvantages occur, then Actor guided share and proposal-centre/covariance
   authority are both set to zero from the next decision through episode reset.

The shadow-versus-veto contrast isolates application of the new veto.  The
RL-off comparison measures the remaining difference between Gaussian-only
Paper fallback and the qualified standard-MPPI baseline; authority zero is not
claimed to be bitwise identical to standard MPPI because their frozen
iteration structures differ.

## Frozen online rule

For a decision with both sources observed:

```text
relative Actor advantage
  = (J_best,Gaussian - J_best,guided)
    / max(abs(J_best,Gaussian), 1).
```

- disadvantage: relative Actor advantage `< 0`;
- patience: three consecutive disadvantages;
- application lag: one control decision;
- recovery: none within the episode; reset only at the next episode;
- shadow arm: identical state update and diagnostics, applied authority remains
  one;
- unavailable comparison: state is retained without update;
- no simulator future, obstacle truth, outcome label, or sealed information is
  consulted.

## Fixed budget and anti-confounding constraints

- exactly 600 model rollouts per controller decision in every arm;
- no additional shadow rollout is authorized;
- guided candidates reuse the existing Stage 4 allocation in the shadow and
  pre-veto periods;
- after active veto, all 300 candidates per Paper iteration are Gaussian and
  Actor authority over proposal mean and variance is zero;
- source-relative elite-yield competence remains disabled because L220 already
  falsified it as a sufficient long-horizon competence signal;
- Actor, HSS sidecar, ICODE/residual artifacts, probability-risk interface,
  task, sensors, and safety configuration remain frozen;
- stop the entire probe on the first collision and preserve all preceding data.

## Outcomes and gates

### Integrity gates

- 9 expected complete episodes unless the first-collision rule stops execution;
- three distinct fresh development seeds and all three arms per seed;
- fixed 600-rollout budget on every decision;
- shadow diagnostics enabled but applied gate authority always one;
- active-veto diagnostics enabled and causal lag exactly one;
- after the first active latch, every later decision has zero guided sequences,
  zero proposal authority, and full proposal fallback;
- sealed seeds remain unopened;
- frozen source and implementation hashes match.

### Safety and directional mechanism gates

- zero collisions;
- RL-off must retain its qualified success behavior;
- active-veto success count must equal RL-off success count;
- active veto must latch in every episode where three comparable consecutive
  disadvantages are observed;
- active-veto final goal distance must be no worse than shadow in at least two
  of three paired seed blocks;
- nominal active-veto planner P95 must remain below 100 ms.

These are development gates, not null-hypothesis significance tests.  Failure
is preserved and motivates further mechanism work; thresholds are not changed
after outcomes are read.

## Explicitly deferred work

Residual dual-Planner timing, dynamic hard-stop feasibility, Actor adaptation,
and any sealed evaluation are separate stages.  They are not modified in this
probe, so a Stage 5 result remains attributable to the proposal-advantage veto.
