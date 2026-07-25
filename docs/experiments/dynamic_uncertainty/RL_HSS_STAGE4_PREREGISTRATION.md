# Dynamic-Uncertainty Stage 4 RL/HSS Preregistration

Date: 2026-07-24  
Status: frozen before the 24-episode development matrix; no Stage 4 episode has
been started by this implementation turn.

## Question

Under the accepted Amendment 17 dynamic-obstacle environment, does the frozen
L217 point-goal Actor with role-aware HSS improve closed-loop navigation when
used with nominal or qualified Stage 3 residual prediction, without weakening
the probabilistic-risk MPPI, residual safety shield, scan guard, or candidate
budget contracts?

This is a development integration study. It does not open sealed seeds and
cannot by itself support a confirmatory generalization claim.

## Frozen lineage and caveats

- Environment: Amendment 17, accepted conditionally by the research lead. Its
  original numerical gate remains 8/9; the completion-regression miss is not
  relabeled as a pass.
- Residual runtime: the Stage 3 parallel-shield gate passed with 12/12 goal
  reaches, zero collisions, maximum planner P95 `97.19 ms`, and exact sequential
  outcome equivalence.
- Actor: the L217 point-goal Actor from L175, not any L219--L285 tracking Actor.
- HSS: the value-aligned L217 HSS thresholds and `policy_rescue` routing. The
  original L193 calibration summary did not pass its small unseen-bin gate;
  the later independent L194 stress gate passed and the complete L217 study is
  retained as the frozen source evidence. Both facts remain visible.
- All artifact paths and SHA-256 values are fixed in the Stage 4 protocol.

## Design

The independent unit is one complete MuJoCo episode. Control steps, MPPI
rollouts, and obstacle observations are repeated measurements, not independent
samples.

The two treatment factors are:

1. prediction dynamics: nominal versus one of the three qualified Stage 3
   residual checkpoint blocks;
2. frozen RL/HSS: off versus on.

Within each obstacle-process seed, the complete set contains the nominal cell
and all three residual model blocks, each with RL/HSS off and on. Three frozen
development obstacle seeds therefore give:

`3 seeds × (1 nominal + 3 residual blocks) × 2 RL/HSS levels = 24 episodes`.

Run order is randomized within each obstacle-seed block using schedule seed
`730199913`. The obstacle seed provides common random numbers across treatments.
All three residual checkpoints are retained as model blocks; none may be
selected or excluded using Stage 4 outcomes.

## Treatment contract

RL-off cells are constructed by the same Stage 3 `configure_condition` path.
Their resolved controller configuration must be exactly unchanged.

RL-on cells use the paper-faithful low-level Actor integration:

- Actor trajectories propose candidate centers and covariance;
- the frozen Actor retains its trained forward-command bounds
  `[0.0, 0.35] m/s`; these are an explicit subspace of the unchanged Stage 3
  controller bounds `[-0.35, 0.35] m/s`. Reverse Gaussian MPPI candidates and
  reverse scan-guard recovery remain available, so the adapter does not narrow
  the controller or safety action space;
- HSS causally allocates proposal authority from completed-transition
  innovation, ensemble disagreement, training support, and Actor OOD score;
- MPPI, not the Actor, selects the command;
- each controller receives exactly `300 × 2 = 600` candidate evaluations per
  decision, equal to the Stage 3 `600 × 1` budget;
- scan guard remains the final external action arbiter.

The HSS evidence ensemble is an explicit frozen sidecar. It is the same in the
nominal and Stage 3 residual RL-on cells, so the dynamics factor changes MPPI
prediction rather than silently changing the reliability estimator. Sidecar
innovation uses only an already completed transition.

For a residual RL-on cell, both internal shield controllers use matched copies
of the Actor, HSS sidecar, optimizer, candidate budget, RNG seed, and mutable
state. The residual controller uses Stage 3 prediction and the nominal fallback
uses nominal prediction. The two copies do not share RNG or mutable policy/HSS
objects. The existing shield certificates and selection rule are unchanged.

## Frozen components

The following may not be tuned from Stage 4 outcomes:

- obstacle generator V3 and Change-Aware IMM;
- Amendment 17 route, localization, Risk V1 definition, hard probability
  threshold `0.20`, and scan guard;
- MPPI horizon `36`, total candidate budget `600`, costs and action limits;
- Stage 3 residual checkpoints, component mask, CUDA Graph runtime and
  parallel residual shield thresholds;
- Actor checkpoint, HSS sidecar members, calibration thresholds,
  `policy_rescue_floor=0.50`, and RL optimizer settings;
- development seeds, run schedule, episode horizon, and gates.

## Outcomes and interpretation

Safety is lexicographically primary: any collision triggers the preregistered
early stop. Episode success, final goal distance, completion, clearance,
planner P95, shield acceptance/fallback, Actor-versus-baseline proposal delta,
HSS authority, and scan-guard override diagnostics are retained.

Analysis is paired at the episode level within obstacle seed. Results are
reported separately for each residual model block and pooled only as a
descriptive blocked summary. With three development seeds, uncertainty and raw
paired values take precedence over significance testing. Model blocks are not
treated as additional environment replicates.

The minimum mechanism checks are:

- every RL-on episode reports HSS enabled and the frozen sidecar active;
- `paper_total_rollouts` is exactly 600 per controller decision;
- RL-off resolved configurations match Stage 3;
- residual cells retain shield diagnostics and nominal cells do not masquerade
  as shielded residual cells;
- scan guard remains present and receives the MPPI/shield-selected command;
- no held-out or sealed seed is imported, scheduled, or emitted.

## Execution rule

The runner defaults to preflight only. The full matrix can start only with the
explicit `--execute` flag. It writes a protocol-hash-bound manifest and resumes
only when the protocol and randomized schedule are unchanged. The current
implementation task runs preflight and tests only; it does not pass `--execute`.
