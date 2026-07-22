# L263 Counterfactual Actor Diagnosis Protocol

Status: preregistered before counterfactual outcomes are read.

## Purpose

L263 is a diagnosis, not another training run.  It separates five possible
causes of the L261/L262 failure in this order:

1. feature, normalizer, action-mapping, entropy, quantile-reduction, or replay
   implementation bias;
2. reward/action ordering in the real MuJoCo environment;
3. Critic action-ranking error;
4. Actor optimization error after conditioning on a credible Critic;
5. observation aliasing and, only after the Critic audit, multi-scene gradient
   interference.

No reward, Actor, Critic, MPPI cost, Actor/Traditional fusion, map, safety
arbiter, or online deployment behavior may be changed during L263.  No
additional SAC or supervised update is allowed before the ranking audit is
complete.

## Frozen artifacts

- repository root: `D:\Projects\mobile-robot-mppi-study`
- execution: Windows native `.venv-cuda` only
- primary checkpoint: L262 seed 20262611 step 6000
- primary checkpoint SHA256:
  `48e662fd728c0d1d62b1ff79681e5cdd924b120d9411c8df6e062a814b4d42a7`
- initialization comparator: L257 seed 20262333 step 30000
- comparator SHA256:
  `fc9166f5c3010156a7c9fad4cb9d155ac222447506ff2ebdfe6c2205250a1547`
- six training and three validation scenes are exactly the L261 scene files.
- diagnostic seeds: 20262731 through 20262739, one fixed seed per scene.

L258 artifacts and final Hairpin, S-Chicane, and Infinity geometries, seeds,
trajectories, and outcomes are forbidden.  They must not enter state
construction, action selection, diagnosis, training, or checkpoint selection.

## Phase 0: implementation and scale audit

The audit records, without tuning:

- the name and index of all 69 observation features;
- raw and normalized mean, standard deviation, min/max, and normalizer clip
  fraction;
- path-preview local-frame sign checks under rigid translations and rotations;
- signed CTE, heading-error, curvature, and angular-action sign checks;
- normalized-to-physical action mapping, rate limits, and executed action;
- deterministic Actor output distributions and saturation fractions;
- learned alpha, target entropy, 25-quantile reduction, CVaR fraction, and
  per-scene Q scale;
- replay counts by scene and by event stratum (center, boundary, moderate and
  far off-track, curvature sign/magnitude, heading error, safety override,
  recovery, and termination).

An implementation-contract failure is reported as such.  It is not described
as evidence against the research method.

## Fixed counterfactual state library

The library uses only the nine allowed L261 scenes.  Each scene contributes
three deterministic, collision-free MuJoCo reset states, constructed from its
polyline rather than from an outcome-selected trajectory:

1. center/low-curvature anchor;
2. center/high-curvature or curvature-reversal anchor;
3. an offset anchor, blocked across scenes at signed magnitudes 0.70 m,
   1.10 m, and 1.80 m.

The offset side is selected by the first collision-free choice in the frozen
order `left`, then `right`; it is not selected from Actor performance.  Invalid
states are retained in the manifest with their rejection reason.  The
Loop-exit validation scene additionally labels the later near-intersection
branch, but its identity is not exposed to the Actor.

Each accepted state stores the complete configured initial state, MuJoCo truth,
raw and normalized 69D observation, normalizer statistics, reference
projection, Actor action, geometric recovery anchor action, Critic quantiles,
scene role, geometry fingerprint, and path progress.  Initial-state noise is
zero.  Previous control is set consistently with the reset twist before the
observation is encoded.

## Frozen action and rollout design

For every accepted state, evaluate:

- the deterministic Actor action;
- a deterministic geometric recovery anchor action;
- the complete Cartesian grid
  `v_norm in {-1,-0.5,0,0.5,1}` by
  `omega_norm in {-1,-0.5,0,0.5,1}`.

Duplicates are retained by semantic label but share one physical rollout.
For every unique action, record requested normalized action, mapped physical
command, rate-limited proposed command, safety-executed command, and applied
plant command.

Two continuation contracts are evaluated from identical resets:

- `constant`: hold the candidate normalized action for up to 40 steps.  This
  diagnoses the reward and physical consequences of an action family.
- `actor_follow`: apply the candidate only at step 0, then follow the frozen
  deterministic L262 Actor.  This is the continuation contract used when
  comparing the Critic with realized returns.

Prefix metrics are recorded at horizons 1, 10, 20, and 40 using gamma 0.99:
discounted return, every reward term, CTE and signed CTE change, path-progress
change, goal-distance change, corridor re-entry, collision, safety override,
and termination.

## Frozen ranking metrics and decision tree

Across the 5x5 grid, report per state, scene, stratum, and horizon:

- Spearman correlation between minimum twin-Q and realized discounted return;
- top-1 action agreement (ties declared within 1e-6);
- recovery-versus-fast-forward pairwise accuracy;
- Critic twin disagreement and quantile spread;
- Actor regret under realized return and under Critic ranking.

Interpretation is constrained as follows:

- If constant-rollout return prefers fast/small-turn actions over recovery in
  off-track states, the frozen reward/action semantics are mis-ranked.
- If realized `actor_follow` return prefers recovery but the Critic prefers
  fast/small-turn actions, the primary failure is Critic ranking.
- If Critic and realized return both prefer recovery but the Actor does not,
  the primary failure is Actor optimization, entropy, or action-head mapping.
- Mixed scene/stratum results must be reported as mixed; no global cause may be
  claimed from a minority subset.

## Conditional Phase 2 analyses

Observation aliasing is audited with standardized 69D k-nearest neighbors and
counterfactual grid-optimal actions.  The report separates within-scene and
cross-scene conflicts and left/right sign conflicts.  Small privileged probes
may compare 69D with longer preview, three-frame history, and progress/branch
information, but are diagnostic only and cannot initialize the formal Actor.

Per-scene Actor gradients are audited only if the Critic ranking is credible in
the relevant stratum.  Equal-size, event-matched batches are used.  Cosine
matrices and cancellation ratios are reported separately for the shared trunk,
linear-speed head, and angular-speed head.  Negative cosine is evidence about
the current learned objective, not proof of an unavoidable task conflict.

## Reporting and stopping rule

L263 stops before any training change.  Raw results, the state/action manifest,
integrity checks, and one concise causal-status file are retained on D:.  No
plots or detailed positive-method report are produced unless the diagnostic
contract completes.  Any later reward, observation, or optimization probe must
be separately preregistered from the L263 evidence; it may not be selected from
final held-out Tracking outcomes.
