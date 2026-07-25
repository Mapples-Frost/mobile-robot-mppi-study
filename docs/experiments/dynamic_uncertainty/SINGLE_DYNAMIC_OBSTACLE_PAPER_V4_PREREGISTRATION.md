# Single Dynamic Obstacle Paper Experiment v4

Status: frozen before qualification, 2026-07-25. Formal outcomes have not
been opened. The 350-episode Candidate--Source study is development evidence
only and is excluded from every v4 estimate.

Revision 2 was frozen after qualification attempt 1 failed before completing
any seed block and before any formal registry existed. That engineering run is
retained. It revealed an omitted scan-layer packaging dependency and two
constructor contracts; no outcome was used to tune an algorithm or select a
seed.

Revision 3 was frozen at 2026-07-25 18:57 +08:00 after qualification attempt 2
passed 16/16 seed blocks and 112/112 episode jobs and after the
controller-independent registry was sealed, but before formal execution and
before any formal outcome existed. The sealed registry SHA-256 is
`91cc59c6025296bcb4aa5f8839bd5bc129159c109730f52c5c9ac13d03b47b56`.
This revision completes the previously skeletal analysis program and expands
the execution manifest to the transitive Python/runtime closure. It changes no
controller, scene, checkpoint, seed, certificate, schedule, arm, rollout
budget or sample size. The passing qualification and sealed registry therefore
remain valid; the formal execution manifest binds the Revision-3 files before
the first formal episode starts.

## Design

The independent unit is one complete simulation seed. Each seed is a block
with common random numbers and is executed by one worker. The four core arms
form a paired 2x2 design:

| Arm | Learning (L) | probability/temporal package (P) |
|---|---:|---:|
| `B00_strong_nominal_mppi` | 0 | 0 |
| `B10_learning_only` | 1 | 0 |
| `B01_probability_only` | 0 | 1 |
| `B11_full_proposed` | 1 | 1 |

The three full-system ablations are `A_full_ordinary_imm`, `A_no_icode`, and
`A_fixed_hss` (fixed guided fraction 0.30). They quantify closed-loop system
increments and are not interpreted as standalone predictor or residual-model
benchmarks.

There are 360 core seeds (180 ID, 180 OOD), giving 1,440 core episodes. A
pre-sealed 140-seed subset (70 ID, 70 OOD) receives all three ablations,
giving 420 more episodes and exactly 1,860 complete episodes. Residual model
checkpoint is a balanced nuisance block over the three qualified ICODE
checkpoints. Arm order uses seeded cyclic Latin rotations within split and
block size. The subset, schedule, certificates, and their SHA-256 hashes are
sealed before formal execution.

## Collision-opportunity scene contract

The controller receives only LaserScan, causal odometry/localization, and the
online tracker/forecast; obstacle identity, future truth, events and intent
are forbidden control inputs. Before any controller runs, a constant-speed
0.35 m/s ghost robot is propagated from (-4.6,-1.8) to (4.2,-1.8). A seed is
eligible only if ghost-to-obstacle center distance is at most 0.50 m (the sum
of radii). Candidate seeds are scanned in ascending order from fixed ranges;
no controller outcome participates in selection.

Conflict windows are constructed from the same frozen ghost and truth where
center distance is at most 0.80 m, padded by 1.00 s, and merged when separated
by at most 0.50 s. All arms use the same seed certificate and windows. A seed
without a valid collision opportunity is excluded before registry creation.

## Factors and resets

`B10-B00` may change only L and `B01-B00` only P. The runner computes a
resolved-config difference audit and requires the same L delta at P=0/P=1
and the same P delta at L=0/L=1. P=0 disables the tracker, probabilistic cost,
temporal emergency, active escape/recovery and their state, retaining only
the common reactive front/side/body scan guard. Each arm constructs a new
ExperimentRunner, planner, Actor, HSS, shield, tracker, RNG and warm-start
state; no state or CUDA input cache crosses an episode boundary.

When P=0, the residual shield cannot consume a forecast. It therefore uses a
pre-frozen causal deterministic certificate computed only from the current
LaserScan local-obstacle layer: both nominal- and residual-model rollouts must
remain collision-free, the candidate's nominal-view clearance cannot regress,
the residual/nominal tube must hold, and nominal progress must be noninferior.
This fallback has no tracker, future truth, Change-Aware signal, or predicted
collision probability. The same shield configuration is present in both L=1
arms; the certificate source follows the explicitly assigned P factor.

`A_fixed_hss` keeps the HSS object active only to satisfy and report the same
proposal-diagnostic contract, but freezes low/medium/high sampling allocations
all to 0.30. Thus reliability observations cannot change the guided candidate
fraction. The same-cycle filter keeps the Gaussian proposal Actor-free in both
Full and fixed-HSS arms.

## Outcomes and inference

Safe success means reaching goal distance <=0.30 m within 400 steps with no
collision and no boundary violation. The observed binary endpoint is called
`collision rate`; `predicted collision risk` is reserved for the planner's
probability output.

For Full minus Baseline, a statistically confirmed success improvement
requires a positive paired difference, two-sided exact McNemar p<0.05, and a
two-sided matched-pair Tango score 95% CI lower bound above zero. It is called
practically meaningful only when its point estimate is also at least +10
percentage points; a confirmed effect below +10 pp is reported as a small
confirmed improvement. Collision-rate noninferiority uses a one-sided 95%
matched-pair Tango upper bound below +2 pp. ID and OOD success/collision harm
margins are -5/+5 pp.

Tango inversion treats paired differences as a trinomial {-1,0,+1}, obtains
the constrained nuisance discordance probability by bounded likelihood
maximization, and inverts the score statistic by bisection with absolute
tolerance 1e-12. Continuous endpoints report paired mean/median, paired-t CI,
Wilcoxon sensitivity, and 200,000 whole-seed bootstrap replicates. Mechanism
and ablation families each use Holm adjustment. Timesteps are never treated
as independent samples.

### Revision-3 executable analysis contract

The analyzer has two commands. `audit` reads only the protocol, registry,
schedule, progress, execution manifest and artifact existence; it never reads
an episode metric or trajectory. `analyze` first requires `complete` progress,
360/360 seed blocks, 1,860/1,860 episode jobs, zero failures, all required
artifacts, the sealed schedule and registry hashes, and unchanged execution
file/runtime manifests. Only after all checks pass does it open the first
outcome. It writes the episode table, input-file hashes, pooled/ID/OOD core
tables, primary analysis, continuous contrasts, factorial effects, ablations
and an output-bundle hash.

The primary behavioral claim is an intersection: pooled success superiority,
pooled collision-rate noninferiority at +2 pp, and ID/OOD one-sided success
and collision harm gates must all pass. Practical importance is a separate
label requiring a pooled success point estimate of at least +10 pp. The final
publication claim remains pending until the independent exclusive RTX-5060
timing cohort is complete.

The confirmatory three-test mechanism family is defined on safe success:

* learning main effect `[(B10-B00)+(B11-B01)]/2`;
* probability/temporal main effect `[(B01-B00)+(B11-B10)]/2`;
* interaction `B11-B10-B01+B00`.

Each is a whole-seed contrast, tested by a two-sided Wilcoxon signed-rank test
with Pratt zero handling, no continuity correction and the normal
approximation; the three p-values receive Holm correction. Whole-seed
bootstrap confidence intervals use 200,000 replicates. Collision-free and
continuous mechanism effects and ID/OOD mechanism strata are supportive and
are always reported.
The confirmatory ablation family is safe-success `B11` minus each of ordinary
IMM, no-ICODE and fixed-HSS, using paired exact McNemar tests with Holm
correction. Its collision and continuous results are supportive closed-loop
system effects, not isolated component-quality claims.

The continuous endpoints are failure-penalized steps (failure=400), final goal
distance, route-projected maximum progress, minimum clearance, frozen-conflict
window q05 clearance, trajectory length, applied-control jerk, stuck/spin
steps and the registered behavior counters. Time to goal is reported only for
pairs in which both arms safely succeed, together with the selected-pair
fraction. Because this frozen point-goal scene contains neither a corridor nor
a world-boundary primitive, boundary violation is structurally zero; if a
future compatible artifact supplies `boundary_violation_steps`, the analyzer
requires zero for safe success. Obstacle contact remains collision.

Direction changes use applied velocity crossing +0.05/-0.05 m/s with the new
direction sustained for two cycles. A three-phase oscillation is a qualified
F-R-F or R-F-R sequence whose first-to-third segment-start span is at most
2.0 s. A reverse segment is premature when its start has clearance >=0.80 m
and temporal-scan TTC >1.50 s. Release delay starts after two non-closing
cycles and ends at two forward cycles; unreleased events are right-censored at
episode end and flagged. Hard-risk exposure is temporal risk alpha=1; raw
closing is a positive temporal closing-rate estimate. Zero-speed exposure uses
|applied v|<=0.05 m/s. Conflict q05 uses samples in the seed's frozen windows;
if termination leaves no such sample, episode minimum clearance is used and a
fallback flag is emitted. Conflict traversal uses the ghost-window entry/exit
route projections and is failure-penalized to 40 s if the robot does not exit.

## Qualification, execution, and timing

Qualification uses 16 development-only certified seeds (8 ID, 8 OOD), all
seven arms, and may inspect only engineering integrity: termination, schema,
600-rollout count, causal-input guard, resolved factor deltas, reset behavior,
artifact completeness, memory and worker health. It is excluded from effect
estimation. A passing report is a hard prerequisite for registry sealing.

The behavior matrix runs headless from local NVMe on the Windows ECS. The
candidate setting is 16 seed-block workers on 32 physical/64 logical CPUs,
four logical CPUs per worker and one Torch/BLAS thread per worker. The complete
seed block stays on one worker. If qualification detects instability, memory
pressure or inconsistency, the frozen engineering fallback is 12 then 8
workers; sample size, rollout count and gates cannot change. Concurrent
behavior timing is health telemetry only.

The publication timing cohort is separate: one exclusive worker on the local
RTX 5060, 16 independent seeds (8 ID/8 OOD) x 7 arms = 112 episodes. It includes
perception, tracker/IMM/forecast, Actor/ICODE/HSS, MPPI, shield and scan guard;
MuJoCo physics and disk I/O are excluded. Capture/warm-up are reported
separately, the first post-reset control cycle is included, and decision
P50/P95/P99, misses, every episode P95 and maximum episode P95 are reported.

No formal outcome may be inspected before all 1,860 episode artifacts are
complete. Algorithmic collision, timeout, stall or boundary failure is a valid
outcome and is never retried. Only documented infrastructure failure may rerun
the identical seed/arm/config while preserving the failed provenance.
