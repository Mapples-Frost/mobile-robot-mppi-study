# Development hypotheses and implementation notes

2026-09-08, active ICODE/SAC branch. These notes distinguish observations from
future hypotheses. They do not change the frozen initial round.

## Observed in the first two training arms

Joint K/H training completed 40,017 physical cycles. H often approached8,
while K varied. H-only completed40,227cycles; intermediate phases favored both
large and small H. These are stochastic training logs with different episode
lengths and changing data distributions, not a controlled comparison of policies.

The complete fixed screen contains challenging moderate-speed static obstacle
contexts and a collision-prone moving crossing. Reliability support confidence
is often nearly1; it is a normalization heuristic, not a calibrated probability.
Innovation and disagreement vary, but their relevance to optimal H remains
untested. Same-context episode outcomes cannot prove that an uncertainty feature
is informative; explicit paired prediction diagnostics and ablations are needed.

## Candidate causes to investigate, not established findings

1. Finite-task time is omitted from the first actor observation. At a stationary
   state, early and late decisions can look identical despite different timeout
   risks. A causal remaining-time feature is appropriate in a new named round.
2. The first SAC has only one update per five physical cycles, around8,000updates
   per arm. Delayed success/timeout credit over up to60decisions may be difficult.
   More updates/interactions or a single-axis initialization could help; these
   are hypotheses and require matched total interaction/optimization accounting.
3. Initial policy exploration covers two dimensions. If H learns a cheap short
   horizon before obstacle-success credit propagates, it may stay in a poor
   region. Tests should compare warm initialization against longer single-axis
   baselines, count all pretraining, and include actor/replay transfer ablations.
4. Shared bottom-controller tuning could matter more than budget selection.
   The prospective factorial protocol preserves safety and applies every change
   to all methods. All original scene outcomes remain visible.

## Prepared tool, not yet an experiment

`experiments/icode_sac_development/axis_transfer.py` copies a one-dimensional SAC
actor/critic into a two-dimensional actor/critic. The old deterministic action
and Q values are preserved exactly up to floating point; the new coordinate
starts at the previously fixed budget and its critic input weight is zero.
Old continuous replay actions are augmented with that actual fixed coordinate.
Optimizers reset, alpha value is retained and target entropy becomes-2.

Two tests verify policy/Q equivalence for both axes and replay ring chronology;
both passed. The tests ran for1.06s at nice19 during the K-only training process.
This short host load is recorded for timing provenance; deployment evaluations
will run without tests/other training in parallel. No transferred policy has yet
been trained or evaluated; no curriculum improvement is claimed.

## Execution note

Further source inspection confirmed current physics9kg/torque2.2/friction1.2
versus L56pretraining13kg/torque1.55/friction.62. This is a physical-domain shift,
not merely different random seeds. A prospective288episode domain screen is
added before the observable-context round. Its13kgtraining domain is chosen by
pretraining provenance before screen outcomes, with identical physics for all
comparators. Old9kgexperiments remain valid transfer/development evidence.
The not-yet-started observable queue waiter was terminated and relaunched to
include this amendment; no running experiment or recorded episode was stopped.

First launch failed before any training due to the local script `queue.py`
shadowing the standard library; renaming fixed import resolution. All4initial
core/learning tests passed afterward. Separate analysis code and subsequent
development code live outside the frozen training source directory. Original
screen source snapshots remain unchanged.

## Diagnostic additions during initial validation

Added after observing partial first-round validation (approximately536/1152
episodes): leave-one-environment-seed-out context-budget selection over the
fixed candidates. Each fold chooses both its global fixed baseline and
family/speed oracle solely on the other seed(s), then scores the held-out seed.
This is a post-hoc developmental diagnostic, not a prospectively registered
gate or unbiased estimate after the full development process. It excludes
within-episode switching, selector overhead and unseen-family generalization.
All fold-specific choices are stored so held-out-label leakage can be checked.

The report generator now excludes prediction-summary dictionaries from episode
counts. Raw auditing additionally checks continuous-to-integer K/H mapping,
reliability masking and extra source hashes for the observable round. Targeted
tests are queued before new physical-domain screening; no concurrent test or
training process is launched during the current deployment-timing evaluation.

Before the288episode physical-domain screen starts, added paired offline
prediction diagnostics at H1/8/20/32 on its common K64/H24 behavior traces.
Nominal and ICODE predictors receive identical recorded action histories, with
separate9kg/13kgand nominal/ICODEbehavior strata. Aggregate within episode
before averaging episodes, and list episodes shorter than32cycles as excluded.
This measures conditional prediction accuracy, not future-feature availability
or counterfactual planning accuracy. It runs after the serial physical screen
and before training, so it cannot contend with deployment timing.

## Reward interpretation

For this point-goal task every episode starts at distance4.5m. With an
undiscounted sum over decision blocks, the progress terms telescope, giving
`training_return = 2 - Q - compute_price * measured_compute_seconds` whenever
the final sensed distance equals the task distance (the current ideal-sensor
condition). Waiting at fixed distance incurs the time and computation costs,
so this reward does not pay a positive per-step reward merely for lingering.
SAC uses gamma.995 per block; its discounted objective is therefore not exactly
that identity and can alter the timing preference. Q is an explicit episode
task metric, not the MPPI running stage cost; both are reported separately.
This is a Bohn-style high-level SAC architecture comparison, not a reproduction
of the original paper's full terminal-value or reward implementation.

## Initial validation completion and memory limitation

Initial validation completed1152episodes in5453.41seconds. Best fixed ICODE
K128/H32 had36/48successes and2collisions; best full joint checkpoint30/48and
4collisions, best K-only34/48and3collisions. No joint advantage is established.

After the completion marker was written, the Python process remained in exit
cleanup with approximately5.9GBRSS and1.1GBswapped. No competing timed experiment
was running. This is evidence of accumulated process memory; it may distort
late-batch latency. Treatment order was randomized within paired contexts, but
that does not remove the timing limitation. Do not treat its latency difference
as a clean hardware deployment measurement without replication.

Before subsequent physical stages begin, added per-episode garbage collection
after closed environment references are removed, outside the measured decision
boundary. Each subsequent stage records RSS,swap and collected object counts.
This changes process resource hygiene and is explicitly prospective. Original
initial-round traces and sources are unchanged.

New development tests initially had9passes and1dependency failure because SciPy
was absent. Installed SciPy1.10.1 (Python3.8/NumPy1.24.4compatible), then all10
tests passed in1.55seconds. Tests ran after all1152validation episodes had been
written, while its process was exiting; no new timed screening had begun.

The first continuation attempt `continuation_234854` failed before creating
any physical-screen episode because Python3.8kept `__file__` relative and its
path could not be relativized against absolute ROOT. Normalize development
source paths with resolve(); preserve the failure log. The repaired continuation
and observable waiter were explicitly restarted. Historic failed attempts are
retained; the waiter now follows the latest continuation attempt.

Raw/source/action auditing passed2298episodes,430896cycles and32315SACblocks.
Generated initial validation figures and inspected the paired outcome figure.
Added a further worker-schedule test after the10-test run; it is queued before
new domain screening with the full development suite.

## Completed common-backbone and domain screens

All11development tests passed before the next domain screen. Shared factorial:
192episodes. At ICODE64/24the original setting had13/24successes and2collisions;
wide noise[.2,.9] with influence.45had21/24and1collision. Fresh-seed budget check:
432episodes; wide noise with original influence.7was more consistent, giving
ICODE64/24:21/24and0collisions;ICODE128/32:23/24and1collision. Nominal64/24also
improved to23/24and0collisions. Thus a substantial gain comes from common MPPI
sampling settings, not from learning or a unique ICODE advantage.

The288episode domain screen selected the same shared planner and13kgICODE
reference128/32. On13kg: ICODE64/24and128/32both22/24successes,1collision;
nominal128/32:23/24successes,0collisions. Domain matching alone did not establish
ICODE closed-loop superiority. The prospective60kcycle joint arm is now running,
with H-only,K-only and masked joint queued serially.

Initial offline predictions: common recorded actions produce lower ICODE mean
error across both speeds at H1/8/20/32. New domain prediction adds8768window/model/
horizon rows across both plants and both behavior models; H32ICODE-minus-nominal
position RMSE is negative in all8domain/behavior/speed strata. These conditional
predictions do not establish that greater H has higher control value.

Stored source/raw checks pass2922episodes and539448cycles through the shared
screens. Command reconstruction also passes allthese cycles under fixed40ms
calibration. The288domain episodes await the next full raw audit. No final test
seeds or additional independent SAC initialization seeds have yet been used.

## Overnight handoff,2026-09-09around00:47Asia/Shanghai

User explicitly requests continuous overnight exploration and a useful morning
result. Desktop heartbeat `icode-k-h` is active every15minutes until10:00local.
Follow this task rather than creating duplicate tasks. Keep unchanged progress
quiet; notify meaningful outcomes/failures. The heartbeat should prepare the
morning report and pause after the requested night.

Current sole timed process queue: `observable_queue_002812`, stage4joint
training. Unified exec session65247runs `run_observable.py`. Joint full is at
10135/60000cycles,68episodes,2047decisions,RSS713MBand0swap. Remaining stages:
H-only,K-only,maskedjoint60k each; then all16checkpoints plus fixed/heuristic
validation on seeds9091202/9091203, with one serial process per paired context.
The evaluator automatically runs full audit/figures/report after completion.
Do not run training,tests or expensive analysis concurrently with timed episodes.

Completed continuation session42312and `continuation_235140` exited0. Initial
session65902also exited0. Old waiters67091/99571failed on the preserved first
continuation path error; they are no longer active. Never restart them blindly.

New-round training has already frozen all files in
`experiments/icode_sac_development/`, all `src/mobile_robot_mppi/`, initial
`experiments/icode_sac_compute/`, and the observable/domain protocols. Do not
edit these while continuing this round; a later experimental change needs a
new separately named module/protocol/artifact. Diagnostic scripts and reports
are outside this frozen source set (except domain_prediction.pyis separately
frozen by the completed domain screen). Current local commit a93f3d3plus the
latest report notes; no push. Task raw artifact directory remains untracked.

After actual complete development evaluation, select paired diagnostics across
KH,H,K,masked,fixed,heuristic and nominal. If newjoint is promising, replicate
initialization seeds and reserve final9090301–9090304until settings freeze. If
not, investigate curriculum/credit assignment/resource pricing with matched
budgets; do not claim success simply because fixed MPPI improved. Prepared
axis_transfer.pyis tested but no curriculum experiment has run. Record total
source training cost for any transferred policy.
# Active continuation, 2026-09-09 morning (supersedes older runtime notes below)

10:46 CST superseding runtime update: independent 624 and mechanism 960 episodes
are COMPLETE; mechanism queue status complete. Both trace and command-history
audits passed. No training/evaluation remains running in these queues. Four
focused tests passed. Do not restart finished queues. Read
docs/reports/icode_sac_morning_2026-09-09.md for the mixed conclusions and next
priorities; new experiments require a separate prospectively recorded protocol.
The old heartbeat's 10:00 limit expired; user now requests continued watching.

09:00 CST update: all eight independent arms completed (480,797 cycles,
3,358 episodes). Queue completed in 15,095 seconds. Independent evaluation
has started on the corrected fresh seeds, 11/624 episodes at 09:00:44;
first worker RSS about 350MB, swap zero. Mechanism queue is waiting for
that evaluator's exit and audit, not running timed simulations concurrently.

User explicitly requested continued work after the morning status discussion.
At 08:58 CST, seven independent training arms are complete; masked KH seed
9091302 is at 56,817/60,000 cycles. Sole timed training queue:
`independent_queue_043708`. Successful interpreter is `.venv/bin/python`.
The initial `independent_queue_043442` failed before training because system
Python had no torch; keep its logs. Old observable/continuation queues finished.

Current waiting independent evaluator: Linux PID 268370, entry point
`experiments/icode_sac_development/evaluate_independent.py`, stdout/stderr
`.codex_tmp/evaluate_independent_audited.*.log`. It waits for all eight arms,
then evaluates all final checkpoints without best-seed selection, on seeds
9194201/9194202, 13 methods x 48 contexts = 624 episodes. The initially planned
9094201/9094202 overlapped incrementing training seeds; this was caught BEFORE
any evaluation, corrected and documented. The two training seed ranges partly
overlap each other; claim independent SAC initialization, not disjoint training
datasets. Final 9090301–9090304 remain unused.

The evaluator now records process RSS/swap, checks training/eval intersections,
checks immutable source/spec equality on resume, and updates progress only for
the active worker. It runs command-history and trace audits plus reports after
completion. Newly changed comparison reporting keeps each initialization and
matches H/K/masked ablations to that same seed; never pools fixed baseline
replicas or selects a best seed. Three focused comparison/schedule tests passed.
They contain no timed simulation. Existing frozen training traces stay intact;
evaluation infrastructure changes are captured in the new evaluation snapshot.

Next serial queue is already launched:
`experiments/icode_sac_mechanism/follow_after.py --wait-pid 268370`.
State: `research_artifacts/icode_sac_compute_2026-09-08/mechanism_queue_2026-09-09/status.json`.
Logs: `.codex_tmp/mechanism_queue.*.log`. It waits for the independent evaluator
process (including reporting) to EXIT and requires its complete 624-episode
trace audit to pass. Then runs focused hold/comparison tests, the 960-episode
mechanism stage, audits, budget diagnostics, mechanism report, and main report.
Do not start overlapping simulators or modify its frozen code after it starts.

Mechanism protocol: `docs/protocols/icode_sac_mechanism_2026-09-09.md`.
Artifact: `mechanism_validation_2026-09-09`; seeds 9195201/9195202.
20 methods: each of two final joint initializations, full/masked, ICODE/nominal;
full ICODE with K held / H held / both held at initial observed choice; six
fixed model/budget comparators including 160/29 and 128/27. Actor is still
computed under hold interventions, so those are matched-overhead interventions.
Nominal policies are transferred WITHOUT retraining and must be labeled that
way. Dedicated analysis: `experiments/icode_sac_diagnostics/mechanism_report.py`.
Outputs: diagnostics/mechanism_comparison.json and
docs/reports/icode_sac_mechanism_results.md, only after completion.

Useful completed diagnostic: observable_validation_budget_behavior.json shows
selected full KH changes budgets in all 48 episodes, mean K/H=159.128/28.630,
within-episode variance fractions K=0.699,H=0.814; masked mean=131.752/27.006.
This shows actual variation, NOT causal utility of switching. Static surrogates
160/29 and 128/27 are rounded from those development means and prospectively
frozen for the next fresh evaluation. No claims of reliable information gain,
ICODE superiority, or ICRA readiness yet.

---
