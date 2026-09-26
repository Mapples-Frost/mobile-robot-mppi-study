# ICODE + SAC joint K/H allocation, continued main research line

2026-09-08. User explicitly requests continued execution until a working research
method is obtained, preserving the joint K/H main line and prioritizing ICODE.
Parent b166aac. Prior 3216-episode campaign and v1/v2 remain frozen. This is a new
prospective development round, not re-scoring or overriding previous conclusions.

## Research question and execution

Can online model-quality information and temporal credit assignment support useful
joint K/H allocation on a common ICODE-MPPI controller? Prior static classifiers
lost task quality and simple feedback failed to beat a medium fixed budget. Here
train actual SAC, with ICODE shared by fixed and adaptive controllers. Do not call
an enabled correction model reliable without measured evidence. No positivity or
publication guarantee. Negative development outcomes inform explicitly versioned
new rounds rather than terminate the overall research request.

Initial full-bank model/grid screen: all 12 prior geometry families, both speed
caps, K/H pairs (16,12),(16,20),(64,20),(64,32),(128,20),(128,32), nominal and ICODE,
seed 9090101, seeded perturbation: 288 episodes. The complete old bank, including
failures, stays visible. ICODE checkpoint is the frozen L57 seed20261201 best.pt;
online shadow ensemble uses seeds20261201/20261202/20261203. Hash every checkpoint.
All treatments compute the same causal reliability context, including nominal via
the frozen shadow ICODE model, so information-construction overhead is comparable.
The active prediction model still differs explicitly by treatment.

Baseline/controller tuning is allowed prospectively but must be shared by every
method and accompanied by a fresh data round. No special cost, safety override or
better rollout model exclusive to proposed. Keep original task score Q from the
exploration: duration/30 + 2*distance/4.5 + 3*timeout + 10*collision. Also retain
all components, original stage cost, path length, clearance and actual timing.
100ms physical cycle; measured command-readiness advances the plant; 40ms native
actuator delay, late commands discarded. These are simulations, not hardware trials.

## SAC development

Use repository SACAgent (tanh Gaussian, twin Q, target Q, entropy tuning), with
all optional robust/distributional/BC correction modes disabled. This is a
common-MPPI SAC-H adaptation, not numerical reproduction of Bohn's nonlinear MPC
or learned quadratic terminal value. No PPO substitution. Raw unrounded actor
actions enter replay; only environment maps them to integer K/H.

Action mapping: K16..256 in steps16; H8..40 step1, linear [-1,1] then half-up
rounding. Each chosen budget lasts5 physical cycles (0.5s); inference is inside
first-cycle readiness timing. Current observation contains sensed goal/scan/motion,
prior planner diagnostics, frozen ICODE residual/support/innovation/ensemble
descriptors and commanded speed cap. No scene ID, future truth or current-plan
statistics in the actor. Training reward uses observed distance progress and
termination events; simulator ground truth is allowed only in retrospective metrics.

Dense reward summed over the block: 2*(observed_distance_before-after)/4.5
- duration/30 - 10*collision - 3*timeout - 0.20*sum(measured_decision_seconds).
This is an explicit resource-priced objective; task Q excludes the price. Preserve
all terms. Hard terminal time limit is part of the finite task, so no bootstrap at
timeout. SAC gamma=.995 per block; hidden64/64, batch128, lr3e-4, tau.005,
initial_alpha=.01, learned entropy. Replay50k; random normalized actions for first
500 block decisions, then one update per decision. Training retains complete
episodes until40k physical cycles are reached; total may exceed by<300cycles.
Use shuffled balanced context blocks, one initial model seed9091001. Save interim
checkpoints each10k cycles without using final test results for selection.

Initial arms: SAC-KH with all context; SAC-H and SAC-K fix the other coordinate to
the ICODE development fixed pair; SAC-KH without dynamics features masks values
and availability only, same acquisition overhead. Same interaction budget, common
scenario sequence and same active ICODE model. A nominal SAC-KH comparator follows
if needed to identify dependence on correction. More training/seeds may be added
as new development rounds with recorded purpose; do not imply a single seed is
sufficient for final publication.

Evaluate checkpoints on development seeds9090201/9090202, all twelve families and
both speeds. Final untouched seeds9090301–9090304 reserved until a candidate and
fixed comparator set are frozen. Train, validation and test seeds cannot overlap.
Compare fixed pairs selected on development, cheap medium fixed, high fixed,
causal feedback, SAC-H/K/KH and masked-context ablation. Matched treatment order
randomized within paired contexts, timing runs serial, BLAS/Torch1thread.
Independent confirmation should replicate with multiple initialization seeds.

## Integrity and progress

Source snapshots, configs, checkpoints, replay, raw cycles, episode summaries,
losses and episode ledger are archived per stage. Failures and code fixes recorded;
no completed raw overwritten. Recovery resumes at complete episode boundaries
including RNG, optimizer, replay and context schedule state. Do not confuse the
many control cycles with independent replicates. Report paired episode and
family/seed hierarchical intervals, task/resource Pareto and failure contexts.
No extra collisions, success drop<=2pp, Q regret<=5%, desired compute saving20%
are descriptive targets, not a rule to stop research at first failure. No claim
of non-inferiority from a point estimate alone. No push or external messaging.
