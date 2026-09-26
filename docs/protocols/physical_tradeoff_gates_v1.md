# Physical-computation development gates v1

Frozen before collecting new mechanism outcomes, 2026-09-07.
Base: 9bcaa77. New branch: codex/physical-tradeoff-sac-mppi.

## Material Passport

User: scientific question, phase ordering, controls and conditional stopping.
Agent: the operational thresholds and serial, seeded development design below.
Data: local MuJoCo CPU simulation. No training/formal statistical claim from these
development experiments. Old outcome files and thresholds remain unchanged.

## Fixed scope and units

Easy=`configs/research/mujoco_clean_goal_3_3.yaml`, Medium=
`configs/research/mujoco_strong_mppi_baseline.yaml`. Keep their geometry, target,
tolerance and episode limit. Use the previous explicitly declared current-state
sensor isolation, with causal LaserScan; it is not evidence for odometry robustness.
Low/high action envelopes: v<=.25/.65 m/s, yaw<=1.25 rad/s, rates1.1/3.5.
The state-dependent realized speeds are measured; speed limit is a treatment,
not assumed achieved velocity. Plant A is the L70 plant; B is original default.
No reliability name is assigned until fresh paired probes pass. Compare A nominal,
B nominal and B frozen-L57 ICODE. Frozen three-member reliability provider is
included as an explicit full-context timing stratum; nominal and residual planner
comparisons keep their own inference costs. Old checkpoints are read-only.

All compute profiling and simulation run serially; randomized order seed=7092701.
One environment seed=0 for L0; sampling seeds0..31; independent probe seed=7092702;
cross-repeat validation seed=1 only if L0 mechanisms justify L1. Steps nested in
episode/state, never counted as independent replicates. Hard timeout per stage
1800s; a timeout is incomplete engineering, not scientific NO-GO.

## New timing bounds (profile first)

K candidates16,64,128,256,512,1024; H5,10,20,40, on the actual configured CPU.
32 repetitions/cell/mode, serial shuffled after explicit single warmup (warmup
reported separately), resetting nominal planner sequence/RNG each sample. Charge
causal context (including ensemble), mapping and diagnostic-enabled MPPI. No SAC
exists yet: estimate a conservative untrained 64/64 Gaussian-sized MLP forward
overhead only for net-headroom accounting, explicitly not a learned controller.
Archive hardware, Python, NumPy, Torch, MuJoCo, threads, code/config/checkpoint SHA.
Admissible maximum K is largest candidate with every H<=40 residual cell's P95<80ms
and P99<100ms. P99 from32 calls is descriptive, not certification. Coarse four K
points are16,64,256,Kmax (if duplicate/too large use four sorted admissible points).
DeltaK=16, H integer5..40. No admissible four-point rectangle => stop for timing
scope; do not create a positive effect through pervasive deadline misses.

## Phase3: sampling diagnostics

Take paired full snapshots after0/20/40 reference cycles from Easy/Medium and
low/high speed (12 states), same higher-plant corrected controller K64/H20.
Replay reference with frozen table latency for repeatable state selection.
At each frozen sensor/planner state run32 independent noise seeds per K at H20.
Preserve nominal sequence, previous action and perception, changing only K/RNG.
Measure repeated full first-action variance, actual clipped-command variance,
U_MC, ESS/K, weight entropy/max and e2e latency. High-K mean is an offline
reference, not ground-truth optimal control; squared control deviation is called
reference disagreement, not task regret. Gate A needs >=20% reduction in aggregate
repeated variance (highest vs lowest K), at least60% states with reduction, and
increasing median latency/stale drift. Task regret uses subsequent closed-loop
L0 performance, not this proxy alone. U_MC must have positive pooled within-state
rank association with squared independent-reference control disagreement; if it
does not, it stays diagnostic and cannot support an uncertainty-informed claim.

## Phase4: fresh model probes

For each low/high envelope draw16 initial states and40-step commands, independent
of old probe banks. Same commands/initial states on A/B physical plants with their
unchanged transport delay. Compare nominal and frozen ICODE to each physical
trajectory at H1,5,10,20,30,40. Report position, wrapped heading, v, omega and
normalized error [scales .25,.25,.35,.25,.6]. Predictions reproduce the transport
delay approximation explicitly; this is not a perfect true-model oracle.
For a validated contrast: B/A nominal H40 position and normalized RMSE>=1.20;
ICODE on B reduces both by>=10%. Report per-speed failures; cannot select the
best speed bin. Gate C additionally requires the L0 long-minus-short H task cost
interaction to worsen with B and improve with correction by>=5% of the appropriate
short-H cost in at least one scene with consistent sign at both speed envelopes.
Prediction contrast alone is insufficient.

## Phase5/6: speed-delay and compensation

Using matched reference-state histories, paired controlled injected latencies
0,.02,.05,.08s compare stale-state norms and10-cycle task consequences at both
speed envelopes. Use frozen expected latency for compensation (exact injected
value is known treatment in this control); never current measured tau or true
future state. Record state vector drift and realized v/omega, not only norm.
Gate D needs a positive latency/stale slope and larger drift at high speed in
>=75% paired moving states. Physical task degradation is evaluated separately:
>=5% cost increase between0 and.08s in at least one geometry, for both speeds.
If compensation removes it, report that fact; no fundamental-delay claim.

## Phase7: L0 coarse landscape

Four K by four H, Easy/Medium x low/high speed x A-nominal/B-nominal/B-ICODE,
environment seed0. Primary measured e2e delay; lambda_compute=0, no stale/U_MC
penalty. Rank fixed/context choices by cumulative executed stage+failure cost;
also display success, collision, final distance and elapsed time to guard against
failure-shortened cost. Failure cost is collision_penalty/max_steps times remaining
steps, declared once for all methods. Compare lambda_compute=1 as secondary
descriptive pricing, not main physical evidence. Randomized serial run order.
If >5% deadline misses for any cell, label timing inadmissible and do not use that
cell to demonstrate the desired high-compute degradation.

Gate B: at fixed K a middle H improves cost>=5% over shortest H, and longest H
loses>=5% vs that middle H in a common scene/condition; no extra collisions or
success loss allowed in the allegedly better arm. Gate A physical K shape is
analogous (at fixed H, some interior K beats both endpoints>=5%). At least two
contexts must have distinct optimal allocations beyond a5% cost indifference
band. Report differences attributable only to a failed episode separately.
L0 is a screening result; no formal positive claim or learning permission from
same-sample best-cell selection.

## Phase8/9: response surface and net headroom

Fit an unconstrained regularized quadratic/interactions model to normalized K,H,
1/sqrt(K), scene, speed and validated reliability; include measured tau for
explanatory fits and frozen expected tau only for executable selection. U_MC can
enter an explanatory surface only; current U_MC is forbidden in online selection.
Report group-held-out CV vs training-fold constant and additive models, coefficients
and failed signs. No forced monotonicity. A heuristic becomes a runnable comparator
only if CV error improves>=10% over context-additive baseline.

Compute contextual-oracle versus global-fixed costs separately with zero/positive
price. L0 oracle is optimistic and not a GO gate. If L0 mechanisms pass, freeze L1
near-optimum cells plus global/end-point anchors; run paired repeat seed1 including
measured, frozen and compensated conditions. Select using seed0, evaluate seed1,
then reverse. Charge full context and a conservative forward-MLP overhead to the
chosen command's physical delay in the validation run. Subtract twice the
cross-repeat absolute difference in paired gain as a noise reserve. Net relative
cost reduction must be>=10%, with>=60% context gains positive, no success reduction
or new collisions; it must survive frozen latency and compensation. Otherwise
stop before SAC. Do not call this finite-context oracle a theoretical upper bound
for all step-varying policies.

## Conditional phases

Only if all above mechanism, interaction, interior-optimum and net-headroom gates
pass: targeted novelty kill-check, standard SAC implementation, T0, then short
one-seed T1. Freeze the T1 schedule before training. T1 must capture>=30% of
validated oracle headroom, improve over fixed>=3%, have context-specific allocations
outside the5% indifference band, and avoid min/max or constant collapse. Failure
triggers observability/Q/replay/quantization diagnostics, no automatic longer run.
T2=3 paired training seeds only on T1 success; T3=10 frozen evaluation seeds only
after stable T2 improvements. Primary comparisons and multiplicity follow the
user's request. No T3 => no formal significance/CI or ICRA superiority claim.

## Stopping and complete reporting

A failed evaluated prerequisite stops dependent experiments. Code may be tested
and evidence analysed/archived afterwards, but no unauthorized RL training.
The final28-section report states every phase as completed, failed or not run,
with Q1–Q12 answers and all positive/negative evidence. Requested plots with no
data are marked not applicable, never fabricated.
