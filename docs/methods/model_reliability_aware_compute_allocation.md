# Model-Reliability-Aware Compute Allocation for MPPI

Research execution update (2026-09-07): development gates, bounded training and
baseline diagnostics are complete. The current paper-claim verdict is **NO-GO**;
see the [full result report](../reports/adaptive_compute_final_report.md). Generic
defaults below remain available; the executed study used explicitly versioned
development overrides, including a final K32..256/delta16 range.

Updated 2026-09-06. Research candidate, **continuous Beta PPO**, replacing the
2026-09-05 categorical prototype. No acronym, priority claim, or validated
superiority is asserted. Historical prototype results remain under their original
artifact directory and commit f0e5c89; its categorical checkpoints/configuration
are intentionally incompatible with this version.

## Scientific problem, motivation and novelty boundary

K is stochastic search breadth: how many candidate control trajectories MPPI
searches. H is model-based planning foresight: how far those trajectories predict.
A fixed K/H spends the same allocation in states whose marginal values of breadth
and foresight may differ. Their trade-off can be non-monotonic and scene-dependent.
Prediction error also accumulates with horizon. Model reliability can therefore
change the useful value of foresight, but the direction and strength of that
relationship are **unresolved hypotheses**, not guaranteed rules.

Candidate contributions are learned state-dependent allocation of MPPI computation
between rollout breadth and prediction foresight, and model-reliability-aware
computation allocation. No first claim or literature priority is established.
Large residuals need not mean a bad corrected model; low disagreement need not
mean correctness. No residual->H or disagreement->K rule appears in code/reward.
The platform must permit results to reject scene context, reliability context,
adaptation of either axis, or ICODE correction itself.

## Architecture and ICODE's two roles

```text
causal RobotObservation
 -> BudgetObservationBuilder(scene + optional dynamics + optional previous planner)
 -> BudgetPolicy (Fixed / Random-test / Learned Beta actor-critic)
 -> a=[a_K,a_H] in [0,1]^2
 -> BudgetActionSpace.map -> integer K,H
 -> AdaptiveBudgetMppiController -> existing standard MPPI, I=1
 -> existing scan guard -> MuJoCo -> existing task reward
 -> ComputeAllocationEnv -> real-latency reward and causal next state
 -> training only: RolloutBuffer -> BudgetPPOTrainer
```

ICODE corrects rollout dynamics: `f_corrected=f_nominal+r_phi`. Independently,
frozen residual/ensemble models provide current model-state descriptors. ICODE
does not choose the budget or gate its authority from those descriptors.
`dynamics_mode: nominal/residual` changes the prediction model in the same controller.

The wrapper reuses MppiPriorEnv with `external_prior_enabled=False`, which bypasses
legacy proposal-actor construction while retaining task reset, safety arbitration,
MuJoCo execution, perception, task reward and termination. RL actions are compute
budgets; robot controls always come from MPPI and its existing safety guard.
No SAC, proposal Actor, cost learning, sampling-distribution learning, memory,
IMM, probability forecasting, or future obstacle truth is introduced.

## Continuous allocation and integer execution

All parameters are loaded from config. Defaults:

| Axis | Minimum | Maximum | Execution step |
|---|---:|---:|---:|
| K | 128 | 1024 | 32 |
| H | 10 | 40 | 1 |

For j in {K,H}, with q_j = (max_j-min_j)/delta_j:

```text
b_j = min_j + a_j*(max_j-min_j)                 # pre-round budget
B_j = min_j + delta_j*floor(a_j*q_j + 0.5)       # round half up
quantized_a_j = (B_j-min_j)/(max_j-min_j)
quantization_error_j = quantized_a_j - effective_a_j
quantization_error_budget_j = B_j - b_j
```

Half-up ties are explicit, rather than depending on language-specific round-to-even.
Non-finite/out-of-range/wrong-shape actions fail instead of silently clipping.
Bounds and deltas must be positive integers, ranges ordered and divisible by delta.
Adaptive axes need nonzero ranges. The default allows **29 K values and 31 H values**,
not a 16-class preset catalog; the actor learns a continuous decision surface over
these quantization cells. Rounding need not be differentiable for policy gradients.

Modes `joint/adaptive_k/adaptive_h/fixed` share this interface. Inactive axes are
replaced with the configured fixed normalized action. Their raw supplied values,
effective values and quantized values are distinguishable in logs. Learned/random
policies sample only active axes, and PPO sums log densities/entropies only over
active axes. Fixed axes receive no actor gradients. To exactly reproduce the old
(600,36), configure fixed ranges min=max=600 and min=max=36 with fixed values 600,36.
Zero-width fixed-axis normalized action is defined as zero.

## State and explicit normalization

Default dimension: **41 = scene 17 + dynamics values 8 + availability flags 8 +
previous planner 8**. Other supported dimensions: scene 17, scene+dynamics 33,
scene+planner 25. Feature names/configuration are checkpoint-checked.

Scene input, in order:

1. Observed-pose distance to configured final goal /10 m, wrapped goal bearing /pi.
2. Observed linear velocity /0.35 m/s and yaw rate /0.9 rad/s.
3. Previous safety-arbitrated executed command divided by each channel's maximum
   absolute bound (two channels; zero at reset).
4. Eight angular scan-sector minima over [-pi,pi), each /4 m.
5. Overall minimum scan distance /4 m, fraction of valid beams within 1 m,
   valid-beam fraction.

All scales and sector count are configurable; scene outputs clip to [-5,5].
+Inf scan returns are no-return, capped at sensor/configured maximum; NaN, -Inf
and below-min returns are missing. Empty sectors default to configured range,
wholly missing scans have zero validity. No ground-truth obstacle geometry is read.
Current default lidar has full angular coverage. Per-sector availability for
partial-FOV lidar remains a limitation. Scene clearance is a scan-distance proxy;
logged outcome minimum clearance is the existing plant metric, not a policy input.

Dynamics values, each with a distinct availability flag:

| Feature | Causal implementation |
|---|---|
| residual_drift_norm | norm of physical output-masked a_phi(z_t) |
| residual_control_map_norm | Frobenius norm of physical output-masked B_phi(z_t) |
| residual_norm | norm of r_phi(z_t,u_executed,t-1), unavailable at reset |
| disagreement | existing normalized RMS ensemble residual standard deviation at current state and previous executed command |
| drift_disagreement | norm of componentwise member-a population standard deviation |
| control_map_disagreement | Frobenius norm of componentwise member-B population standard deviation |
| support | existing normalization heuristic on current state and previous executed command |
| innovation | EMA of normalized RMS corrected-model prediction error on completed observed transitions |

The actual feature order is residual_norm, drift norm, control-map norm,
disagreement, drift disagreement, control-map disagreement, support, innovation.
Defaults divide each descriptor by explicit scale 1.0 and clip to [0,5]. These
physical norms can mix state-channel units; they are descriptors, not calibrated
reliability probabilities. Values missing online are zero with availability false.

Physical affine decomposition follows the existing network normalization:
`B=diag(residual_scale)*G*diag(1/control_scale)`;
`a=residual_mean+residual_scale*d-B*control_mean`; both apply residual_output_mask.
No arbitrary future/test control is selected. Single PlatformResidualDynamics and
PlatformResidualEnsemble expose this API; general MLP residuals and other wrappers
without an explicit affine API retain unavailable a/B masks.

Single-model support explicitly calls the existing normalization-confidence
formula without using its gated derivative: maximum absolute state/control
training z-score <=3 gives 1, >=7 gives 0, linear between. Ensemble support is the
minimum member confidence. This is an uncalibrated support heuristic. Single-model
mode has no ensemble: all three disagreement masks are false. Support/residual_norm
are unavailable at reset; innovation is unavailable until a transition completes.

`online_ensemble.yaml` loads three existing frozen L57 checkpoints and evaluates
them online, without reading pilot post-processing. It leaves the active single
planner residual unchanged. Its residual disagreement uses the existing scales
[0.0192860868,0.0061503653,0.0276655704,0.2438328117,0.6734842658]. Innovation describes
the active/context corrected model, not the shadow ensemble's future error.
Innovation scales are [0.25,0.25,0.35,0.25,0.60], EMA decay .90, with theta wrapped.
The first completed error initializes EMA. Prediction uses the existing causal
command-delay approximation: `(delay/dt)*previous_command+(1-delay/dt)*executed`.
It uses observation transitions, not plant hidden state or applied-control truth.
It is affected by localization noise and the delay approximation.

Nominal mode has unavailable dynamics features by default. An explicit
`reliability.context_checkpoint` preserves frozen ICODE information in nominal
rollouts, allowing information access to be held constant in future comparisons.
Calibrated reliability probabilities, horizon-specific accuracy and future ranking
quality are not available.

Planner channels are exclusively from t-1:
`[K/max_K,H/max_H,min(rho,5),ESS/K,deadline_violation,guard_override,available,ESS_available]`.
They are zero at reset. No current latency/ESS and no raw cost_min enters z_t.
Cross-H raw costs do not have comparable scales.

PPO then applies running mean/variance normalization (epsilon 1e-8, clip +/-5).
Statistics are frozen over each collected rollout and its update, then updated
from raw collected observations. Evaluation never updates them. Availability flags
are included in normalization; raw flags remain recorded in logs.

## Continuous PPO formulation

Shared MLP defaults: input ->64 Tanh ->64 Tanh, four actor outputs and scalar V.
`alpha=softplus(raw_alpha)+1`, `beta=softplus(raw_beta)+1` for each of K/H.
The two Beta distributions are conditionally independent given the shared state.
Deterministic evaluation uses their **means**, alpha/(alpha+beta), then quantization.
The distribution/policy scoring interfaces can later be replaced, but this round
adds no autoregressive coupling or RL innovation.

Standard [PPO clipped surrogate](https://arxiv.org/abs/1707.06347) and
[GAE](https://arxiv.org/abs/1506.02438) are used:

```text
logp = sum_active_axes log Beta(a_j;alpha_j,beta_j) # raw continuous action, not rounded K/H
ratio = exp(new_logp-old_logp)
delta = reward+gamma*(1-terminated)*V(final_next_observation)-old_V
A_t = delta_t+gamma*lambda*(1-terminated_t)*(1-truncated_t)*A_(t+1)
value_target = A+old_V
actor_loss = -mean(min(ratio*A,clip(ratio,1-epsilon,1+epsilon)*A))
value_loss = 0.5*mean((V-value_target)^2)
loss = actor_loss+value_coef*value_loss-entropy_coef*mean(sum_active Beta_entropy)
```

Continuous differential entropy can be negative; that is not a failure. Inactive
endpoints are excluded safely before scoring, avoiding -Inf*0. Actual Beta samples
are stored unchanged with their log density. There is no likelihood surrogate on
rounded actions and no smoothing penalty by default. Returns stop at true terminal
states, bootstrap final observations at time limits, and never bridge resets.

Defaults: Adam lr=.0003/eps=1e-5; rollout128, epochs4, minibatch64, gamma=.99,
lambda=.95, clip=.2, entropy_coef=.01, value_coef=.5, max_grad_norm=.5. Advantages
are standardized per rollout (except single samples). Exact requested additional
steps are collected. Python/NumPy/Torch seeds and deterministic Torch algorithms
are controlled; actual timing-based rewards cannot be bitwise guaranteed.

Atomic update-boundary checkpoints store actor/critic, optimizer, normalization,
RNG states, step/update counts, next episode seed, mapping contract, and CLI feature/
reward/model-hash metadata. Resume starts a fresh episode, not a bitwise-restored
MuJoCo hidden state. No unfinished rollout is checkpointed. Old categorical
checkpoints are explicitly rejected. Only project-owned checkpoints should be loaded.
The existing continuous SAC implementation is not used; there is only one new
compute-policy RL algorithm, Beta-PPO, implemented with existing PyTorch dependencies.

## Reward and timing boundary

```text
rho = T_plan/T_control
computation_penalty = beta*rho
deadline_penalty = eta*max(0,rho-1)
switch_cost = lambda_switch*(abs(K-K_prev)/(K_max-K_min)+abs(H-H_prev)/(H_max-H_min))
r_RL = task_scale*r_task-computation_penalty-deadline_penalty-switch_cost
```

A zero-width fixed range contributes zero switch cost (implementation divisor is
max(span,1)); the first cycle has none. Defaults task_scale=1, beta=.1, eta=1,
lambda_switch=0. No direct residual, disagreement, K, H or K*H penalty exists.
Raw task reward, normalized reward, each penalty and total reward are logged.

Task reward is unchanged MppiPriorEnv._reward / RewardConfig. Default point-goal
terms are `8*(d_prev-.99*d)-.02`, goal+100, collision-100, clearance
`-1.5*max(0,.45-clearance_truth)^2`, guard-.15, effort `-.02*||u||^2`, change
`-.04*||u-u_prev||^2`, stuck-.03. Other existing path/heading terms default to zero,
intrinsic exploration is disabled. The inherited discounted potential can be
positive while stationary; it was not altered/tuned here. Complete resolved
coefficients are saved. PPO/task shaping use the same gamma.

T_plan uses perf_counter from action mapping through budget configuration and MPPI
action production, including sampling, correction, costs and final rollout; it
stops before safety/plant execution. Context and actor inference are timed separately.
The reward is specifically a planner computation price, not a full-cycle realtime
guarantee. The simulator does not inject extra actuation delay for slow host cycles.
All cold starts and deadline violations remain recorded.

## Dynamic K/H implementation and buffer strategy

The same MppiController retains RNG, previous solution, delay and safety state.
H changes shift the old solution first, truncate if shorter, or retain the valid
prefix and repeat the old final control if longer; the ordinary previous_sequence_blend
with goal warm start is unchanged. Equal-H behavior follows the original path.
H=1 is supported; candidate ranges must also satisfy existing safety-prefix validation.

The compute wrapper preallocates one float64 [max_K,max_H,action_dim] sample-control
scratch buffer (default **655,360 bytes**) and slices it per cycle. Addition and
clipping use this buffer in place. Original RandomState noise generation is
preserved exactly and still allocates noise; rollout outputs remain independently
owned to prevent a later plan from overwriting returned trajectories. Old controllers
do not enable this opt-in scratch by default. No MPPI objective/noise changes occur.
Buffer setup allocation time/bytes and per-cycle horizon allocation/copy/fill and
configuration times are logged. Setup time is a repeated metadata field, not a
per-cycle charge; normal sampling/trajectory allocation remains in stage timing.
No planner reconstruction or cache of all 899 execution pairs is needed.

## Complete online and training algorithm

```text
Algorithm: Model-Reliability-Aware Continuous Compute Allocation for ICODE-MPPI
Input: nominal f0, frozen ICODE residual, optional frozen online ensemble,
       Beta budget policy, configured K/H bounds/deltas/masks, ordinary MPPI settings.
Reset: sensor/reference/planner state, unavailable historical context, innovation EMA.
For each t:
 1 receive causal robot observation
 2 build scene context
 3 compute ICODE residual/reliability descriptors using current/history only
 4 retrieve previous planner diagnostics
 5 concatenate configured groups and availability into z_t
 6 sample active Beta actions a_K,a_H (evaluation: mean); fill fixed axes
 7 start planning timer; quantize/map a to K,H
 8 configure K,H and shift/resize/blend warm start
 9 sample K ordinary control trajectories
10 rollout H steps using f0+r_phi (or f0 in nominal mode)
11 evaluate unchanged MPPI objective and weighted update; produce first action
12 stop timer; pass action through existing guard and execute it
13 observe task transition/reward/termination
14 compute actual planning utilization and RL reward
15 update completed-transition innovation; publish planner t diagnostics for z_(t+1)
16 log raw/effective/quantized actions, errors, consumed context, distribution and outcomes
17 training: store raw continuous action/logp/value/reward/next-value/boundary flags
At rollout boundary:
18 compute GAE/targets using frozen rollout normalization
19 perform shuffled PPO minibatch updates and gradient clipping
20 update running observation statistics, save learning state at update boundary
```

The timer is stopped before execution so plant time is not charged as planning.
The feature API never reads simulator truth, auxiliary future arrays, current
planning output or pilot post-processing. Ground truth remains solely in existing
task reward/termination/outcome logging. Current dynamics corrections and diagnostics
are frozen model inference; no ICODE training/checkpoint writes occur.

## Ablations, commands and outputs

Eight config files under configs/compute_allocation implement the same backbone:

| Internal comparison | Config |
|---|---|
| A fixed nominal | fixed_nominal.yaml |
| B fixed ICODE | fixed_residual.yaml |
| C scene only joint | scene_only.yaml |
| D scene+dynamics joint | scene_dynamics.yaml |
| E scene+planner joint | scene_planner.yaml |
| F full joint | full.yaml (base defaults) |
| G adaptive K only | adaptive_k.yaml |
| H adaptive H only | adaptive_h.yaml |

All other plant/scene/cost/reward settings remain common. For evaluation use the
training mapping/input config and checkpoint. CLI overrides also support mode and
input groups. Published literature baselines will be implemented separately after
selecting appropriate sources; these configs are not published baselines.

From the WSL project root (fresh output directories required):

```bash
.venv/bin/python experiments/compute_allocation/run.py smoke --output results/continuous/smoke
.venv/bin/python experiments/compute_allocation/run.py train --steps 8 --max-episode-steps 4 --seed 0 --output results/continuous/train
.venv/bin/python experiments/compute_allocation/run.py evaluate --policy learned --checkpoint results/continuous/train/checkpoints/latest.pt --max-episode-steps 4 --seed 0 --output results/continuous/eval
.venv/bin/python experiments/compute_allocation/run.py train --steps 4 --checkpoint results/continuous/train/checkpoints/latest.pt --max-episode-steps 4 --seed 0 --output results/continuous/resume
.venv/bin/python experiments/compute_allocation/profile_compute.py --steps 4 --output results/continuous/profile
```

CSV/JSON cycles contain raw a_K/a_H, effective/quantized actions, normalized and
budget-unit quantization errors, current/previous K/H, I/K*H, actual planning time,
rho/deadline, raw/normalized task reward, separated penalties, total RL reward,
success/collision/goal/clearance, ESS/ESS/K, all dynamics values/masks, consumed scene
context, actor alpha/beta, action log probability/value, mapping/actor/context/
configuration/resize timing, sample-buffer allocation metadata and planner profiles.
Fixed/random test policies have null learned distribution/value fields.

Episodes include completion time (null on failure), path length including initial
pose, minimum clearance, task/RL returns, summed planning seconds, latency quantiles,
deadline count/rate, K/H means/histograms/joint distributions, total theoretical K*H,
and descriptive K/H-to-scene/model Pearson correlations. Correlations use only
available finite inputs, record valid-pair counts, and are null with fewer than
three pairs or zero variance. They are descriptive, not causal or independent
statistical replicates. Resolved config, manifests/model hashes, PPO update records,
and checkpoints accompany logs. JSON is finalized on normal close; partial episodes
are explicitly labelled.

## Pilot evidence and unresolved hypothesis

The [K/H pilot](../../research_artifacts/kh_pilot_2026-09-05/report.md) is unchanged.
Easy's equal-K*H allocations (1024,10)/(512,20)/(256,40) performed similarly (D1
134/133/131 steps, all successful). Complex's balanced (512,20) achieved much more
goal progress than either extreme, but **all Complex conditions still failed the
unchanged .30 m success criterion**. This suggests context-dependent, non-monotonic
breadth/foresight trade-offs, not a proven adaptive-controller advantage.
D1/D2 did not establish a clear reliability contrast. Thus ICODE reliability ->
value of H is unresolved. The pilot's offline ensemble cannot prove it and its
visited-state/low-motion confounding remains. Equal K*H did not imply equal time;
residual planner ~38-78 ms versus nominal ~4-6 ms motivates actual value-of-computation
measurement, not a K*H proxy. Those pilot numbers are not a universal latency model.

## Validation, profile and next bounded screen

See [continuous implementation report](model_reliability_aware_continuous_implementation.md)
and research_artifacts/continuous_compute_2026-09-06. Only unit/integration checks,
short random-control smoke and 8+4 PPO CLI transitions were run. No reward tuning,
formal learning run, multi-seed qualification or ICODE modification occurred.

Profile instruments nominal derivative, residual inference, sampling, batch/final
rollout, cost, resize, mapping and actor inference. Derivative times are nested in
rollout; actor/context are outside T_plan. Percentages are therefore not additive.
Residual inference remains the bottleneck, already batched across K. H and RK4
substeps are sequential dependencies, so flattening them changes semantics. No
residual core rewrite is justified by this smoke.

Next proposed experiment (not executed): freeze bounds/scales/reward/platform and
one development scene, then run `train --steps 2048 --seed 0` with a new output,
inspect finite losses, Beta concentration/entropy, raw versus quantized actions,
availability, histogram and actual latency. Evaluate its frozen mean policy against
matched fixed budgets, with serial randomized run order. Do not count control steps
as independent replicates or tune beta/eta for positive results. A separate
reliability manipulation/held-out study is needed before claiming the mechanism.
GPU/hardware, long-run stability, partial-FOV sensing, calibrated reliability,
bitwise simulator resume and full-cycle realtime effects remain unqualified.
