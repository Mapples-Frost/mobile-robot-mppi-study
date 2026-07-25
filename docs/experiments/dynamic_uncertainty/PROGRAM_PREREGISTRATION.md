# Dynamic-Uncertainty Program Preregistration

Status: development protocol frozen before probability-sandbox data generation  
Platform: native Windows only  
Formal sealed seeds: reserved and prohibited in this stage

## Research questions

1. Does Change-Aware IMM improve accuracy and calibration after stochastic
   motion changes relative to deterministic CV and Gaussian CV?
2. Does calibrated candidate collision probability reduce collision and
   dangerous approach without excessive waiting?
3. Under the same predictor and risk cost, does the frozen RL--ICODE--HSS stack
   improve efficiency relative to vanilla risk-aware MPPI?
4. Does the full system remain stable when robot dynamics and obstacle motion
   are simultaneously out of distribution?

## Experimental unit and blocking

One independently seeded obstacle trajectory is the independent unit.
Horizons and time steps from that trajectory are repeated measurements, not
independent samples. Predictors are paired on the exact same truth and
observation history. Process, noise level, and horizon are within-seed strata.

The run schedule is seeded and written before model evaluation. Run order will
be randomized within split. Analyses aggregate each trajectory first and then
use seed-cluster effects; time steps will never be counted as independent
replicates.

## Motion processes

- P1 Noisy CV: approximately constant velocity with stochastic acceleration.
- P2 Speed Change: one random acceleration, deceleration, or stop event.
- P3 Direction Change: one random turn from {-90, -45, +45, +90} degrees.
- P4 Combined Change: direction and speed changes plus short missing-observation
  intervals.

Process noise changes the real obstacle state. Observation noise changes only
what the estimator sees. Separate deterministic random-number streams are used
so the two sources cannot silently contaminate one another.

## Predictors

1. Deterministic CV;
2. Gaussian CV Kalman filter;
3. IMM with CV, Turn-L, Turn-R, and Brake/Stop modes;
4. Change-Aware IMM with NIS detection and covariance inflation.

The first implementation stage is restricted to predictors 1 and 2.

## Splits and seed rules

The committed registry `configs/seeds/dynamic_uncertainty_splits.yaml` defines:

- Development: tuning, smoke tests, and threshold selection;
- Held-out ID: frozen in-distribution predictor and calibration evaluation;
- Held-out OOD: frozen process/noise shift evaluation;
- Sealed: separate existing registry; never imported by development scripts.

Split membership is determined by seed namespace before trajectories are
generated. No trajectory, observation realization, or window may cross splits.
The proposed full predictor dataset is 4 processes x 3 splits x 300 independent
trajectories. The first smoke is only 4 processes x 2 development noise
settings x 5 development seeds = 40 trajectories.

## Prediction outcomes

- ADE and FDE;
- Gaussian negative log likelihood;
- 50%, 90%, and 95% position coverage;
- probability-ellipse area;
- change detection delay and false-alarm rate;
- dominant-mode accuracy;
- post-change recovery time.

Metrics that require a Gaussian distribution do not apply to deterministic CV
and will be marked not applicable, not replaced by zero.

## Predictor Gate

Before collision-risk work:

1. Change-Aware IMM improves ADE/FDE or NLL over Gaussian CV on change
   processes without systematic degradation on Noisy CV;
2. held-out-ID coverage is within preregistered tolerances of nominal coverage;
3. post-change uncertainty is not persistently underestimated;
4. held-out OOD preserves the expected method ordering or is explicitly
   reported as a failure;
5. every covariance is finite, symmetric, and positive semidefinite;
6. every mode-probability vector is finite, nonnegative, and sums to one;
7. no predictor has access to future truth or true mode/change labels;
8. all relevant tests pass.

Exact numerical tolerances for the IMM Gate must be frozen before its held-out
data are opened.

## Collision-probability calibration Gate

Candidate trajectories are fixed before empirical obstacle futures are
generated. Each candidate/initial-state pair uses 1,000 independent true
futures. Primary outcomes are Brier score, reliability curve, expected
calibration error, Spearman ranking, pairwise ranking, high-risk recall, and
low-risk false alarm.

The Gate requires stable high-risk identification, better risk ranking than
deterministic minimum distance, no systematic risk underestimation, and stable
ordering across the preregistered Monte Carlo sample-count sensitivity. Risk
thresholds remain development-only until this Gate passes.

## Risk-planning Gate

The paired development comparison uses Noisy Crossing, Sudden Stop, Sudden
Turn-In, and Sudden Reverse with fixed MPPI budget, base cost, safety chain,
initial robot state, and obstacle realization. M4 must reduce collision or
dangerous approach relative to M1/M2 without systematic completion collapse,
unacceptable waiting, or planner p99 deadline failure. Event timelines must
agree with predictor uncertainty and risk outputs.

## Qualification and sealed design

Qualification crosses robot seen/unseen with obstacle seen/unseen, giving four
joint domains. DRA-MPPI Adapted, Simple, and Full use paired seeds and equal
compute. The planned qualification size is 3 methods x 4 scenes x 4 domains x
3 development seeds = 144 episodes.

Only a passed qualification authorizes the frozen sealed matrix:
3 methods x 4 scenes x 4 domains x 10 fresh sealed seeds = 480 episodes.
Seed is the independent unit; scene/domain/change type are within-seed repeated
conditions. Primary effects use paired seed-cluster estimates and 10,000
cluster bootstrap replicates.

## Stop conditions

Stop and retain all artifacts when:

- a Gate fails;
- covariance or probability invariants fail;
- any future truth enters an online predictor input;
- a development script imports or emits a sealed seed;
- resolved configuration, seed, or provenance disagrees with the schedule;
- NaN/Inf, duplicate experimental keys, or missing required artifacts occur;
- a safety layer is bypassed;
- the agreed compute or real-time budget is exceeded.

No failed method, seed, scene, or threshold may be removed after outcomes are
observed. Any design change requires an amendment written before new outcomes
are inspected.

## Probability concepts in control language

- The mean is the nominal predicted obstacle state.
- The covariance is a belief-state confidence envelope, not an extra physical
  obstacle trajectory.
- Process noise is model/plant mismatch in obstacle dynamics.
- Observation noise is sensor error.
- Calibration asks whether the claimed confidence matches long-run frequency.
- IMM is a bank of motion models with a Bayesian probability over which model
  is currently plausible.

