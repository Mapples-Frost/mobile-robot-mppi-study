# Ordinary IMM Predictor Preregistration

Status: frozen before ordinary-IMM evaluation  
Date: 2026-07-23  
Scope: offline obstacle prediction only

## Objective

Compare three online predictors on the frozen V3 obstacle process:

1. deterministic constant velocity;
2. Gaussian constant-velocity Kalman filter;
3. ordinary four-mode interacting multiple model (IMM).

No MuJoCo controller, collision-risk estimator, MPPI, RL, ICODE, or HSS module
is enabled.

## Predictor input contract

The online predictor may receive only:

- observation timestamp;
- current/past noisy two-dimensional position observations;
- whether each observation is available.

It may not receive:

- true state or velocity;
- V3 process name;
- true mode or event type;
- waypoint coordinates or waypoint identifiers;
- current or future waypoint goal;
- future event schedule;
- future dropout schedule;
- robot start, robot goal, or future robot trajectory.

Offline evaluation may use hidden labels only after forecasts have been written.

## Ordinary IMM model bank

State:

```text
x = [px, py, vx, vy].
```

The four registered linear-Gaussian motion models are:

1. CV;
2. coordinated left turn with fixed positive yaw rate;
3. coordinated right turn with fixed negative yaw rate;
4. exponentially decaying velocity for brake/stop.

The IMM performs standard probability-weighted state mixing, per-model
prediction/update, measurement-likelihood mode update, and Gaussian-mixture
forecasting.

## Reverse and multimodality interpretation

Endpoint reversal in P2 remains in-distribution because it is preceded by a
visible stop at a registered spatial structure. Random mid-route reversal is
not included in ID and is reserved for future OOD evaluation.

Before a stopped obstacle restarts or a branch becomes observable, the
four-mode IMM is not required to guess one unique future waypoint. A valid
forecast should retain probability in multiple modes and expand mixture
uncertainty. After new motion becomes observable, probability should move
toward the corresponding model.

## Development design

The independent unit is one complete trajectory seed. Forecast times and
horizons inside one trajectory are repeated measurements.

Development evaluation:

```text
4 V3 processes x 2 noise levels x 10 new development seeds = 80 trajectories
```

Seeds: 730100041--730100050.  
Prediction horizon: 3.0 s at 0.05 s resolution.  
Forecast evaluation stride: 0.20 s.  
Warm-up: 1.0 s.

Run order is randomized with the committed schedule seed. V1/V2/V3
qualification seeds, held-out ID/OOD seeds, and sealed seeds are excluded.

## Metrics

Report each process separately, then macro-average process-level values:

- ADE and 3-second FDE;
- Gaussian/mixture NLL;
- 50%, 90%, and 95% coverage;
- mean 95% prediction-region area;
- IMM mixture spread;
- mode probabilities;
- mode-switch response summaries;
- observation-available versus occluded performance.

Event-relative forecast-origin windows:

```text
pre_change:       -2.0 <= delta_t < 0.0 s
post_change_0_1:   0.0 <= delta_t < 1.0 s
post_change_1_3:   1.0 <= delta_t < 3.0 s
steady:            otherwise
```

Trajectory metrics are computed first. Time steps are not treated as
independent replicates.

## Ordinary Predictor Gate

The ordinary IMM engineering gate requires:

1. all 80 runs complete with unique keys;
2. exact deterministic replay;
3. finite state, forecast, likelihood, and NLL values;
4. every covariance symmetric and positive semidefinite;
5. mode probabilities non-negative and summing to one;
6. no latent-intent or future-schedule access;
7. event-window, occlusion, and per-process reports are complete;
8. macro 90% coverage is between 0.55 and 0.995;
9. macro ADE is no more than 1.25 times Gaussian CV macro ADE.

Passing this gate does not establish that ordinary IMM is the final method.
It establishes a valid probabilistic baseline for the later Change-Aware IMM.

## Non-adaptation rule

If ordinary IMM performs poorly, investigate or revise the predictor using only
the declared development workflow. Do not alter V3 behavior to improve IMM
results.
