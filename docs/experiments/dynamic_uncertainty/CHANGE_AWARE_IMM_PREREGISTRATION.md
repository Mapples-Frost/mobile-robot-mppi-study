# Change-Aware IMM Development Preregistration

Status: frozen before Change-Aware IMM pilot outcomes are inspected  
Date: 2026-07-23  
Scope: offline obstacle prediction only

## Objective

Test whether an online NIS detector can make the frozen ordinary four-mode IMM
respond faster and become less overconfident after unannounced V3 motion
changes.

The frozen comparators are:

1. Gaussian CV Kalman predictor;
2. ordinary four-mode IMM;
3. Change-Aware IMM.

V3, the Gaussian CV baseline, and every ordinary-IMM parameter remain
unchanged. No collision-risk, MPPI, RL, ICODE, HSS, or MuJoCo controller module
is enabled.

## Online information boundary

The Change-Aware predictor receives exactly the same online inputs as ordinary
IMM: timestamps, current/past noisy 2-D positions, and observation
availability. It may not receive truth, V3 process identity, true change flags,
events, waypoints, future schedules, or robot state.

True change flags are used only by the offline evaluator after all predictions
and trigger records have been written.

## Detector in control language

For a position observation `z`, the detector computes

```text
innovation = z - predicted_position
NIS = innovation' * innovation_covariance^-1 * innovation
```

NIS is a dimensionless, covariance-normalized residual. For two measured
coordinates and a correct Gaussian model, it approximately follows a
chi-square distribution with two degrees of freedom. A large NIS therefore
means: "the new measurement is too surprising for the confidence claimed by
the current model bank."

The detector uses only past and current observations. It does not identify the
true V3 event type.

## Registered response

When the NIS rule triggers:

- reset the four mode probabilities to `[0.40, 0.20, 0.20, 0.20]`;
- multiply each component state covariance by 6;
- multiply process covariance by 4 during a 1.0 s recovery interval;
- suppress a second trigger for a 1.0 s refractory interval.

The recovery inflation decays linearly to one. It is applied both to online
filter propagation and to the portion of a future forecast lying inside the
remaining recovery interval.

Observation dropout is not falsely called a motion change. After 0.25 s of
continuous missing observations, a separate dropout guard multiplies component
covariance by 2 once and activates the same recovery process-noise inflation.
Dropout-guard activations are reported separately from NIS triggers.

## Frozen pilot candidates

Only the NIS decision rule varies:

| Candidate | NIS threshold | Required exceedances | Window |
|---|---:|---:|---:|
| persistent_95 | 5.991465 | 2 | 3 available observations |
| single_99 | 9.210340 | 1 | 1 available observation |
| persistent_90 | 4.605170 | 2 | 3 available observations |

These are fixed packages, not a continuous threshold search.

## Development split

The independent unit is one complete trajectory seed. All predictors are
paired on identical truth and observations.

- Pilot selection: seeds 730100041--730100043, giving
  4 processes x 2 noise levels x 3 seeds = 24 units per candidate.
- Confirmation: untouched seeds 730100044--730100050, giving
  4 processes x 2 noise levels x 7 seeds = 56 units.

Forecast samples within one trajectory are repeated measurements, not
independent replicates. Held-out ID, held-out OOD, and sealed seeds remain
closed.

## Pilot selection rule

A candidate is feasible only when:

1. all numerical and causal invariants pass;
2. P1 false NIS triggers do not exceed 3 per trajectory-minute;
3. P1 ADE is no more than 1.10 times ordinary-IMM P1 ADE;
4. overall mean 95% region area is no more than 2.5 times ordinary IMM.

Among feasible candidates, select the lowest equal-weight mean of P2--P4 NLL in
the `post_change_0_1` and `post_change_1_3` windows. Ties within `1e-9` are
resolved by lower P1 false-trigger rate, then lexicographic candidate name.

If no candidate is feasible, stop without opening confirmation results.

## Confirmation metrics

Report process-macro metrics and paired candidate-versus-ordinary differences:

- ADE, 3 s FDE, exact mixture NLL;
- 50%, 90%, and 95% moment-matched coverage;
- mean 95% region area and mixture spread;
- event-window and observation-availability metrics;
- NIS trigger count, true-change recall, false triggers per minute;
- median and p90 detection delay.

Detection matching uses the V3 `change_flags` offline. A trigger matches at most
one previously unmatched true change occurring from 0 to 1.5 seconds before
the trigger. Dropout-guard activations are excluded from motion-change recall
and false-trigger calculations.

## Change-Aware Predictor Gate

On the 56 confirmation units:

1. every run is complete, unique, exactly replayable, finite, symmetric, PSD,
   and probability-normalized;
2. the ordinary-IMM replay agrees exactly with its frozen implementation;
3. P1 Change-Aware ADE / ordinary-IMM ADE is at most 1.10;
4. P2--P4 macro NLL is at most 0.98 times ordinary IMM;
5. both post-change NLL windows are at most 0.98 times ordinary IMM;
6. P2--P4 macro 90% coverage is not lower than ordinary IMM;
7. overall mean 95% region area is at most 2.0 times ordinary IMM;
8. P1 false NIS triggers do not exceed 3 per trajectory-minute;
9. true-change recall is at least 0.20 and median matched delay is at most
   1.25 seconds.

Passing is an offline predictor result only. Failing preserves all artifacts
and triggers a documented amendment; it must not cause V3 to be modified.

