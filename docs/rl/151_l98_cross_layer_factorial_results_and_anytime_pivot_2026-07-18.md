# L98 cross-layer factorial: results and bounded anytime pivot

Date: 2026-07-18

Status: completed development experiment; integrity audit passed; primary Gate failed

Confirmation status: L99 remains sealed and was not executed

## Question and design

L98 tested whether the independently retained ICODE and contextual-sampling
modules compose into one useful MPPI controller. The blocked 2 x 3 factorial
crossed:

- prediction dynamics: nominal or frozen ICODE residual dynamics;
- sampling: fixed `K=50`, contextual covariance at `K=50`, or fixed `K=100`.

The experiment used two held-out high-dynamic routes, four MuJoCo physics
domains and three development seeds. Each route x physics x seed block received
all six arms. Controller steps were not treated as independent replicates.

## Integrity audit

- observed/expected episodes: 144/144;
- complete blocks: 24/24;
- unique episode keys: 144/144;
- run-order balance: every arm appeared four times in every position;
- successes: 144/144;
- collisions: 0/144;
- required numeric metrics finite: yes;
- optional `minimum_clearance`: missing in all episodes by design because this
  experiment is clean path tracking without obstacles.

The integrity audit passed. The missing clearance values were retained as
missing rather than replaced by invented values.

## Preregistered primary results

Paired deltas are treatment minus reference. Confidence intervals are the
frozen hierarchical bootstrap intervals over route x physics and seed.

| Gate | Paired contrast | RMSE delta, mean [95% CI] | elapsed delta | compute delta | Result |
|---|---|---:|---:|---:|---|
| A: ICODE contribution | ICODE contextual K50 - nominal contextual K50 | -14.79 mm [-20.92, -8.56] | -0.254 s | +20.603 ms | pass |
| B: half-budget contextual policy inside ICODE | ICODE contextual K50 - ICODE fixed K100 | +0.94 mm [-2.20, +4.78] | -1.738 s | -6.094 ms | **fail** |
| C: complete package | ICODE contextual K50 - nominal fixed K100 | -22.37 mm [-26.37, -18.53] | -1.800 s | +20.363 ms | pass |

All three contrasts preserved success and collision outcomes. The complete
package's absolute mean planner compute was 24.811 ms, below the frozen 50 ms
real-time threshold.

Gate A provides strong development evidence that ICODE improves closed-loop
tracking under the same contextual `K=50` sampler. Gate C shows that the
complete ICODE + contextual-K50 package is more accurate than the conventional
nominal-K100 comparator while remaining within the absolute compute budget.

Gate B does **not** pass: although contextual `K=50` reduced time and compute,
the upper RMSE interval was +4.78 mm, above the preregistered +2 mm
noninferiority margin. Therefore the joint primary Gate failed. L99 is not
opened, and this development experiment does not support a universal claim
that `K=50` preserves ICODE-K100 precision.

## Secondary interpretation

The failure is structured rather than a general rejection of the research
direction:

- on the held-out hairpin, contextual `K=50` was already within the RMSE margin
  in all 12 blocks;
- on the held-out reverse-S, a `K=100` budget was needed in 5 of 12 blocks;
- the difference therefore points to state-dependent compute demand, not to a
  need to discard either ICODE or contextual sampling.

The exploratory hindsight oracle selected contextual `K=50` in 15 blocks,
fixed `K=50` in 4 blocks and fixed `K=100` in only 5 blocks. Relative to always
using fixed `K=100`, this non-deployable oracle had:

- RMSE delta -2.11 mm, 95% CI [-3.78, -0.79] mm;
- elapsed delta -1.763 s, 95% CI [-2.550, -0.963] s;
- mean planner-compute delta -5.098 ms, 95% CI [-6.202, -3.739] ms.

This is only a headroom test because the oracle uses completed-episode outcomes.
It must not be reported as an online controller result. Simple one-feature
diagnostics also showed weak rank association with the K50 error, so no single
threshold is justified by L98.

## Bounded next mechanism

The next implementation should preserve the successful modules while keeping
two causal responsibilities separate. More samples can reduce Monte Carlo
sampling error, but cannot repair a biased dynamics model.

### Model-reliability gate

A causal innovation statistic compares the previous ICODE prediction with the
new observation. It controls a residual trust coefficient `beta in [0, 1]`:

```text
f_used = f_nominal + beta * f_ICODE_residual
```

Persistent prediction error may reduce `beta`, trigger a nominal fallback or
increase a safety margin. It is not, by itself, evidence that another 50
rollouts will help.

### Compute-budget gate

The budget gate operates on MPPI sampling uncertainty:

1. freeze the state, prior, covariance and prediction model;
2. execute the first 50 MPPI samples;
3. compute causal diagnostics such as normalized effective sample size,
   split-half control disagreement, high-weight action dispersion and sample
   saturation;
4. stop when the control estimate is sufficiently stable;
5. otherwise draw 50 independent samples from the same proposal, retain the
   first batch and renormalize weights jointly over all 100 samples;
6. leave safety arbitration unchanged.

ICODE innovation may be included as a predictor only if same-state evidence
shows that it predicts the *benefit of additional sampling*. It must not be
presented as though sample count directly removes model bias.

This can be described as **reliability-aware anytime ICODE-MPPI**: model
reliability controls which prediction model is trusted, while planner sampling
uncertainty controls how much Monte Carlo computation is allocated. The
existing route-context LinUCB policy remains a fixed baseline and may select the
first-batch covariance; it does not receive simulator physics labels.

The current episode-level hindsight analysis only establishes preliminary
headroom. Before fitting a gate, the next development experiment must compare
nested `K=50` and `K=100` decisions at the same recorded control states using
common random numbers: the first 50 samples of `K=100` must be exactly the
`K=50` batch. It must test whether the benefit of samples 51--100 is stable and
predictable from first-batch information. A hand-written threshold is a required
baseline; a learned or RL stop/continue policy is justified only if temporal
decision effects leave measurable headroom beyond that baseline.

The mechanism is deliberately small: it adds an auditable model-trust decision
and stop/continue decision, not another end-to-end controller. A new development
study must be preregistered before fitting thresholds or a policy. Confirmation
data remain untouched until that study passes its own frozen Gate.

## Reproducibility boundary

The versioned repository contains the configuration, runner, analysis and unit
tests. Raw `results/` artifacts remain ignored because they are large and
reproducible; each executed result directory contains its config snapshot,
episode table, summary and audit outputs. This is a development result, not a
paper-level final estimate and not evidence of stability or convergence.
