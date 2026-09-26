# L100 nested anytime-budget Oracle: preregistration

Date: 2026-07-18

Status: frozen before execution

## Research question

L98 showed that a universal contextual `K=50` budget was not noninferior to
ICODE fixed `K=100`, but that comparison also changed sampling covariance. L100
isolates the budget factor:

> At the same physical state and under the same ICODE model, proposal mean and
> covariance, are samples 51--100 selectively valuable, and can their value be
> predicted using information available after the first 50 samples?

This is a development mechanism experiment. It is not a controller confirmation
and does not open the sealed L99 data.

## Experimental unit and blocking

The independent unit is one MuJoCo closed-loop episode identified by route,
physics domain and seed. Eight anchor states are repeated measurements nested
inside that episode and are never counted as eight independent replicates.

- routes: held-out hairpin and reverse-S;
- physics: anchor, high friction, long delay and combined moderate;
- discovery seeds: `20271201--20271202`;
- untouched evaluation seeds: `20271211--20271212`;
- total episodes: 32;
- target anchor records: 256.

Episode order is randomized with a frozen seed. Model, memory, perception and
safety settings are fixed. Counterfactual open-loop branches are allowed only in
obstacle-free scenes; the live trajectory retains ordinary safety arbitration.

## Common-random-number contract

At each anchor, the MuJoCo plant is captured with its complete integration,
actuator and command-delay state. One pool of 100 candidate sequences is drawn.

```text
K50  = candidates[0:50]
K100 = candidates[0:100]
```

The first 50 candidates must be byte-identical. The current state, ICODE model,
target, prior mean, covariance and preceding action are shared. Each budget
renormalizes MPPI weights over its own complete pool. There is no update of mean
or covariance between batches. Both weighted sequences are then executed from
the same restored MuJoCo snapshot for the full planning horizon.

This makes `true_cost_K50 - true_cost_K100` a paired estimate of additional
Monte Carlo budget value. It does not claim that more samples correct ICODE
model bias.

## First-batch predictors

Only quantities available after the first 50 samples may enter a deployable
budget policy:

- normalized effective sample size;
- normalized weight entropy and maximum weight;
- split-half first-control disagreement;
- high-weight first-control dispersion;
- normalized cost spread;
- sample saturation fraction;
- weighted perturbation norm;
- local route-geometry features;
- causal ICODE reliability diagnostics already available before execution.

Outcome costs, physics-domain names, seeds and future states are forbidden
inputs. ICODE innovation is treated as a possible predictor, not as proof that
additional sampling repairs model error.

## Frozen policies and analyses

The hindsight Oracle chooses `ADD50` only when K100 improves true branch cost by
at least 1% and does not introduce a collision. Two deployable development
baselines are fit using discovery episodes only:

1. a single-feature decision stump selected from frozen first-batch features;
2. a ridge-linear benefit predictor with fixed penalty `1.0`.

Both are evaluated without refitting on the held-out evaluation seeds. A random
policy matched to the learned policy's ADD50 rate is reported as a compute-
matched negative control.

Paired policy contrasts are first averaged within episode. Hierarchical
bootstrap intervals then resample route x physics blocks and episode seeds.
Anchor timesteps are not treated as independent observations.

## Primary development Gate

All clauses must pass:

1. integrity: all episodes and anchors present, nested-prefix hashes match,
   required values are finite and no counterfactual branch collides;
2. selective headroom: Oracle ADD50 fraction lies in `[0.10, 0.75]`;
3. practical headroom: Oracle mean relative true-cost gain is at least 1%;
4. predictability: the ridge policy improves evaluation true cost over fixed
   K50 with a 95% interval upper bound below zero;
5. efficiency: ridge mean budget is no greater than 80;
6. utility retention: ridge captures at least 30% of the evaluation Oracle gain.

The stump, matched-random policy, classification accuracy and correlations are
secondary. If the Gate fails, no confirmation run is opened. Thresholds or
features may be redesigned only in a new numbered development study that
retains the L100 result.

## Claim boundary

A pass would justify building an online stop/continue policy. It would not yet
show that RL is necessary, that the policy improves full closed-loop episodes,
or that the result transfers to obstacles or a real robot. Those require a
threshold comparison, a sequential RL ablation and a separately sealed
closed-loop confirmation.
