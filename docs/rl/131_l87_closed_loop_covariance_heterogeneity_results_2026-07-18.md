# L87 closed-loop covariance heterogeneity results

Date: 2026-07-18

## Outcome

L87 passed every preregistered gate.  This establishes closed-loop headroom
for a context-dependent MPPI sampling policy; it does **not** yet establish
that a learned policy can recover the oracle mapping.

- 480/480 scheduled MuJoCo episodes completed;
- 480 unique `(split, scene, physics domain, candidate, seed)` keys;
- 240 selection and 240 strictly held-out evaluation episodes;
- all episodes reached the goal and none collided;
- the globally selected fixed comparator was `speed = [1.75, 0.75]`;
- the context oracle used two candidates and differed from the global fixed
  comparator in 8/16 contexts.

On the 48 paired evaluation episodes, context oracle minus global fixed was:

- cross-track RMSE: -0.772 mm, hierarchical 95% CI
  [-1.469, -0.152] mm;
- time to goal: -1.419 s, hierarchical 95% CI
  [-2.125, -0.712] s;
- control jerk: -0.00769, hierarchical 95% CI
  [-0.01157, -0.00384];
- success delta: 0;
- collision delta: 0.

The mapping was stable across the four tested physics domains: `narrow` was
selected for chicane and hairpin, while `speed` was selected for accel-turn
and reverse-S.  On the eight contexts that changed action, time improved by
approximately 2.5--3.0 s per episode.

## Interpretation boundary

This result rejects the hypothesis that one global covariance is already
sufficient for all clean path geometries.  It supports learning a high-level
sampling decision above MPPI.  The oracle used selection outcomes and is
therefore non-deployable; its result is an upper-bound/learnability gate, not
the RL contribution itself.

Artifacts:

`results/research_platform/rl/l87_covariance_context_oracle_20260718_v1`

