# L285 Frozen-Actor Proposal Gate protocol

## Question

Does the recovery-capable L276 Actor already provide a useful frozen proposal
distribution to the unchanged value-consistent ICODE + HSS + MPPI stack?
L276 versus the pre-imitation source Actor is the only confirmatory contrast.
L281 and L284 are descriptive forgetting-trajectory controls and may not be
selected by their L285 outcome.

## Design

- Platform: Windows native only; no WSL/Linux MuJoCo or benchmark execution.
- Independent unit: one paired Actor/environment seed cluster (`n=3`).
- Repeated strata: three L260 development-validation scenes in the nominal
  seen physics domain.  Episodes are not treated as independent replicates.
- Blocking: within every seed x scene block, all five arms use the same scene,
  environment seed, physics domain, rollout budget, and episode budget.  Arm
  order is generated from the committed schedule seed.
- Arms: value-aligned ICODE without RL proposal, and Full with the source,
  L276, L281, or L284 frozen Actor.
- Actor parameters are never updated.  Reward, 69D observation, ICODE models,
  value models, HSS, MPPI costs, rollout count, fusion, scan guard, and safety
  chain are unchanged across Full arms.
- Development seeds are `20264511`, `20264512`, and `20264513`.  They are
  disjoint from the L260/L277 validation seeds and all final or sealed seeds.
- The final Hairpin, S-Chicane, and Infinity geometries and all L258 artifacts
  are prohibited.

The complete design is 3 seed blocks x 3 scenes x 5 arms = 45 episodes.  The
runner writes the frozen schedule before MuJoCo starts and supports exact
episode-level resume without replacing a completed experimental key.

## Outcomes and frozen Gate

Primary paired effects are L276 Full minus source Full.  The Gate requires:

1. all 45 unique episodes, exact checkpoint hashes, finite metrics, and the
   frozen runtime contract;
2. no per-seed mean-completion regression beyond 0.02 and nonnegative median
   completion change;
3. median CTE improvement of at least 0.05 m or median final-goal-distance
   improvement of at least 0.10 m;
4. the intended CTE/goal direction in at least two of three seed clusters and
   positive CTE improvement in at least two of three scenes;
5. no collision or boundary-violation increase relative to source Full or the
   value-aligned ICODE control;
6. L276 Full completion noninferior to ICODE within 0.02 per seed and
   nonnegative at the median; and
7. nonzero proposal use: median proposal authority at least 0.01 and median RL
   elite fraction at least 0.001.

L281 and L284 cannot rescue a failed L276 Gate.  A pass authorizes only a
larger preregistered independent-seed development validation of the frozen
L276 Actor.  A failure authorizes only the preregistered early-stopping study;
neither outcome opens the final maps automatically.

