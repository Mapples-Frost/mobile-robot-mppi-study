# L285 Frozen-Actor Proposal Gate protocol

## Question

Does the recovery-capable L276 Actor already provide a useful frozen proposal
distribution to the unchanged value-consistent ICODE + HSS + MPPI stack?
L276 versus the pre-imitation source Actor is the only confirmatory contrast.
L281 and L284 are descriptive forgetting-trajectory controls and may not be
selected by their L285 outcome.

## Original confirmatory design (superseded by the amendment below)

- Platform: Windows native only; no WSL/Linux MuJoCo or benchmark execution.
- Independent unit: one paired Actor/environment seed cluster (`n=3`).
- Repeated strata: three L260 development-validation geometries in the nominal
  seen physics domain, loaded through L285 wrappers that add only the frozen
  Full-MPPI corridor contract (`completion_corridor=0.75`,
  `corridor_half_width=0.80`, `footprint_radius=0.20`, and boundary
  termination). Episodes are not treated as independent replicates.
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

## Original outcomes and frozen Gate (withdrawn for the shortened screen)

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

## Pre-outcome engineering amendment

The first two five-step engineering smokes failed before producing an episode
at the path-boundary fail-closed check because the raw L260 manifests, and then
their L261 value-stability wrappers, did not instantiate the reference's
`corridor_half_width` and `footprint_radius`. Before any L285 method outcome
existed, dedicated L285 wrappers were added. They inherit each L260 geometry
and initial state and add the same 0.80/0.20 Full Tracking corridor/footprint
contract used by the existing safety chain, plus the pre-existing 0.75
completion corridor. No seed, checkpoint, MPPI cost weight, Gate threshold, or
arm changed; both failed smoke outputs are retained.

## Pre-outcome resource-budget amendment

After exactly one of the originally scheduled 45 episodes completed, and
before any episode metric or method effect was inspected, the user requested a
substantially shorter run.  The original process was stopped; its completed
episode and interrupted second-episode directory are retained as an aborted
confirmatory raw run and are not used by the revised analysis.

The revised L285 study is explicitly a **direction screen**, not a
confirmatory Gate.  It uses the first schedule block fixed independently of
outcomes: evaluation seed `20264511`, all three scenes, and all five arms (15
episodes).  Its within-seed blocking, arm order, checkpoints, runtime budgets,
costs, and safety contracts are unchanged.  A promising screen may authorize
only a separately preregistered multi-seed confirmation; it cannot establish
cross-seed stability, open final maps, or be reported as the original 45-run
Gate.  A non-promising screen stops this frozen-Actor direction and authorizes
only the already specified early-stopping diagnosis.
