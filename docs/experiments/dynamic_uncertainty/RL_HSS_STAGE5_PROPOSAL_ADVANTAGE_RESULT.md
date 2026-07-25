# Stage 5 proposal-advantage HSS mechanism-probe result

## Decision

The preregistered Stage 5 gate fails because the shadow arm collided in
`seed 730100064`.  The registered first-collision rule stopped execution after
`7/9` episodes; the remaining RL-off and active-veto cells for that seed must
not be run or imputed.

Within this failed safety gate, the two complete blocked comparisons provide a
clear but bounded mechanism signal: applying the episode-latched proposal-
advantage veto restored goal completion in `2/2` fresh development blocks,
while the otherwise identical shadow treatment reached the goal in `0/2`.
This supports continuing the safe-rejection design, but two complete pairs are
not sufficient for a confirmatory or publication-level effect estimate.

## Completed episodes

| Seed | Arm | Outcome | Steps | Final distance (m) | Planner P95 (ms) | Veto latch step |
|---:|---|---|---:|---:|---:|---:|
| 730100068 | active veto | reached | 328 | 0.2980 | 56.66 | 12 |
| 730100068 | shadow | max steps | 400 | 1.0803 | 81.40 | 12 (counterfactual only) |
| 730100068 | RL/HSS off | reached | 367 | 0.2970 | 25.37 | n/a |
| 730100066 | active veto | reached | 299 | 0.2987 | 56.57 | 12 |
| 730100066 | shadow | max steps | 400 | 0.6262 | 80.38 | 12 (counterfactual only) |
| 730100066 | RL/HSS off | reached | 340 | 0.2974 | 22.83 | n/a |
| 730100064 | shadow | collision | 381 | 0.8356 | 77.83 | 11 (counterfactual only) |

No held-out or sealed seed was opened.

## Paired descriptive effects

The complete blocks are seeds `730100066` and `730100068`.

### Active veto minus shadow

- success delta: `+1` in both blocks;
- step delta: `-101` and `-72`, median `-86.5`;
- final-goal-distance delta: `-0.3275 m` and `-0.7823 m`, median
  `-0.5549 m`;
- planner P95 delta: median `-24.28 ms`.

### Active veto minus RL/HSS-off

- both arms reached the goal in both complete blocks;
- active veto used 41 and 39 fewer steps, median `-40`;
- final-goal-distance delta was only `+0.00116 m` at the median because all
  four episodes stopped at the shared goal tolerance;
- active-veto planner P95 was `+32.51 ms` slower at the median but remained
  below the 100 ms deadline in both episodes.

No p-value, confidence interval, or standardized effect size is reported.  With
only two complete blocks and informative missingness from the safety stop, an
inferential test would be misleading.

## Veto integrity

Both completed active-veto episodes latched at zero-based decision 12.  The
decision that accumulated the third disadvantage retained the causal current
authority.  From the next decision through termination:

- applied advantage-gate authority was exactly zero;
- guided candidate count was exactly zero;
- Actor authority over proposal centre and variance was exactly zero;
- proposal fallback fraction was exactly one;
- total rollout budget remained exactly 600;
- 315 and 286 post-latch decisions respectively satisfied all invariants.

All three shadow episodes computed the same counterfactual rule and recorded a
would-be authority of zero, but their applied gate authority remained exactly
one.  This confirms that the shadow-versus-active contrast changed application
of the veto rather than its evidence stream.

## Collision audit

The collision occurred in the shadow arm for seed `730100064` at `38.1 s`:

- the counterfactual veto had latched at zero-based decision 11 but was not
  applied by design;
- proposal-advantage disadvantage fraction was `0.9860` over observed
  comparisons;
- mean dynamics confidence was `0.00142`;
- the episode used 49 safety interventions before collision;
- minimum clearance was `-0.02066 m`;
- during the final approach the predicted collision probability was
  approximately one while proposed, executed, and applied velocity were all
  zero;
- the moving obstacle continued closing and contacted the stationary robot.

This is consistent with the previously identified stopping-feasibility defect:
zero robot velocity is not necessarily the minimum-risk action against a moving
obstacle.  It does not prove that the unrun active-veto cell would have avoided
the collision.  The first-collision rule makes that counterfactual unavailable.

## Interpretation

The fresh-seed comparison directly strengthens the Actor-HSS coupling
diagnosis.  In two paired blocks, a veto based only on completed current-planner
candidate costs changed the old treatment from `0/2` to `2/2` success without
changing the Actor, probability predictor, MPPI objective, safety chain, or
rollout budget.  This makes persistent authority over demonstrably inferior
Actor proposals a credible cause of the Stage 4 completion collapse.

The result does not establish that the frozen Actor is useful in any dynamic
state, does not separate all Actor and HSS effects, and does not qualify the
method for sealed evaluation.  It establishes a narrower engineering result:
the implemented veto can reject this bad Actor quickly, preserves the fixed
budget, and recovered baseline-level completion in the two complete prospective
blocks.

The minimum-cost source comparison is deliberately conservative and should not
be interpreted as a calibrated Actor-competence estimator.  Guided and Gaussian
sources have different unique candidate counts, and guided candidates are
reused across Paper iterations; minimum order statistics therefore do not have
identical sampling distributions.  That limitation is acceptable for this
one-way safety veto, where a false rejection falls back to Gaussian search, but
the signal must not be used by itself to restore Actor authority or claim that
the Actor is globally incompetent.

## Regression and forensic audit

- targeted HSS, Paper planner, metrics, runtime-contract, and Stage 5 protocol
  tests: `65 passed`;
- full repository suite: `1150 passed, 8 failed`;
- one full-suite failure is the expected Stage 4 historical preflight rejecting
  the Stage 5 change to its formerly frozen `rl_driven_mppi.py` hash;
- five failures require old RL checkpoint files absent from this workspace;
- one failure contains a Windows absolute path that cannot replay under WSL;
- one failure is an existing CRLF-versus-LF byte-exact curriculum comparison;
- Stage 5 result bindings: 49 files checked, zero missing and zero hash
  mismatches, including formal-run and regression-test logs.

The historical Stage 4 hash check was not weakened and missing checkpoints were
not fabricated to make the full suite green.

## Next decision

Keep the active veto implementation and preserve this stopped probe unchanged.
Before any broader Actor experiment:

1. add an independent Gaussian-only Paper fallback versus standard-MPPI
   equivalence study, because authority zero is not bitwise RL-off equivalence;
2. repair and geometrically test moving-obstacle stopping feasibility;
3. only then run a new, separately preregistered active-veto safety matrix on
   untouched development seeds;
4. defer residual dual-Planner timing and dynamic Actor adaptation to their own
   stages.

Formal artifacts are in
`research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/`.
