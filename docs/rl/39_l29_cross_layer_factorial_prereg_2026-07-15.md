# L29 cross-layer ICODE--RL--Gate factorial preregistration

Date frozen: 2026-07-15, before L29 development outcomes are collected.

## Research question

Does the proposed division of labor hold in one common MuJoCo platform?

- ICODE should reduce prediction/control degradation caused by physical-model mismatch.
- The RL sampling prior should improve candidate-sequence quality in locally blocking scenes.
- The LaserScan complexity gate should recover traditional MPPI exactly in clean geometry and activate the learned prior only when local geometry warrants it.

L29 is an attribution experiment. It does not assume that the three learned components interact synergistically; that interaction is an estimand.

## Design

This is a blocked, repeated-measures mixed factorial. Every episode seed within a model block receives every method under every scene and physics domain.

| Factor | Levels |
|---|---|
| Residual prediction | nominal, ICODE |
| Policy sampling mode | traditional, target-LCB RL, complexity-gated target-LCB RL |
| Obstacle setting | clean, static single blockage, prescribed dynamic crossing |
| Physics domain | nominal seen, combined unseen mismatch |
| Model block | three paired RL/ICODE training seeds |

The six method cells are therefore:

1. `traditional_nominal`
2. `traditional_icode`
3. `lcb_nominal`
4. `lcb_icode`
5. `gated_lcb_nominal`
6. `gated_lcb_icode`

Memory is disabled. MPPI uses `K=100`. All methods retain the same MuJoCo plant, LaserScan, local obstacle layer, scan guard and safety arbitration. The planner never receives the dynamic obstacle trajectory or MuJoCo obstacle truth.

## Experimental unit and blocking

The independent learned-model replication unit is the model-training block, not an individual control step. Episode seeds are repeated measures nested inside each of three model blocks. Control steps are technical measurements used for exact fallback audits and must not be analyzed as independent experimental replicates.

Run order is shuffled reproducibly within each model block using schedule seed `20260791 + block index`. All six methods share episode seeds, scene definitions and physics domains.

## Dynamic obstacle contract

The crossing obstacle is a MuJoCo mocap collision body following a deterministic linear ping-pong path at approximately 0.21 m/s. The same current geometry is used by MuJoCo contact, clearance measurement, ray-cast LaserScan and visualization. Only LaserScan-derived obstacles reach MPPI and the gate.

## Development and confirmation separation

- Development seeds: `20282101--20282105`.
- Sealed confirmation seeds: `20282106--20282120`.
- L25--L28 protected seeds remain forbidden.
- Confirmation is opened only if all integrity/safety checks and at least three of four fixed efficacy checks pass.

The initial development matrix contains `3 model blocks x 5 episode seeds x 3 scenes x 2 physics domains x 6 methods = 540 episodes`.

## Primary estimands

1. ICODE effect under unseen physics: paired final-goal-distance difference between ICODE and nominal prediction, holding policy, scene, block and episode fixed.
2. Gated-RL effect in static blockage: paired success-count and final-distance difference versus traditional MPPI, holding residual mode fixed.
3. Gated-RL effect in dynamic crossing: the same paired contrast, reported separately rather than pooled with static scenes.
4. Gate fallback: stepwise equality of gated and traditional methods in clean geometry for each residual mode.
5. Combined ranking: number of blocking scene x physics strata in which `gated_lcb_icode` is best or tied by success, with final distance as the secondary outcome.

## Fixed development gates

Integrity and safety, all required:

- no missing or duplicate episode keys;
- zero protected/confirmation seeds used;
- zero clean-scene fallback step mismatches;
- no collision regression relative to the residual-matched traditional method.

Efficacy, at least three of four required:

- unseen-physics ICODE mean paired final-distance improvement at least 0.05 m;
- static-blocking gated-RL net success gain at least 6 episodes across residual modes;
- dynamic-crossing gated-RL net success gain non-negative across residual modes;
- combined method best or tied in at least three of four blocking scene x physics strata.

These thresholds are development eligibility criteria, not final paper hypothesis tests.

## Interpretation boundaries

- Passing development permits sealed confirmation; it does not establish an ICRA-ready claim.
- Failing one module contrast does not invalidate unrelated contrasts.
- Dynamic-obstacle results concern reactive LaserScan-based avoidance; the planner has no explicit obstacle-velocity predictor in L29.
- The ICODE checkpoint is a control-affine residual model trained on the existing seen physics-domain dataset. This does not reproduce every theoretical guarantee of the original ICODE work.
- No theorem, stability, contraction or convergence claim is made.
