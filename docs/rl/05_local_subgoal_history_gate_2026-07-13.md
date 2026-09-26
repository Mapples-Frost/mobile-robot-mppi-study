# Local-Subgoal and Short-History RL Gate (L6)

Date: 2026-07-13  
Status: implementation and controlled smoke evaluation complete; strict complex-scene gate **not passed**

## 1. Research question

L2-L5 showed that an RL sampling prior could consistently guide MPPI out of the
U-trap, but a four-dimensional control-knot action did not produce reliable
terminal success. L6 tested two narrowly scoped hypotheses:

1. a two-dimensional local subgoal is easier to learn than a short open-loop
   control waveform;
2. a short observable history can distinguish task phases that look similar in
   a single LaserScan frame.

The predeclared smoke gate was at least `4/5` U-trap successes, zero collisions,
and no material simple-scene regression. Failure means that physics OOD,
dynamic-obstacle, ICODE+RL, and larger-seed claims must not be started from this
checkpoint family.

## 2. Delivered mechanism

### 2.1 Two-dimensional local-subgoal prior

The SAC action is

```text
a_RL = [normalized_distance, normalized_bearing]
```

It is decoded into a body-frame subgoal `(rho, beta)`. A deterministic bounded
turn-then-drive generator converts that subgoal into the full 36-step MPPI mean
sequence. MPPI still samples and optimizes the actual controls; RL neither sends
`/cmd_vel` nor bypasses `scan_guard`.

```text
LaserScan + robot-relative goal + state
                    |
                    v
              SAC: (rho, beta)
                    |
                    v
      bounded local-subgoal sequence generator
                    |
                    v
             MPPI sampled rollouts
                    |
                    v
            safety arbitration -> MuJoCo
```

The default parameterization remains `control_knots`, so older configurations
and checkpoints keep their original interpretation. Local-subgoal dimensions,
gains, bounds, and integration interval are configuration-driven and reject
invalid, NaN, or infinite values.

### 2.2 Short observable history

The history variant stacks three consecutive observation frames in
oldest-to-newest order. Each frame contains only deployable signals: relative
goal features, measured motion, the previous command, safety state, and
sectorized LaserScan. Simulator obstacle truth is not exposed. History is reset
at each episode and the first frame is repeated to avoid undefined padding.

The default remains one frame, preserving existing checkpoint dimensions.

### 2.3 Isolation boundaries

All L6 comparisons keep the following fixed:

- MuJoCo differential-drive plant;
- LaserScan -> local obstacle layer -> MPPI obstacle chain;
- `scan_guard` and final safety arbitration enabled;
- nominal MPPI rollout dynamics;
- ICODE, memory cost, and learned covariance disabled;
- point goal `(3.0, 3.0)`, `0.30 m` success radius;
- `K=100`, 360 maximum control steps;
- no simulator ground-truth obstacle input to RL.

## 3. Training protocol

Two SAC policies were trained for 40,000 MuJoCo/MPPI environment steps:

| Policy | Action dimension | Observation history | Training seed |
|---|---:|---:|---:|
| L6 subgoal | 2 | 1 frame | 20260718 |
| L6 subgoal + history | 2 | 3 frames | 20260719 |

Both use the L5 reverse curriculum: terminal and U-trap-exit starts first,
route expansion second, then 75% original U-trap starts. Validation is always
from the original configured start. Fixed internal validation occurs every
5,000 steps; independent evaluation uses seeds 41-45.

These are smoke-scale learnability runs, not formal paper results.

## 4. Internal validation observations

The feed-forward subgoal policy improved late in training. At 35k it reached
`5/5` in the clean scene and `1/5` in the U-trap; at 40k it reached `3/5` and
`0/5`, respectively. The 40k checkpoint was retained for the independent
comparison because its U-trap final distance was lower and more consistent.

The history policy was strongly non-monotonic. Its 20k checkpoint reached
`2/5` internal U-trap successes but `0/5` clean successes. By 35k it reached
`4/5` clean successes but lost all U-trap successes. The 20k checkpoint was
selected before the independent comparison on the stated complex-scene gate.

This checkpoint dependence is itself evidence: adding raw frame history did
not yield a stable multi-scene representation.

## 5. Independent four-way comparison

Seeds 41-45, identical `K=100` and safety settings:

| Scene | Method | Success | Collision | Final distance (m) | Safety interventions | Planner time (ms) |
|---|---|---:|---:|---:|---:|---:|
| clean | MPPI | 1/5 | 0/5 | 1.840 | 135.6 | 3.72 |
| clean | L5 control knots 35k | **5/5** | 0/5 | **0.297** | **0.0** | 4.23 |
| clean | L6 subgoal 40k | 3/5 | 0/5 | 0.497 | **0.0** | 4.40 |
| clean | L6 history 20k | 0/5 | 0/5 | 1.626 | 102.2 | 4.54 |
| U-trap | MPPI | 0/5 | 0/5 | 3.434 | 264.6 | 3.91 |
| U-trap | L5 control knots 35k | 0/5 | 0/5 | 0.570 | 16.4 | 4.56 |
| U-trap | L6 subgoal 40k | 0/5 | 0/5 | 0.524 | **8.8** | 4.82 |
| U-trap | L6 history 20k | **1/5** | 0/5 | **0.355** | 27.0 | 4.75 |

Planner times are machine-specific smoke measurements. Every evaluated method
had zero observed collisions, but five seeds are too few to estimate a safety
rate for a paper.

## 6. Near-goal fallback diagnostic

The history 20k policy often finished just outside the `0.30 m` success radius.
Three pre-bounded distance gates therefore faded RL into the conventional MPPI
prior near the goal:

| Full fallback / full RL distance (m) | Success | Collision | Final distance (m) |
|---|---:|---:|---:|
| 0.40 / 0.80 | 1/5 | 0/5 | 0.559 |
| 0.60 / 1.00 | 0/5 | 0/5 | 0.718 |
| 0.80 / 1.20 | 0/5 | 0/5 | 0.676 |

The diagnostic did not improve the ungated history result. The failure is not
only a last-centimetres hand-off problem, and additional threshold sweeps are
not justified by these data.

## 7. Gate decision and interpretation

The L6 strict gate is **failed**:

- best independent U-trap result: `1/5`, below `4/5`;
- all collision counts were zero;
- the history policy materially degraded the clean scene;
- the feed-forward subgoal policy was safer and more balanced, but still did
  not reliably enter the success radius.

The defensible conclusion is:

> A two-dimensional local-subgoal action gives MPPI a smooth, useful escape
> direction and reduces safety interventions. Three-frame stacking can improve
> U-trap terminal distance at an intermediate checkpoint, but currently causes
> severe cross-scene interference and does not provide reliable success.

This does **not** support claims of robust RL planning, OOD generalization, or
an ICODE+RL performance gain.

## 8. Bounded next scientific action

Do not add dynamic obstacles, physics OOD, recurrent policies, or joint ICODE
training yet. The next experiment should first separate representation failure
from multi-task optimization failure:

1. verify the local-subgoal decoder's attainable upper bound with a fixed,
   auditable waypoint/subgoal sequence;
2. freeze the L5 and L6 checkpoints as reported baselines;
3. test one balanced multi-scene training change with a validation rule that
   requires both clean and U-trap performance, rather than aggregate score;
4. retain a checkpoint only if it meets the existing `4/5` U-trap criterion
   without clean-scene regression;
5. stop again if that bounded test fails.

This keeps the paper story focused: RL proposes useful exploration directions,
MPPI optimizes controls, and safety remains external to the learned policy.

## 9. Reproducible artifacts

Versioned inputs and code:

- `configs/rl/sac_mppi_subgoal_l6.yaml`
- `configs/rl/sac_mppi_subgoal_history_l6.yaml`
- `src/mobile_robot_mppi/rl/parameterization.py`
- `src/mobile_robot_mppi/rl/observation.py`
- `tests/rl/test_observation_and_prior.py`

Ignored run artifacts:

- feed-forward run: `results/research_platform/rl/subgoal_l6_20260713/`
- history run: `results/research_platform/rl/subgoal_history_l6_20260713/`
- independent comparison:
  `results/research_platform/rl/l6_four_way_ablation_20260713/`
- exact machine-readable table:
  `results/research_platform/rl/l6_four_way_ablation_20260713/summary.csv`
- trajectories:
  `results/research_platform/rl/l6_four_way_ablation_20260713/learnability_gate_trajectories.png`
- training curves: `training_curves.png` inside each training-run directory.

The artifact metadata records git SHA
`4dcd6f26a2bfbef8d4335603be2fe8f54af5f8d7`; the working tree also contained
intentional uncommitted research changes, so the configs and run metadata are
both required for reproduction.
