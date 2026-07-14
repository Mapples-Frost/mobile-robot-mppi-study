# RL Sampling-Prior Learnability: L2-L5 Evidence

Date: 2026-07-13  
Status: engineering delivery complete for this iteration; complex-scene research gate **not passed**

## 1. Question tested

Can a SAC policy improve MPPI exploration by predicting only the sampling-prior
parameters, while MPPI, LaserScan perception, `scan_guard`, MuJoCo execution,
and the final safety arbitration remain unchanged?

All primary comparisons in this note use:

- MuJoCo differential-drive plant;
- the standard LaserScan -> local obstacle layer -> MPPI chain;
- nominal rollout dynamics and memory disabled, to isolate the RL prior;
- `K=100` unless explicitly stated otherwise;
- point goal `(3.0, 3.0)` and success tolerance `0.30 m`;
- zero access to simulator obstacle truth in the policy observation;
- fixed validation seeds during training and independent seeds 41-45 afterward.

This is a framework/learnability gate, not a paper-result claim.

## 2. Delivered mechanisms

### L2: curriculum, entropy floor, and near-goal fallback

- Piecewise scene curriculum.
- Minimum SAC entropy coefficient.
- Smooth distance gate that can fade the RL prior back to the conventional
  goal warm-start near the target.
- Gate diagnostics and checkpoint-safe configuration.

Result: the 8k model reliably escaped the U-trap and reduced mean final distance
from `3.434 m` to `1.002 m`, but achieved `0/5` independent successes.

### L3: reward correctness

The original discounted potential expression gives a positive dense term even
when distance is unchanged. At `d=4 m`, `gamma=0.99`, and weight 8, standing
still contributes approximately `+0.32` per step before other terms. L3 added a
backward-compatible `distance_delta` mode:

```text
progress = previous_distance - current_distance
```

Standing still therefore earns zero progress, approaching is positive, and
moving away is negative. Small distance and path-length costs were also added.

Result: reward semantics improved, but none of 49 U-trap training episodes or
the fixed validations succeeded. The L3 critic also became numerically unstable
late in training, so this checkpoint family is not a candidate.

### L4: training-only reset-state curriculum

A reproducible initial-state curriculum was implemented. It can sample named,
collision-free states along a feasible route during training, records the state
and phase in every episode log, and saves the schedule in run metadata.
Validation always starts from the original configured state.

Result: terminal-approach starts succeeded `5/6`, but the complete U-trap start
remained `0/28`. This demonstrated that terminal behavior was learnable but had
not propagated through the long task.

### L5: lower-dimensional high-level prior and staged reverse curriculum

- Prior knots reduced from 6 to 2, reducing the SAC action from 12 to 4
  dimensions. MPPI still computes all 36 control steps.
- Rewards and learning rates scaled down to stabilize critic targets.
- A strict reverse curriculum first trained terminal/exit states, then route
  states, and finally used 75% original-start episodes.
- Total training: 50,000 MuJoCo/MPPI environment steps.

This version remained numerically stable and produced the first U-trap success
from the original start in internal validation, but not reliably.

## 3. L5 fixed validation history

| Step | Clean success | Clean distance (m) | U-trap success | U-trap distance (m) |
|---:|---:|---:|---:|---:|
| 5k | 0/5 | 2.456 | 0/5 | 1.521 |
| 10k | 5/5 | 0.298 | 0/5 | 2.029 |
| 15k | 5/5 | 0.289 | 0/5 | 3.387 |
| 20k | 5/5 | 0.296 | 0/5 | 3.596 |
| 25k | 5/5 | 0.289 | 0/5 | 3.677 |
| 30k | 4/5 | 0.355 | 0/5 | 1.210 |
| 35k | 5/5 | 0.298 | 1/5 | 0.624 |
| 40k | 5/5 | 0.295 | 0/5 | 0.597 |
| 45k | 5/5 | 0.298 | 1/5 | 0.760 |
| 50k | 4/5 | 0.483 | 0/5 | 0.796 |

The best internal checkpoint is 35k by aggregate score. Internal `1/5` success
is evidence of learnability, not evidence of reliability.

## 4. Independent U-trap benchmark

Seeds 41-45, `K=100`, 360 control steps, no near-goal fallback:

| Method | Success | Collision | Final distance (m) | Safety interventions | Planner time (ms) |
|---|---:|---:|---:|---:|---:|
| MPPI | 0/5 | 0/5 | 3.434 | 264.6 | 3.77 |
| L2 8k | 0/5 | 0/5 | 1.002 | 16.2 | 4.82 |
| L5 35k | 0/5 | 0/5 | **0.570** | 16.4 | 4.53 |
| L5 40k | 0/5 | 0/5 | 0.615 | 23.2 | 4.87 |
| L5 45k | 0/5 | 0/5 | 0.695 | 39.0 | 4.64 |
| L5 50k | 0/5 | 0/5 | 0.799 | 61.8 | 4.59 |

Interpretation:

- L5 learned a repeatable escape direction and substantially reduced safety
  interventions and final distance.
- It did not meet the `0.30 m` success condition on independent seeds.
- Extending the horizon to 600 steps did not solve the terminal behavior.
- Increasing MPPI sample count from 100 to 200/400 did not solve L2.
- Near-goal gate sweeps did not solve L2 or L5.

## 5. Goal-cost diagnostic

The goal lies near obstacle-influence regions. Terminal-weight sweeps were run
without weakening collision costs or `scan_guard`.

- Terminal weight 200: L5 35k achieved `1/5`.
- Terminal weight 800: L5 35k achieved `2/5`; equal-weight MPPI remained `0/5`.
- Intermediate and larger running-weight combinations were non-monotonic.

Because these weights were selected after looking at seeds 41-45, the `2/5`
result is a diagnostic, not a held-out benchmark and not a valid headline
result. It does show that RL supplies useful global direction that goal-cost
tuning alone does not supply.

## 6. Gate decision

The complex-scene learnability gate remains **failed** under the strict
criterion. The current best defensible statement is:

> The RL sampling prior consistently escapes the U-trap and approaches the
> target with zero observed collisions, but terminal success is not yet stable
> across independent seeds.

No claim of robust RL planning, OOD generalization, or ICRA-level performance is
supported by these runs.

## 7. Next controlled experiment (L6)

Do not continue blind SAC hyperparameter sweeps. The next change should address
task phase explicitly while keeping the scientific story small:

1. Freeze L5 35k as the feed-forward control-prior baseline.
2. Add a low-dimensional local-subgoal prior (direction and distance) so RL
   selects *where MPPI should explore*, not a short control waveform.
3. Add only observable temporal context (short scan/goal history or progress
   memory); do not expose simulator obstacle truth.
4. Compare, one change at a time:
   - 4-D control-knot prior;
   - 2-D local-subgoal prior;
   - local-subgoal prior plus short history.
5. Keep ICODE and memory cost disabled in this ablation.
6. Require at least `4/5` smoke successes before expanding to 10-20 seeds,
   physics OOD, dynamic obstacles, or joint ICODE experiments.

## 8. Reproducible artifacts

- L2 run: `results/research_platform/rl/learnability_gate_l2_20260713/`
- L3 run: `results/research_platform/rl/learnability_gate_l3_20260713/`
- L4 run: `results/research_platform/rl/learnability_gate_l4_20260713/`
- L5 run: `results/research_platform/rl/learnability_gate_l5_20260713/`
- L5 independent benchmark:
  `results/research_platform/rl/learnability_gate_l5_20260713/fixed_seed_u_trap/`
- L5 training figure:
  `results/research_platform/rl/learnability_gate_l5_20260713/training_curves.png`
- L5 representative trajectories:
  `results/research_platform/rl/learnability_gate_l5_20260713/fixed_seed_u_trap/learnability_gate_trajectories.png`

Large run artifacts are intentionally ignored by Git. Configs, source, tests,
and this evidence note are versionable.
