# Scripted Local-Subgoal Upper-Bound Gate (L7)

Date: 2026-07-13  
Status: diagnostic implementation complete; decoder/planning-interface Gate **not passed**; balanced SAC training intentionally not started

## 1. Decision being tested

L6 showed that a learned two-dimensional local-subgoal prior could guide MPPI
out of the U-trap, but did not reach the terminal tolerance reliably. Before
changing replay sampling or training longer, L7 asked a more basic causal
question:

> If a privileged offline route supplies a reasonable local subgoal at every
> control step, can the exact L6 distance/bearing decoder and unchanged MPPI
> stack complete the task?

If the answer were yes, the remaining failure could reasonably be assigned to
SAC optimization. If the answer were no, balanced replay would be unable to
repair the downstream representation/control bottleneck by itself.

The conditional plan was fixed before running the experiment:

1. run a scripted-subgoal upper-bound diagnostic;
2. start scene-balanced SAC only if both clean and U-trap tasks are feasible;
3. otherwise stop learning changes and locate the failed interface.

## 2. Diagnostic boundary

The offline route is privileged and **diagnostic-only**. It is not a proposed
deployment method and is not given to the learned policy. During each run:

```text
offline collision-free polyline
             |
             v
  scripted lookahead target
             |
             v
 normalized (distance, bearing)
             |
             v
 exact L6 local-subgoal decoder
             |
             v
 MPPI sampling mean -> MPPI optimization
             |
             v
 scan_guard -> safety arbitration -> MuJoCo
```

The MPPI task remains the original point goal `(3.0, 3.0)`. The offline route
does not replace the MPPI cost reference, and planner obstacles still come only
from LaserScan -> local obstacle layer. ICODE and memory remain disabled.

The scripted policy projects noisy odometry onto the route, enforces monotonic
bounded route progress, selects a fixed-distance lookahead point, and converts
that point into the same normalized two-value action consumed by SAC.

## 3. Geometry audit

The U-trap goal has only `0.1209 m` geometric clearance after subtracting the
robot collision radius. Consequently, an extra `0.15 m` route margin marks the
goal occupied by construction. The audit found valid routes at margins from
`0.00` through `0.10 m`; L7 fixed the largest feasible tested margin,
`0.10 m`, at `0.02 m` grid resolution. The point goal and obstacle geometry
were not moved.

## 4. Development-seed lookahead calibration

Seed `0`, `K=100`, 360 maximum steps, zero configured initial-state noise:

| Lookahead (m) | Clean success | Clean final distance (m) | U-trap success | U-trap final distance (m) | U-trap safety interventions |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0/1 | 0.441 | 0/1 | 3.168 | 258 |
| 0.35 | 1/1 | 0.299 | 0/1 | 2.768 | 232 |
| 0.45 | 1/1 | 0.300 | 0/1 | **0.934** | **40** |
| 0.70 | 1/1 | **0.292** | 0/1 | 1.038 | 71 |
| 0.95 | 1/1 | 0.287 | 0/1 | 2.546 | 207 |

All ten runs had zero collision. No lookahead passed the U-trap even on the
development seed, so no parameter was selected for seeds 41-45. This avoids
spending held-out seeds on a candidate that had already failed the prerequisite
and avoids selecting a final setting after seeing held-out results.

The trajectories show a structural trade-off:

- short lookahead stays local but repeatedly activates the safety layer before
  clearing the early board;
- medium lookahead escapes the obstacle field but overshoots the high-curvature
  terminal turn and circles outside the success radius;
- long lookahead cuts the route too aggressively and accumulates many safety
  interventions.

## 5. Sampling-budget diagnostic

The best development lookahead (`0.45 m`) was rerun in the U-trap without any
training change:

| Samples K | Success | Final distance (m) | Collision |
|---:|---:|---:|---:|
| 100 | 0/1 | 0.934 | 0 |
| 200 | 0/1 | 0.940 | 0 |
| 400 | 0/1 | 0.755 | 0 |

More samples slightly improved the `K=400` terminal distance but did not repair
the task. The failure therefore cannot be attributed only to the smoke-scale
sample count.

## 6. Privileged-reference control

A second causal control changed the MPPI reference itself to the offline
polyline. This is more privileged than the scripted sampling prior and is not a
candidate method. At `K=100`, seeds 41-45:

| Scene | Success | Collision | Final distance (m) | Safety interventions |
|---|---:|---:|---:|---:|
| clean | 4/5 | 0/5 | 0.372 | 1.6 |
| U-trap | 0/5 | 0/5 | 2.567 | 192.6 |

Thus the low-sample point-goal stack is not a strong path follower in this
U-trap, while L6 learned priors nevertheless reached `0.355-0.524 m`. This
supports the claim that learned exploration direction is useful, but it does
not establish a reliable controller.

## 7. Gate decision

The scripted-subgoal upper-bound Gate is **failed**. Scene-balanced replay and
constrained checkpoint selection were therefore not implemented or trained in
this iteration.

This is an intentional stop, not an incomplete conditional branch:

- no scripted local-subgoal candidate solved the development U-trap;
- increasing `K` to 400 did not solve it;
- the privileged polyline-reference control also failed at `K=100`;
- changing SAC sampling cannot, by itself, prove that the same downstream
  decoder will become reliable.

The defensible L7 conclusion is:

> The current two-value action contains useful global direction, but its
> kinematic turn-then-drive decoding is not a validated attainable upper bound
> for the dynamic MuJoCo U-trap task. Training interference is therefore not
> yet the only identified bottleneck.

## 8. Most likely interface limitations

The experiment identifies hypotheses, not a unique root cause:

1. the decoder starts its internal rollout from zero dynamic state and does not
   condition sequence generation on measured `v` and `omega`;
2. the generated mean does not model actuator lag or command delay, which is
   most visible at the terminal high-curvature turn;
3. one static subgoal expresses a point but not desired route curvature;
4. a sampling prior biases candidates but does not add a local-subgoal tracking
   objective, so the global point-goal cost can pull the optimized sequence
   away from the privileged route.

These mechanisms must be separated one at a time. Adding all of them together
would make the ablation uninterpretable.

## 9. Next bounded change

The next admissible experiment is a dynamics-conditioned local-subgoal decoder:

- retain the same two-dimensional `(distance, bearing)` RL action;
- initialize the decoder rollout with measured `v` and `omega`;
- propagate nominal velocity/yaw dynamics with configured time constants;
- keep actuator limits, MPPI cost, LaserScan, and safety arbitration unchanged;
- rerun the scripted upper-bound Gate before any SAC retraining.

Only if this decoder reaches at least `4/5` U-trap successes with zero collision
and no clean regression should scene-balanced replay be implemented. Dynamic
obstacles, physics OOD, uncertainty gating, and ICODE+RL remain downstream.

## 10. Reproducible artifacts

Versioned code:

- `src/mobile_robot_mppi/rl/scripted_subgoal.py`
- `experiments/rl/run_scripted_subgoal_upper_bound.py`
- `tests/rl/test_scripted_subgoal.py`

Ignored run artifacts:

- consolidated development calibration:
  `results/research_platform/rl/l7_scripted_subgoal_gate_dev_final_20260713/`
- `K=200` diagnostic:
  `results/research_platform/rl/l7_scripted_subgoal_k200_final_20260713/`
- `K=400` diagnostic:
  `results/research_platform/rl/l7_scripted_subgoal_k400_final_20260713/`
- independent privileged-reference control:
  `results/research_platform/rl/l7_polyline_reference_control_complete_20260713/`

Each completed scripted run stores a resolved configuration, trajectory CSV,
per-run summary, exact route CSV, aggregate CSV/JSON, git SHA, and figure.
