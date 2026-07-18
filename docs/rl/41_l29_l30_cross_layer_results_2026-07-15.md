# L29/L30 cross-layer development results

Date: 2026-07-15

Status: development evidence only.  No sealed confirmation seed was opened.

## Question

The experiment tests the intended division of labor in one common MuJoCo
platform:

- ICODE corrects prediction under physical-model mismatch;
- RL biases MPPI toward more useful candidate control sequences;
- the gate suppresses RL in easy geometry and activates it when local risk
  warrants learned guidance.

The planner receives obstacles only through ray-cast LaserScan and the local
obstacle layer.  The prescribed dynamic obstacle is a collidable MuJoCo mocap
body, but neither its pose nor its motion script is exposed to MPPI, RL or the
gate.  Memory is disabled, and scan guard plus safety arbitration remain
active for every method.

## Design and provenance

L29 contains 540 episodes:

`3 model-training blocks x 5 matched episode seeds x 3 scenes x 2 physics
domains x 6 methods`.

L30 adds 180 mechanism-remediation episodes:

`3 model-training blocks x 5 matched episode seeds x 3 scenes x 2 physics
domains x 2 temporal-gate methods`.

The three model blocks use independently trained RL and ICODE checkpoints.
Episode seeds are repeated measures within model block; controller time steps
are technical observations and are not treated as independent replicates.

Primary artifacts:

- `results/research_platform/rl/l29_cross_layer_development_multiblock_20260715_v1/`
- `results/research_platform/rl/l30_temporal_gate_development_multiblock_20260715_v1/`
- `configs/rl/cross_layer_factorial_l29.yaml`
- `configs/rl/cross_layer_temporal_gate_l30.yaml`

## L29 result: spatial gate failure in lateral dynamic crossing

L29 passed every integrity and safety audit:

- 540/540 expected episode keys, with zero missing or duplicate keys;
- no protected or sealed seed used;
- 7,806 clean-scene matched control steps were exactly equal between the
  spatial-gated and traditional methods;
- no learned-policy collision regression relative to the matched traditional
  method.

It nevertheless failed the fixed development efficacy gate.  The spatial gate
was best or tied only in the two static strata, and the pooled unseen-physics
ICODE final-distance improvement was `0.012 m`, with a block/episode bootstrap
95% interval of `[-0.188, 0.218] m`.

The mechanism audit showed why: a single obstacle crossing laterally is a
sparse, one-sided LaserScan feature.  The original score intentionally ignored
a single side wall, so the gate weight remained approximately `0.05` on
average and rose too late.  Always-on target-LCB RL was substantially safer in
this dynamic stratum.

## L30 method: scan-only temporal closing risk

L30 retains the original spatial score and adds a temporal activation derived
from the positive closing rate of the front, left and right LaserScan sector
minima.  It uses consecutive scans only; it has no simulator-truth or obstacle
identity input.  The activation uses fixed physical thresholds frozen before
L30 outcomes:

- soft closing rate `0.05 m/s`;
- full activation rate `0.20 m/s`;
- relevant clearance `1.50 m`;
- hold time `0.60 s`.

The spatial and temporal activations are combined by `max`.  This raises the
RL weight early when an obstacle is closing from one side while preserving an
exact zero-complexity fast path.

## Main development outcomes

Across the four blocking strata (static/dynamic x seen/unseen), each method has
60 matched episodes.

| Method | Success | Collision | Mean final distance |
|---|---:|---:|---:|
| Traditional nominal | 0/60 | 30/60 | 2.628 m |
| Traditional + ICODE | 2/60 | 30/60 | 2.536 m |
| Always-RL nominal | 23/60 | 1/60 | 1.631 m |
| Always-RL + ICODE | 8/60 | 0/60 | 2.033 m |
| Spatial gate nominal | 24/60 | 24/60 | 1.605 m |
| Spatial gate + ICODE | 31/60 | 24/60 | 1.330 m |
| Temporal gate nominal | 58/60 | 2/60 | 0.357 m |
| **Temporal gate + ICODE** | **60/60** | **0/60** | **0.289 m** |

The proposed temporal-gate + ICODE combination achieved:

- static seen: 15/15 success, 0 collision;
- static unseen: 15/15 success, 0 collision;
- dynamic seen: 15/15 success, 0 collision;
- dynamic unseen: 15/15 success, 0 collision;
- clean seen/unseen: 30/30 success, 0 collision.

In clean geometry, all 7,806 temporal-gate/traditional paired control steps
matched exactly, including executed controls, goal distance, safety decision
and gate alpha.  This supports the intended conditional-computation behavior:
the learned actor is not merely given a small weight; it is skipped and the
controller reduces exactly to traditional MPPI.

Within the temporal gate under unseen physics, ICODE improved mean final
distance by `0.038 m` and produced one additional success.  The two-stage
block/episode bootstrap 95% interval is `[-0.002, 0.155] m`; therefore this
development result is directionally positive but does not establish a
standalone population-level ICODE effect.

![L29/L30 development interaction](../../results/research_platform/rl/l30_temporal_gate_development_multiblock_20260715_v1/fig_l29_l30_cross_layer_development.png)

The vector version is saved as
`fig_l29_l30_cross_layer_development.pdf` in the same result directory.

## Fixed gate decision

L30 is formally recorded as `development_fail`, despite the strong primary
combination.  The reason is narrow and explicit: temporal-gate nominal had one
collision in the seen dynamic stratum, whereas its residual-matched always-RL
comparator had zero.  The preregistered rule required no collision increase for
both residual levels, so it cannot be relaxed after observing the result.

All other fixed L30 conditions passed:

- complete unique 180-episode remediation matrix;
- protected and sealed seeds untouched;
- exact clean fallback;
- dynamic success not worse than always-RL;
- static success improved by 17 episodes over the spatial gate;
- proposed temporal-gate + ICODE method best in all four blocking strata.

## What the evidence supports

The development evidence strongly supports three engineering claims:

1. A static spatial-complexity gate is insufficient for a sparse lateral
   crossing obstacle.
2. A real-robot-compatible scan-temporal risk signal repairs that failure in
   this prescribed dynamic benchmark without sacrificing clean fallback.
3. The complete temporal-gate + RL + ICODE combination is materially more
   robust than any individual baseline in these development cells.

It does **not** yet support an ICRA-level final claim because:

- all L30 results reuse development seeds selected for mechanism repair;
- only one prescribed dynamic obstacle trajectory is evaluated;
- the temporal rule is a reactive closing-risk heuristic, not a tracked
  obstacle-velocity predictor;
- no sealed confirmation result or real-robot result exists yet;
- the ICODE implementation reproduces the publicly described control-affine
  residual structure, not every theoretical guarantee of the original ICODE
  work;
- no stability, contraction, convergence or safety theorem is claimed.

## Next decision

Do not tune the four temporal thresholds again on these seeds and do not open
the L29/L30 sealed seeds.  The next protocol should be frozen as a new study:

1. declare temporal-gate + ICODE as the primary deployment candidate and keep
   temporal-gate nominal as a diagnostic ablation, not a safety-qualified
   controller;
2. add new development scenes varying crossing direction, phase, speed and
   obstacle size, still using LaserScan-only planner inputs;
3. require zero collision for the primary method and non-inferior success
   against always-RL plus the spatial gate;
4. only after that screen passes, allocate a new sealed multi-seed confirmation
   set and then proceed to real-robot shadow-mode testing.

