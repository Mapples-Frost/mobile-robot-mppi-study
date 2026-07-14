# Safety-envelope and route-reference gate

## Outcome

The narrow-corridor collision from the waypoint diagnostic was reproduced at
step 276 and converted into a no-collision regression. The frozen clean MuJoCo
baseline still succeeds on 5/5 seeds at its original `K=600` sampling budget.

Terminal heading gates and continuous polyline references were also evaluated.
They remain optional ablation components because no single hand-tuned reference
variant dominated across `lab_complex`, `narrow_corridor`, and
`u_trap_long_board`. This negative result is retained as evidence for a learned
sampling prior rather than hidden by scene-specific tuning.

## Collision root cause

The research robot collision radius is 0.25 m, while the former scan settings
used 0.18 m for side stop and 0.20 m for the all-direction near-body stop. A
side obstacle could therefore enter the physical collision envelope before
either stop activated. In the reproduced trajectory, the controller changed
from `hard_stop` to `front_soft_block`, then to `front_clear` while the vehicle
still had forward velocity. Clearance progressed from 0.0186 m to -0.0014 m.

The revised MuJoCo research settings are:

| Quantity | Former | Revised |
|---|---:|---:|
| hard/front stop distance | 0.28 m | 0.35 m |
| side stop distance | 0.18 m | 0.35 m |
| all-direction near-body stop | 0.20 m | 0.35 m |
| soft-block translation | scale 1.0 | capped at 0.04 m/s |

Hard, side, and near-body stops force exactly `v=0` while preserving omega.
The 0.04 m/s soft-block cap mirrors the bounded-creep concept already present
in the protected hardware bridge. No PyTorch dependency or research-runtime
module was introduced into the ROS/Python-2 scripts.

## Safety regression evidence

For `narrow_corridor / direct goal / nominal / seed 0 / K=400`:

| Version | Collision | Termination | Final distance |
|---|---:|---|---:|
| former envelope | yes | collision at step 276 | 3.609 m |
| revised envelope | no | max steps | 3.849 m |

Failure to reach the goal is expected in this causal case; the assertion is
that a poor local plan cannot penetrate the body envelope.

For the unchanged clean single-obstacle baseline at `K=600`, seeds 0–4:

- success: 5/5;
- collisions: 0/5;
- mean final distance: 0.294 m;
- mean planner time: 7.60 ms;
- deadline misses at 100 ms: 0;
- mean safety interventions: 21.8.

An exploratory `K=400` run reached only 4/5. It is not a regression comparison
because it uses a smaller sampling budget than the frozen baseline.

## Reference variants evaluated

### Discrete waypoint reference

Intermediate and final tolerances are now separate. Every executed step logs
the active `reference_id`, which made missed-waypoint and terminal-handoff
failure modes observable.

### Heading-gated warm start

`GoalWarmStartPrior` optionally suppresses translation and retains yaw control
when heading error exceeds a configured gate. The recovery is smooth rather
than binary. It can be restricted using reference phases:

- `tracking`;
- `terminal_approach`;
- `terminal`.

A global 0.90 rad gate made both nominal and ICODE reach the U-trap endpoint,
but caused ICODE failures in lab/narrow. A 1.10 rad gate still regressed
lab/narrow. The strong baseline therefore leaves this gate disabled.

### Continuous polyline reference

`PolylineReference` projects the vehicle onto a route, enforces monotonic route
progress, and supplies a metric lookahead target. It prevents chasing a waypoint
that has already been passed and only exposes final tolerance at the route end.

At seed 0 and `K=400`, it produced:

| Scene | Nominal | ICODE |
|---|---:|---:|
| `lab_complex` | fail, 3.587 m | fail, 3.675 m |
| `narrow_corridor` | fail, 1.151 m | fail, 0.481 m |
| `u_trap_long_board` | success, 0.298 m | success, 0.272 m |

This is an ablation result, not an accepted replacement for the default task
reference.

## Research decision

Accepted into the default research runtime:

- body-consistent scan thresholds;
- bounded soft-block creep;
- retained angular control during translation stops;
- reference phase and `reference_id` diagnostics;
- no change to the protected ROS bridge behavior;
- no change to collision radius or success tolerance.

Available but disabled by default:

- translation heading gate;
- continuous polyline reference.

The cross-scene failure of fixed route heuristics supports the next controlled
question: can an RL sampling prior propose useful control-sequence families in
topologically difficult states while MPPI, ICODE, and scan safety retain their
separate roles? Formal evaluation must compare this against the fixed waypoint
and polyline ablations across multiple seeds.

Artifacts are under
`results/research_platform/safety_reference_gate_20260712/` and remain ignored
by Git as generated experiment output.
