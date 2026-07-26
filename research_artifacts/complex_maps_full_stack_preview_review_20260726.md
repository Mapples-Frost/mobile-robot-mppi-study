# Complex maps current/full-stack preview review

These runs are opened development evidence. They are not effect estimates and
must never be reused as fresh qualification or confirmatory seeds.

## Chapter 1, original scene configuration

- Config: `mujoco_complex_static_three_dynamic_v2.yaml`
- Seed: `780500001`
- Result: max-steps failure, no collision
- Steps: 900
- Trajectory length: 0.964 m
- Final goal distance: 12.117 m
- Stuck steps: 871

This run confirms severe conservative stagnation in the original scene
configuration. It did not use the new exact static-map cost.

## Chapter 2, original scene configuration

- Config: `mujoco_irregular_spiral_three_dynamic_v1.yaml`
- Seed: `781700001`
- Result: intentionally aborted when the user requested the new mechanism
  stack; only resolved configuration and provenance were produced.

No outcome claim may be made from this incomplete run.

## Chapter 1, migrated full-stack preview

- Config: `mujoco_complex_static_three_dynamic_v2_full_stack_preview.yaml`
- Seed: `780500001`
- Result: collision at step 665
- Trajectory length: 6.884 m
- Final goal distance: 8.114 m
- Stuck steps: 406
- Minimum physical clearance: 0.0398 m
- Minimum dynamic centre distance: 0.5216 m
- Temporal emergency trigger steps: 86
- Vetted emergency-candidate selections: 7

The migrated perception/static-map mechanisms improved movement relative to
the original preview, but did not complete safely.

## Chapter 2, migrated full-stack preview

- Config: `mujoco_irregular_spiral_three_dynamic_v1_full_stack_preview.yaml`
- Seed: `781700001`
- Result: collision at step 115
- Trajectory length: 0.812 m
- Final goal distance: 7.535 m
- Stuck steps: 58
- Minimum physical clearance: -0.0041 m
- Minimum dynamic centre distance: 0.9036 m
- Temporal emergency trigger steps: 94
- Vetted emergency-candidate selections: 9

The large dynamic-centre distance at collision indicates that this terminal
contact was with static geometry, despite the exact static-map cost being
active. This is a static-reference/short-horizon execution failure, not a
missed dynamic-track classification at the terminal step.

## Decision

Do not proceed to ablations or baseline comparisons. The last permitted
complex-scene repair is restricted to:

1. one shared static/dynamic perception contract across all three maps;
2. smooth probabilistic-risk scheduling of the existing A* reference cost;
3. standard static-only A* replanning after sustained deviation or stagnation.

No new learned model, high-level discrete policy, maneuver state machine, or
space-time search algorithm is permitted.
