# Complex Full soft-reference live failure review

Status: failed development evidence, outcomes opened, ineligible for later
confirmatory use.

## Run

- Map: `chapter1`
- Seed: `790200005`
- Arm: actual `B11_full_proposed`
- Result: collision after 72 control steps (7.2 simulated seconds)
- Success: false
- Final goal distance: 7.612128884659623 m
- Trajectory length: 0.5791929183960594 m
- Actual minimum clearance: -0.0005011746817166762 m

## Mechanism evidence

- The closest dynamic-obstacle center was approximately 0.826 m away at
  termination, so this was a static-geometry collision rather than a dynamic
  obstacle impact.
- The final scan guard repeatedly returned `near_body_hard_stop`.
- Safety intervened on 48/72 decisions.
- The residual safety shield fallback fraction was 0.9027777777777778.
- The exact known-static-map weighted update was infeasible on
  0.4027777777777778 of decisions.
- The planner proposed turning/escape controls, but the final arbiter often
  replaced them with zero commanded speed. Because the plant retained negative
  velocity, a zero command did not prevent continued motion into the wall.
- `static_astar_replan_trigger_count` was zero. The immediate failure therefore
  does not support blaming A* or global-reference replanning.

## Decision

Treat this as an execution-contract defect between the statically constrained
planner and the final Safety arbiter. Preserve this run without rerunning or
overwriting it. The next change may make the final arbiter respect a
planner-certified static-feasible escape, and must retain dynamic emergency
protection. It must not introduce a new navigation algorithm.

