# Chapter 1 long-window failure review

Status: failed development evidence, outcomes opened, ineligible for
confirmatory use.

## Run

- Map / seed: `chapter1` / `790200013`
- Window: 750 steps (75 simulated seconds)
- Success / collision: false / false
- Termination: `max_steps`
- Goal-distance change: approximately 0.029 m
- Trajectory length: 0.696 m
- Actual minimum clearance: 0.229 m
- Stuck steps: 646

## Cross-layer diagnosis

- Exact static-map feasibility remained valid on every decision
  (`known_static_map_weighted_update_infeasible_fraction = 0`), so the failure
  was not caused by an absent collision-free static candidate.
- The context-free Safety reverse collision from the previous checkpoint did
  not recur.
- The robot nevertheless spent 344 decisions in `front_obstacle_slow` and 146
  in `front_soft_block`.
- 361/750 decisions had predicted maximum dynamic collision probability below
  0.05. Even in those low-risk windows the planner frequently proposed zero or
  negative translation, so probability thresholding alone cannot explain the
  stall.
- Static A* correctly detected stagnation, but replanning failed because the
  map's manually authored spiral reference is feasible while a start-to-final
  grid search over the ring geometry has no direct topology. This does not
  justify adding a new navigation layer.
- The perception pipeline currently sends known-static scan returns into the
  local point-obstacle layer while also supplying the same geometry to the
  exact static-map cost/filter. Static walls are therefore represented twice:
  once exactly and once as a dense set of inflated local circles. This
  double-counting explains why low-risk, exact-statically-feasible decisions
  still prefer near-zero motion.

## Next falsifiable repair

Keep the raw scan for the final Safety guard. For planner-local obstacle
generation and dynamic tracking only, remove returns matched to the injected
known-static geometry. The exact static-map cost/filter remains authoritative
for those returns. This is a perception-interface correction, not a new
planner, state machine, or map-specific rule.

