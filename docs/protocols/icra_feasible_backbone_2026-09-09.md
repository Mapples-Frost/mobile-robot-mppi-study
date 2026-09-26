# Feasibility calibration after finding a guard/geometry boundary conflict

Static centerline probes (clearance_probe.json) showed that0.7m openings have
half-width0.35m exactly equal to the configured near-body hard-stop radius.
4/26 centerline poses for that family had positive physical clearance but
forced zero translation. This is a task/safety-envelope conflict; not evidence
that any specific planner is worse. Keep all previous narrow results visible.

Known-route24 showed both models succeed3/6. The0.9m alternating layout's
remaining timeouts had passed the last gate and were approaching the final
goal at30s (nominal x4.35; ICODE x3.71 in the inspected seed), unlike the
hard-stopped narrow layout. This motivates a small backbone calibration before
training any further adaptive method.

Use only the two0.9m layouts (alternating_gates_level0, gate_then_turn_level0),
known-map route for BOTH fixed models, nominal64/40 and ICODE16/40. Same .65cap,
same safety/physics/costs/route logic; set maximum mission time60s for every
controller. Two fresh seeds10110101/10110102:2layouts x2models x2seeds=8episodes.
The task is now explicitly known-map navigation in clearance-qualified passages.
This calibration does not retroactively redefine earlier tests or claim broad
generalization. No learned policy is involved.

Log complete duration, success/collision, measured compute and trajectories.
For arithmetic comparability the diagnostic Q retains its existing duration/30
normalization even though the hard timeout is60s; it is not a fraction of this
new timeout. Success and duration are primary. Raw audit can therefore use the
existing fixed normalization without silently changing old metrics.

If successful, this only establishes an executable common backbone. It is not
a new algorithm contribution; all subsequent learned and fixed comparators
must receive the identical route, safety envelope and time limit. Final test
episodes require a separate frozen definition. Do not expand parameter sweeps.
