# Real-robot cause-chain audit (v3)

Every armed cycle now records a compact, causal command chain in
`cycles.jsonl` under `control_cause_chain`, alongside the existing tracker,
planner, and safety records.

The chain is ordered as:

1. `planner_proposed`: the command returned by MPPI after the weighted update.
2. `planner_optimizer_*`: best candidate, selected sequence, cost gap, feasible
   counts, and emergency-candidate index.
3. `probabilistic_obstacle_*`: risk-selected fallback, admission evidence,
   scan support/rejected-jump quality, prefix/hold settings, and the command
   before/after the risk fallback.
4. `safety_reason`/`safety_executed`: the geometric safety arbiter's decision.
5. `path_guard_*`: the start-goal path guard's output and reason.
6. `heartbeat_target`/`pi_applied`: what crossed the PC-to-Pi boundary and what
   the gateway reports as applied.

`dominant_cause` is a reporting label only. It does not change control. The
precedence is `goal_stop`, unconditional hard stop, path guard, probabilistic
emergency candidate, safety override, then nominal MPPI.

For emergency admission, the physical runner keeps ordinary probabilistic risk
inside MPPI's cost. Direct emergency takeover additionally requires the causal
temporal scan to meet its support and rejected-jump thresholds, plus forecast
corroboration when enabled. The critical near-body distance remains an
independent hard safety path.

Emergency lattice prefixes are slew-limited from the previous command. This
prevents a direct forward/reverse sign flip and makes the first command in a
fallback explainable from the preceding cycle.

Temporal hard-stop diagnostics now include `temporal_scan_safety_quality_ok`.
Sparse or jump-heavy temporal flow can still contribute a slowdown, but cannot
alone convert into a translation hard stop in the physical profile. Geometric
`near_body_hard_stop` remains unchanged.
