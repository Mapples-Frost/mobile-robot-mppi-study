# Known-route feasibility diagnostic — 24 episodes

Temporal-sampling64 completed and raw-audited: all eight variants fail both
alternating-gates levels on both seeds. Coarse perturbations sometimes help
gate_then_turn, but guarded mixing adds no consistent value. Do not scale it
or call it an established contribution.

Question: can the current plant/planner/safety stack traverse these passages
when given a valid approximate route, or does a lower-level issue prevent it?

Compare direct final goal versus a known-map waypoint reference, for nominal
K64/H40 and ICODE K16/H40. For each wall pair, use the known gate center and
waypoints .4m before/after the wall plane; then final(4.5,0). Advance a waypoint
within .22m. This route uses the configured static obstacle map and is an
explicitly privileged upper-bound diagnostic relative to local sensing. It is
not a new algorithm and cannot be presented as a fair advantage over direct
navigation. A later known-map comparison must give the same route to all methods.

Three existing diagnostic layouts: alternating_gates levels0/1 and
gate_then_turn level0, fast .65cap, fresh seeds10010101/10010102.3x2x2x2=24.
Shared iid samples, physical model, cost coefficients, guard, robot dimensions,
maximum30s, initial state and final success criterion. Only the reference used
inside planning is changed. Executed task Q still uses the original final goal;
route index/waypoints are explicitly logged. Timing includes reference selection.

All paired methods run serially in random order; raw traces/config/source kept.
Success/collision/timeouts and progress separate, no removal of failed cells.
If known-route fails too, inspect trajectory/clearance/guard consistency before
any RL or new budget experiment. If it succeeds, establish a shared feasible
navigation backbone and reassess where dynamics or resource adaptation can help.
