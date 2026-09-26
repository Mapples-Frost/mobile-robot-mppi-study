# Door-budget resolution: development, not final confirmation

Written after complete 480 geometry and 288 compound episodes. All raw audits
passed. Door-offset gains repeat across the two existing joint initializations,
but masked policies also succeed. Compound layouts largely time out and joint
performance varies strongly with initialization. Do not train on compound
layouts or expand them further until the base navigation limitation is resolved.

Immediate question: does a sufficiently broad fixed-budget bank or a single-axis
policy remove the apparent advantage in the one promising door family?

Use door_offset with all three previously defined lateral scales .8/1./1.2,
both speed caps and fresh seeds 9790101/9790102. This is a targeted development
replication chosen after seeing prior results; it is NOT a final test set.

Methods (40): nominal and ICODE with complete Cartesian K={16,64,128,256},
H={8,16,28,40} (32); both final initializations for KH-full, KH-masked, H-only,
and K-only (8). All inherited planner/plant/safety settings are identical to
the prior geometry screen. Existing training is reused without new tuning;
all final checkpoints and seeds retained. No novel method or paper reproduction
is claimed. Forty methods x 3 widths x 2 speeds x 2 environment seeds = 480
episodes, serial within randomized complete paired contexts.

Report all cells; success, collision, duration, Q, per-cycle and per-episode
measured compute. Inspect whether large constant H suffices before claiming
online horizon adaptation is needed. Select the global fixed bank and
speed-specific budgets only as development candidates; do not call an oracle
per-test-episode budget a deployable baseline. Final independent comparison
must freeze selected fixed settings and include both relevant model classes.

This is the last budget-resolution grid under the current deadline plan. If
fixed or single-axis methods explain the gain, simplify the claim or change
the method on that evidence; do not launch another geometry sweep by default.
Independent verification will require a separately frozen plan and new data.
