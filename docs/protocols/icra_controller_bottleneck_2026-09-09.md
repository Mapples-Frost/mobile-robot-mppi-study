# Controller bottleneck intervention — 48 episodes

Motivation: complete compound-failure census finds 243 timeouts among288 runs.
Mean guard intervention fractions over each timeout's last50cycles range .603
to .860 by layout; deadline fractions are zero. This is association, not proof
that safety caused failure. Both proposed and accepted velocities are low.
Increasing rollout budgets is not yet a justified remedy.

Question: are repeated stalls caused by the planner's broad soft obstacle field,
insufficient incorporation of safety feedback into its sampling prior, or both?

Two shared planner factors, fully crossed: obstacle_influence .70/.35 metres;
safety_recovery_prefix_steps 0/5. The latter uses existing tested MPPI support
for a zero-forward prefix after safety feedback, retaining angular control.
The soft influence distance changes a cost, NOT physical robot clearance,
hard collision penalties, sensor data, or safety guard rules. All variants
retain exactly the same physical and safety settings. Any collision is reported.

Two strong fixed controllers: nominal K64/H40 and ICODE K16/H40, chosen from the
completed development grid. Three diagnostic layouts: alternating_gates levels
0/1 and gate_then_turn level0; fast cap .65 only; fresh seeds9890101/9890102.
3 layouts x2seeds x2models x2influences x2recovery prefixes =48 episodes.
Selection of layouts and parameters is explicitly development informed by
previous failures. No claims about RL, novel algorithms, or broad generalization.

Methods are randomized within each paired layout/seed block; serial timing.
Primary outcomes: success/collision/timeouts, terminal progress and guard
fraction; report compute and physical speed too. Every variant and failure
retained. A positive interaction motivates a shared controller correction and
new comparison, not an advantage assigned only to the proposed method.

Do not automatically expand to hundreds of RL episodes. Inspect raw trajectories
and summarize the complete factorial before any follow-up. This isolates a
specific engineering limitation so that later tasks can actually exercise the
candidate method instead of testing a stalled navigation backbone.
