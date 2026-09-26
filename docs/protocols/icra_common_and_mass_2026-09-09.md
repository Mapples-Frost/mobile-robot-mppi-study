# Common-anchor qualification followed by mass/inertia ladder

Wait for all3RL-H training/qualification audits. Primary nominalMPPI H36 and
RL-H+value adaptation, proposed previewH36. Include nominalH28 as a strong
development-motivated fixed-H control; do not choose the winner on test data.
K100 and shared physical sensing/safety remain. Three paired blocks link
RL seeds10801001/2/3 with residualseeds20261201/2/3. This diagonal pairing is
not a fullcrossed trainingseed design; nominal repeated blocks are dependent.

Phasecommon:20developmentseeds10710001–20 x3blocks x4methods=240. Require
each method/block>=19/20success andzero collisions. If fail, stop before test
ladder, retain all qualification outputs, diagnose and amend development only.

Phasemass: only after anchorpasses,10held-outseeds10720001–10 x3blocks x
4methods x4mass/inertiascales1.0/1.15/1.30/1.45=480. Same obstacle geometry,
speedcap.35,controller models/checkpoints frozen. Scale chassis mass and all
chassis inertia axes together, preserving approximate shape; wheelmass,torque,
friction and commanddelay unchanged. No guarantees of monotonic actual mismatch.
No intermediate outcome selection, retraining or dropped cells. Development
and heldout outputs stored separately. Allmasslevels remain physically feasible
only if observed qualification supports it; report stalls/safety interventions.

All methods use SAME h.Environment step implementation for comparisons; preview
loads its existing residual and selector, RL uses nominalpredictor+learnedvalue,
fixednominal has no terminalvalue. Include the RL terminalvalue as part of its
method, with prior separate ablation. No claim exact originalpaper reproduction.
Timer includes planner,selector andexplicit observationfeature cost for every
method (common feature overhead); excludes perception/plant andtraining. Fixed-
step dynamics, no compute delay injected. Retain source hashes,configs,rawtraces.
Report error,compute,success,guardreason/H/modeluse per cell; independently audit
recorded H and raw RMSE. This does not yet establish the mathematical mismatch
mechanism; a common heldoutcommand predictor-error probe must follow separately.
