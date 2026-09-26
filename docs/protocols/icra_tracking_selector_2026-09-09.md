# Continuous tracking model-selector pilot — 30 new episodes

Point-goal model-selection30 finished: all methods succeeded6/6; innovation
reduced compute versus alwaysICODE but not versus nominal with essentially
equal completion time. This does not establish model-selection utility.
Archived L59 shows predictive-model utility in continuous tracking RMSE. That
old evidence motivates task choice; its scores are not reused as new results.

Use the resolved L58/L56 fixed high-dynamic path-tracking setup, including
its inherited K/H, physical plant, known .04s command delay and common costs.
Three paths: sweep, chicane, reverse-S. Fresh seeds10310101/10310102; five
methods nominal, alwaysICODE, observed-turn selector, paired-innovation selector,
and periodic10-cycle alternating selector.3x2x5=30, one fresh serial worker per
episode. Explicitly assign the same episode seed to planner and plant.
Use existing model20261201 for this pilot; any confirmation needs all3model
training seeds and additional path/episode conditions. No actor training.

Turn/innovation thresholds unchanged from the previous pilot. Reconstruct
the .04s fixed motor delay from the last two sent commands and integrate the
two command segments for each paired predictor; no applied actuator truth.
Only innovation mode pays for paired prediction. All gating and reference
work is inside an outer planner timer overriding compute_ms for EVERY method.
Log sensed states, sent-command history, selected model, errors and compute.

ExperimentRunner uses fixed simulated control intervals. Measured computer
runtime is an outcome, not inserted as physical command-readiness delay.
Do not mix this timing scope with earlier readiness-delay experiments. Physical
actuator delay is still present and shared. The point-goal Q/audit is inapplicable.

Primary question: retain learned-model cross-track accuracy at lower measured
compute, while improving over nominal and simple periodic/turn selectors on
the same accuracy-compute comparison. Report all episodes, success/collision,
cross-track RMSE and maxima, completion, control smoothness if available,
compute distribution and actual model-use fraction. Audit raw trajectory RMSE,
step counts, compute sums and selector decisions before interpreting effects.
No best-path/seed selection, significance or publication guarantee from this pilot.
