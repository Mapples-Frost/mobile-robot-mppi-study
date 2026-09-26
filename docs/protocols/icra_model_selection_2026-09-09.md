# Causal model-fidelity selector — 30-episode development pilot

The shared known-route,0.9m-passages,60s-cap calibration succeeded for both
models on all4cases, with zero collisions. This only establishes a feasible
backbone. It motivates testing selective learned-model use, not a renewed KH
grid. ADP already studies adaptive dynamics fidelity; this prototype is not
claimed novel without subsequent literature positioning and strong evidence.

Five controllers: fixed nominal K16/H40; fixed nominal K64/H40; fixed ICODE
K16/H40; observed-turn-triggered model switch K16/H40; paired-innovation model
switch K16/H40. All share the same known route, geometry, physical settings,
safety guard and objective. Innovation and turn selectors start nominal.

Turn selector: use ICODE when observed absolute yaw rate>=.25rad/s; switch
back below.15rad/s. Innovation selector: compare one-step nominal and ICODE
predictions on the same past state and causally reconstructed average sent
command; the simulator's applied-actuator value is discarded. Reconstruction
retains the explicit .04s calibration assumption. Normalize five state errors
by[.25,.25,.35,.25,.60], wrap heading, take RMS and exponential average(decay.8).
After5transitions, enter ICODE if eR+.0001<.8eN; leave if eR>=.95eN.
Thresholds are fixed before new data, engineering choices, not calibrated
confidence probabilities. Warm-start sequence is carried across model switches.

Disable unused shadow-ensemble configuration for all native fixed models.
Innovation-selection paired prediction and reconstruction overhead are inside
the context/readiness timer. Per-cycle selected model, two error averages and
sample count are logged; actual mode must match the planner dynamics in use.
Fixed methods do not pay for the optional paired predictor. Frozen model
parameters remain unchanged. No actor is trained or used in this pilot.

Two clearance-qualified layouts, alternating_gates_level0 and gate_then_turn_level0;
fast .65cap; fresh seeds10210101/10210102/10210103.2x3x5=30episodes, serial random
method order inside each paired environment. Max60s; Q keeps duration/30 as
already documented. All outcomes retained. Primary outputs are success,
collision, duration, measured compute, model-use fraction and per-cycle error
gain. Same-count nominal16 separates model utility from sample-count effects;
strong nominal64 remains a deployment comparator.

Any positive signal must survive new environment seeds, independent learned
model training seeds, task/domain variations and simple-heuristic comparisons.
Do not tune thresholds on these results and label them final confirmation.
