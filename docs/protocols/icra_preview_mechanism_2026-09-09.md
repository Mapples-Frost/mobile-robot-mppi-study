# Mechanism and stronger compute baselines: prospective384

User requests more experiments, stronger baselines and scenarios where preview
can be useful. Freeze384 before outcomes:4 geometries x2 speed limits x3
pretrained model seeds x2 fresh episode seeds x8 methods. Seed10610200 randomizes
context and method order. Episode seeds10610101/2; model seeds20261201/2/3.

Geometries: straight, long straight followed by a bend, alternating bends,
continuous arc. The late bend and alternation test anticipating demand; straight
and arc expose overhead and saturation boundaries. All8 scene/speed cells must
be reported, including failures. These are designed open-space simulations, not
an external navigation benchmark. No obstacle/safety robustness claim.
Speed caps .4/.65m/s; identical450-step caps, plant, sensing and safety per cell.
Use achieved speed as well as configured cap; this is not a commanded-speed test.
Sensors remain ideal simulation observations as in the preceding confirmation.

Methods: nominal MPPI K100; full ICODE K100; full ICODE K32 andK64; reactive
yaw-rate selector; current-path-heading selector; periodic60%; preview.
All H36,dt.1. Lower-K full models directly test whether fewer samples are a
better accuracy/compute tradeoff. They are budget brackets, not claimed exact
wall-clock matches before measuring. No baseline tuning on new outcomes.

Heading ablation uses the same projected reference and hysteresis .25/.12rad,
but only current tangent (zero preview lead). Preview remains .6m/7samples.
Both include heading tracking error. This isolates preview lead from changing
thresholds or merely measuring a different instantaneous signal. Turn remains
yaw-rate on.25/off.15; periodic6/10 cycles. All gating overhead included.

Primary report: every method's success/collision, RMSE, compute, invocation,
achieved speed and command smoothness per geometry/speed/model. Compare preview
to each full-model budget and heading/periodic controls using paired effects.
Explore accuracy-compute tradeoffs; do not reduce them to a winner by ignoring
one metric. Report the prior confirmation separately, never pool development
and validation for a larger nominal sample count. At2 seeds per cell, emphasize
raw paired values and model consistency rather than population significance.

Serial executor, per-episode worker, frozen configs/sources/checkpoints and all
raw artifacts. Expected about2h depending on system load; individual worker
timeout900s. No concurrent heavy analysis or training. Fixed-step physics;
measured compute is not injected as actuation delay. This protocol does not
claim hardware deadline robustness or a new learned dynamics architecture.

Baseline provenance: nominal and ICODE are the common local MPPI backbone and
local paper-structure adaptation, not exact author-code replications. K32/K64,
periodic, turn and heading are explicit local controls. Original-paper/code
verification is documented separately; do not relabel heuristics as literature
baselines. No new fitting or checkpoint selection in this stage.
