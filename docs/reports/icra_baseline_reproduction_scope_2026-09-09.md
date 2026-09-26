# Baseline implementation and reproduction scope

## Source-to-equation inspection (September9,16:05)

The shared local controller at src/mobile_robot_mppi/planning/mppi.py computes
clipped effective perturbations (line1094), adds its importance-sampling cost
(1101), normalizes exponential negative costs (1119–1125), and updates the prior
with weighted perturbations (1140–1141). These implement the general sampling,
weighting and update structure in original-paper equations5–9, with additional
local clipping, costs and proposal conventions. They are not a proof of identical
sampling distributions or identical original control-cost corrections.

Original-paper text following equation9 describes Savitzky–Golay smoothing.
The inspected local update instead clips the sequence and applies terminal
speed/heading/alignment rules before action clipping (1142 onward); no
Savitzky–Golay step was found in this controller. This difference must remain
visible in baseline naming and any numerical comparison to the original paper.
Do not silently add smoothing to an active frozen experiment. A later isolated
adaptation can compare smoothing under the same control/safety contract, after
the window/order are verified or explicitly documented as local choices.

ResidualNetwork in src/mobile_robot_mppi/learning/models.py:48 constructs
separate drift and gain networks (62–68); components():104 onward applies
normalized control-affine composition and converts to physical units. Its
output mask restricts learned channels. The local L57 protocol fixes64-64
Softplus and H36 training; these are adaptation details, not replication of
the original vehicle data or convergence guarantees.

Source inspection is complete for these links; independent numerical tests of
paper-update equivalence and a verified original smoothing configuration remain
outstanding. No original-code baseline has been newly trained or evaluated in
this session. The active384 strengthens local controls, not author-code fidelity.

Accessed September9,2026. Queries: "ICODE-MPPI" github; "Shugen Song" "ICODE"
code; ICODE residual MPPI vehicle. Primary manuscript inspected:
[ICODE-MPPI arXiv2605.03260v1](https://arxiv.org/html/2605.03260v1).

The original uses bicycle states x,y,heading,speed,steering and acceleration/
steering-rate inputs. It describes control-affine residual learning, iterative
data collection and smoothed MPPI updates. Our present baseline transfers the
residual form to a dynamic unicycle, with residual correction only in speed and
yaw-rate derivatives. It uses our MuJoCo data, three saved64-64 Softplus models,
our task/cost/safety backbone and budgets. This is a structure-level adaptation,
not numerical replication of the paper's tracking scores or original plant.
The local training choices are recorded in docs/rl/
82_l56_data_results_and_l57_training_prereg_2026-07-16.md.

[Author ICODE repository](https://github.com/EEE-ai59/ICODE) is available for
the underlying Input Concomitant Neural ODE work. Its displayed examples cover
single-link robot, converters, rigid body and other dynamics. The inspected
README describes per-example training and plotting. It is not identified as
the ICODE-MPPI vehicle control implementation. Targeted search did not find a
verified author MPPI repository; this is not proof that none exists. Do not
claim downloading this repository reproduces the vehicle baseline.

Current comparison labels:
- Nominal MPPI: existing shared local controller, explicitly report prior,
  safety, control clipping and cost modifications.
- ICODE-MPPI adaptation K100/K64/K32: same local learned dynamics, differing
  Monte Carlo budgets; K64/K32 are strong budget controls, not separate papers.
- Turn/heading/periodic: local selectors for mechanism and compute allocation.
- Preview: proposed local heuristic, residual itself is not a new contribution.

Outstanding for a publication-quality reproduction package: source-to-equation
mapping for local MPPI weighting/smoothing and residual training, author-code
availability provenance, original-vehicle reproduction feasibility, and at least
one independently implemented relevant comparison if justified by the mechanism
results. Do not run extra training during the timed384 stage. Preserve both
positive and negative outcomes; literature labels cannot substitute for evidence.
