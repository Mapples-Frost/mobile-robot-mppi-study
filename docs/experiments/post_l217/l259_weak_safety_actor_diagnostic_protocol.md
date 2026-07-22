# L259 Weak-Safety Actor Source Diagnostic Protocol

Date: 2026-07-22  
Status: preregistered MuJoCo development diagnostic; no outcome read

## Question

The completed L258 Gate showed two coupled failures: repeated scan-guard
deadlock and almost no Actor-guided elite selection.  This diagnostic asks:

1. does removing the omnidirectional and lateral hard-stop trap restore motion
   near the first S-Chicane obstacle; and
2. when Actor candidates are actually offered, are they rejected because their
   frozen MPPI costs or boundary feasibility are worse than Gaussian/traditional
   candidates, or only because reliability later suppresses their allocation?

## Frozen scope

- MuJoCo development only; never real hardware.
- Scene: frozen L239 W=4.0D S-Chicane geometry.
- Seed: `923301021`, which is not a sealed seed and was not used by L258.
- Arm: `full_proposed` only.
- Maximum: 250 control steps, 50 candidates x 2 iterations.
- Actor: frozen L258-selected seed20262441 step60000 checkpoint.
- ICODE, RL/Traditional fusion, reliability/HSS, network, reward, MPPI cost,
  candidate boundary filter, dynamics, map, and action bounds remain unchanged.
- The L258 results and the invalid chained seed42 artifacts are read-only.

## Deliberately weak simulation guard

The diagnostic sets `near_body_stop_radius=0` and `side_stop_distance=0`.
Only a narrow 20-degree frontal last-resort guard remains: hard stop 0.24 m,
soft block 0.30 m, and mild slowdown beginning at 0.50 m.  This is an explicit
development treatment and is unsafe for hardware.

No file under `mppi_hardware_bridge`, no hardware launch configuration, and no
real-robot scan-guard default may be changed by L259.

## Required source diagnostics

For every planner decision, retain separately for Actor-guided and Gaussian
candidates:

- actual opportunity and elite counts;
- minimum, mean, and median frozen MPPI cost;
- candidate-boundary feasible fraction;
- Actor and baseline first actions;
- Actor-versus-baseline sequence and first-action differences;
- reliability proposal authority and next-cycle guided allocation.

The braking candidate label remains excluded from source accounting.  The
diagnostic must also expose the actual post-replacement opportunity counts so
the known first-slot replacement bias is visible rather than hidden.

## Interpretation

- Actor cost gaps greater than zero together with low guided elite yield mean
  the Actor proposal is mismatched to the deployed MPPI objective/distribution.
- Comparable or lower Actor cost with low later allocation means reliability
  suppression is the dominant failure.
- Lower Actor boundary feasibility means the candidate filter is the immediate
  rejection mechanism.
- Restored translation after the old deadlock region supports the safety-trap
  diagnosis; it is not evidence that the resulting weak guard is acceptable.

Negative results retain raw data and a short status only.  This diagnostic may
not be described as formal evidence and may not be used to select a checkpoint.

