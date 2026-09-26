# L204 Development Result and L205 Bounded-Correction Amendment

## L204 development-only result

L204 verified the full causal data path and exact step-zero migration, but the
unconstrained full-Actor SAC update failed the preregistered advancement gate.
The initial validation achieved 100% success, zero collision and 0.1063 m mean
cross-track RMSE. Every trained checkpoint had worse validation return and
cross-track RMSE; the best checkpoint therefore remained step zero.

This rejects **unconstrained full-network fine-tuning** for this coupling. It
does not test whether residual context is useful under a competence-preserving
policy parameterization, and it is not paper-positive evidence.

## Diagnosed failure mode

The L185 policy is already competent. Updating every Actor parameter gives SAC
far more freedom than required for physics-conditioned adaptation and permits
catastrophic forgetting. The residual context begins with zero input weights,
so useful context sensitivity and preservation of the base behavior must be
learned simultaneously.

## L205 frozen remediation

L205 freezes the zero-extended L185 Actor and trains a separate SAC correction
policy. The composed normalized action is

\[
a_t = a_t^{\mathrm{L185}} + \Delta a_t,
\]

with componentwise hard authority bounds induced by
`correction_scale=[0.10, 0.15]` and `correction_gate_alpha=0.50`. The correction
is exactly zero at initialization. The base Actor cannot change, and scan_guard
and MPPI remain downstream authorities.

The reward increases path-precision and smoothness weights because L204 showed
that progress-dominated return did not preserve cross-track accuracy. Actor
updates use scene-group-robust aggregation so a frequent/easy geometry cannot
dominate the correction.

## Decision rule

L205 uses the same L204 development family and is permitted to tune only this
bounded-correction parameterization. It must improve on its exact step-zero
base before any closed-loop MPPI experiment. If it does, the selected
checkpoint advances to fresh MPPI development seeds; sealed confirmation
remains untouched. If it does not, this remediation is rejected and no result
will be described as supporting the paper.
