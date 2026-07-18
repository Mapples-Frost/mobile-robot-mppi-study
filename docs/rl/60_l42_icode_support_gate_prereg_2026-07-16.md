# L42 normalized-support-gated ICODE — preregistration

## Motivation

L40 substantially improved held-out H=36 prediction. L41 then produced a small pooled closed-loop gain without collision increase, but one of three model seeds regressed. Because MPPI optimizes through its prediction model, even a good on-policy model can steer candidate rollouts into regions unsupported by its training data. L42 tests a fail-toward-nominal residual support gate.

## Gate definition

For the checkpoint-normalized state features and control, let

\[
z_{\max}=\max\left(\left|\frac{\phi(x)-\mu_x}{\sigma_x}\right|,
\left|\frac{u-\mu_u}{\sigma_u}\right|\right).
\]

Residual confidence is fixed before evaluation:

\[
\alpha_{\mathrm{res}}=
\begin{cases}
1, & z_{\max}\le3,\\
(5-z_{\max})/2, & 3<z_{\max}<5,\\
0, & z_{\max}\ge5.
\end{cases}
\]

The rollout derivative is

\[
f_{\mathrm{pred}}=f_{\mathrm{nom}}+\alpha_{\mathrm{res}}f_{\mathrm{ICODE}}.
\]

This gate uses only training normalization statistics. It has no access to collisions, simulator state, obstacle truth or task outcome, and exactly recovers nominal dynamics at zero confidence.

## Frozen design

- Conditions: nominal, ungated L40 ICODE, and support-gated L40 ICODE.
- Three independent ICODE seeds, three calibrated dynamic scenes, two delay strata and five new episode seeds.
- Common MPPI random seeds and identical temporal safety chain.
- 270 planned episodes; experimental units are model-block × scene × physics × seed.
- Development seeds: `20960731`–`20960735`; confirmation seeds `20960736`–`20960745` remain sealed.

## Frozen gate

Against nominal, gated ICODE must show positive success in at least two model blocks, collision noninferiority in all three, pooled success gain at least three, no pooled collision increase and at least 0.10 m mean final-distance improvement. Against ungated ICODE it must not increase collisions and must not worsen mean final distance. Mean support reduction must exceed 1%, disabled support must remain below 50%, and mean planner compute must remain below 50 ms.

Passing this development gate permits an independent residual-only confirmation. It does not authorize RL composition.
