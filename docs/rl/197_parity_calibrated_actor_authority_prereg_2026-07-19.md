# L197 parity-calibrated Actor-authority preregistration

Date frozen: 2026-07-19
Status: frozen before running seeds 572--574
Parent: failed L196 multi-physics development Gate

## Diagnosis and single permitted change

L196 completed all 48 episodes with zero collisions, equal budgets, and the
expected lower dynamics confidence in the unseen combined domain. It failed
the tracking Gate: pooled Full Proposed cross-track RMSE was 7.99% above
ordinary fixed, and the unseen-domain ratio was 1.1332.

The frozen diagnostic showed that the source-relative Actor elite-yield ratio
was generally below parity while the original identity mapping still assigned
medium or high sampling authority. A ratio of 0.60 means that, after Beta
smoothing, Actor-guided candidates produced elite samples at only 60% of the
Gaussian-source rate; it must not be interpreted as 60% competence.

L197 changes only this dimensional interpretation:

\[
c_\pi =
\operatorname{clip}\left(
\frac{\rho_\pi-0.75}{1.00-0.75},0,1
\right),
\qquad
\rho_\pi =
\frac{\text{guided elite yield}}
     {\text{Gaussian elite yield}}.
\]

Thus a source at least 25% below Gaussian parity receives no learned
competence authority, parity receives full authority, and the interval is
linear. The causal EMA, one-step lag, terminal minimum guidance, ICODE models,
Actor, MPPI implementation, cost, safety chain, and compute budget are
unchanged. The implementation retains the raw ratio and mapped confidence as
separate diagnostics. Default thresholds remain 0 and 1, so historical
configurations retain their exact mapping.

The thresholds are fixed from the semantic meaning of relative yield, not
selected by searching L197 outcomes.

## Design

The design and Gate are identical to L196 except for:

- fresh development seeds: 572, 573, 574;
- source-competence config:
  `configs/rl/source_relative_actor_competence_l197.yaml`;
- output root:
  `results/research_platform/rl/path_aware_multiphysics_dev_l197`.

Each seed is a paired block across all four arms and all four physics domains:
`high_mass_seen`, `low_friction_seen`, `long_delay_seen`, and
`combined_unseen`. There are 48 episodes in total. Physics domains are
repeated strata within seed; the inferential cluster is seed.

## Frozen development Gate

The Gate passes only if:

1. Full Proposed succeeds in all 12 episodes and has zero collisions;
2. pooled Full Proposed cross-track RMSE is no greater than ordinary fixed;
3. Full Proposed is within 5% of ordinary fixed in at least three of four
   domains;
4. combined-unseen Full Proposed cross-track RMSE is no greater than ordinary
   fixed;
5. pooled Full Proposed jerk is no more than 10% above ordinary fixed;
6. unseen dynamics confidence is below the mean of the three seen domains;
7. raw source ratio, mapped competence, and applied allocation are finite and
   exercised;
8. all arms use exactly `K=100` and two paper iterations.

No sealed L186 geometry or seeds 561--565 may be inspected or used during this
remediation. Passing L197 permits one sealed confirmation; failing L197
rejects this adaptive hybrid-sampling realization rather than triggering
further threshold search.
