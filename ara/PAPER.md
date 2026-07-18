# Research Artifact Manifest

- Working title: Cross-Layer Uncertainty-Gated Learned Dynamics and Policy-Guided MPPI
- Target venue: ICRA 2027
- Submission target: 2026-09-15
- Repository: `mobile-robot-mppi-study`
- Current evidence stage: sealed parameter-matched evidence that control-affine ICODE improves MPPI tracking across frozen plant and bounded observation shifts; L87 established path-dependent covariance headroom, and L89 independently confirmed that a reward-trained contextual bandit improves both time and tracking precision over the strongest fixed covariance on held-out route geometries with no detectable compute or safety regression

## Layers

- `logic/`: falsifiable claims, problem framing and implementation heuristics.
- `trace/`: chronological research decisions, failed gates and session provenance.
- `evidence/`: pointers to immutable experiment artifacts and reports.
- `staging/`: observations not yet promoted to claims or decisions.
