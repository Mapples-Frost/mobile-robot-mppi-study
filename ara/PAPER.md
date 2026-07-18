# Research Artifact Manifest

- Working title: Cross-Layer Uncertainty-Gated Learned Dynamics and Policy-Guided MPPI
- Target venue: ICRA 2027
- Submission target: 2026-09-15
- Repository: `mobile-robot-mppi-study`
- Current evidence stage: sealed parameter-matched evidence that control-affine ICODE improves MPPI tracking across frozen plant and bounded observation shifts; route-context covariance and half-budget efficiency have independent support; L103 now confirms that a constrained contextual bandit can allocate nested K50-to-K100 compute non-randomly at held-out ICODE-MPPI states, while whole-episode and ICODE-interaction claims remain open

## Layers

- `logic/`: falsifiable claims, problem framing and implementation heuristics.
- `trace/`: chronological research decisions, failed gates and session provenance.
- `evidence/`: pointers to immutable experiment artifacts and reports.
- `staging/`: observations not yet promoted to claims or decisions.
