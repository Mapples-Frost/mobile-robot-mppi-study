# Local approximation: prior-art check

[Salzmann et al., Real-time Neural MPC](https://arxiv.org/html/2203.07747v3), published with DOI10.1109/LRA.2023.3246839, already separates expensive neural dynamics from online optimization through parallel local approximations. Its optimization is gradient-based, whereas this repository's candidate uses the approximation inside sampling-based MPPI. Therefore batching/linearizing a neural residual is not, by itself, a new contribution. Numerical gains here must be described as an implementation result until a specific additional methodological contribution is established.

The present prototype retains nonlinear nominal dynamics, approximates residual drift/control-gain around mean-candidate nominal trajectories, and evaluates the final weighted trajectory with the full model. Whether these implementation distinctions justify a publishable contribution is unresolved; do not describe them as first-of-kind.

[GPU-Parallel Linearization Error Bounds](https://arxiv.org/abs/2607.01203) is also a relevant primary-source lead. Only its abstract page was inspected in this check; no reproduction or theorem equivalence is claimed. A positive controller result would require closer review of approximation-error treatment and strong corresponding controls.

Accessed2026-09-10. This note does not alter active frozen experiments or their acceptance criteria.
