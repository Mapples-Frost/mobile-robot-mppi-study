# MPOPI fidelity report

Source: Asmar et al., ICRA 2023, [Algorithm 1](https://arxiv.org/abs/2203.16633).
Author source: `sisl/MPOPIS`, commit `22b04d0`, `mppi_mpopi_policies.jl`.

Implemented joint-sequence Gaussian proposal, repeated weighted mean/full-covariance
AIS, final weighted command and per-cycle covariance reset. Defaults in the author
moment-AIS constructor are 10 iterations and AIS temperature20. Our development
port uses 3 iterations, common task temperature2.5, jitter1e-8 and alpha.95;
these are declared port settings, not claims of original hyperparameter replication.
The source's general default alpha1 makes the disputed correction zero.

Paper correction uses proposal mean and original covariance. Current author code
uses original mean and current proposal covariance. `paper` and `author_code`
variants expose this difference; no silent reconciliation. Full covariance uses
normalized weighted population moments, not a claim of bitwise Julia equivalence.

Original tasks include MountainCar, car racing and MuJoCo locomotion. The NumPy
kernel runs on the project's navigation cost, dynamics, clipping, warm-start,
rate limits and guard. It does not reproduce those original tasks. Configs provide
nominal and ICODE prediction; the latter is an internal fairness control.
Post-selection task-specific terminal overrides of the repository's standard
solver are not part of this kernel port; current study configurations have no
heading/speed terminal gate. This must be checked before porting other tasks.

Tests: analytic L=1 information-theoretic update; full-covariance moments/PSD;
paper/author correction disagreement; nominal/residual two-cycle MuJoCo sanity.
Source-derived mechanism implemented, original published performance unverified.
