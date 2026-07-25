# Table 21: Stage 4 frozen RL/HSS integration preflight

| Check | Frozen contract | Preflight evidence | Status |
|---|---|---|---|
| Development design | 3 obstacle seeds x (nominal + 3 residual blocks) x RL/HSS off/on | Reproducible 24-cell blocked schedule; complete MuJoCo episode is the independent unit | Passed |
| RL-off control | Exact Stage 3 `configure_condition` result | Configuration equality tested for every RL-off cell | Passed |
| Candidate budget | 600 candidates per controller decision | RL-on uses `300 x 2`; RL-off uses `600 x 1`; no additional candidates authorized | Passed |
| Frozen Actor | L217 SHA-256 `eaea2e2975de0794da4ad6876fbff3e6e85e53427e5471982e857585fff97ae4` | Forward-action bounds retained as explicit `[0.0, 0.35]` subspace of the unchanged `[-0.35, 0.35]` controller | Passed |
| Frozen HSS | Three value-aligned L192 sidecar members with L193/L194 provenance | Sidecar loads independently of MPPI prediction dynamics and updates only after a completed transition | Passed |
| Residual treatment | Matched nominal/residual Paper RL controllers inside the existing shield | Actor, HSS mutable state and RNG objects are independent; shield selection rule is unchanged | Passed |
| Safety chain | MPPI -> residual shield when applicable -> scan guard | Factory construction retains the residual shield and final scan guard | Passed |
| Execution boundary | Preflight by default; explicit `--execute` required | Status is `preflight_passed_matrix_not_started`; no formal output directory or sealed-seed access | Passed |

Software validation completed before any Stage 4 episode: seven focused Stage 4 tests passed; relevant regression groups passed 27 and 65 tests; the broad repository run passed 922 tests. Its five failures are unchanged tests requiring historical L34/L70/L72 checkpoint artifacts absent from this working copy, not failures of the Stage 4 interface. Real frozen checkpoints constructed a nominal `PaperRLDrivenMppiController` and a parallel `ResidualSafetyShieldController` containing two independent Paper RL controllers and three-member HSS sidecars.

This table establishes treatment construction and preflight integrity only. It contains no Stage 4 closed-loop result and does not authorize sealed seeds.
