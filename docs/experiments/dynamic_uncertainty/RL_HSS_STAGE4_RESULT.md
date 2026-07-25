# Dynamic-Uncertainty Stage 4 RL/HSS Result

Date: 2026-07-24  
Status: **failed the preregistered safety gate and stopped after the first
collision; the sealed-seed stage is not authorized.**

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-07-24
- Verification Status: ANALYZED
- Version Label: stage4_rl_hss_amendment1_result_v1
- Independent Unit: complete MuJoCo episode
- Protocol:
  `configs/research/dynamic_uncertainty_rl_hss_stage4_amendment1.yaml`
- Protocol SHA-256:
  `d039047f9be1ba744a2d4171186e30d73f0eed1722d4559c8a3fbba8c9a68874`

## Decision

Stage 4 does not pass. The amended formal run completed `22/24` scheduled
episodes, then the RL/HSS-on residual-block-0 cell at obstacle seed
`730100010` collided. The runner correctly applied the registered first-
collision stop, leaving the final two RL/HSS-on cells unrun. They must not be
resumed, imputed, or silently excluded.

The result is stronger than a marginal gate miss. All 12 completed RL/HSS-off
episodes reached the goal without collision. None of the 10 completed
RL/HSS-on episodes reached the goal: nine ended at the 400-step horizon and
one collided. In the 10 complete common-random-number pairs, the success
difference was `-1` in every pair.

This is a development result. It supports rejecting the frozen L217 Actor/HSS
integration for this environment, but it does not estimate held-out
generalization and does not authorize opening sealed seeds.

## Execution accounting

| Arm | Completed | Goal reached | Max steps | Collision | Unrun |
|---|---:|---:|---:|---:|---:|
| RL/HSS off | 12 | 12 | 0 | 0 | 0 |
| RL/HSS on | 10 | 0 | 9 | 1 | 2 |
| Total | 22 | 12 | 9 | 1 | 2 |

The two intentionally unrun cells are:

- `residual_block2::seed730100010::rl_hss_on`
- `residual_block1::seed730100010::rl_hss_on`

The original pre-amendment failed launch remains excluded from treatment
analysis because it produced no completed episode and no control command.

## Episode-level paired result

The table uses only complete RL-off/RL-on pairs. Medians are descriptive. The
three residual checkpoints are reported as model blocks and are not counted as
independent environment replicates.

| Dynamics block | Complete pairs | Off success | On success | On collision | Median final-distance delta, on-off | Median completion delta | Median planner-P95 delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal | 3 | 3 | 0 | 0 | +0.699 m | -0.079 | +56.90 ms |
| Residual block 0 | 3 | 3 | 0 | 1 | +0.835 m | -0.095 | +249.80 ms |
| Residual block 1 | 2 | 2 | 0 | 0 | +0.562 m | -0.064 | +222.53 ms |
| Residual block 2 | 2 | 2 | 0 | 0 | +0.443 m | -0.050 | +196.78 ms |
| All complete pairs | 10 | 10 | 0 | 1 | +0.609 m | -0.069 | +208.66 ms |

Every paired final-goal-distance delta was positive, so every completed
RL/HSS-on treatment ended farther from the goal than its RL/HSS-off control.
The raw paired values are retained in `paired_analysis.json`; no p-value is
reported because only three development obstacle seeds were registered and
the matrix stopped early.

## Gate and treatment-integrity audit

| Check | Result | Evidence |
|---|---|---|
| Zero collisions | **FAIL** | 1 observed; threshold 0 |
| Full matrix | **INCOMPLETE BY RULE** | 22/24; two later cells unrun |
| First-collision stop | PASS | collision at run order 21; no retry or later episode |
| Frozen hashes | PASS | protocol, original-failure evidence, Stage 3 evidence, Actor/HSS/checkpoints, and five implementation files revalidated |
| Complete artifact sets | PASS | 22 config/metrics/trajectory sets; 0 partial or unexpected sets |
| RL-off reproduces Stage 3 configuration | PASS | 12/12 resolved configurations exactly match the frozen generator, excluding only the host-dependent source-path annotation |
| All resolved treatment configurations frozen | PASS | 22/22 match the preregistered generator and runtime contract |
| HSS active in RL-on | PASS | enabled fraction 1.0 in 10/10 completed RL-on episodes |
| Candidate budget | PASS | 300 candidates per iteration, 2 iterations, 600 total rollouts on every RL-on decision |
| Residual shield | PASS | enabled fraction 1.0 in every completed residual cell |
| Nominal/residual separation | PASS | nominal cells report no residual shield |
| Probability-risk parity | PASS | risk enabled and required diagnostics present in 10/10 completed RL-on episodes |
| Scan guard | PASS | frozen configuration and runtime diagnostics present in 22/22 episodes |
| Sealed seeds | PASS | not imported, scheduled, emitted, or opened |

Thus, the treatment-integrity mechanisms passed while the scientific safety
gate failed. The collision cannot be dismissed as a budget change, missing
HSS, disabled risk path, absent shield, scan-guard bypass, or sealed-seed
contamination.

## Collision forensics

The collision occurred in
`residual_block0::seed730100010::rl_hss_on` after 214 control steps. Minimum
clearance was `-0.02486 m`, and final goal distance was `4.456 m`.

The terminal hard-risk interval began at approximately `20.1 s`. During the
following second:

- the probabilistic maximum-step risk rose to `1.0`;
- the active-avoidance fallback was selected;
- the residual shield selected its nominal controller near the collision;
- the scan guard issued repeated `dynamic_active_escape` commands, including
  bounded reverse motion;
- at `21.2--21.4 s`, the near-body hard stop commanded exactly zero velocity
  for three consecutive control periods;
- the moving obstacle nevertheless entered the combined collision radius.

This evidence shows that the safety stack was active rather than bypassed. It
also shows that, from the RL/HSS-induced terminal geometry, the existing
escape-then-stop policy did not preserve collision avoidance. The single
collision does not isolate whether the principal cause is proposal authority,
optimizer state evolution, escape geometry, or their interaction.

## Mechanism signal

Across the 10 completed RL/HSS-on episodes:

- mean HSS proposal authority had median `0.811` and range `0.611--0.922`;
- mean dynamics confidence had median `0.00245` and range `0--0.0160`;
- Actor-versus-baseline first-action L2 difference had median `0.468`;
- the episode mean of best guided cost minus best Gaussian cost was positive
  in `10/10` episodes, with range `45.31--618.99`;
- nominal RL/HSS-on planner P95 was `73.89--82.06 ms`;
- residual RL/HSS-on planner P95 was `239.75--328.19 ms`, and all residual
  RL/HSS-on decisions missed the 100 ms deadline.

The consistent positive guided-minus-Gaussian minimum-cost diagnostic means
that the Actor-guided subset was worse than the Gaussian subset under the
controller's own cost in every completed RL-on episode. At the same time, the
`policy_rescue` routing retained high proposal authority despite near-zero
dynamics confidence. This is a strong diagnostic hypothesis for the uniform
completion regression, not a separately randomized causal decomposition.

## Validation report

- Overall Confidence: **RED_FLAG**
- Reproducibility Method: no rerun; prohibited by the preregistered
  first-collision stop
- Reproducibility Verdict: **CANNOT_VERIFY by rerun**
- Artifact/contract verification: passed

### Warnings

| Type | Detail | Affected inference |
|---|---|---|
| Safety failure | One RL/HSS-on collision versus zero in its paired control | Stage 4 pass decision |
| Uniform task regression | RL/HSS-on success 0/10 versus RL/HSS-off 10/10 in complete pairs | Frozen integration eligibility |
| Informative missingness | Two RL/HSS-on residual cells are absent because the safety stop occurred first | Full-matrix averages and block-1/block-2 seed-010 effects |
| Small development sample | Only three obstacle seeds; model blocks are not extra environment samples | Statistical precision and generalization |
| Runtime regression | Residual RL/HSS-on P95 exceeds the 100 ms control period | Real-time eligibility |

### Statistical fallacy scan

Coverage: **11/11 checked**.

| Fallacy | Severity | Finding |
|---|---|---|
| Simpson's paradox | NOTE | Block-level raw effects are reported beside the pooled descriptive summary; no direction reversal is hidden. |
| Ecological fallacy | NOTE | Inference is limited to complete development episodes, not individual control steps or a broader robot population. |
| Berkson's paradox | NOTE | Seeds and schedule were frozen before outcomes; no outcome-selected episode subset was introduced. |
| Collider bias | NOTE | No adjusted regression or post-treatment conditioning is used. |
| Base-rate neglect | NOTE | No diagnostic-accuracy claim is made. |
| Regression to the mean | NOTE | Seeds were not selected from extreme Stage 4 outcomes. |
| Survivorship bias | **CAUTION** | The two post-collision cells are explicitly retained as unrun; the report does not replace them with completed survivors. |
| Look-elsewhere effect | NOTE | Preregistered endpoints and the first-collision rule determine the decision; no significance search was run. |
| Garden of forking paths | NOTE | Amendment 1 repaired an implementation interface before any episode outcome; the amended schedule and gates remained frozen. |
| Correlation versus causation | **CAUTION** | Pairing supports an in-matrix treatment comparison, but the tiny development sample and early stop do not support a general causal-performance claim. |
| Reverse causality | NOTE | Not applicable to the randomized treatment injection; mechanism attribution remains explicitly unresolved. |

## Required next boundary

Do not open sealed seeds and do not continue the two missing cells under this
protocol. Any further RL/HSS work is a new development intervention with a new
preregistered amendment or stage. The evidence motivates, but does not itself
authorize, three bounded investigations:

1. gate Actor proposal authority on observed proposal advantage rather than
   preserving a high `policy_rescue` floor under near-zero dynamics confidence;
2. profile and reduce matched dual-controller Paper optimization latency before
   another closed-loop safety matrix;
3. test moving-obstacle escape feasibility when stopping is collision-prone,
   without changing the frozen Risk V1 threshold or using simulator truth.

## Artifacts

- Raw and resolved run:
  `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/`
- Episode summary:
  `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/episode_summary.csv`
- Paired analysis:
  `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/paired_analysis.json`
- Automatic gate and collision forensics:
  `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/gate.json`
- Forensic SHA-256 manifest:
  `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/result_manifest.json`
- Progress/early-stop record:
  `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/progress.json`
- Original implementation-only failed launch:
  `research_artifacts/dynamic_uncertainty_rl_hss_stage4_development/`
- Reproducible analyzer:
  `experiments/dynamic_uncertainty/analyze_rl_hss_stage4.py`
