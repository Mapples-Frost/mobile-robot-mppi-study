# Table 23: Stage 5 proposal-advantage veto mechanism probe

## Protocol and execution boundary

| Item | Registered or observed value |
|---|---|
| Data scope | Fresh development seeds only; nominal dynamics; sealed seeds unopened |
| Arms | RL/HSS off; RL/HSS shadow; RL/HSS active advantage veto |
| Veto rule | Best guided cost greater than best Gaussian cost for three consecutive comparable cycles; zero margin; next-cycle application; episode latch |
| Fixed budget | 600 Paper rollouts per RL/HSS decision |
| Stop rule | Stop the complete matrix at the first collision |
| Execution | 7/9 episodes completed; stopped after `rl_hss_shadow::seed730100064` collided |
| Unrun cells | `rl_hss_off::seed730100064`; `rl_hss_advantage_veto::seed730100064` |
| Formal gate | **FAIL**: zero-collision gate false and complete matrix unavailable |

Stage 4 retrospective EDA selected the veto threshold: 3378/3395 comparable decisions (99.499%) had a cost-inferior guided minimum, and all ten episodes began with at least five consecutive disadvantages. Those observations are threshold-selection data, not independent Stage 5 outcomes.

## Completed outcomes

| Seed | Arm | Outcome | Steps | Final distance (m) | Planner P95 (ms) | Latch step (zero-based) |
|---:|---|---|---:|---:|---:|---:|
| 730100068 | active veto | goal reached | 328 | 0.2980 | 56.66 | 12 |
| 730100068 | shadow | max steps | 400 | 1.0803 | 81.40 | 12, counterfactual only |
| 730100068 | RL/HSS off | goal reached | 367 | 0.2970 | 25.37 | n/a |
| 730100066 | active veto | goal reached | 299 | 0.2987 | 56.57 | 12 |
| 730100066 | shadow | max steps | 400 | 0.6262 | 80.38 | 12, counterfactual only |
| 730100066 | RL/HSS off | goal reached | 340 | 0.2974 | 22.83 | n/a |
| 730100064 | shadow | collision | 381 | 0.8356 | 77.83 | 11, counterfactual only |

## Paired descriptive results on complete blocks

| Contrast | Complete pairs | Success delta | Median step delta | Median final-distance delta | Median planner-P95 delta |
|---|---:|---:|---:|---:|---:|
| Active veto minus shadow | 2 | +1 in both pairs | -86.5 | -0.5549 m | -24.28 ms |
| Active veto minus RL/HSS off | 2 | 0 in both pairs | -40.0 | +0.00116 m | +32.51 ms |

No p-value, confidence interval or standardized effect size is reported because the safety stop leaves only two complete blocks with informative missingness.

## Integrity and collision forensics

- Both active episodes latched at zero-based step 12. The current decision remained causal; veto application began on the next decision.
- All 315 and 286 post-latch decisions respectively had zero guided candidates, zero Actor mean/variance authority, fallback fraction one and exactly 600 rollouts.
- Every shadow episode observed the counterfactual latch while applied gate authority remained one.
- In the seed-730100064 shadow collision, the disadvantage fraction was 0.9860 and mean dynamics confidence was 0.00142. Near contact, predicted collision probability was approximately one while proposed, executed and applied robot velocities were zero; this is consistent with, but does not prove, a dynamic stopping-feasibility defect.
- The result manifest binds 38 artifact files and 11 external files: 49/49 hashes matched, with zero missing files.

## Interpretation boundary

The two complete prospective blocks support the narrow claim that the implemented one-way veto can rapidly reject this transferred bad Actor and recover RL/HSS-off-level completion without changing the fixed rollout budget. They do not establish global Actor incompetence, active-veto safety on the unrun collision seed, confirmatory effect size, or readiness for sealed evaluation. Because guided and Gaussian minima come from unequal candidate populations, the signal is not a calibrated competence estimator and must not automatically restore authority.

Primary evidence: `research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/analysis.json`, `paired_analysis.json`, `integrity_audit.json`, `gate.json`, `result_manifest.json`, and `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_RESULT.md`.
