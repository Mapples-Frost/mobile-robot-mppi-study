# Table 22: Stage 4 RL/HSS Amendment 1 development result

| Evidence item | RL/HSS off | RL/HSS on | Interpretation |
|---|---:|---:|---|
| Completed episodes | 12 | 10 | Matrix stopped at 22/24 after the first collision; two later RL-on cells remain unrun |
| Goal reaches | 12/12 | 0/10 | Frozen RL/HSS treatment regressed every completed pair |
| Collisions | 0 | 1 | Safety gate threshold was zero; Stage 4 failed |
| Complete common-random-number pairs | 10 | 10 | Independent unit is a complete episode; residual checkpoints are model blocks, not environment replicates |
| Median paired final-goal-distance delta | reference | +0.60898 m | Every raw on-minus-off delta was positive |
| Median paired completion delta | reference | -0.06920 | Descriptive only; no inferential test was run |
| Median paired planner-P95 delta | reference | +208.66 ms | Residual RL-on P95 was 239.75--328.19 ms |

## Treatment integrity

| Check | Result |
|---|---|
| RL-off exact Stage 3 generator match | 12/12 passed |
| All completed resolved configurations match frozen generator | 22/22 passed |
| HSS enabled in completed RL-on episodes | 10/10 at fraction 1.0 |
| Fixed RL-on candidate budget | 300 candidates x 2 iterations = 600 on every decision |
| Probability-risk diagnostics and enabled path | 10/10 passed |
| Residual shield / nominal separation | passed |
| Final scan guard | present in 22/22 episodes |
| First-collision stopping rule | enforced; no retry or later job |
| Sealed seeds | closed |

## Mechanism and collision forensics

- Mean HSS proposal authority across RL-on episodes had median `0.81063` and range `0.61057--0.92204`, while mean dynamics confidence had median `0.00245` and range `0--0.01603`.
- Best guided-minus-Gaussian cost was adverse in 10/10 completed RL-on episodes; the episode mean ranged from `45.31` to `618.99`.
- The collision was `residual_block0::seed730100010::rl_hss_on`. Terminal hard risk began near `20.1 s`; active escape ran before three final zero-command near-body hard-stop steps, yet minimum clearance reached `-0.02486 m`.
- Mechanism integrity passed, but the scientific gate failed. This does not authorize sealed confirmation or resuming the two missing cells.

## Bindings

- `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/episode_summary.csv`
- `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/paired_analysis.json`
- `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/gate.json`
- `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/result_manifest.json`
- `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE4_RESULT.md`
