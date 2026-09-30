# Executor responses to independent reviews

Updated: 2026-09-29T23:37:44.061580+00:00 by GPT-5.5 executor after reading `docs/bohn2021_takeover/astra_reviews/LATEST.md` and `docs/bohn2021_takeover/astra_reviews/20260929T153837Z.md` at a safe boundary. This log records dispositions; it does not authorize sealed-test access or change acceptance criteria.

## Report `20260929T153837Z`

Evidence hashes available this cycle include report `c22291382519d4da417272a178cff749a23914c360c1ec63dcb4f39a35f92f6a`, manifest `3b388a3f3938832efe24296813f78a8a11ba00975b7f45438d9a335f06000cdf`, v16b raw `d391540ae567138d8ce0e6c940a322c29300d7968751b89063c2faae3a19a4d0`. v17 follow-up headline: strict_nested_save=0.000000, strict_nested_bad=0, pass5=False.

| ID | disposition | verified evidence | concrete action / result |
|---|---|---|---|
| `A1_ORIGINAL_incomplete_not_final_success` | accepted | SOURCE_MAP/run.py limitations and project status: ORIGINAL exact paper config/test files remain unavailable; no final-test success claimed. | Continue reporting as core author-code reconstruction and/or IMPROVED only. |
| `A2_current_vehicle_selector_not_original_SAC` | accepted | Latency-tree/true-variable-H/v10-v17 scripts are finite search/refit diagnostics with no RL gradient training. | All true-variable-H/risk selector results remain labeled IMPROVED. |
| `A3_original_masked_AHMPC_does_not_reduce_NLP_dimension` | accepted | Reviewer-cited controller mask behavior verified; case5 true-variable-H smoke showed opt_x H10 [182] < H15 [267] only for the IMPROVED true-H branch. | Do not infer original AHMPC runtime reduction from action H; true-variable-H stays IMPROVED. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted | v16b and v17 are offline branch-decision sums; no online selector overhead included. | No speed claim until a blocked/randomized overhead smoke measures feature+selector+solver whole decision time. |
| `A5_pendulum_three_seed_training_incomplete` | deferred | Pendulum inventory still has s0 complete, s1 interrupted, s2 absent; vehicle remains prioritized. | Do not claim pendulum completion; revisit after vehicle diagnostic line reaches a decision point. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted/deferred | Current v15-v17 compare mainly fixed true H15 in development; formal fixed-H grid/per-H terminal baselines remain required for final claims. | Keep final claims gated on full strong baselines and independent validation/test. |
| `A7_targeted_risk_banks_are_not_population_estimates` | accepted | v15/v16b/v17 use mined opened boundary/development rows. | Interpret only as mechanism/development diagnostics, not population estimates. |
| `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted | v16b in-sample bad=0/save=0.2677 but strict nested bad=4; v17 tests a consensus veto as follow-up. | Do not use in-sample/global repair as confirmation. |
| `A9_validation64_exposed_for_latency-tree_development` | accepted | Existing validation64 was used for latency-tree development; current true-variable-H diagnostics avoid validation64 and sealed test. | If method changes continue after exposure, require fresh independent confirmation before any final test request. |
| `A10_runtime_semantic_modifications_affect_original_fidelity` | accepted | Local runtime semantic fixes are documented in project protocol and reviewer report. | Scope ORIGINAL claims to reconstructed core author code with local semantic fixes unless exact author config evidence is found. |
| `A11_training_failure_modes_need_separation` | accepted/deferred | Terminal/reward audits show terminal-profile sensitivity; v16b/v17 address representation/uncertainty without new gradients. | If v17 fails or is too conservative, prioritize terminal/risk-value training/refit or controlled terminal-value ablation over more static sweeps. |
| `A12_registry_backup_schema_contract` | accepted | Schema false positives and backup requests are preserved; latest supervisor backup was verified before v17. | Continue external backup gating after v17 artifacts. |

## Follow-up through v18d/v19/v20b boundary

Updated at the safe post-v20b boundary by GPT-5.5 executor after verifying that `LATEST.md` still points to report `20260929T153837Z`. This addendum records concrete follow-up evidence; it does not open validation64 or sealed test and does not change any acceptance criterion.

| linked recommendation(s) | disposition after follow-up | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; still open | v20b summary `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/summary.md` reports strict leave-bank/source decision savings from offline H12/H15 branch sums. Its own `overhead_semantics` state that static/history families have no measured online selector overhead yet; no validation64/test was accessed. | Do not claim deployed speed from v20b alone. After post-v20b backup, run a bounded online-overhead smoke for the selected deployable H12/H15 feature/selector path, blocked/randomized against H15, measuring whole decision and solver timing. |
| `A7_targeted_risk_banks_are_not_population_estimates` and `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; v20b positive remains development-only | v19 summary `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/summary.md`: fixed H12 vs H15 saved 14.32% decision time but had 6 catastrophic/high-cost H12 rows, all from `fresh_v11/fresh_case05_slot1_mid_late_control`. v20b summary: after repeat averaging, H12 positive rows=21, catastrophic rows=3, negative sources only `fresh_v11/fresh_case05_slot1_mid_late_control`; strict leave-bank pass5=True with save=5.21%, bad=0; strict leave-source pass5=True with save=7.21%, bad=0. | The first H12/H15 selector result is encouraging but not validation-ready because the only negative source is held safe largely by the no-catastrophe-training uncertainty fallback. Next unique science should be source-independent H12 negative/support acquisition before any validation claim. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; partially sharpened by v19 | v19 measured true H12/H15 from identical opened boundary states. Fixed H12 was not an acceptable fixed baseline on these rows (`physical_delta` 566.4 vs tolerance 96; catastrophic rows=6), while an oracle H12/H15 switch saved 15.72% decision time and 17.19% solver time with physical delta -1.265. | Future adaptive claims must still compare against strong fixed-H, including H12/H15 as relevant true-variable-H fixed comparators. v19 supports H12/H15 adaptation as a mechanism, not final superiority. |
| `A11_training_failure_modes_need_separation` | accepted; next diagnostic narrowed | v17 removed H10 catastrophes only by choosing no H10; v18d H10 probe telemetry failed strict useful-safety; v19 showed intermediate H12 creates a useful oracle tradeoff; v20b showed static/history/probe H12/H15 selector can pass opened strict splits but with only one negative source. | Before new gradient training, acquire source-independent H12 labels and overhead because the current bottleneck is not yet proven to be representation/terminal-value learning rather than missing negative coverage. If fresh acquisition shows selector false positives or no robust labels, pivot to bounded terminal-risk/value refit/training ablation. |
| `A12_registry_backup_schema_contract` | accepted; active blocker | Post-v20b repository listing for `research_artifacts/aws_backup_proofs/backup_proof_20260930*` was empty in this API iteration. Existing backup request is `REQUEST_BACKUP_AFTER_H12_H15_SELECTOR_REFIT_V20B_FAST_20260930T0100Z.json`; latest verified supervisor backup in context predates v20b (`2026-09-30T00:34:46Z`, commit `a8713ed1eda0cc6388926564fd026a454ba7729a`). | No further unique simulations/refits/validation should run until a verified external backup proof covers v20b and this response-log/state addendum. |

## Follow-up through v21 source-independent acquisition boundary

Updated at the safe post-v21 boundary by GPT-5.5 executor. The v21 run used the verified supervisor-context backup commit `d77c6f952021feaaa55a24c1033ae253850dcc82` before execution. It did not access validation64 or sealed test, and it did not perform training/refit.

| linked recommendation(s) | disposition after v21 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A7_targeted_risk_banks_are_not_population_estimates` and `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; risk reduced but not closed | v21 summary `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/summary.md` and completed marker report 40 development episodes / 1962 control steps over 16 fresh branch states selected before H12 outcomes. H12-beneficial rows=15/16, catastrophic/high-cost H12 rows=0, H15 unsafe rows=0. Fixed H12 vs H15 saved 15.05% decision time and 18.82% solver time with physical delta -21.293 and pass5=True; oracle H12/H15 saved 16.47%. | This is encouraging fresh source-independent development evidence and weakens the hypothesis that all safe H12/H15 gains were an artifact of one known negative source. It remains targeted stress-pool evidence, not a population estimate or validation. Do not open validation/test yet; continue with overhead and broader baselines/confirmation. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; sharpened by v21 | v21 fixed H12 itself passed the H15-referenced safety/physical/5% timing gate on this batch, whereas v19 fixed H12 failed due to the `fresh_v11` cluster. | Treat true fixed H12 as a serious strong baseline on any revised H12/H15 benchmark. Future adaptive selector claims must report whether adaptation beats fixed H12, not only fixed H15, and must disclose per-H terminal/baseline opportunities. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; still open and now higher priority | v21 measured true H12/H15 branch decision/solver timings, but it did not execute a deployed selector path with feature extraction/model decision overhead. v20b static/history selectors also still have no measured online overhead. | Next unique experiment after verified backup should be a bounded blocked/randomized online-overhead smoke for fixed H15 vs fixed H12 vs the v20b selected/static-history H12/H15 selector path, measuring selector feature time, selector decision time, whole decision time, solver time, p50/p95 and safety. No speed claim before this. |
| `A11_training_failure_modes_need_separation` | accepted; training deferred with evidence, not indefinitely | v21 found no fresh H12 catastrophes in the targeted source-independent batch and suggests the immediate blocker may be compute-overhead/baseline comparison rather than representation/terminal-risk failure. | Defer new gradient training for the next step. If online-overhead or broader confirmation shows false positives, unstable savings, or no advantage over fixed H12, pivot to bounded terminal-risk/value refit/training or scenario redesign with a frozen hypothesis. |
| `A12_registry_backup_schema_contract` | accepted; active blocker | v21 wrote `REQUEST_BACKUP_AFTER_V21_SOURCE_INDEPENDENT_H12_H15_ACQUISITION_20260930T010632.171731+0000.json`; docs/state/registry were updated by the script, and this response-log addendum is a further artifact. | Before running the overhead smoke or any other unique science, require a verified external backup covering v21 outputs, run registry `research_artifacts/aws_runs/20260930T010156_d3bb52dd/registry.json`, updated docs including this response log, state, registry and backup requests. |

## Follow-up through v22 online-overhead diagnostic

Updated by GPT-5.5 executor at `2026-09-30T01:19:07.174133+00:00`. `LATEST.md` still points to `20260929T153837Z`; stable recommendation IDs are preserved.

| linked recommendation(s) | disposition after v22 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; partially resolved for static/history micro-overhead, still open for full closed-loop deployment | v22 raw/summary: Python in-memory history feature+selector overhead mean 0.001956693s, p95 0.002653475s over 20000 calls. The combined v19/v21 branch-level selector proxy retains 6.50% overhead-adjusted decision saving with zero catastrophic H12 false positives. | Do not claim final speed yet; next full confirmation must include closed-loop whole-decision timing and selector overhead in blocked/randomized order. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; sharpened | v22: v21-only fixed H12 saves 15.05% vs H15 with zero bad, while combined fixed H12 has 3 bad rows and physical_gate=False. | Treat fixed true H12 as a primary strong baseline on v21-like states; adaptive selector value is currently safety against known v19 negatives, not v21-only superiority. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v22 uses only opened v19/v21 targeted development rows. Combined selector pass remains development-only and source/risk enriched. | Before validation, acquire broader source-independent confirmation or freeze a fresh confirmation protocol with fixed H12/H15/selector baselines; sealed final test remains closed. |
| `A11_training_failure_modes_need_separation` | accepted; deferred by current evidence | v22 did not show overhead erasing branch-level selector savings, but fixed H12 dominates v21-only fresh states. | If broader confirmation shows fixed H12 safe, adaptivity may be unnecessary for that distribution; if new negatives recur, train/refit richer terminal-risk/value selector instead of static sweeps. |
| `A12_registry_backup_schema_contract` | accepted; active | v22 wrote new code/results/docs and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_ONLINE_OVERHEAD_V22_20260930T0125Z.json`. | Require verified external backup covering v22 before further unique science. |

## Follow-up through v23 broader H12/H15 confirmation

Updated by GPT-5.5 executor at `2026-09-30T01:32:17.706415+00:00`. Stable Astra IDs are preserved; v23 did not access validation64 or sealed test.

| linked recommendation(s) | disposition after v23 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; further addressed for pre-outcome selector choice overhead on fresh branch states | v23 preoutcome selector manifest measured feature+decision overhead mean 0.002620919s and p95 0.003181456s over 24 states; policy evaluation charged measured overhead. | Still no final speed claim; future confirmation/validation must measure full closed-loop whole-decision timing. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; fixed H12 remains primary comparator | v23 fixed H12 save 17.08%, bad 0, pass5 True; selector save 8.46%, bad 0, pass5 True. | Do not compare only against H15. If fixed H12 dominates safely, adaptivity/scenario opportunity must be reassessed; if fixed H12 fails and selector remains safe, plan fresh confirmation with fixed H12 primary. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v23 selected 12 new source_candidate_index values by metadata only and excluded prior sources, but still used the stress-v1 diagnostic pool. | Treat as broader development confirmation only, not population validation. |
| `A11_training_failure_modes_need_separation` | accepted; conditional | v23 decision: v23 fixed H12 is safe and faster than the selector on this broader fresh batch; fixed H12 must be the primary simple baseline and adaptivity is not yet justified for this stress distribution. | If v23 shows false positives or missed oracle opportunity, pivot to terminal-risk/value refit/training; if fixed H12 safely dominates, investigate scenario/comparison design rather than forcing adaptive switching. |
| `A12_registry_backup_schema_contract` | accepted; active | v23 wrote new source/results/docs/state/registry and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_BROADER_CONFIRMATION_V23_20260930T0145Z.json`. | Require verified external backup covering v23 before further unique science. |

## Follow-up through v24 H12/H15 adaptivity-opportunity audit

Updated by GPT-5.5 executor at `2026-09-30T01:41:17.634738+00:00`. Stable Astra IDs preserved. v24 is offline development analysis of v19/v21/v23 only; validation64 and sealed test remained closed.

| linked recommendation(s) | disposition after v24 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; fixed H12 is now the primary comparator on fresh stress-pool rows | v24 summary: v21+v23 fixed H12 bad=0, pass5=True, save=16.22%; v19 fixed H12 bad=3 and fails; all-row oracle H12/H15 pass5=True. | No adaptive validation until a fresh protocol shows adaptive value beyond fixed H12. Next diagnostic should target v19-like H12-risk-family reproduction with fixed H12 as the first baseline. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v24 shows v21/v23 zero-cat evidence is source-independent but still stress-pool/development; all H12 catastrophes remain concentrated in opened v19 boundary rows. | Treat v21/v23 as evidence that broad stress-pool support favors fixed H12, not as a population guarantee. |
| `A11_training_failure_modes_need_separation` | accepted; training deferred for one more discriminating scenario diagnostic | The current bottleneck is ambiguous: no fresh H12 negatives in v21/v23, but v19 has a reproducible negative cluster. Static selector conservatism loses fixed-H12 savings on v23. | After backup, freeze/run v25 H12-risk-family acquisition. If fresh negatives recur and static selector fails, pivot to bounded terminal-risk/value refit or training; if not, prioritize scenario/comparison design and fixed-H12-primary confirmation. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; unchanged by v24 | v24 performs no new timing; it relies on v22/v23 overhead evidence and row-level branch timings. | Continue to require full closed-loop whole-decision timing for any final speed claim. |
| `A12_registry_backup_schema_contract` | accepted; active | v24 wrote new source/results/docs/state/registry and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_ADAPTIVITY_OPPORTUNITY_AUDIT_V24_20260930T0205Z.json`. | Require verified external backup covering v24 before v25 or other unique science. |

## Follow-up through v25 dry-run/protocol freeze

Updated by GPT-5.5 executor at `2026-09-30T01:57:29.103230+00:00`. v25 dry-run did not access validation64/sealed test and ran no simulations.

| linked recommendation(s) | disposition after v25 dry-run | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; baked into v25 protocol | Protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_preoutcome_frozen_20260930T0210Z.json` makes fixed H12, fixed H15, oracle H12/H15 and v20b/v22 selector the planned comparisons. | After backup, run v25; report fixed H12 first and do not claim adaptivity unless it beats/avoids a fixed-H12 failure. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; v25 remains targeted development | Dry-run selected 8 unused source72-neighborhood cases by metadata only from the stress-v1 bank; selected indices [234, 17, 222, 162, 11, 135, 177, 167]. | Interpret v25 as risk-family mechanism acquisition, not population validation. |
| `A11_training_failure_modes_need_separation` | accepted; next discriminating experiment frozen | v24 showed fixed H12 dominates broad fresh rows but fails on opened v19 source72 cluster; v25 tests whether that risk reproduces in unused neighbors before choosing refit/training. | If v25 fixed-H12 risk recurs and selector fails, pivot to terminal-risk/value refit or bounded training; if no risk recurs, prioritize scenario/comparison design and fixed-H12-primary confirmation. |
| `A12_registry_backup_schema_contract` | accepted; active | v25 dry-run wrote new source/protocol/dry-run docs and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_DRYRUN_20260930T0210Z.json`. | Require verified external backup covering v25 dry-run before actual v25 simulations. |

## Follow-up through v25 H12-risk-family acquisition run

Updated by GPT-5.5 executor at `2026-09-30T02:04:13.379250+00:00`. v25 did not access validation64/sealed test.

| linked recommendation(s) | disposition after v25 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; updated by v25 | Fixed H12 save 19.16%, bad 0, pass5 True; selector save 12.11%, bad 0, pass5 True. | Keep fixed H12 primary in subsequent confirmation. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v25 is source-independent but deliberately source72-neighborhood stress-pool development evidence. | No population/validation claim from v25 alone. |
| `A11_training_failure_modes_need_separation` | accepted; decision recorded | v25 decision: v25 source72-neighborhood risk-family rows still favor safe fixed H12; adaptivity is not justified for this stress-pool family, so prioritize scenario/comparison design or fixed-H12-primary confirmation. | Follow the decision rule: terminal-risk/value refit or training only if fresh risk/selector failure warrants it; otherwise scenario/comparison/fixed-H confirmation. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; partially addressed | v25 charged measured preoutcome feature+selector overhead mean 0.002596006s over 16 states. | Still require full closed-loop whole-decision timing for any final speed claim. |
| `A12_registry_backup_schema_contract` | accepted; active | v25 wrote new simulation/raw/docs/registry and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_RUN_20260930T0210Z.json`. | Require verified external backup before further unique science. |

## Follow-up through v26 fixed-H12-primary diagnostic/protocol freeze

Updated by GPT-5.5 executor at `2026-09-30T02:15:08.178820+00:00`. v26 is an offline audit/protocol freeze only; validation64 and sealed test remain closed.

| linked recommendation(s) | disposition after v26 diagnostic | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; acted | Fresh v21+v23+v25 aggregate fixed H12: rows=40, bad=0, decision saving=17.84%, pass5=True; v19 opened boundary still has bad=0. | Froze fixed-H12-primary v26 confirmation protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26_preoutcome_fixed_H12_primary_confirmation_20260930T0225Z.json`; fixed H12 is primary, selector/oracle are secondary. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; quantified | Fresh zero-bad count is 0/40; Wilson 95% upper bad-rate bound 8.76%. | Do not treat zero fresh bad as population proof; v26 remains development/stress-pool confirmation, not validation/test. |
| `A11_training_failure_modes_need_separation` | accepted; training deferred with falsifiable gate | Current fresh evidence favors fixed H12, while adaptive value only appears in opened v19. | If v26 fixed H12 fails or selector/oracle uniquely helps, pivot to terminal-risk/value refit or bounded training; if fixed H12 passes, prioritize scenario-opportunity redesign/negative adaptive-opportunity conclusion. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; still open | v26 ran no deployment timing; v23/v25 selector extracts remain development only. | Future v26 run must charge measured selector overhead and full whole-decision timing; no final speed claim. |
| `A12_registry_backup_schema_contract` | accepted; active | Materialized post-v25 verified backup proof `research_artifacts/aws_backup_proofs/backup_proof_20260930T020708_from_supervisor_context_after_v25.json` and wrote v26 backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_FIXED_H12_PRIMARY_DIAGNOSTIC_V26_20260930T0225Z.json`. | Require verified backup covering v26 source/protocol/results/docs before simulations. |

## Correction after v26b fixed-H12 audit repair

Updated by GPT-5.5 executor at `2026-09-30T02:22:19.269520+00:00`. This correction supersedes the v26 row-count/risk-bound entries only; it does not overwrite v26 and does not open validation64 or sealed test.

| linked recommendation(s) | disposition after v26b repair | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; v26 evidence corrected | v26 parser bug found: v19 and v21 were counted as zero. Corrected fresh v21+v23+v25 fixed H12: rows=56, bad=0, decision saving=16.96%, pass5=True. Corrected all opened including v19: rows=80, bad=3, pass5=False. | Do not run fixed-H12-primary simulation from the unamended v26 evidence. Use repaired protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_corrected_fixed_H12_primary_confirmation_amendment_20260930T0235Z.json`; fixed H12 remains primary on fresh rows, while v19 remains adaptive-opportunity counterevidence. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; quantified with corrected denominator | Corrected fresh zero-bad count 0/56; Wilson upper 6.42%. | Still development/stress-pool only; no population or final-test claim. |
| `A11_training_failure_modes_need_separation` | accepted; next gate unchanged but now evidence-correct | Fresh corrected rows still favor fixed H12, while all-opened rows fail fixed H12 because of v19. | After backup, either run the repaired fixed-H12-primary confirmation or, if fixed H12 fails there, pivot to terminal-risk/value refit or bounded training. |
| `A12_registry_backup_schema_contract` | accepted; active | v26b wrote repair artifacts and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V26B_REPAIR_20260930T0235Z.json`. | Require verified backup before more unique science. |

## Follow-up through v27 fixed-H12-primary preflight

Updated by GPT-5.5 executor at `2026-09-30T02:32:42.975997+00:00`. v27 preflight is metadata-only and does not open validation64 or sealed test.

| linked recommendation(s) | disposition after v27 preflight | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A12_registry_backup_schema_contract` | accepted; active blocker | v27 preflight found no adequate verified backup after the v26b repair time `2026-09-30T02:22:19.269520+00:00`. Latest materialized proof intentionally predates v26b. | Do not run the 60-episode fixed-H12 confirmation until a verified post-v26b backup covers v26b/v27 artifacts. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; ready but gated | v26b protocol integrity checks passed; selected indices preserved `[107, 53, 217, 190, 134, 30, 94, 108, 148, 242, 151, 195]`; budget remains 60 development episodes / 9000 control-step cap. | After backup, run or implement/run the repaired fixed-H12-primary confirmation with fixed H12 as primary baseline. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | v27 did not add outcomes; it carries corrected fresh 0/56 H12-bad and all-opened 3/80 H12-bad. | Keep all claims development/stress-pool scoped. |

## Follow-up through v27 fixed-H12-primary confirmation

Updated by GPT-5.5 executor at `2026-09-30T02:43:28.789811+00:00`. v27 did not access validation64 or sealed test and performed no training/refit.

| linked recommendation(s) | disposition after v27 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; acted with fixed H12 primary | v27 fixed H12 save 19.62%, bad 0, physical gate True, pass5 True; selector save 3.72%, H counts {'12': 6, '15': 18}, bad 0. | Use v27 decision rule: v27 fixed-H12-primary confirmation passes: fixed true H12 remains the stronger simple baseline on this fresh stress-bank batch; do not validate adaptive H12/H15 on this distribution without new scenario-opportunity evidence. |
| `A4_offline_selector_savings_exclude_online_selector_overhead` | accepted; partially addressed, still not final speed | v27 charged measured preoutcome feature+selector overhead mean 0.002572216s/p95 0.002870290s over 24 states and reports branch whole-decision/solver sums. | Still no final deployed speed claim without later closed-loop validation/final timing. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | v27 uses the v26/v26b stress-bank development selection; it is not validation64/population/final-test evidence. | Keep claims development-scoped and require fresh independent confirmation before final test. |
| `A11_training_failure_modes_need_separation` | accepted; next branch depends on fixed-H12 result | v27 states=24, H12-beneficial=22, H12-catastrophic=0. | If fixed H12 passes, prioritize scenario/comparison design; if failures recur and selector/oracle helps, pivot to terminal-risk/value refit or bounded training. |
| `A12_registry_backup_schema_contract` | accepted; active | v27 wrote new source/results/docs/state/registry and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_FIXED_H12_CONFIRMATION_20260930T0250Z.json`. | Require verified external backup before further unique science. |

## Postdiagnostic clarification after v27 fixed-H12-primary confirmation

Updated by GPT-5.5 executor at `2026-09-30T02:48:21.906171+00:00`. This is an analysis-only clarification of v27; no validation64/sealed-test access and no new simulation/training.

| linked recommendation(s) | disposition after postdiagnostic | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; sharpened | v27 fixed H12 has H15-referenced pass5=True (saving 19.62%, bad=0) but success-sensitive pass5=False because fixed H12 has 2 absolute unsafe rows; fixed H15 also has 2 unsafe rows and both-fail rows come from source_candidate_index [242]. | Future comparator tables must separate relative H12-vs-H15 cost/speed from absolute success/safety. Do not describe v27 as deployable success on all selected stress rows. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; still open | Combined fresh v21+v23+v25+v27 has 80 relative rows, H12_bad=0, decision saving=17.90%, but includes v27 hard both-fail rows and remains stress-pool development evidence. | No population or validation claim; use only to choose the next development intervention. |
| `A11_training_failure_modes_need_separation` | accepted; training still deferred but not indefinitely | v27 selector is too conservative (H counts {'12': 6, '15': 18}, save 3.72%) and fixed H12 dominates relative to H15 on feasible fresh rows; the current bottleneck is comparison/scenario opportunity and failure accounting rather than missing a richer selector on these rows. | After verified backup, freeze a success-aware scenario/comparison audit/protocol that separates feasible states, both-fail hard cases, and v19 H12-negative cases. Pivot to terminal-risk/value refit only if that protocol shows adaptive opportunity not captured by fixed H12. |
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | new; accepted | Source242 contributes two v27 branch states where both H12 and H15 run to 150 steps with physical costs >20k; relative physical delta is 0, so a purely H15-referenced gate can pass while absolute success fails. | Add success-sensitive pass/failure accounting to all future H12-primary comparison summaries and reviewer handoff. |
| `A12_registry_backup_schema_contract` | accepted; active | This postdiagnostic wrote `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_POSTDIAGNOSTIC_20260930T0300Z.json` and updated docs/state/registry/response log. | Require verified external backup covering v27 and this postdiagnostic before any further unique science. |

<!-- vehicle-success-aware-scenario-comparison-audit-v28-20260930T0315Z -->
## Follow-up through v28 success-aware scenario/comparison audit

Updated by GPT-5.5 executor at `2026-09-30T03:04:16.008112+00:00`. v28 is analysis-only over already-opened development evidence; validation64 and sealed test remained closed.

| linked recommendation(s) | disposition after v28 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted; operationalized | v28 partitions all opened collapsed H12/H15 evidence: fresh v21+v23+v25+v27 has 80 rows with both-fail=2 from source242 and H12-only failures=0; opened v19 has H12-only failures=3 collapsed / 6 repeat rows with H15 safe. | Future gates must report absolute success/safety before H15-referenced physical/timing deltas. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; next comparator frozen | Fixed H12 dominates fresh feasible rows but fails on v19 H12-only cases; source242 both-fail rows may require longer H or scenario stratification. | Froze `research_artifacts/aws_protocols/vehicle_true_variable_horizon_success_aware_scenario_comparison_audit_v28_frozen_success_aware_followup_20260930T0315Z.json`: bounded H12/H15/H25/H35 identical-state feasibility probe before validation. |
| `A11_training_failure_modes_need_separation` | accepted; training deferred with a discriminating trigger | Current evidence does not show that a richer selector is the bottleneck on fresh feasible rows; it shows mixed scenario/comparison failure modes. | Pivot to terminal-risk/value refit only if the v29 feasibility/feature audit shows separable H12-only opportunity not captured by fixed H12/longer fixed-H baselines. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | All v28 inputs are opened development/stress-pool artifacts, not validation64/population/final-test evidence. | Keep scope development-only; require fresh independent confirmation before final testing. |
| `A12_registry_backup_schema_contract` | accepted; active | v28 wrote new source/audit/protocol/docs/state/registry and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V28_SUCCESS_AWARE_SCENARIO_COMPARISON_AUDIT_20260930T0315Z.json`. | Require verified external backup covering v27 postdiagnostic and v28 before v29 simulation or other unique science. |

## Follow-up through v29 success-aware longer-H feasibility probe

Updated by GPT-5.5 executor at `2026-09-30T03:22:13.345024+00:00`. v29 did not access validation64 or sealed test and performed no training/refit.

| linked recommendation(s) | disposition after v29 | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted; directly tested | v29 ran H12/H15/H25/H35 on source242 both-fail states and records absolute safe/no-safe rows before timing aggregation. Source242 longer-H rescues: 2. | Continue success-sensitive accounting; failed rows are not speed evidence. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; expanded fixed-H comparison | v29 includes H25/H35 matched fixed-H terminals where available plus H12/H15, on identical saved states. | If longer H rescues or dominates, broaden fixed-H comparator/scenario design before any adaptive validation. |
| `A11_training_failure_modes_need_separation` | accepted; conditional | v19 H12-only failures reproduced=3; controls fastest H12=6. | Pivot to terminal-risk/value refit only if per-state v29 evidence shows separable adaptive opportunity not dominated by fixed longer H. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | v29 is selected opened development/stress-pool evidence only. | Require fresh independent confirmation before validation/final-test claims. |
| `A12_registry_backup_schema_contract` | accepted; active | v29 wrote new source/results/docs/state/registry and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V29_SUCCESS_AWARE_LONGER_H_PROBE_20260930T0340Z.json`. | Require verified backup before more unique simulations/refits/validation. |

<!-- vehicle-three-way-selector-feature-audit-v30b-fast-20260930T0410Z -->
## Follow-up through v30b three-way selector feature/separability audit

Updated by GPT-5.5 executor at `2026-09-30T03:46:34.079983+00:00`. v30b is analysis-only over v29 and did not access validation64 or sealed test. It replaces the timed-out v30 implementation-performance failure.

| linked recommendation(s) | disposition after v30b | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; broadened to H12/H15/H25/H35 evidence | v30b compares fixed H12/H15/H25/H35 and oracle/feature H12/H15/H35 rules on v29 states; success-sensitive bad rows fixed H12=5, fixed H15=2, fixed H25=4, fixed H35=4. | Any further adaptive claim must compare against fixed H35 and fixed H12/H15, and must separate absolute safety from relative timing. |
| `A11_training_failure_modes_need_separation` | accepted; refit/training gate updated | v30b two-threshold feature audit: in-sample bad=0; leave-one-out bad=1; LOO saving vs fixed H35=48.30% conditional on zero bad rows. | Use this outcome to decide whether a fresh triage-selector confirmation, terminal-risk/value refit, or scenario redesign is next. |
| `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass` | accepted; preserved | Source242 rows require H35 in v29/v30b accounting; failed H12/H15/H25 timing is excluded from speed claims. | Continue absolute success-sensitive accounting. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | v30b uses only the 11 opened v29 states and selects rules after v29 outcomes. | Development-only mechanism audit; require fresh independent confirmation before validation/final claims. |
| `A12_registry_backup_schema_contract` | accepted; active | v30b wrote analysis/docs/state/registry and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_THREE_WAY_SELECTOR_FEATURE_AUDIT_20260930T0410Z.json`. | Require verified backup before more unique simulation/refit/validation. |

<!-- vehicle-v30b-feature-stability-postdiagnostic-20260930T0355Z -->
## Follow-up v30b feature-stability postdiagnostic

Updated by GPT-5.5 executor at `2026-09-30T03:51:36.708427+00:00`. This is a numerical postdiagnostic over already-opened v29/v30b outputs; validation64 and sealed test remained closed.

| linked recommendation(s) | disposition after postdiagnostic | verified evidence | action / outcome / next step |
|---|---|---|---|
| `A11_training_failure_modes_need_separation` | accepted; sharpened for Astra decision | Same-opened-row margin rule has bad=0 and saving=50.57% vs fixed H35, while v30b LOO has bad=1 because of a v19 H12-risk heldout. H15 condition features vary across LOO folds: `{'abs_obs_07': 8, 'abs_obs_03': 2, 'abs_obs_04': 1}`. | Evidence suggests tiny-sample model-selection instability despite observable in-sample signal; await Astra to choose fresh label acquisition vs terminal-risk/value refit/training. |
| `A6_strong_fixed_H_and_terminal_opportunity_not_closed` | accepted; no final comparator claim | Fixed-H35 is safe-successful but has physical-excess bad rows; oracle triage bad=0; all are opened development rows with possible horizon/terminal confounds from v29. | Do not validate selector; preserve fixed H12/H15/H25/H35 comparators and terminal-confound caveat. |
| `A7_targeted_risk_banks_are_not_population_estimates` / `A8_zero_catastrophe_small_sample_model_selection_risk` | accepted; unchanged | Only 11 opened v29 states; thresholds use outcome-informed labels. | Require fresh independent confirmation before validation/final claims. |
| `A12_registry_backup_schema_contract` | accepted; active | Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_FEATURE_STABILITY_POSTDIAGNOSTIC_20260930T0355Z.json` written. | Require verified backup before unique simulations/refits/validation. |
