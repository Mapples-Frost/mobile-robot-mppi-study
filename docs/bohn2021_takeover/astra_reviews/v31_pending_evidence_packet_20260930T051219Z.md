# v31 pending-Astra evidence packet

UTC: `2026-09-30T05:12:19.416907+00:00`. This is branch-neutral metadata analysis only over already-opened development artifacts. It ran **0** simulations, **0** control steps, **0** selector refits, **0** training/gradient steps, opened no validation64 bank and accessed no sealed test.

## Status-line values for next report
- Elapsed service lifetime since `2026-09-26T10:55:29.419331Z`: `3d 18h 16m 49.998s`.
- Cumulative server API total_tokens from research.sqlite: `unknown (research.sqlite not found in checked locations)`; desktop conversation tokens excluded.

## Astra gate
- Current `NEXT_REVIEW_REQUEST.json` request_id: `v31-source-coverage-budget-bounds-20260930T044819Z`.
- Expected v31 request_id: `v31-source-coverage-budget-bounds-20260930T044819Z`.
- `ANALYSIS_READY.json` exists: `False`; matches/supersedes current: `False`; report: `None`.
- `LATEST.md` still points to the old 2026-09-29 report: `True`. That report predates v29/v30b/v31 and is not current analysis of the source-coverage finding.

## Evidence inspected
- v29 opened-development identical-state probe: {'absolute_no_safe_horizon_states': 0, 'episodes': 44, 'horizon_counts': {'12': 11, '15': 11, '25': 11, '35': 11}, 'safe_control_rows_fastest_H12': 6, 'source242_bothfail_rows_rescued_by_H25_or_H35': 2, 'states': 11, 'v19_h12_only_failure_rows_reproduced': 3}
- v30b opened-row oracle H12/H15/H35 metric: bad_count `0`, h_counts `{'12': 6, '15': 3, '35': 2}`, decision_sum_s `28.177247803483624`, decision_saving_vs_fixed_H35 `None`.
- v31 source-family cluster headline: rows `11`, families `5`, row-level LOO bad `0`, grouped LOGO bad `5`, label family counts `{'12': 3, '15': 1, '35': 1}`.
- v31 source-budget lower bound: current counts `{'12': 3, '15': 1, '35': 1}`; basic additional families `{'12': 0, '15': 1, '35': 1}`; three-source additional families `{'12': 0, '15': 2, '35': 2}`.

## Consistency checks
{
  "all_required_input_files_exist": true,
  "expected_current_counts_H12_3_H15_1_H35_1": true,
  "v30b_oracle_h_counts_match_expected_rows": true,
  "v31_cluster_and_source_counts_match": true
}

## Backup gate
- Latest verified backup found by local proof scan: `{'path': 'research_artifacts/aws_backup_proofs/backup_proof_20260930T050506_from_user_context_after_v31_0459_gate_recheck.json', 'time': '2026-09-30T05:05:06.407045+00:00', 'verified_like': True, 'status': 'verified', 'remaining_changed_files': 0, 'commit': 'f952a812d377ef5ac183a6757878d013231a6cdf', 'package_sha256': 'd01979be2f6629c04820bcfe362e7809df8c7112a9b2c5c7313796d02d7b020a'}`.
- 05:07 metadata request plus 05:09 note fully covered by that backup: `False`.
- Files not covered by latest verified backup count: `16`.
- New backup request for this packet and pending files: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_PENDING_EVIDENCE_PACKET_20260930T051219Z.json`.

## Branch-neutral carry-forward
- Treat v29/v30b/v31 as development/opened-row diagnostics only, not validation/final evidence and not a reproduction claim.
- Do not deploy or validate the v30b row-level two-threshold selector: non-default H15/H35 labels each have only one independent source family.
- Same-family densification cannot repair grouped-CV missing-class structure.
- If Astra selects fresh source-independent acquisition, use the already computed lower-bound budgets; if Astra selects value/refit/training or scenario/comparison redesign, freeze a versioned protocol before any new simulation/refit/training.

## Next action
No matching Astra report is available; do not choose source-label acquisition, value/refit/training, or scenario/comparison redesign independently. Continue only reversible integrity/preparation until Astra responds or an already-frozen authorized action exists. Latest metadata outputs are not externally backed up, so unique simulation/refit/training/validation/final-test work remains blocked pending the new backup request.
