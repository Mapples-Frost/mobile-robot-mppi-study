# Server-to-local Opus handoff for Bøhn 2021 adaptive MPC horizon research

Captured handoff context: 2026-10-01. The autonomous server research service is frozen for this handoff. No physics/training/validation/test experiment should be run as part of this handoff. Packaging/upload/restore is to be performed by the outer operator; this report does **not** claim that GitHub upload, local restore, or local execution has already succeeded.

## Executive status

No reproduction success has been established. The project remains an incomplete vehicle-first Bøhn 2021 adaptive MPC prediction-horizon reproduction/improvement effort. The strongest verified current vehicle evidence is negative for the existing IMPROVED latency-tree/gated selectors: two learned seeds are behaviorally fixed H25, and the only switching seed is slower/worse and has a preserved case43 failure. Pendulum evidence is incomplete: pendulum_s0 has a completed historical latency-tree training artifact, pendulum_s1 is interrupted/stale, and pendulum_s2 is absent/unstarted in the latency-tree training inventory. Sealed/final test is not authorized for the current work.

The latest operational state is not scientific progress: after the backup release-rotation repair, backup recoverability is verified, but the source242 microcontinuation still failed before any solver or plant use because the legacy per-version backup gate rejected despite verified global backup. The progress watchdog explicitly reports stagnation: the recent two-hour window had zero measured solver calls, plant steps, training steps, validation episodes, or test episodes; last measured solver use was 2026-09-30T17:54:10Z and no plant/training transition has been recorded.

## Repository and frozen metadata

- Branch/capture metadata supplied by operator: `codex/bohn-aws-20260926`, source commit `cf654dcd43984cee9fb49a5bf32675cbbb685f00`, 417 registered runs, 327 complete, 86 failed, 4 interrupted, registered run wall time about 209,876.61 s. Dirty/untracked files include `.venv`, the handoff directory, docs figures, real robot obstacle kit files, and other docs artifacts.
- Local handoff locator files already exist under `docs/bohn2021_takeover/local_opus_handoff_20261001/`: `AUDIT_FACTS.json`, `REGISTERED_RUNS.json`, `CHECKPOINT_LOCATOR.json`, `EVIDENCE_LOCATOR.json`, `SERVER_FILE_INVENTORY.jsonl`, `WATCHDOG_AT_FREEZE.json`, `INTERRUPTED_ITERATION_POINTER.json`.
- I did not fully audit every locator entry in this final pass; treat those locator files as indices and verify specific evidence before relying on it.
- Latest backup status in supplied state reports verified at `2026-10-01T02:09:10.721129+00:00`, remaining_changed_files=0, commit `cf654dcd43984cee9fb49a5bf32675cbbb685f00`, package `20261001T020900_629e7f6f.tar.gz`, package SHA256 `e4f2ca3bf557b8214743d0324604ae87c7cbee65bf599ca027018d3678837df9`, release tag `bohn-aws-evidence-20260926-r001`. This is recoverability/operations evidence, not science.

## Working language and roles

All new working text between agents must be English. Historical Chinese reports/source excerpts are evidence and should not be rewritten merely for language.

Current server roles before freeze: GPT-5.5 was active lead/executor in temporary user-authorized solo mode; Opus and Astra reviewers were paused by the user. Local Opus now takes over as the only active research agent after repository upload/restore by the operator. The paused server roles and API receipts do not constitute independent acceptance.

## ORIGINAL Bøhn 2021 reconstruction versus IMPROVED variants

Keep these separated.

### ORIGINAL / closest-to-paper SAC reconstruction

Evidence exists in historical artifacts, especially under `research_artifacts/bohn2021_reproduction_2026-09-17/results/` and source under `experiments/bohn2021_reproduction/`. I read only selected locator and audit files in this final cycle, not the full original result corpus. Therefore:

- `research_artifacts/bohn2021_reproduction_2026-09-17/results/results_audit.json` says a historical audit passed for fixed-H and RL runs for vehicle and pendulum, with checks including 15000 transitions/14901 updates, final test traces, recomputed cost, horizon/proxy cost, and termination/solver-failure counts. This is historical evidence and must be rechecked from raw files before any publication claim.
- `research_artifacts/bohn2021_reproduction_2026-09-17/results/diagnosis_finalized.json` says historical diagnostic groups `refined` and `paper_defaults` were verified, 28 new training runs/420k transitions plus 26 old preserved runs/390k transitions, total 810k transitions, with a note that diagnostic interventions are separate from final comparison and no test-based checkpoint/seed selection. This is not current fresh sealed-test evidence.
- `research_artifacts/bohn2021_reproduction_2026-09-17/results/full/` contains directories/logs for `vehicle_rl_s0/s1/s2`, `pendulum_rl_s0/s1/s2`, and fixed-H runs. I only listed this directory; I did not audit the contents in this handoff pass.
- Historical final/test traces, if present, are already opened/contaminated for current decision-making and cannot be counted as a fresh sealed final test for a new claim.

Conclusion for ORIGINAL: **not reproduced yet under current acceptance criteria**. A valid original claim would require both tasks, all required seeds, exact method fidelity, accepted validation and a fresh sealed/final test under a frozen gate. That has not happened.

### IMPROVED variants

IMPROVED work includes the latency-tree protocol, safe-shortening, gated-horizon risk/actual-time reselection, true-variable-H experiments, terminal/objective diagnostics, and source242 microcontinuation engineering. These are modified methods and must not be labeled ORIGINAL SAC.

The main inherited IMPROVED protocol is `docs/protocols/bohn2021_latency_tree_2026-09-26.md`; recovery/migration rules are in `docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md`. The amendment explicitly states that WSL/AWS timing must not be mixed, behaviorally fixed trees are fixed-H comparators, validation64 and sealed test remain controlled, and the latency-tree is IMPROVED, not ORIGINAL SAC.

## Vehicle evidence summary

### Formal validation64 for IMPROVED latency-tree vehicle

The vehicle validation64 campaign completed all 12 shards, 2688 validation episodes and 236,348 represented control steps, with no sealed/final test access. This is validation-output/development model-selection evidence for an IMPROVED method, not final evidence.

Key aggregate from `research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/summary.md`:

- `learned_s0`: 64/64 success, 0 failures, 4967 steps, physical+constraint mean/episode 24.433, total mean 26.3732, decision mean 0.16884 s/step, horizons `{'25': 4967}`. Behaviorally fixed H25.
- `learned_s1`: 64/64 success, 0 failures, 4960 steps, physical+constraint mean/episode 18.7272, total mean 20.6647, decision mean 0.17276 s/step, horizons `{'25': 4960}`. Behaviorally fixed H25.
- `learned_s2`: 63/64 success, 1 failure, 5014 steps, physical+constraint mean/episode 628.572, total mean 630.595, decision mean 0.17714 s/step, horizons `{'25': 4605, '35': 409}`. It switches, but fails acceptance.
- Same-seed/adaptive gate diagnostics: s0/s1 fail because not adaptive; s2 fails because physical/control cost exceeds the fixed-H nomination and it does not achieve >=10% decision-time reduction.
- Strong fixed-H nominations in that aggregate favor matched-terminal H25 and independent-terminal seed0 H30, depending on comparator family. Any future comparison must keep those strong baselines and terminal-source distinctions.

Case43 is important. The aggregate and zero-resource later summaries preserve: learned_s2 case43 failed with 150 steps, physical_constraint_cost about 39125.9231, total about 39129.6831, horizon counts `{'25':149,'35':1}`; fixed H25/H35 comparators succeeded in 96 steps. This remains a preserved failure, not something to average away.

Zero-resource confirmation summaries read in this handoff:

- `research_artifacts/aws_diagnostics/vehicle_validation64_all_shards_pairwise_summary_solo_v0_20260930T225821Z/summary.md` confirms all-shard learned-vs-fixed metrics and case43 scans.
- `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_v3_receipt_repair_solo_v0c_20260930T231719Z/summary.md` confirms s0/s1 structural H25 collapse and s2 H35 use/failure; resources zero.

### Current gated-horizon policies

`research_artifacts/aws_diagnostics/vehicle_current_gated_horizon_training_audit_v1_receipt_repair_solo_v0_20261001T002636Z/summary.md` preserves that current vehicle policies are finite-search IMPROVED gated selectors, not ORIGINAL SAC and not the older latency tree:

- seed0 policy `h20_p1_g5`, gradient steps 0.
- seed1 policy `h15_p2_g5`, gradient steps 0.
- seed2 policy `h10_p0_g5`, gradient steps 0.
- Selection used mean raw cost with physical-cost gates and no measured wall-clock objective.

This audit was zero-resource and did not open validation bank/generator content.

### Source242 true-variable-H microcontinuation / pending engineering failure

This is the current immediate frontier and the current failure is engineering/backup/interface, not a scientific result.

Relevant source/plan paths actually read:

- Current inherited active plan: `docs/bohn2021_takeover/solo_gpt55/PLAN_READY.json` and `docs/bohn2021_takeover/solo_gpt55/solo_500adfc8b53031a4fa3722a3.execution_plan.json`.
- Existing wrapper: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0j_task_id_binding_repair.py`, script SHA256 in latest registry `3d41115e45e47cbdb6ed6431011c8ae2d2c8d66686abb9100b836c280b535dcf`.
- v0i raw TVP repair: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0i_env_tvp_format_repair.py`.
- v0h high-level env.step repair: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0h_highlevel_env_step_repair.py`.
- Legacy env sources read in prior/current cycle: `research_artifacts/bohn2021_reproduction_2026-09-17/sources/gym-horizon/gym_let_mpc/let_mpc.py`, `simulator.py`, `controllers.py`.

Verified recent artifacts:

- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_v0i_raw_tvp_unit_diagnostic_v0_20260930T214943Z/summary.md` passed zero-resource implementation checks: high-level horizon action present; raw list-of-dicts TVP restore writes `true`/`forecast` keys; this is not solver/control evidence.
- Latest actual registered source242 attempt: `20261001T014703_27a3c026`, `research_artifacts/aws_runs/20261001T014703_27a3c026/outcome_receipt.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34z2_source242_microcontinuation_v0j_task_id_binding_repair_20261001T014704Z/failed.json`, and `backup_dependency_failure.json`.
  - Outcome: engineering failure dependency, resources all zero.
  - Error: `post-S-TC2G external backup proof is not verified before solver/plant resources`.
  - It preserved previous repair evidence keys (high-level env-step applied, nested 2x1 payload inherited, raw TVP format restored, task-id binding repaired), but did not run arms.
- The latest compatible backup proof after release rotation exists: `research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_AFTER_RELEASE_ROTATION_PATCH_SOLO_V0B_20261001T013958Z.json` and `research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_FROM_SUPERVISOR_CONTEXT_20261001T014627Z.json`. Despite this, v0j's legacy gate rejected because it scans/version-gates differently.

Important: do **not** claim the top-level action shape is the proven remaining cause. `docs/bohn2021_takeover/PROGRESS_SUPERVISION_20261001.md` states that v0h/v0i already construct `np.asarray([float(h)], dtype=float)`, and a mock author-step test reaches a controller boundary. The remaining float-not-subscriptable issue can be deeper in controller/TVP/observation path. The decisive missing evidence is a full sanitized traceback from a bounded real env.step attempt after a correct content-based backup gate.

### Other IMPROVED true-variable-H / terminal-objective diagnostics

Many v33/v34 terminal/objective diagnostics exist. I read selected summaries/state only enough to understand current history; they are not the immediate next branch. Their key role is background: they show terminal/H interactions, objective reconstruction repairs, epsilon/slack provenance, and numerous operational gates, but they are opened-development diagnostics, not validation/final evidence. Do not rerun or rely on them without targeted reason.

## Pendulum evidence summary

Actual pendulum inventory read: `research_artifacts/aws_diagnostics/pendulum_inventory_metadata/summary.md`.

- `pendulum_s0`: exists and is a completed training artifact. Selected `g1_c10`; tree leaves `[5,20,25,30]`; 66 completed markers; generations and selection directories complete. Many recorded PID attempt files are dead because the old PID is stale, but root completion marker is present.
- `pendulum_s1`: exists but is `started_partial_or_interrupted`; no root completed marker; threshold reference progress only `episodes=1`, `expected=12`, `steps=13`, dead PID 1694525, tmp file present.
- `pendulum_s2`: directory absent at train root; treated as absent/unstarted unless future evidence contradicts.

This inventory is metadata-only and not a validation result, sealed-test result, or reproduced control-performance evidence. Therefore whole two-task reproduction is unresolved.

Historical original-style pendulum/vehicle directories exist under `research_artifacts/bohn2021_reproduction_2026-09-17/results/full/`; I listed them but did not audit the raw contents in this final handoff. Treat any claims from those folders as unaudited until checked.

## Backups and current operational state

The backup blocker was real and consumed substantial engineering time. The sequence:

1. Backup status diagnostics showed HTTP 422 / apparent GitHub asset cap, then timeout/write failures. Evidence includes `research_artifacts/aws_diagnostics/backup_failure_status_capture_s_tc2h8_current_status_v0c_20260930T221311Z/summary.md` and related current-status summaries.
2. `experiments/bohn2021_aws/backup_release_rotation_patch_apply_solo_v0b_static_check_repair.py` applied a source patch to `scripts/research_service/backup.py` without running the uploader; `research_artifacts/aws_runs/20261001T013008_2c0202f6/outcome_receipt.json` confirms zero scientific resources and no validation/test access. After hash: `fb5a21945ab8441007ffcbb34b456cd59053e93243e6532d98bf2af493bc3d8e`.
3. `research_artifacts/aws_diagnostics/backup_post_rotation_status_recheck_solo_v0b_gate_key_repair_20261001T013958Z/summary.md` and `research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_AFTER_RELEASE_ROTATION_PATCH_SOLO_V0B_20261001T013958Z.json` show adequate verified backup after the release-rotation patch.
4. Nevertheless, the v0j microcontinuation wrapper still failed at a legacy per-version backup gate. This is the known unproductive wrapper/gate repair loop. The intended shared fix is in `scripts/research_service/backup_contract.py`: content-based verification of immutable inputs against pushed Git/ref and verified release index, while treating mutable live logs as point-in-time evidence rather than clean-tree prerequisites.

Current operator-supplied `backup_status.json` reports verified at `2026-10-01T02:09:10Z`, remaining_changed_files=0, commit `cf654dcd...`, rotated release `bohn-aws-evidence-20260926-r001`. This operational fact enables future work only after local restore/packaging verification; it is not scientific progress.

## Test and validation exposure / contamination

- Current IMPROVED vehicle validation64 was intentionally opened and fully run for the latency-tree method: 12 shards, 2688 episodes. It is no longer fresh for new methods. It can be used as historical/development/validation-output evidence but not as a fresh independent validation for later variants.
- Sealed/final test for the current branch remains closed and unauthorized. No current `final_test_gate.json` is accepted. Do not open unused banks.
- Historical full results may include final/test traces for original-style work; they are already opened and contaminated for current final-test purposes. They cannot count as new sealed tests.
- Multiple development banks were opened across true-variable-H/safe-shortening/risk-reselection work. Treat them as development/model-selection data only.

## Training versus engineering time

- Historical original-style audits report substantial training artifacts (e.g. `diagnosis_finalized.json` says 810k total preserved/new transitions), but this final pass did not independently validate the entire corpus.
- The AWS short SAC feasibility probe (`docs/bohn2021_takeover/TRAINING_FEASIBILITY.md`) is engineering-only: vehicle 200 steps/101 updates in 38.9 s, pendulum 200 steps/101 updates in 21.8 s, peak RSS ~417.6 MiB. It is not a training-performance or control-success result.
- The IMPROVED latency-tree vehicle validation64 campaign consumed real validation rollouts (2688 episodes) and produced negative candidate evidence.
- Recent two-hour service activity is mostly metadata, backup repair, wrapper gates, and pre-resource failures. Watchdog at freeze says zero measured solver, plant, training, validation, or test resources in the two-hour window. Service/API activity is not scientific progress.
- Operator metadata reports model usage: GPT-5.5 6278 calls / 411,816,463 recorded tokens; GPT-6-Astra 380 calls / 28,318,555 tokens; Claude Opus 417 calls / 40,966,296 tokens. These are operational costs, not evidence.

## Environment constraints and authoritative entrypoints

- Legacy TF1/MPC runtime: `/home/mapples/.local/share/bohn2021-python37/bin/python`. Use this for controller/plant/TF1 rollouts if experiments are later allowed.
- Modern interpreter is for metadata/offline audits only.
- Hardware policy on server: t3a.medium, one experiment at a time, no IAM/EC2/EBS/network changes, no parallel training. Local Opus should adapt to local machine but preserve split/budget/acceptance rules.
- Source roots and snapshots:
  - Current repo branch: `codex/bohn-aws-20260926`.
  - Historical WSL frozen artifacts: `research_artifacts/bohn2021_reproduction_2026-09-17/`.
  - IMPROVED AWS scripts: `experiments/bohn2021_aws/`.
  - Original/reconstruction scripts: `experiments/bohn2021_reproduction/`.
- Key protocol files: `docs/protocols/bohn2021_latency_tree_2026-09-26.md`, `docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md`, `REPRODUCTION_PROTOCOL.md`.
- Current source242 plan context: `docs/bohn2021_takeover/solo_gpt55/solo_500adfc8b53031a4fa3722a3.execution_plan.json`. It has already produced a failed attempt and max_attempts=1, so it is context, not necessarily relaunch authorization. The likely successor should use request id `execution-result:20261001T014703_27a3c026` and replace per-wrapper backup gates with `backup_contract.verify`.

## Confirmed causes versus hypotheses

Confirmed:

- The current latency-tree vehicle candidate does not meet adaptive acceptance: s0/s1 H25-only, s2 H25/H35 but failed case43 and is worse/slower than fixed H25.
- Pendulum latency-tree training inventory is incomplete: s0 complete, s1 interrupted, s2 absent.
- Modern interpreter cannot run TF1 formal shard runner; the shard02 modern attempt failed with no episodes/control steps and was archived.
- Backup release rotation fixed external backup recoverability at the service level.
- The latest v0j source242 attempt failed before resources at a legacy backup dependency gate; no scientific outcome.
- Prior v0g solved one first decision per H but did not advance plant; prior wrappers created a versioned gate/repair loop.

Likely but not confirmed:

- The remaining source242 env.step failure is deeper than top-level action shape, likely in controller/TVP/observation path. Need full sanitized traceback after content-based backup verification.
- True-variable-H methods may still be scientifically useful, but current evidence is opened-development and terminal/objective-confounded; no deployable generalization is established.
- Scenario/reward/training/comparison causes all remain plausible. Do not reduce diagnosis to action-shape or backup alone.

## Obsolete or duplicated code paths to avoid

- Do not keep adding v0k/v0l-style backup wrapper gates that scan live logs or named proof files. Use `scripts/research_service/backup_contract.py` for immutable-input recoverability if future server work resumes.
- Treat `v0` through `v0j` source242 wrappers as historical failure evidence, not as a clean codebase. Preserve their receipts but consolidate before new real measurement.
- Old Opus/Astra plans under `docs/bohn2021_takeover/opus_lead/` and `docs/bohn2021_takeover/astra_reviews/` are mostly historical and reviewers are paused; do not treat them as active local authority unless the user reinstates them.
- Do not rerun broad validation64 summaries, learned-collapse receipt repairs, or backup status audits unless evidence mismatch appears.
- Do not use modern interpreter for TF1 plant/controller rollouts.

## Recommended next actions for local Opus, bounded and prioritized

1. **Verify local restore/readability only.** Confirm Git commit/dirty state, locator files, and key artifact hashes. Do not run physics/training/validation/test during this verification.
2. **Write a successor design note for source242 microcontinuation** using current evidence, not a new broad audit. The successor should:
   - cite `execution-result:20261001T014703_27a3c026`;
   - use shared `backup_contract.verify` or local equivalent for immutable inputs;
   - preserve prior failure evidence;
   - not require mutable logs to be Git-clean;
   - keep horizons H12/H15/H35, source242 goal endpoint[61], V15_shared terminal, no training/refit, no validation64, no sealed/final test;
   - capture full sanitized stack traces for every env/controller/TVP exception;
   - count solver calls and plant steps exactly.
3. **If experiments are later authorized locally**, run only the smallest source242 diagnostic that can either advance at least one plant step for each H or provide the full traceback. Do not use old validation64/sealed test.
4. **If source242 plant transitions are obtained**, analyze local costs/safety/solver/timing versus fixed H baselines before any training. Strong fixed-H H25/H30/H35 baselines and terminal-source confounds must remain explicit.
5. **If source242 still fails pre-transition**, repair the exact traceback, not a guessed top-level action shape.
6. **Do not claim reproduction or improvement** until original acceptance gates are satisfied, including both tasks and fresh sealed testing under explicit authorization.

## Read-only verification commands for local Opus

Run only after restore, as read-only checks:

```bash
git rev-parse HEAD
git status --short
python -m json.tool docs/bohn2021_takeover/local_opus_handoff_20261001/AUDIT_FACTS.json >/dev/null
python -m json.tool docs/bohn2021_takeover/local_opus_handoff_20261001/REGISTERED_RUNS.json >/dev/null
python -m json.tool research_artifacts/aws_runs/20261001T014703_27a3c026/outcome_receipt.json >/dev/null
python -m json.tool research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_AFTER_RELEASE_ROTATION_PATCH_SOLO_V0B_20261001T013958Z.json >/dev/null
python -m json.tool research_artifacts/aws_backup_proofs/BACKUP_VERIFIED_FROM_SUPERVISOR_CONTEXT_20261001T014627Z.json >/dev/null
sha256sum scripts/research_service/backup_contract.py scripts/research_service/backup.py
sed -n '1,220p' docs/bohn2021_takeover/PROGRESS_SUPERVISION_20261001.md
sed -n '1,220p' research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/summary.md
sed -n '1,180p' research_artifacts/aws_diagnostics/pendulum_inventory_metadata/summary.md
```

Avoid opening unused validation/test banks during verification.

## Handoff limitations

This final handoff used actual files read in the repository plus the operator-provided verified audit metadata, but it did not fully audit every one of the 151,997 files, every checkpoint, every original SAC run, or every generated figure/doc. Claims above are bounded to evidence paths cited here. The `EVIDENCE_LOCATOR.json`, `CHECKPOINT_LOCATOR.json`, and `SERVER_FILE_INVENTORY.jsonl` are large indices, not proof of completeness by themselves. Packaging/upload/restore is not verified by this report. No new experiment ran during this handoff, and the server remains frozen for research execution.

Final conclusion: **no reproduction success yet**. Vehicle IMPROVED evidence is negative for the current selector and blocked by pending engineering diagnosis for source242. Pendulum is incomplete. Historic opened tests are contaminated for current final acceptance. The next useful scientific/engineering work is a bounded, backup-contract-based source242 microcontinuation traceback/transition diagnostic, after local restore and explicit authorization.