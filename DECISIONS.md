# 决策记录

2026-09-26：优先审计 D 盘，但采用 WSL 已存在的作者复现资产作为服务器科研输入。证据：D相关25文件为早期原型；WSL含数十个实验组、多seed模型、原始轨迹和协议。两个工作树保留，既有修改不重置。避免从原型重复昂贵训练。

2026-09-26：复制冻结源码和原始绝对路径兼容链接，不修改已登记源码。新部署文件单列。source SHA、dirty patch 与迁移后的SHA重新登记，不能把新快照提交当原实验提交。

2026-09-26：保留四个完成浅树模型与中断训练，但不静默拼接两台主机的测时。恢复策略先基于源码和未完成边界审计登记；可能需重新完整计量参与排名的配对块，重复预算单列。

2026-09-26：长期控制器只允许gpt-5.5/xhigh，单实验串行，4h wall上限、3.1GB服务内存、48次API/日、每轮6次、150万报告token/日。预算耗尽等待下一个UTC日，不能换模型。连续失败进入诊断。服务器重启由systemd恢复。

2026-09-26追加：服务器已逐文件核验vehicle seed1训练内复选10个条件、160回合、12640控制步，3/4树候选行为始终H25，另1个使用H25/H45且成本更高，最终按冻结规则选择fixed。该保存策略不可能满足逐seed回合内自适应门槛；这不是独立test结论。优先诊断候选退化/覆盖和测时噪声，避免仅完成剩余长训练就误称成功。未读取validation/test结果。



2026-09-26用户覆盖：取消所有API调用、token、API费用日上限，去掉本地单请求输出token上限。仅使用GPT-5.5/xhigh；每轮最多12次调用后保存状态，5秒后继续；保留实验公平预算、单任务执行、实验超时、失败退避、资源限制。旧48次/日与150万token/日规则作废。

<!-- latency-tree-recovery-migration-amendment-20260926 -->
## 2026-09-26 latency-tree recovery/migration amendment

Decision: freeze a recovery/migration amendment before any resumed timing-sensitive latency-tree work. The historical preregistration is preserved unchanged. WSL and AWS wall-time measurements must not be mixed for ranking; any timing-influenced selection requires a whole same-host paired AWS block. Behaviorally fixed trees are fixed-H comparators, not adaptive policies. Adaptive claims now require actually used multiple horizons plus inherited safety/non-inferiority gates and paired cost or same-host timing benefit. The current vehicle training-selection evidence is development-only: seed0 is timing-noise-susceptible fixed-H25 behavior, seed1 is fixed, and only seed2 is a switching candidate requiring independent validation.

<!-- pendulum-inventory-interpretation-20260926 -->
## 2026-09-26 pendulum inventory finalized

UTC: 2026-09-26T12:23:05.106778+00:00. Metadata-only inventory `experiments/bohn2021_aws/pendulum_inventory_metadata.py` completed with no simulations, no validation reads, and no sealed-test reads. pendulum_s0 is a completed training artifact (selected `g1_c10`, 66 completed markers). pendulum_s1 is a stale/interrupted threshold-reference run (progress {'episodes': 1, 'expected': 12, 'pid': 1694525, 'steps': 13}, dead PIDs [1694525], tmp files 1). pendulum_s2 is absent/unstarted at train root. This is not control-performance evidence. The partial WSL pendulum_s1 timing-sensitive work must be counted as interrupted budget and must not be spliced into AWS timing objectives. Final test remains sealed/unauthorized; validation64 remains unopened for post-amendment model selection.

Decision: treat pendulum_s1 as failed/interrupted historical work and pendulum_s2 as unstarted unless future metadata contradicts this. Formal all-seed pendulum latency-tree evidence requires an AWS-only fresh recovery block or an explicitly vehicle-only development scope; behaviorally fixed trees remain fixed-H comparators.

<!-- post-amendment-vehicle-freeze-diagnostic-20260926 -->
## 2026-09-26 post-amendment vehicle freeze diagnostic

UTC: 2026-09-26T12:34:18.073579+00:00. Metadata-only diagnostic completed with no simulations, no validation64 content/outcome read, and no sealed-test read. Vehicle learned candidates s0/s1/s2 and fixed-H comparator inventory were hashed. Next actual experiment is frozen as a non-formal AWS-only vehicle smoke paired timing/control block versus fixed H25; formal validation remains unopened and whole two-task validation remains blocked by pendulum_s1/s2 recovery. Artifacts: `research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/raw.json`, `research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/summary.md`, `research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/vehicle_development_timing_freeze.json`. New artifacts require backup before unique formal evidence accumulates.

<!-- vehicle-development-smoke-pairing-20260926 -->
## 2026-09-26 vehicle development smoke pairing

UTC: 2026-09-26T12:51:06.985559+00:00. Non-formal AWS-only vehicle smoke completed on `vehicle_smoke_bank` with 24 episodes and 1764 control steps. Replay passed=True; validation_accessed=false; test_accessed=false. Use only for engineering readiness/timing-boundary checks, not validation/model selection. Artifacts: `research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing/raw.json`, `research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing/summary.md`. New artifacts require backup before formal evidence.

<!-- vehicle-smoke-artifact-digest-20260926 -->
## 2026-09-26 vehicle smoke artifact digest

UTC: 2026-09-26T12:56:46.885426+00:00. Metadata-only audit of the non-formal AWS vehicle smoke outputs completed. Top-level completed hash check passed=True; episode completed hash check passed=True; independent replay passed=True with 12 pairs. Seed2 H35 trigger count=8 and intervals=[{'episode_index': 2, 'repeat': 0, 'case': 1, 'start_step': 14, 'end_step': 16, 'length': 3}, {'episode_index': 2, 'repeat': 0, 'case': 1, 'start_step': 31, 'end_step': 31, 'length': 1}, {'episode_index': 13, 'repeat': 1, 'case': 1, 'start_step': 14, 'end_step': 16, 'length': 3}, {'episode_index': 13, 'repeat': 1, 'case': 1, 'start_step': 31, 'end_step': 31, 'length': 1}]. No simulations were run; validation_accessed=false; test_accessed=false. This remains engineering/development evidence only, not model selection or reproduction evidence. Artifacts: `research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest/raw.json`, `research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest/summary.md`, `research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest/completed.json`. New digest artifacts require backup before formal evidence.

<!-- vehicle-validation-gate-freeze-20260926 -->
## 2026-09-26 vehicle validation gate freeze

UTC: 2026-09-26T13:05:41.673579+00:00. Metadata-only no-validation gate frozen at `research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json`. Validation bank content opened=false; sealed test content opened=false; simulations=0. Gate froze 42 unique rollout arms, 2688 planned validation episodes over case indices only, and 12 bounded shards. Vehicle learned s0 and s1 are preclassified as fixed/nonadaptive by structure; only s2 is structurally switching. External backup of this new gate is required before formal validation64 rollout; final test remains unauthorized.

<!-- vehicle-validation64-shard-runner-dryrun-20260926 -->
## 2026-09-26 vehicle validation64 shard runner dry-run

UTC: 2026-09-26T13:16:00.900755+00:00. New runner `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py` dry-run completed with validation_accessed=false, test_accessed=false, simulations=0. It verified the frozen gate, source/model/policy hashes, bank stat metadata without opening validation/test content, and formal-run backup-proof requirements. Artifacts: `research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/dry_run.json`, `research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/summary.md`. Runner and dry-run outputs now require external backup before any validation64 content access.

<!-- post-dryrun-backup-blocker-audit-20260926 -->
## 2026-09-26 post-dry-run backup blocker audit

UTC: 2026-09-26T13:21:03+00:00. Metadata-only blocker audit completed with no validation/test bank content opened and no simulations. The dry-run/gate/runner hashes remain consistent, but no adequate post-dry-run external backup proof is present in the repository. A new backup request was written at `research_artifacts/aws_backup_proofs/REQUEST_POST_DRYRUN_BACKUP_20260926T132103.json`. Formal vehicle validation64 shard0 remains blocked until the supervisor provides a verified external backup proof with `backup_verified=true`, `remaining_changed_files=0`, commit, GitHub release asset/download SHA256, runner/gate hashes, and dry-run artifact hashes. Sealed test remains closed.

<!-- vehicle-validation64-shard-complete-20260926-shard00 -->
## 2026-09-26 vehicle validation64 shard 00

UTC: 2026-09-26T14:35:01.431574+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19832. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard00-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard00 audit

UTC: 2026-09-26T14:40:37+00:00. Post-run audit of formal shard00 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard00 has 224 episodes and 19832 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard00: s0 fixed H25 (408 steps), s1 fixed H25 (390 steps), s2 used H25/H35 ({'25': 287, '35': 60}) with 18 switches. This is only 1/12 validation evidence and not final model selection. New formal evidence requires external backup before shard01; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_20260926T144050.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard00-audit-v2-schema-repair-20260926 -->
## 2026-09-26 vehicle validation64 shard00 audit-v2 schema repair

UTC: 2026-09-26T14:45:06+00:00. Corrective audit-v2 classified audit-v1's nonzero exit as a registry schema false negative: the shard run registry records `commit_sha=523ec69d0986ebde3f10f24fd6dd8ad7df30e1f9` while audit-v1 required a `git_commit` field. Substantive v1 checks were complete and passed: completed-hash audit=True, episode trace/hash audit=True, aggregate replay=True, schedule mismatches=[]. validation_accessed=true because already-created shard00 results were read; sealed test accessed=false; simulations=0; training steps=0. Shard00 remains only 1/12 validation evidence, not model selection. External backup covering shard00, audit-v1, audit-v2, docs, registries and backup requests is required before shard01.

<!-- vehicle-validation64-shard-complete-20260926-shard01 -->
## 2026-09-26 vehicle validation64 shard 01

UTC: 2026-09-26T16:00:03.047684+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=20036. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard01/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard01/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard01-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard01 audit

UTC: 2026-09-26T16:04:04+00:00. Post-run audit of formal shard01 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard01 has 224 episodes and 20036 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard01: s0 fixed H25 (457 steps), s1 fixed H25 (420 steps), s2 used horizons {'25': 526, '35': 18} with 16 switches. This is only 2/12 validation evidence and not final model selection. New formal evidence requires external backup before shard02; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD01_AUDIT_20260926T160417.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard02-modern-runtime-failure-recovery-20260926 -->
## 2026-09-26 vehicle validation64 shard02 modern-runtime failure recovery

UTC: 2026-09-26T16:17:32+00:00. The planned shard02 run `20260926T161356_fb71c8d7` failed before any episode/control step because it was accidentally launched with the modern interpreter, where TensorFlow is unavailable (`ModuleNotFoundError: No module named 'tensorflow'`). The failed run had already opened the validation64 bank and wrote only `run_started.json` plus `schedule.json`; no `episodes/`, `progress.json`, `raw.json`, `summary.md`, or `completed.json` existed. Recovery archived the empty partial directory to `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/failed_shard02_20260926T161356_modern_tf_missing` so the frozen runner can later create `shard02` cleanly. Recovery itself reopened no validation bank content, opened no sealed test content, and ran 0 simulations / 0 control steps / 0 gradient steps. Legacy runtime import under `/home/mapples/.local/share/bohn2021-python37/bin/python` passed with TensorFlow `None`. A verified external backup covering the failure archive, recovery artifacts, docs, registry and backup request is required before retrying shard02 with the legacy interpreter.

<!-- vehicle-validation64-shard-complete-20260926-shard02 -->
## 2026-09-26 vehicle validation64 shard 02

UTC: 2026-09-26T17:32:27.856583+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19542. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard02/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard02/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard02-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard02 audit

UTC: 2026-09-26T17:37:53+00:00. Post-run audit of formal shard02 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard02 has 224 episodes and 19542 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard02: s0 fixed H25 (497 steps), s1 fixed H25 (403 steps), s2 used horizons {'25': 346, '35': 46} with 15 switches. Vehicle validation64 progress is now 3/12 completed shards with 672 completed episodes and 59410 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard03; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD02_AUDIT_20260926T173807.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard-complete-20260926-shard03 -->
## 2026-09-26 vehicle validation64 shard 03

UTC: 2026-09-26T18:59:14.269869+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19887. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard03-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard03 audit

UTC: 2026-09-26T19:04:50+00:00. Post-run audit of formal shard03 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard03 has 224 episodes and 19887 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard03: s0 fixed H25 (460 steps), s1 fixed H25 (550 steps), s2 used horizons {'25': 685, '35': 72} with 34 switches. Vehicle validation64 progress is now 4/12 completed shards with 896 completed episodes and 79297 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard04; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD03_AUDIT_20260926T190503.json`. Sealed test remains closed.

<!-- post-shard03-audit-backup-gate-state-20260926 -->
## 2026-09-26 post-shard03 audit backup gate state

UTC: 2026-09-26T19:13:04+00:00. Metadata-only state preservation after shard03 audit and user continuation. No simulations, no control steps, no gradient steps, no validation-bank reopen, and no sealed-test access/open/hash occurred in this action. Existing shard03 formal/audit metadata may be hashed for provenance only; this is not model selection, not final-test evidence, and not an ORIGINAL SAC result.

Backup gate result: shard04 formal validation is BLOCKED at this state check. Required proof time remains after `2026-09-26T19:05:30Z` and must cover shard03 formal outputs, shard03 audit outputs, audit run registry/stdout/stderr, docs/registry updates, backup requests, final addendum, and the blocker note. Latest local proof observed: `research_artifacts/aws_backup_proofs/backup_proof_20260926T190416_after_shard03_formal_before_audit.json` at `2026-09-26T19:04:16.474763+00:00`; adequate proofs found: 0.

Vehicle validation64 progress remains 4/12 completed shards, 896 formal validation episodes, 79297 formal validation control steps, plus the archived modern-interpreter shard02 failed attempt with 0 episodes/control steps. Sealed final test remains closed and unauthorized.

Next action if and only if a verified external backup proof satisfying the gate appears: run exactly one formal experiment, `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 4 --backup-proof <post-shard03-audit-proof> --i-accept-validation-access`, with the legacy interpreter; then audit shard04 before any further shard. Do not use the modern interpreter for the TF1 runner.

<!-- vehicle-validation64-shard-complete-20260926-shard04 -->
## 2026-09-26 vehicle validation64 shard 04

UTC: 2026-09-26T20:27:57.550241+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=20101. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard04/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard04/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard04-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard04 audit

UTC: 2026-09-26T20:32:40+00:00. Post-run audit of formal shard04 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard04 has 224 episodes and 20101 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard04: s0 fixed H25 (440 steps), s1 fixed H25 (321 steps), s2 used horizons {'25': 220, '35': 32} with 14 switches. Vehicle validation64 progress is now 5/12 completed shards with 1120 completed episodes and 99398 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard05; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD04_AUDIT_20260926T203240.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD04_AUDIT_FINAL_ADDENDUM_20260926T203240.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard-complete-20260926-shard05 -->
## 2026-09-26 vehicle validation64 shard 05

UTC: 2026-09-26T21:55:13.414681+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19660. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard05/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard05/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard05-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard05 audit

UTC: 2026-09-26T21:59:16+00:00. Post-run audit of formal shard05 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard05 has 224 episodes and 19660 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=False. Learned candidates in shard05: s0 {'episodes': 4, 'steps': 289, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 289}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 3, 'steps': 211, 'success_count': 3, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 211}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 3, 'steps': 195, 'success_count': 3, 'episode_failure_count': 0, 'switches': 4, 'horizon_counts': {'25': 193, '35': 2}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 6/12 completed shards with 1344 completed episodes and 119058 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard06; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_20260926T215916.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_FINAL_ADDENDUM_20260926T215916.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard05-audit-v2-schema-repair-20260926 -->
## 2026-09-26 vehicle validation64 shard05 audit v2 schema repair

UTC: 2026-09-26T22:05:23+00:00. Repaired post-run audit of formal shard05 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. V2 preserves failed v1 audit `research_artifacts/aws_diagnostics/vehicle_validation64_shard05_audit_20260926/completed.json` and fixes only audit-schema false positives: absent derived gross-decision mean is auxiliary, and shard05 registry's closed-test budget is accepted because episodes/control_steps are zero, sealed_test_bank_content_opened=false, test_authorization=false, and raw/completed/run_started all show test_accessed=false. Shard05 has 224 episodes and 19660 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard05: s0 {'episodes': 4, 'steps': 289, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 289}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 3, 'steps': 211, 'success_count': 3, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 211}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 3, 'steps': 195, 'success_count': 3, 'episode_failure_count': 0, 'switches': 4, 'horizon_counts': {'25': 193, '35': 2}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 6/12 completed formal shards with 1344 episodes and 119058 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and the failed v1 audit with 0 simulations. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard06; v2 request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_20260926T220523.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_FINAL_ADDENDUM_20260926T220523.json`. Sealed test remains closed.

<!-- post-shard05-v2-backup-gate-recheck-20260926T221104 -->
## 2026-09-26 post-shard05 v2 backup gate recheck

UTC: 2026-09-26T22:11:04+00:00. Metadata-only backup inventory recheck before shard06: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. No adequate repository-local verified external backup proof after shard05 v2 audit finalization was found (`backup_proof_20260926T22*.json` count 0; adequate proofs 0). Latest supervisor backup in context remains `2026-09-26T22:04:47.633765+00:00`, which predates shard05 v2 audit/blocker finalization and is insufficient. Shard06 remains backup-gated. New recheck artifacts: `research_artifacts/aws_diagnostics/post_shard05_v2_backup_gate_recheck_20260926T221104/raw.json`, `research_artifacts/aws_diagnostics/post_shard05_v2_backup_gate_recheck_20260926T221104/summary.md`, `research_artifacts/aws_diagnostics/post_shard05_v2_backup_gate_recheck_20260926T221104/completed.json`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_SHARD05_V2_GATE_RECHECK_20260926T221104.json`. Next proof must be after this recheck and after the metadata run registry/stdout/stderr finalize, and must cover shard05 formal outputs, failed v1 audit, passed v2 audit, docs/registry, all shard05-v2 backup requests/addenda/blocker notes, this recheck/request, runner SHA `cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e`, and gate SHA `5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b`. No reproduction or final-test conclusion is permitted; method remains IMPROVED, not ORIGINAL SAC.

<!-- vehicle-validation64-shard-complete-20260926-shard06 -->
## 2026-09-26 vehicle validation64 shard 06

UTC: 2026-09-26T23:24:17.125157+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19137. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard06-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard06 audit

UTC: 2026-09-26T23:29:08+00:00. Post-run audit of formal shard06 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard06 has 224 episodes and 19137 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard06: s0 359 steps, s1 634 steps, s2 horizons {'25': 270, '35': 33} with 7 switches. Vehicle validation64 progress is now 7/12 completed formal shards with 1568 completed episodes and 138195 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard07; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_20260926T232908.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_FINAL_ADDENDUM_20260926T232908.json`. Sealed test remains closed.

<!-- post-shard06-audit-backup-gate-recheck-20260926T233430+0000 -->
## 2026-09-26 post-shard06 audit backup gate recheck

UTC: 2026-09-26T23:34:30+00:00. Metadata-only backup inventory recheck before shard07: validation_accessed=true only for existing shard06 output hashing/counting, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. No adequate repository-local verified external backup proof after shard06 audit finalization and persisted state was found (`backup_proof_20260926T23*.json` count 0; adequate proofs 0). Latest supervisor backup in context `2026-09-26T23:28:51.819071+00:00` predates the shard06 audit and is insufficient. Key shard06 formal/audit/runner/gate hashes matched expected values: True. Shard07 remains backup-gated. New recheck artifacts: `research_artifacts/aws_diagnostics/post_shard06_audit_backup_gate_recheck_20260926T233430+0000/raw.json`, `research_artifacts/aws_diagnostics/post_shard06_audit_backup_gate_recheck_20260926T233430+0000/summary.md`, `completed.json` in the same directory. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_SHARD06_AUDIT_GATE_RECHECK_20260926T233430+0000.json`. Next proof must be after this recheck and after the metadata run registry/stdout/stderr finalize, and must cover shard06 formal outputs, shard06 audit outputs, formal/audit/recheck run logs, docs/registry, shard06 backup request/final addendum/blocker note, this recheck/request, runner SHA `cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e`, and gate SHA `5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b`. Current status remains partial validation evidence only for an IMPROVED latency-tree method, not ORIGINAL SAC, not final-test evidence, and not a reproduction-success claim.

<!-- vehicle-validation64-shard-complete-20260926-shard07 -->
## 2026-09-26 vehicle validation64 shard 07

UTC: 2026-09-27T00:49:15.611490+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19532. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard07/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard07/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard07-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard07 audit

UTC: 2026-09-27T00:53:19+00:00. Post-run audit of formal shard07 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard07 has 224 episodes and 19532 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard07: s0 {'episodes': 2, 'steps': 170, 'success_count': 2, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 170}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 9, 'steps': 658, 'success_count': 9, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 658}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 6, 'steps': 551, 'success_count': 5, 'episode_failure_count': 1, 'switches': 14, 'horizon_counts': {'25': 532, '35': 19}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 8/12 completed formal shards with 1792 episodes and 157727 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard08; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_20260927T005319.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_FINAL_ADDENDUM_20260927T005319.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard07-audit-v2-schema-repair-20260927 -->
## 2026-09-27 vehicle validation64 shard07 audit v2 schema repair

UTC: 2026-09-27T00:56:35+00:00. Repaired post-run audit of formal shard07 completed with validation_accessed=true (reading existing shard outputs and failed v1 audit artifacts), sealed test accessed=false, simulations=0, training steps=0. V2 preserves failed v1 audit `research_artifacts/aws_diagnostics/vehicle_validation64_shard07_audit_20260927/completed.json` and fixes only a registry-schema false positive: shard07's run registry records validation budget as `episodes_exact=224` rather than `episodes`/`validation_episodes`. Shard07 has 224 episodes and 19532 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard07: s0 {'episodes': 2, 'steps': 170, 'success_count': 2, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 170}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 9, 'steps': 658, 'success_count': 9, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 658}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 6, 'steps': 551, 'success_count': 5, 'episode_failure_count': 1, 'switches': 14, 'horizon_counts': {'25': 532, '35': 19}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 8/12 completed formal shards with 1792 episodes and 157727 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard08; v2 request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_V2_20260927T005635.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_V2_FINAL_ADDENDUM_20260927T005635.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard-complete-20260926-shard08 -->
## 2026-09-26 vehicle validation64 shard 08

UTC: 2026-09-27T02:11:46.710508+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19703. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard08-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard08 audit

UTC: 2026-09-27T02:15:20+00:00. Post-run audit of formal shard08 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard08 has 224 episodes and 19703 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard08: s0 {'episodes': 4, 'steps': 290, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 290}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 3, 'steps': 208, 'success_count': 3, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 208}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 4, 'steps': 314, 'success_count': 4, 'episode_failure_count': 0, 'switches': 8, 'horizon_counts': {'25': 307, '35': 7}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 9/12 completed formal shards with 2016 episodes and 177430 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard09; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_20260927T021520.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_FINAL_ADDENDUM_20260927T021520.json`. Sealed test remains closed.

<!-- post-shard08-audit-backup-gate-recheck-20260927T022136+0000 -->
## 2026-09-27 post-shard08 audit backup gate recheck

UTC: 2026-09-27T02:21:36+00:00. Metadata-only backup proof inventory check before shard09: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. Required proof was after `2026-09-27T02:15:35+00:00`. Repository-local `backup_proof_20260927T02*.json` count was 0; all `backup_proof_20260927T*.json` files were ['research_artifacts/aws_backup_proofs/backup_proof_20260927T005920_after_shard07_audit_v2_run_finalized.json']; adequate post-shard08 proofs found: 0. The supervisor-context backup at `2026-09-27T02:14:50.580286+00:00` is insufficient because it predates the shard08 audit run-finalized threshold and no local proof file exists. Shard09 remains backup-gated. No formal validation evidence was created; vehicle validation64 remains 9/12 shards, 2016/2688 episodes, 177430 control steps, final test closed. New artifacts: `research_artifacts/aws_diagnostics/post_shard08_audit_backup_gate_recheck_20260927T022136+0000/raw.json`, `research_artifacts/aws_diagnostics/post_shard08_audit_backup_gate_recheck_20260927T022136+0000/summary.md`, `research_artifacts/aws_diagnostics/post_shard08_audit_backup_gate_recheck_20260927T022136+0000/completed.json`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_POST_SHARD08_AUDIT_GATE_RECHECK_20260927T022136+0000.json`. Next proof must be after this recheck and this metadata run's finalized registry/stdout/stderr/cloudwatch snapshot before shard09 can run. Method remains IMPROVED latency-tree, not ORIGINAL SAC, and no reproduction-success claim is supported.

## User steering 2026-09-27: continue evidence-driven repair and retraining
User requested sustained evidence-driven method revision and retraining until final acceptance or fully evidenced inability. Operational details are in docs/bohn2021_takeover/ITERATIVE_RESEARCH_STEERING_20260927.md and the per-iteration MISSION. Frozen ongoing experiments remain unchanged.

## User clarification 2026-09-27: conceptual fidelity, methodological flexibility

The user explicitly permits changing the method; exact algorithmic identity is not required as long as the core idea remains consistent. Pursue learning-based, state-dependent adaptive MPC prediction horizon to improve the control-performance versus measured-computation tradeoff against strong, fairly tuned fixed-horizon baselines. Vehicle first, then inverted pendulum; no return to mobile robot joint K/H research.

Do not treat author-specific SAC, network architecture, critic/value estimator, terminal value method, reward processing, optimizer, exploration, normalization, state representation or policy extraction as immutable. Evidence-supported changes to these are authorized routine research decisions and need no further user approval. Retain useful author-method reconstruction as a reference and preserve negative results, but do not force exact ORIGINAL algorithm reproduction to be a prerequisite for pursuing and completing a successful modified approach.

Keep provenance explicit: ORIGINAL denotes faithful author-method reconstruction; IMPROVED denotes substantive modifications. An improved method can satisfy the user's intended scientific objective by demonstrating the core adaptive-horizon idea with rigorous evidence. Report that outcome as support for the core idea using a modified method; do not claim exact reproduction of the original algorithm or all paper conclusions. State precisely which original findings are and are not supported.

This permission changes implementation flexibility, not experimental rigor. Preserve strong fixed-H baselines and fair budgets, at least3 independent training seeds, all failures/negative results, actual wall-clock and solver timing, frozen versioned protocols, independent final tests, and complete audit/backup. Do not change the ongoing frozen validation campaign mid-run. Introduce revisions in new versioned development/training campaigns, record their reasons and validation/test exposure, and apply the original scientific acceptance requirements without post-hoc relaxation. A change to the entire research question or a material scientific fork still follows existing user-escalation rules.

<!-- vehicle-validation64-shard-complete-20260926-shard09 -->
## 2026-09-26 vehicle validation64 shard 09

UTC: 2026-09-27T03:37:07.147485+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19876. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard09/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard09/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard09-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard09 audit

UTC: 2026-09-27T03:40:23+00:00. Post-run audit of formal shard09 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard09 has 224 episodes and 19876 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard09: s0 {'episodes': 10, 'steps': 742, 'success_count': 10, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 742}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 4, 'steps': 293, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 293}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 6, 'steps': 474, 'success_count': 6, 'episode_failure_count': 0, 'switches': 17, 'horizon_counts': {'25': 387, '35': 87}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 10/12 completed formal shards with 2240 episodes and 197306 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard10; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD09_AUDIT_20260927T034023.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD09_AUDIT_FINAL_ADDENDUM_20260927T034023.json`. Sealed test remains closed.

<!-- vehicle-case43-shard07-trajectory-diagnostic-20260927 -->
## 2026-09-27 vehicle shard07 case43 trajectory diagnostic

UTC: 2026-09-27T03:47:33.055258+00:00. Metadata-only diagnostic of already-created validation outputs completed: learned_s2 shard07 case43 versus same-seed terminal25 fixed H25 case43. No validation bank reopen, no sealed-test access/hash, no simulations/control steps/gradient steps.

Key diagnostic facts: learned_s2 case43 failed at 150 steps with physical+constraint cost 39125.9 and horizons {'25': 149, '35': 1}; fixed seed2 terminal25 H25 succeeded in 96 steps with physical+constraint cost 90.4598. H35 steps were [3]. Prefix max previous-state/input diffs before the singleton H35 were 0 / 0; first input/post-state divergence steps were 3 / 4. Learned solver calls were all successful/accepted with retry_rows=0. Objective/terminal-value components and warm-start vectors were not persisted, so an instrumented non-formal replay is required for that part of the diagnosis.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_case43_shard07_trajectory_diagnostic_20260927/raw.json`, `research_artifacts/aws_diagnostics/vehicle_case43_shard07_trajectory_diagnostic_20260927/summary.md`, `research_artifacts/aws_diagnostics/vehicle_case43_shard07_trajectory_diagnostic_20260927/completed.json`. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_CASE43_SHARD07_DIAGNOSTIC_20260927T034733.json`; shard10 remains blocked until an adequate verified external backup proof covers shard09 audit run-finalized evidence and subsequent diagnostic artifacts.

<!-- vehicle-validation64-shard-complete-20260926-shard10 -->
## 2026-09-26 vehicle validation64 shard 10

UTC: 2026-09-27T05:01:42.259853+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19831. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard10-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard10 audit

UTC: 2026-09-27T05:05:50+00:00. Post-run audit of formal shard10 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard10 has 224 episodes and 19831 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard10: s0 {'episodes': 2, 'steps': 148, 'success_count': 2, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 148}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 5, 'steps': 425, 'success_count': 5, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 425}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 5, 'steps': 344, 'success_count': 5, 'episode_failure_count': 0, 'switches': 13, 'horizon_counts': {'25': 327, '35': 17}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 11/12 completed formal shards with 2464 episodes and 217137 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard11; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_20260927T050550.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_FINAL_ADDENDUM_20260927T050550.json`. Sealed test remains closed.

<!-- post-shard10-audit-backup-gate-recheck-20260927T051156+0000 -->
## 2026-09-27 post-shard10 audit backup gate recheck

UTC: 2026-09-27T05:11:56+00:00. Metadata-only backup proof inventory check before shard11: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. Required proof was after `2026-09-27T05:06:10+00:00`. Repository-local `backup_proof_20260927T05*.json` count was 0; all `backup_proof_20260927T*.json` files were ['research_artifacts/aws_backup_proofs/backup_proof_20260927T005920_after_shard07_audit_v2_run_finalized.json', 'research_artifacts/aws_backup_proofs/backup_proof_20260927T022317_after_post_shard08_gate_recheck_run_finalized.json', 'research_artifacts/aws_backup_proofs/backup_proof_20260927T034828_after_shard09_audit_and_case43_diagnostic_run_finalized.json']; adequate post-shard10 proofs found: 0. The supervisor-context backup at `2026-09-27T05:05:16.430103+00:00` is insufficient because it predates shard10 audit completion/finalization. Shard11 remains backup-gated. No formal validation evidence was created; vehicle validation64 remains 11/12 shards, 2464/2688 episodes, 217137 control steps, final test closed. New artifacts: `research_artifacts/aws_diagnostics/post_shard10_audit_backup_gate_recheck_20260927T051156+0000/raw.json`, `research_artifacts/aws_diagnostics/post_shard10_audit_backup_gate_recheck_20260927T051156+0000/summary.md`, `research_artifacts/aws_diagnostics/post_shard10_audit_backup_gate_recheck_20260927T051156+0000/completed.json`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_POST_SHARD10_AUDIT_GATE_RECHECK_20260927T051156+0000.json`. Next proof must be after this recheck and this metadata run's finalized registry/stdout/stderr/cloudwatch snapshot before shard11 can run. Method remains IMPROVED latency-tree, not ORIGINAL SAC, and no reproduction-success claim is supported.

<!-- vehicle-validation64-shard-complete-20260926-shard11 -->
## 2026-09-26 vehicle validation64 shard 11

UTC: 2026-09-27T06:26:48.030739+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19211. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard11/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard11/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.

<!-- vehicle-validation64-shard11-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard11 audit

UTC: 2026-09-27T06:31:22+00:00. Post-run audit of formal shard11 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard11 has 224 episodes and 19211 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard11: s0 {'episodes': 9, 'steps': 707, 'success_count': 9, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 707}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 6, 'steps': 447, 'success_count': 6, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 447}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 7, 'steps': 541, 'success_count': 7, 'episode_failure_count': 0, 'switches': 18, 'horizon_counts': {'25': 525, '35': 16}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 formal shard execution is now 12/12 complete with 2688 episodes and 236348 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. Full paired comparison/model-selection analysis is backup-gated until this final shard audit and finalized audit run logs are externally recoverable. Request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD11_AUDIT_20260927T063122.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD11_AUDIT_FINAL_ADDENDUM_20260927T063122.json`. Sealed test remains closed.

<!-- vehicle-learned-policy-collapse-diagnostic-v2-full-validation64-20260927 -->
## 2026-09-27 vehicle learned-policy collapse diagnostic v2 (full validation64)

UTC: 2026-09-27T06:36:13+00:00. Existing-output diagnostic for learned_s0/s1/s2 across validation64 shards 00-11 completed with validation_accessed=true only because saved validation output traces/summaries were read; validation bank reopened=false; sealed final test accessed/opened/hashed=false; simulations/control_steps/gradient_steps=0/0/0. Learned trace episodes read: 192; scanned episode directories: 2688.

Key classifications: {"learned_s0": {"actual_non25_steps": 0, "classification": "nonH25_leaves_exist_but_validation_states_never_reach_them", "falsifiable_next_step": "Compare feature distributions with thresholds and training states; non-H25 opportunity may be outside visited validation distribution.", "policy_trace_mismatch_count": 0, "predicted_non25_steps": 0, "structurally_constant_policy": false, "unique_leaf_horizons": []}, "learned_s1": {"actual_non25_steps": 0, "classification": "nonH25_leaves_exist_but_validation_states_never_reach_them", "falsifiable_next_step": "Compare feature distributions with thresholds and training states; non-H25 opportunity may be outside visited validation distribution.", "policy_trace_mismatch_count": 0, "predicted_non25_steps": 0, "structurally_constant_policy": false, "unique_leaf_horizons": []}, "learned_s2": {"actual_non25_steps": 409, "classification": "adaptive_horizon_used_on_existing_validation_traces", "falsifiable_next_step": "Use paired aggregate validation analysis and targeted replays to decide whether non-H25 decisions improve cost/time tradeoff or induce failures.", "policy_trace_mismatch_count": 0, "predicted_non25_steps": 0, "structurally_constant_policy": false, "unique_leaf_horizons": []}}

Artifacts: `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v2_full_validation64/raw.json`, `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v2_full_validation64/summary.md`, `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v2_full_validation64/completed.json` (SHA256 `d45a4e09bd616489751d163c2b43be89bee72194891b36854c556d63ff81203c`). Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_LEARNED_POLICY_COLLAPSE_DIAGNOSTIC_V2_FULL_VALIDATION64_20260927T063613+0000.json`. Method remains IMPROVED latency-tree, not ORIGINAL SAC; no final-test or reproduction-success claim is supported. Next scientific step is full paired validation aggregation/model selection and then targeted case43/instrumentation diagnostics or versioned method revision as indicated by the aggregate evidence.

<!-- vehicle-learned-policy-collapse-diagnostic-v3-full-validation64-20260927-decisions -->
### Decision: treat learned_s0/s1 as extracted-policy collapse, not runtime noise (2026-09-27T06:40:45+00:00)

Before evidence: validation64 traces showed learned_s0 and learned_s1 used H25 on every stored step; gate summary listed s0/s1 as fixed-by-structure/fixed-H25, but v2 diagnostic failed policy lookup.

Change in v3: no controller or data change. Only diagnostic locator repaired to hash exact frozen policy files from the gate summary and re-evaluate stored tree features from all 12 validation shards.

Decision rule: if policy hashes match the gate summary and trace/policy mismatches are zero, H25-only behavior is attributed to extracted policy structure. This does not imply adaptive superiority and does not authorize final test.

Outcome: `passed` with conclusions `{'learned_s0': {'classification': 'extracted_policy_structurally_constant_H25', 'structurally_constant_policy': True, 'unique_leaf_horizons': [25], 'actual_non25_steps': 0, 'predicted_non25_steps': 0, 'policy_trace_mismatch_count': 0, 'falsifiable_next_step': 'Inspect training/selection objective and candidate extraction logs; rollout timing noise is not needed to explain H25-only behavior.'}, 'learned_s1': {'classification': 'extracted_policy_structurally_constant_H25', 'structurally_constant_policy': True, 'unique_leaf_horizons': [25], 'actual_non25_steps': 0, 'predicted_non25_steps': 0, 'policy_trace_mismatch_count': 0, 'falsifiable_next_step': 'Inspect training/selection objective and candidate extraction logs; rollout timing noise is not needed to explain H25-only behavior.'}, 'learned_s2': {'classification': 'adaptive_horizon_used_and_trace_matches_policy', 'structurally_constant_policy': False, 'unique_leaf_horizons': [25, 35], 'actual_non25_steps': 409, 'predicted_non25_steps': 409, 'policy_trace_mismatch_count': 0, 'falsifiable_next_step': 'Use full paired aggregate analysis and targeted deterministic replays to test cost/time benefit and failure causality.'}}`. Next action is full paired validation64 aggregation against strong fixed-H baselines.

<!-- vehicle-validation64-full-aggregate-model-selection-20260927-decision -->
### Decision after full vehicle validation64 aggregate (2026-09-27T06:48:01+00:00)

Before evidence: all 12 validation shards had passed audits, v3 policy-collapse diagnostic showed learned_s0/s1 structurally constant H25 and learned_s2 adaptive with one catastrophic validation failure.

Aggregate change: no controller/model/data change; this diagnostic derived paired validation/model-selection summaries from existing per-episode summary files only.

Decision: do not open final test for the current frozen candidate. Treat current IMPROVED latency-tree as insufficient for a robust 3-seed adaptive superiority claim unless a later independent validation revision changes the evidence. Next bounded diagnostics should target (1) training/selection collapse for s0/s1 and (2) learned_s2 case43 replay/ablation causality, then freeze a versioned IMPROVED method revision.

Evidence: `research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/completed.json` headline `{'matched_terminal_all_seeds_performance_optimal': 'matched_terminal_allseeds_H25', 'matched_terminal_all_seeds_speed_within_3pct_cost': 'matched_terminal_allseeds_H25', 'independent_terminal_seed0_performance_optimal': 'fixed_seed0_terminal30_controllerH30', 'adaptive_gate_diagnostic': {'learned_s0': {'same_seed_speed_nomination': 'fixed_seed0_terminal25_controllerH25', 'adaptive_horizons_used': False, 'unique_horizons': [25], 'success_noninferior_proxy': True, 'cost_within_3pct_proxy': True, 'decision_at_least_10pct_faster_proxy': False, 'passes_against_same_seed_speed_nomination': False, 'reasons': ['not adaptive on validation: only one horizon used', 'decision mean does not show >=10% reduction vs same-seed speed nomination']}, 'learned_s1': {'same_seed_speed_nomination': 'fixed_seed1_terminal25_controllerH25', 'adaptive_horizons_used': False, 'unique_horizons': [25], 'success_noninferior_proxy': True, 'cost_within_3pct_proxy': True, 'decision_at_least_10pct_faster_proxy': False, 'passes_against_same_seed_speed_nomination': False, 'reasons': ['not adaptive on validation: only one horizon used', 'decision mean does not show >=10% reduction vs same-seed speed nomination']}, 'learned_s2': {'same_seed_speed_nomination': 'fixed_seed2_terminal25_controllerH25', 'adaptive_horizons_used': True, 'unique_horizons': [25, 35], 'success_noninferior_proxy': True, 'cost_within_3pct_proxy': False, 'decision_at_least_10pct_faster_proxy': False, 'passes_against_same_seed_speed_nomination': False, 'reasons': ['physical/control cost exceeds +3% proxy vs same-seed speed nomination', 'decision mean does not show >=10% reduction vs same-seed speed nomination']}}}`.

<!-- vehicle-training-selection-collapse-diagnostic-v2-20260927-decision -->
### Decision: current vehicle latency-tree failure is selection/objective collapse, not runtime dispatch (2026-09-27T06:55:45+00:00)

Before evidence: validation64 aggregate showed learned_s0/s1 were H25-only and learned_s2 adaptive but unsafe on case43; v3 policy diagnostic showed validation traces match stored policies exactly; v1 diagnostic failed on a policy schema bug and is preserved.

Diagnostic repair/change: no controller/algorithm/data change. V2 only changed diagnostic parsing to accept constant-policy key `h` as well as `horizon` and wrote fresh v2 outputs. Validation/test access remained zero.

Decision: current frozen IMPROVED latency-tree candidate is insufficient for final test. A versioned IMPROVED method revision should modify the horizon policy selection/extraction objective and risk/robustness handling. Preserve ORIGINAL/old negative results, keep strong fixed-H baselines, and use fresh independent validation for any changed candidate.

Evidence: `research_artifacts/aws_diagnostics/vehicle_training_selection_collapse_diagnostic_20260927_v2/completed.json` headline `{'cross_seed_classifications': {'0': 'selection_chose_structurally_constant_H25_tree', '1': 'selection_chose_fixed_H25_baseline', '2': 'selection_chose_adaptive_tree'}, 'selected': {'0': 'g3_c09', '1': 'fixed', '2': 'g3_c08'}, 'final_horizons': {'0': [25], '1': [25], '2': [25, 35]}, 'eligible_adaptive_counts': {'0': 15, '1': 17, '2': 18}, 'best_eligible_adaptive': {'0': 'g1_c00', '1': 'g2_c05', '2': 'g3_c08'}, 'interpretation_short': 'selection/objective-collapse rather than runtime-dispatch bug; current frozen candidate insufficient for final test'}`.

<!-- vehicle-case43-instrumented-replay-v1-20260927 -->
## 2026-09-27 vehicle case43 instrumented replay/ablation v1

UTC: 2026-09-27T07:14:24.502108+00:00. Development diagnostic intentionally reopened the already-used vehicle validation bank case43 and ran 5 deterministic replay/ablation episodes (588 new control steps, 0 gradient steps, sealed test closed). This is not fresh independent validation/model-selection evidence.

Outcome: forcing H25 at the singleton H35 rescues the learned replay, and a single H35 perturbation on the constant-H25 path fails; evidence supports the step3 horizon intervention as a direct cause of this case43 failure under this terminal/source setup.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/raw.json`, `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/summary.md`, `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/episode_summary.csv`, completed marker `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/completed.json`. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_CASE43_INSTRUMENTED_REPLAY_V1_20260927T071424+0000.json`.

<!-- vehicle-h35-leaf-validation-diagnostic-v4-20260927 -->
## 2026-09-27 vehicle H35 leaf validation diagnostic v4

UTC: 2026-09-27T07:30:08.460730+00:00. Existing validation64 rollout outputs only; no new simulation/control steps/gradient steps, no validation bank reopen, sealed test closed. v4 preserves failed v1/v2/v3 and fixes compact-tree parsing plus first-H35 step-index handling.

Finding: learned_s2's H35 branch follows rule `heading_error_5 <= 0.0957597175326 and abs_yaw_input <= 0.250382459863`. It appeared in 52/64 validation cases (409 total H35 steps). First-H35 rollout-step stats: `{'count': 52, 'sum': 1497.0, 'mean': 28.78846153846154, 'median': 31.0, 'p95': 63.80000000000001, 'min': 0.0, 'max': 67.0}`. Logged-vs-replayed policy horizon mismatches: 0. Case43 remains the only learned_s2 validation failure, with H35 steps `[3]`; prior deterministic replay showed that forcing H25 at the singleton H35 rescues it while forcing H35 into the constant-H25 path reproduces the failure. Excluding case43, learned_s2 vs same-seed fixed H25 physical deltas are summarized by `{'count': 63, 'sum': -17.59428504099685, 'mean': -0.2792743657301088, 'median': 0.0, 'p95': 0.0005317749462619757, 'min': -12.132815113962833, 'max': 0.0031670370110390422}`.

Decision: next IMPROVED vehicle revision should test safe-shortening/risk-sensitive extraction rather than allowing H>25 cost-seeking branches to compete as adaptive-horizon acceleration. Artifacts: `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v4/summary.md`, `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v4/raw.json`, `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v4/per_case_h35.csv`.

<!-- vehicle-safe-shortening-v1-smoke-20260927-v3 -->
## 2026-09-27 vehicle safe-shortening v1 smoke

UTC: 2026-09-27T08:02:25.089325+00:00. Engineering smoke for IMPROVED safe-shortening wrapper completed on vehicle smoke bank only: 24 episodes, 1764 control steps, replay passed=True. Validation64 and sealed test remained closed. This is not model-selection/final evidence. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v3/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v3/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v3/completed.json`.

<!-- vehicle-safe-shortening-v1-case43-replay-v3-collate-20260927-decision -->
### Decision: v3 collation recovers v2 case43 replay without rerunning simulations

Before evidence: v2 repaired the v1 arm-filter bug and ran all six already-opened case43 episodes, but failed after simulation because `_aggregate_pair()` read `decision_timing_s` from an aggregate that exposes `decision_total_s` and `decision_mean_s_per_step`.

Change: v3 is collation-only from completed v2 episode summaries and uses the aggregate timing schema already produced by the smoke helper. Controller, policies, terminal models, case43, seeds, horizon rule, solver/recovery behavior and validation/test access policy are unchanged.

Outcome: see `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v3_collate/summary.md`. This confirms only an engineering precondition: safe-shortening v1 avoids H>25 on contaminated case43 in preserved v2 episodes. It does not authorize final test; backup and fresh development-validation remain required.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard00 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 00

UTC: 2026-09-27T09:24:18.053445+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13784 control steps, cases [14, 29, 42, 63], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard01 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 01

UTC: 2026-09-27T10:24:23.361302+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 16086 control steps, cases [21, 26, 33, 52], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard01/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard01/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard01/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard02 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 02

UTC: 2026-09-27T11:25:22.996476+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15754 control steps, cases [8, 15, 30, 61], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard02/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard02/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard02/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard03 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 03

UTC: 2026-09-27T12:20:50.233484+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13603 control steps, cases [5, 24, 32, 56], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard03/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard03/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard03/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard04 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 04

UTC: 2026-09-27T13:13:53.294129+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 12912 control steps, cases [6, 11, 49, 53], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard04/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard04/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard04/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard05 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 05

UTC: 2026-09-27T14:13:22.323523+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15126 control steps, cases [22, 25, 44, 51], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard05/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard05/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard05/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard06 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 06

UTC: 2026-09-27T15:11:23.792340+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14178 control steps, cases [7, 17, 37, 41], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard07 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 07

UTC: 2026-09-27T16:08:45.227251+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14624 control steps, cases [23, 31, 46, 62], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard08 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 08

UTC: 2026-09-27T17:02:28.089742+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13165 control steps, cases [13, 35, 38, 43], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard09 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 09

UTC: 2026-09-27T17:56:53.522498+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14176 control steps, cases [4, 12, 18, 57], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard10 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 10

UTC: 2026-09-27T18:54:43.290131+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14237 control steps, cases [10, 28, 47, 54], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard11 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 11

UTC: 2026-09-27T19:50:07.888027+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13756 control steps, cases [1, 34, 48, 58], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard12 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 12

UTC: 2026-09-27T20:47:34.091485+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15129 control steps, cases [0, 2, 55, 59], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard12/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard12/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard12/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard13 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 13

UTC: 2026-09-27T21:44:10.683619+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14479 control steps, cases [9, 36, 40, 50], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard13/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard13/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard13/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard14 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 14

UTC: 2026-09-27T22:47:15.250406+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15632 control steps, cases [3, 19, 20, 39], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard14/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard14/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard14/completed.json`.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard15 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 15

UTC: 2026-09-27T23:44:53.534841+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14802 control steps, cases [16, 27, 45, 60], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard15/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard15/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard15/completed.json`.

<!-- vehicle-safe-shortening-v1-devval-progress-digest-v2-00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15 -->
## 2026-09-27 vehicle safe-shortening v1 devval64 progress digest v2

UTC: 2026-09-27T23:51:01.751104+00:00. Metadata-only digest over completed fresh devval shards [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]: 2752 episodes, 231443 control steps, 64/64 cases scored. No rollout/training, no historical validation64 bank reopen, and no sealed-test access. Primary same-seed H25 preliminary deltas: {'0': {'paired_case_count': 64, 'adaptive_success_count': 63, 'fixed_success_count': 63, 'physical_delta_sum': 8.701375885893775, 'total_delta_sum': 4.9963758858937695, 'decision_ratio_overall_total_decision_s': 0.9904014853310217, 'adaptive_horizon_counts': {'20': 741, '25': 4131}, 'used_adaptive_h_above_25': False}, '1': {'paired_case_count': 64, 'adaptive_success_count': 64, 'fixed_success_count': 64, 'physical_delta_sum': 9.592265011229458, 'total_delta_sum': -1.357734988770634, 'decision_ratio_overall_total_decision_s': 0.9867955839764387, 'adaptive_horizon_counts': {'15': 1095, '25': 3693}, 'used_adaptive_h_above_25': False}, '2': {'paired_case_count': 64, 'adaptive_success_count': 63, 'fixed_success_count': 64, 'physical_delta_sum': 753.7911792487655, 'total_delta_sum': 754.8811792487656, 'decision_ratio_overall_total_decision_s': 1.017903663842976, 'adaptive_horizon_counts': {'25': 4812, '10': 49}, 'used_adaptive_h_above_25': False}}. Continue frozen shards before model selection/final-test gate. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_devval_progress_digest_20260927_v2_shards00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_devval_progress_digest_20260927_v2_shards00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_devval_progress_digest_20260927_v2_shards00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15/completed.json`.

<!-- vehicle-safe-shortening-v1-shard13-case9-seed2-trace-diagnostic-20260927T235655Z -->
## 2026-09-27 vehicle safe-shortening v1 shard13 case9 seed2 trace diagnostic

UTC: 2026-09-27T23:56:55.994529+00:00. Ran a metadata-only diagnostic over already-created shard13 raw evidence; no rollout/training, no historical validation64 bank reopen, and no sealed-test access. Adaptive seed2 case9 failed with summary horizons {'10': 1, '25': 149} and raw horizons {'10': 1, '25': 149}, clamped=0, solver_fallback=0, solver_fail_steps=0, retries=0; same-seed H25 succeeded. Adaptive-minus-H25 deltas: steps 73, physical 751.34848, total 753.15848. Conclusions: ['The episode-level fields show no clamp, solver-failure fallback, solver-failure steps, or retries for the adaptive failure; an execution fallback/clamp explanation is not supported by these aggregate raw fields.', 'The raw artifact did not expose a recoverable step-level horizon sequence despite aggregate horizon_counts showing H10 once; this prevents exact localization from raw metadata alone.', 'The adaptive case9 failure is a real paired degradation relative to same-seed H25 in this development shard: adaptive exhausted the 150-step cap, while H25 reached the goal earlier.', 'Same-case fixed-grid evidence suggests very short horizons are risky on this scenario while H25+ succeeds, consistent with but not proving that the rare H10 decision could be harmful.', 'Existing raw fields appear insufficient for terminal-value/objective/state-normalization causality; a bounded instrumented deterministic replay is the appropriate next diagnostic after the frozen campaign/backup gate.']. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_shard13_case9_seed2_trace_diagnostic_20260927T235655Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_shard13_case9_seed2_trace_diagnostic_20260927T235655Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_shard13_case9_seed2_trace_diagnostic_20260927T235655Z/completed.json`.

<!-- vehicle-safe-shortening-v1-trace-policy-diagnostic-20260928-decision -->
### Decision: safe-shortening v1 collapse is a policy/guard objective failure, not a runtime horizon-dispatch bug

Before evidence: full v1 devval64 failed the frozen gate; metadata-only case9 diagnostic showed a real seed2 failure but could not localize the rare H10 decision.

Diagnostic action: parsed existing step traces, guard values, raw and executed horizons, training-selection records, and fixed-grid devval summaries. No new rollouts/training/test access.

Outcome: see `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_trace_policy_diagnostic_20260928T000351Z/summary.md`. No trace-level dispatch inconsistency or H>25 request was found. The near-collapse to H25 follows the selected gated policies and thresholds; seed2's strict h10_p0_g5 profile produces extremely rare H10 states. This supports a versioned IMPROVED v2 change focused on learned/selection objective and state-level safety/value estimation, after one bounded deterministic counterfactual replay for case9/case43.

<!-- vehicle-safe-shortening-v1-case9-counterfactual-v1-20260928-decision -->
### Decision: case9 seed2 v1 failure is a horizon-choice causal hazard, not dispatch/noise

Before evidence: full devval64 rejected safe-shortening v1; metadata diagnostic showed adaptive and matched-H25 case9 trajectories were identical before step57, where adaptive selected H10 once, then adaptive failed while matched H25 succeeded.

Diagnostic change: two one-variable deterministic replays on the already-opened case only. Force H25 at the adaptive H10 step, and inject H10 into the fixed-H25 path at the same step. No training, no controller/terminal/model/source mutation, no sealed test.

Outcome: see `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_counterfactual_v1_20260928T000944Z/summary.md`. This supports revising the IMPROVED method toward state-level safety/value estimation rather than revalidating v1. Evidence remains development-only and contaminated by case9 diagnosis.

<!-- vehicle-safe-shortening-v1-case9-switch-sensitivity-v1-20260928-decision -->
### Decision: case9 switch-sensitivity informs IMPROVED v2 horizon-change safety

Before evidence: v1 full devval64 failed the gate; case9 one-step H10 at step57 was causally implicated by force/inject counterfactuals.

New diagnostic: one-step H15, one-step H20, and H10-from-step57 onward replays on the same already-opened case. No training, no controller/terminal mutation, no sealed test.

Outcome: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_switch_sensitivity_v1_20260928T001549Z/summary.md`. Use this as development-only evidence for v2 design; it cannot serve as fresh validation.

<!-- vehicle-safe-shortening-v1-case9-ramp-sensitivity-v1-20260928-decision -->
### Decision input: case9 H10-return/ramp sensitivity for IMPROVED v2

Before evidence: one-step H10 at step57 caused the case9 seed2 failure; one-step H15/H20 and continued H10 were safe on the same case.

New diagnostic: transition/ramp schedules after H10 on the same already-opened case. No training, no controller/terminal mutation, no sealed test.

Outcome: `The H10-return/ramp schedules still failed on this case; v2 should avoid H10 in comparable states unless a stronger continuation/risk model justifies it.`. Use only as development evidence for v2 transition-safety design.

<!-- vehicle-safe-shortening-v1-case9-h10-hold-sensitivity-v1-20260928T002551Z -->
## 2026-09-28 vehicle safe-shortening v1 case9 H10 hold diagnostic

Development-only case9 seed2 H10 hold-length diagnostic completed: 4 new episodes, 308 new control steps, no training and no sealed-test access. Result: At least one finite H10 hold repaired the case9 transition; shortest successful hold among tested lengths is 3 steps. This supports considering transition dwell/hysteresis in v2, but only as development evidence from one already-opened case. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_20260928T002551Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_20260928T002551Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_20260928T002551Z/completed.json`.

<!-- vehicle-safe-shortening-v2-transition-hold-smoke-20260928 -->
## 2026-09-28 vehicle safe-shortening v2 transition-hold smoke

UTC: 2026-09-28T00:39:40.931469+00:00. IMPROVED v2 transition-hold protocol frozen and engineering smoke completed on a fresh v2 smoke bank: 24 episodes, 1800 control steps, replay passed=True. No validation64 bank or sealed test was opened. This is not model-selection/final evidence. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928/completed.json`.

<!-- vehicle-safe-shortening-v2-transition-hold-devval-preflight-20260928T004500Z -->
## 2026-09-28 vehicle safe-shortening v2 transition-hold devval preflight

UTC: 2026-09-28T00:46:17.422154+00:00. Ran metadata/runtime preflight for the frozen IMPROVED v2 transition-hold development-validation campaign. No devval bank generation, no rollout/control steps, no historical validation64 bank reopen, and no sealed-test access. Schedule/arm dimensions verified: 2752 episodes, 16 shards, 43 arms; terminal-source metadata verified for 12 unique sources. Decision: Preflight passed. Do not run v2 devval shard until supervisor reports verified external backup for smoke, protocols, runner and this preflight. After backup, next concrete action is shard00 with legacy interpreter. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/completed.json`.

<!-- decision-v2-devval-backup-gate-20260928T005101+0000 -->
## 2026-09-28 decision: hold v2 devval until post-preflight backup proof

Evidence: metadata-only gate recheck at 2026-09-28T00:51:01+00:00 found 0 adequate repository-local backup proofs after `2026-09-28T00:46:20+00:00`. The latest supervisor-context verified backup at `2026-09-28T00:45:40.181961+00:00` predates the v2 devval preflight and cannot cover its outputs/logs. Decision: do not start v2 devval shard00 until a verified external backup after this recheck covers the v2 protocols/source, smoke, preflight, docs/registry, backup requests, this gate recheck, and this run's finalized logs. This is a storage/recoverability gate only; it does not alter the frozen v2 controller, schedule, selection rules, or acceptance criteria.

## Priority user steering 2026-09-28: shift from local patches to training-level diagnosis

The user explicitly redirects research emphasis: local horizon-transition repairs have been explored; if they do not yield meaningful improvement, investigate whether training and method design are the real bottleneck. Exact fidelity to the original paper's algorithm is NOT required. Preserve the core learned adaptive MPC-horizon idea, but freely revise training, objectives, value estimation, exploration, representation, policy class and selection when evidence supports it. This is authorized routine IMPROVED research, not a major goal change needing approval.

At the next bounded iteration after the currently running v2 shard finishes, prioritize a concrete training/selection diagnostic before automatically launching another long validation shard. The frozen controller and existing campaign data must remain unchanged. Inserting a separate diagnostic does not permit retrospective changes to selection/acceptance rules; record campaign scheduling and any later amendment transparently. Do not wait for all16 shards merely to inspect existing training records. Do not discard the existing campaign or silently declare an early-stop outcome to be full independent validation.

Required next work: distinguish the CURRENT reused gated-horizon policies from the older latency-tree policy. Trace how each current seed's short_h/profile/guard and H25 terminal checkpoint were trained/searched/selected, including candidate coverage, budgets, discarded candidates and objective components. Diagnose state/feature coverage and gate-trigger scarcity, exploration of alternative horizons, objective scaling and runtime noise, value/terminal estimation bias, and whether training evaluates the consequences of transitions and continuation costs. A diagnostic of the OLD latency-tree selection alone does not establish a diagnosis of the CURRENT gated-horizon training. Separate verified causes from hypotheses and missing evidence.

Produce and execute the smallest informative training-level experiment supported by that audit: for example a controlled re-selection/re-fitting experiment using identical training scenarios and a better justified risk/timing objective, or a bounded re-training/value-estimation ablation with one principal change. Freeze its hypothesis, candidate method, train/validation IDs, budgets and comparison criteria before execution; use smoke before expansion. If current implementation uses finite candidate search rather than gradient RL, call the operation search/refit/reselection accurately instead of claiming neural re-training. Do not re-run completed expensive training without a stated new hypothesis. If a gradient-training change is justified, actually run and log it rather than repeatedly describing a future training plan.

Report the concrete evidence and decision, then extend the promising revised method to all3 independent training seeds with strong fairly tuned fixed-H baselines. Preserve negative results and every additional simulation/training/model-selection budget. Do not force horizon switching for appearance, remove difficult scenarios, weaken acceptance after results, or use sealed test to guide development. Newly modified methods must remain labeled IMPROVED, and an improved-only success may support the core idea without falsely claiming exact ORIGINAL reproduction.

Do not let another sequence of local dwell/guard patches and whole-grid validations replace this priority. If there is evidence that another action is more informative, explicitly record that evidence and the reason before choosing it. User absence is not a blocker. API/token caps remain removed; resource and scientific safeguards remain in force.

<!-- vehicle-safe-shortening-v2-transition-hold-devval64-shard-20260928-v1-shard00 -->
## 2026-09-28 vehicle safe-shortening v2 transition-hold fresh development-validation shard 00

UTC: 2026-09-28T01:48:44.635740+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v2 transition-hold: 172 episodes, 14487 control steps, cases [3, 44, 52, 56], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval64_20260928_v1/shard00/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval64_20260928_v1/shard00/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval64_20260928_v1/shard00/completed.json`.

<!-- vehicle-current-gated-horizon-training-audit-v1-20260928 -->
## 2026-09-28 current gated-horizon training/search audit v1

UTC: 2026-09-28T01:56:06+00:00. Metadata-only audit of the CURRENT reused gated-horizon policies used by AWS safe-shortening v1/v2, not merely the older latency-tree policy. No rollouts/control steps, no training/gradient steps, no historical validation64 reopen, and no sealed-test access/hash occurred. Audited finite search/reselection over 36 structured candidates plus fixed H25 per seed and selected/fixed training traces; already-opened v2 shard00 adaptive traces were used only for development coverage comparison. Key result: current policies came from finite candidate search with gradient_updates=0; objective is mean raw total_cost among hard-gated admissible candidates, with no measured runtime term and no transition/dwell-risk model. Candidate tables and coverage summaries are in `research_artifacts/aws_diagnostics/vehicle_current_gated_horizon_training_audit_v1_20260928T015606+0000/summary.md` / `research_artifacts/aws_diagnostics/vehicle_current_gated_horizon_training_audit_v1_20260928T015606+0000/raw.json`. Next: freeze and run a bounded IMPROVED re-selection/refit diagnostic before any further unchanged validation shard.

<!-- vehicle-gated-horizon-risk-reselection-v1-20260928T020201+0000 -->
### Vehicle gated-horizon risk-first re-selection v1 (2026-09-28T02:02:01+00:00)
Metadata-only re-selection over existing current gated-search training traces completed: research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_20260928T020201+0000/completed.json. No rollout/control/training/test access. Frozen decision `freeze_and_run_small_smoke_after_backup`; adaptive nominated seeds=3, changed-from-current adaptive seeds=3. Summary: research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_20260928T020201+0000/summary.md. Backup request: research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_20260928T020201+0000.json.

<!-- vehicle-gated-horizon-risk-reselection-v1-smoke-20260928 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 smoke

UTC: 2026-09-28T02:21:04.988578+00:00. IMPROVED risk-first gated-horizon re-selection smoke completed on a fresh engineering bank: 36 episodes, 3258 control steps, hard_pass=True, risk_reselected_below_H25_seed_count=3. No validation64 bank or sealed test was opened. This is not model-selection/final evidence and does not modify the frozen v2 shard campaign. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_smoke_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_smoke_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_smoke_20260928/completed.json`.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard00 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 00

UTC: 2026-09-28T03:27:13.683697+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16091 control steps, cases [18, 42, 43, 50], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard00/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard00/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard00/completed.json`.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard01 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 01

UTC: 2026-09-28T04:29:51.388015+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16541 control steps, cases [19, 20, 28, 49], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard01/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard01/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard01/completed.json`.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard02 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 02

UTC: 2026-09-28T05:27:19.235465+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 15023 control steps, cases [2, 5, 14, 56], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard02/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard02/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard02/completed.json`.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard03 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 03

UTC: 2026-09-28T06:29:54.353947+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16619 control steps, cases [26, 31, 36, 39], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard03/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard03/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard03/completed.json`.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard04 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 04

UTC: 2026-09-28T07:34:31.779002+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 17494 control steps, cases [22, 37, 52, 62], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard04/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard04/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard04/completed.json`.

## Priority user steering 2026-09-28: experimental scenarios may be redesigned

The user explicitly permits repeated revisions to experimental design and scenario generation, not only the algorithm. Weak effects may reflect scenarios that offer little useful state-dependent prediction-horizon tradeoff. Treat experimental-design limitations as an active hypothesis alongside training, selection, value bias and runtime overhead. Routine evidence-driven scenario revisions within the same vehicle/pendulum adaptive-MPC-horizon question are authorized without further approval.

Before committing to another large revised-method validation campaign, inspect whether scenario difficulty and dynamics make horizon choice consequential: state/goal-distance distributions, curvature or heading changes where supported by the original environment, constraint proximity, initial speed and disturbance/model uncertainty where justified, transient versus steady phases, and the balance of easy/hard cases. Verify the actual environment capabilities before adding any factor. From identical development states compare candidate horizons with continuation rollouts and actual timing, accounting for terminal-value quality and solver/selection overhead. If strong fixed H offers the same Pareto tradeoff everywhere, document that rather than force artificial switching. Distinguish absence of adaptive opportunity from failure of learning to exploit existing opportunity.

Scenario revisions must be hypothesis-driven and versioned BEFORE generating/scoring the new campaign. Preserve the paper-aligned reference benchmark and all its negative results. Keep modified/distribution-shift/stress scenarios as explicitly separate IMPROVED or diagnostic benchmarks. For every revision record the motivating evidence, exact generator/parameter changes, intended scientific question, original-task fidelity, affected claims, split IDs/seeds, selection rules, metrics, train/validation/test budgets and acceptance criteria. Re-tune strong fixed-H baselines fairly on the revised training/validation distribution rather than reusing a disadvantaged fixed H. Include both favorable and unfavorable conditions; never delete difficult cases, search test outcomes, or design a benchmark solely to make adaptive H win.

Do not edit the current frozen campaign's cases/controllers/acceptance rules mid-run. Additional diagnostics may be interleaved after a bounded shard without changing that campaign; transparent scheduling amendments must preserve existing evidence. Previously seen data may inform development only. Use fresh confirmation validation when design changes used validation findings; keep final test sealed, regenerate it if contaminated, and freeze the complete final method and scenario protocol before opening it. Any claimed support on modified scenarios must be scoped to those scenarios and not represented as exact reproduction of the original paper's environment.

Before the next major long-run allocation, report an explicit diagnosis across all three layers: (1) scenario opportunity/design, (2) training/selection/terminal value, (3) implementation and real compute overhead. Choose the most informative controlled experiment, not another unchanged full campaign by default. Exact algorithm identity with the paper is not required; fairness, multiple seeds, actual timing, independent evaluation and honest claims remain required.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard05 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 05

UTC: 2026-09-28T08:35:14.280065+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16079 control steps, cases [0, 34, 54, 55], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard05/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard05/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard05/completed.json`.

<!-- vehicle-risk-reselection-partial-opportunity-diagnostic-20260928 -->
## 2026-09-28 risk-reselection partial opportunity/runtime diagnostic

UTC: 2026-09-28T08:50:05.424948+00:00. Read-only diagnostic over fresh devval shards [0, 1, 2, 3, 4, 5]; no simulations, no training, no sealed-test access, and no historical validation64 bank access. Parsed 1104 per-episode summaries and shard-reported 1104 episodes / 97847 control steps.

Key result: Do not allocate another unchanged long risk-reselection shard by default. Preserve the partial campaign and prioritize a versioned training/selection or scenario-opportunity diagnostic that can create a stronger candidate or explain absent adaptive opportunity.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z/completed.json`. New diagnostic artifacts and doc updates require external backup before further simulations.

<!-- vehicle-risk-reselection-partial-opportunity-diagnostic-v2-20260928 -->
## 2026-09-28 risk-reselection partial opportunity/runtime diagnostic V2

UTC: 2026-09-28T08:55:56.583581+00:00. Corrected read-only diagnostic over fresh devval shards [0, 1, 2, 3, 4, 5]; no simulations, no training, no sealed-test access, and no historical validation64 bank access. V2 supersedes v1 fixed-H timing-opportunity statements because v1 did not parse nested per-episode timing. Parsed 1104 per-episode summaries and shard-reported 1104 episodes / 97847 control steps.

Decision: Pause additional unchanged long risk-reselection devval shards after backup; freeze an outcome-informed scheduling amendment and design a smaller versioned training/selection or scenario-opportunity experiment with actual measured-time objectives.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/completed.json`. New diagnostic artifacts and doc updates require external backup before further simulations.

<!-- vehicle-gated-horizon-actual-time-reselection-v2-20260928 -->
## 2026-09-28 vehicle actual-time-aware gated-horizon re-selection V2

UTC: 2026-09-28T09:00:46+00:00. Metadata-only IMPROVED finite re-selection using existing training candidate metrics and corrected V2 timing/opportunity diagnostics; no simulations, no training, no historical validation64 bank reopen, and no sealed-test access. Adaptive nominations: 2/3; acceptance_for_smoke_met=True. Nominations: {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}. Next action after backup: `after_backup_freeze_and_run_small_actual_time_reselection_v2_smoke`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z/completed.json`.

<!-- vehicle-gated-horizon-actual-time-reselection-v2b-seed0-overhead-repair-20260928 -->
## 2026-09-28 vehicle actual-time-aware re-selection V2b seed0-overhead repair

UTC: 2026-09-28T09:05:35+00:00. Metadata-only repair of V2 seed0 selection-overhead bookkeeping; no simulations, no training, no validation64 bank reopen, and no sealed-test access. Corrected nominations: {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}; adaptive_nominated_seed_count=2/3; acceptance_for_smoke_met=True; changes_vs_V2={'seed0_overhead_fraction_applied_now': 0.0016665805221004085, 'previous_v2_acceptance_for_smoke_met': True, 'corrected_acceptance_for_smoke_met': True, 'acceptance_changed': False, 'previous_v2_nomination_ids': {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}, 'corrected_nomination_ids': {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}, 'nomination_ids_changed': False}. Next action after backup: `after_backup_freeze_and_run_small_actual_time_reselection_v2b_smoke`.

<!-- vehicle-gated-horizon-actual-time-reselection-v2b-smoke-preflight-20260928 -->
## 2026-09-28 vehicle actual-time-aware V2b smoke preflight

UTC: 2026-09-28T09:12:46+00:00. Metadata-only preflight for the frozen IMPROVED V2b smoke completed: hard_pass=True, nominations={'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}. No simulations, training, validation64 bank access, or sealed-test access. Next action after verified backup: `after_verified_backup_run_legacy_actual_time_v2b_smoke`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/completed.json`.

<!-- vehicle-actual-time-v2b-candidate-failure-surface-diagnostic-20260928 -->
## 2026-09-28 vehicle actual-time V2b candidate failure-surface diagnostic

UTC: 2026-09-28T09:17:06+00:00. Metadata-only diagnostic completed over V2b candidate metrics and preflight artifacts: rows=111, nominations={'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}, adaptive_nominated_seed_count=2/3. No simulations, no training, no validation64 bank access, and no sealed-test access. Verified seed0 has no eligible adaptive candidate under the current finite V2b gates; seeds1/2 both nominate h15_p1_g5. Next action after verified backup: run the already-frozen 36-episode legacy V2b smoke; do not resume the unchanged long risk-reselection devval shards by default. Artifacts: `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/completed.json`.

<!-- vehicle-gated-horizon-actual-time-reselection-v2b-smoke-20260928 -->
## 2026-09-28 vehicle actual-time-aware gated-horizon V2b smoke

UTC: 2026-09-28T09:31:58.719179+00:00. IMPROVED actual-time-aware gated-horizon V2b smoke completed on a fresh engineering bank: 36 episodes, 2826 control steps, hard_pass=True, selected_below_H25_seed_count=2. No validation64 bank or sealed test was opened. This is not model-selection/final evidence and does not modify any frozen devval campaign. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/completed.json`.

<!-- vehicle-actual-time-v2b-smoke-postdiagnostic-20260928 -->
## 2026-09-28 vehicle actual-time V2b smoke postdiagnostic

UTC: 2026-09-28T09:36:20.558519+00:00. Metadata-only postdiagnostic of the V2b smoke completed; no simulations/training, no validation64 bank access, and no sealed-test access. Smoke hard_pass=True; episodes=36; control_steps=2826. Weighted selected/fixed decision-time ratio=0.975365; adaptive seed ratios={'1': 0.923164173083399, '2': 1.016109314669204}. Decision: V2b is engineering-ready but not strong enough to justify a long validation campaign by default. Next protocol draft: `research_artifacts/aws_protocols/vehicle_scenario_opportunity_capability_diagnostic_v0_after_v2b_smoke_20260928.md`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/completed.json`.

<!-- vehicle-scenario-opportunity-capability-diagnostic-v0-20260928T0950Z -->
## Vehicle scenario-opportunity/capability diagnostic V0

UTC: 2026-09-28T09:50:19.198779+00:00. Metadata/source-only audit; no simulations, training, validation64 bank access, or sealed-test access. Finding: the vehicle environment supports straight-line goal/path length and heading variation plus three obstacle constraints/noisy forecasts, but the current registered vehicle config fixes initial x/y/theta, lacks plant process noise/model randomization, has no direct initial-speed state, and has no native curved-path generator. Decision: Run a bounded fixed-H opportunity probe only after backup; do not redesign scenarios or resume long adaptive validation until fixed-H Pareto opportunity is measured. Artifacts: `research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/completed.json`.

<!-- vehicle-fixed-h-opportunity-probe-v0-preflight-20260928T1000Z -->
## Vehicle fixed-H opportunity probe V0 preflight

UTC: 2026-09-28T09:55:04.355094+00:00. Metadata-only preflight froze the next bounded fixed-H opportunity-probe design: 80 planned fixed-H episodes, <=12000 control steps, 24 fresh candidate-bank resets, full H grid 5..50, independent seed0 terminal per H. Terminal grid ready=True. No simulations/training/validation64/test access occurred. External backup is required before the rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z/completed.json`, `research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_frozen_20260928.md`.


## Priority user steering: thorough causal analysis and retraining when warranted

User instruction (2026-09-28): fully analyze the causes, and retrain when necessary. This explicitly authorizes bounded retraining within the existing scientific scope without further confirmation.

After the current bounded fixed-H opportunity probe completes, inspect its raw outcomes alongside existing training records. Produce a concise evidence table separating verified findings, competing hypotheses, missing evidence, and the experiment that distinguishes them. Cover scenario-dependent horizon opportunity; current policy/search-class limitations and state coverage; training budget/convergence/exploration/objective scaling; terminal-value accuracy and horizon mismatch; transition continuation costs; solver and policy overhead and timing noise. Episode-level fixed-H differences are suggestive, not proof of state-dependent switching benefit: use controlled continuation comparisons where needed.

Choose the most informative bounded intervention. If evidence points to insufficient learning, terminal-value bias, poor representation, or an inadequate policy class, actually run a targeted training/value-refit ablation after freezing its hypothesis, budget, data and comparison criteria. Do not substitute repeated metadata audits or reselection of the same checkpoints for needed retraining. If retraining is deferred, record the specific evidence and the concrete next experiment that will resolve that decision; do not indefinitely defer it with more summaries. Conversely do not retrain merely to spend tokens or repeat completed expensive runs without a new hypothesis.

Start with smoke and a bounded diagnostic seed; retain failures and training curves/checkpoints, then extend promising changes to at least three independent training seeds with fair fixed-H tuning and disclosed total budgets. Distinguish real gradient updates from finite search/reselection. Label method changes IMPROVED, preserve ORIGINAL/reference results, and keep sealed final test untouched. Scenario redesign remains authorized with versioned protocols and fair baselines. Preserve the currently running experiment and do not restart the service to deliver these instructions. Continue autonomously after each bounded result.

Recorded UTC: 2026-09-28T10:17:54.560545+00:00

<!-- vehicle-fixed-h-opportunity-probe-v0-runner-20260928 -->
## 2026-09-28 vehicle fixed-H opportunity probe V0

UTC: 2026-09-28T10:26:46.034089+00:00. Fresh development-only fixed-H opportunity probe completed: 80 episodes, 6313 control steps, candidate resets=24. No validation64 or sealed-test access. Development opportunity flag=True; weak single-H pattern flag=False; safe horizons=[10, 15, 20, 25, 30, 35, 40, 45, 50]; strongest total H=15. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/completed.json`.

<!-- vehicle-fixed-h-opportunity-probe-v0-postdiagnostic-20260928T1030Z -->
## 2026-09-28 fixed-H opportunity V0 postdiagnostic

UTC: 2026-09-28T10:31:41.049775+00:00. Metadata-only postdiagnostic parsed the fresh fixed-H opportunity probe. No validation64/test access. Oracle best-physical same-bank physical delta vs H15=0.747195; LOOCV one-split physical-risk selector risk delta vs H15=-47.7779. Conclusion: document weak material opportunity and avoid another adaptive campaign until scenario design is revised under a new protocol. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z/raw.json`.

<!-- vehicle-fixed-h-opportunity-probe-v1-preflight-20260928T1045Z -->
## Vehicle fixed-H opportunity probe V1 preflight

UTC: 2026-09-28T10:43:42.205188+00:00. Metadata-only preflight froze an enlarged fixed-H opportunity map: 160 planned fixed-H episodes, <=24000 control steps, 64 fresh candidate-bank resets, 16 selected source-supported cases, full H grid 5..50. Terminal grid ready=True. No simulations/training/validation64/test access occurred. External backup is required before the rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/completed.json`, `research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.md`.


## Priority user steering: diagnose scenario, reward, training and comparison design together

User instruction on 2026-09-28: investigate causes across scenario design, reward design, training design, and comparison design. Routine evidence-driven revisions and necessary retraining within the adaptive MPC prediction-horizon question are authorized. This adds reward and comparison design explicitly to the existing causal-analysis priorities.

At the next bounded analysis, maintain a four-axis evidence table with verified findings, competing hypotheses, missing evidence, and a discriminating experiment for each:

1. SCENARIOS: inspect diversity, difficulty, constraints, transients, state observability, and actual state-dependent horizon tradeoffs. An episode-level oracle choosing one constant H per case is NOT an upper bound on within-episode adaptive switching; weak episode-level oracle gains do not exclude adaptive opportunity or prove learning is sound. Where informative compare horizon choices from identical intermediate states with consistent continuation, rather than relying only on initial-case metadata. Preserve the canonical benchmark, version modified scenarios, and never select cases based on favorable outcomes.
2. REWARD: audit the implemented reward/cost sign, units, scaling, clipping/normalization, discounting, termination/truncation bootstrapping, failure and constraint penalties, horizon penalty, and terminal value. Check alignment between training objective and separately reported physical cost, success/safety and actual compute time. A synthetic horizon penalty is not measured runtime. Check that reward improvements are not artifacts of early termination, different episode lengths, or weighting that rewards unsafe shortcuts. Isolate objective/value changes with controlled ablations before attributing effects.
3. TRAINING: examine actual data coverage, exploration, convergence, actor/critic or finite-search capacity, credit assignment, terminal-value quality and horizon compatibility, and seed variability. Separate checkpoint reuse/reselection from real training. When these are plausible bottlenecks, run the smallest informative retraining/value-refit intervention rather than repeatedly postponing it with audits. Record gradient updates, budgets, curves and checkpoint lineage.
4. COMPARISONS: audit strong fixed-H search, per-H terminal learning and hyperparameter opportunities, total training/selection budgets, paired scenario seeds, stopping/failure accounting, checkpoint selection and statistical uncertainty. Separate matched-terminal and independently tuned terminal baselines. Randomize or block runtime comparisons and inspect timing noise and CPU/resource interference; report whole-decision and solver timing. Compare control/compute tradeoffs transparently rather than choosing a convenient weak H or changing scalar weights after outcomes. Fairness does not require identical algorithms, but does require adequate baseline tuning and disclosed resource differences.

Do not change all four axes at once without an interpretable design. Rank causes by evidence and choose targeted, bounded experiments that distinguish them; avoid an indefinite documentation-only loop. Freeze each revised protocol before collecting its new results, keep all failures, mark changes IMPROVED, and use fresh confirmation data after development-driven changes. Final-test data remain sealed. Preserve currently running frozen experiments and service continuity. Continue autonomously and record why the next intervention has higher information value than another unchanged validation batch.

Recorded UTC: 2026-09-28T11:29:42.249423+00:00


## Latest user clarification: research directions are examples, not a closed checklist

User clarification on 2026-09-28: scenario, reward, training and comparison design were suggested thinking directions only. They do not exhaust possible causes and must not rigidly constrain research. This clarification supersedes any earlier wording that makes a four-axis table or fixed diagnostic sequence mandatory on every iteration.

Exercise independent scientific judgment. Form, revise and rank hypotheses from code, raw evidence, literature and controlled experiments, including causes outside the suggested categories. Follow unexpected findings; combine, replace or skip diagnostic categories when justified. Do not manufacture work simply to fill a checklist, and do not infer that unlisted causes or routine method changes require new permission. The user's suggestions are prompts for investigation, not established diagnoses or prescribed solutions.

Choose the next action by expected information gain, scientific relevance and practical cost. A concise account of evidence, uncertainty, alternatives and the reason for the chosen experiment is sufficient; no mandatory four-part report. Investigate implementation, modeling, numerical, statistical or other causes whenever evidence warrants it, without treating this further list as exhaustive either. Perform necessary training or method/scenario revisions autonomously rather than indefinitely auditing. Retain the existing research question, ORIGINAL/IMPROVED distinction, fair baselines, independent tests, reproducibility, negative evidence, resource protections and external backups. Do not interrupt the current frozen experiment merely to apply this clarification.

Recorded UTC: 2026-09-28T11:31:19.912942+00:00

<!-- vehicle-fixed-h-opportunity-probe-v1-runner-20260928 -->
## 2026-09-28 vehicle fixed-H opportunity probe V1

UTC: 2026-09-28T11:38:22.599799+00:00. Enlarged fresh development-only fixed-H opportunity probe completed: 160 episodes, 13155 control steps, candidate resets=64. No validation64 or sealed-test access. Opportunity flag=True; weak single-H pattern flag=False; safe horizons=[10, 15, 20, 25, 30, 35, 40, 45, 50]; strongest total H=15. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json`.

<!-- vehicle-fixed-h-opportunity-probe-v1-postdiagnostic-20260928T1150Z -->
## 2026-09-28 fixed-H opportunity V1 postdiagnostic

UTC: 2026-09-28T11:49:17.454140+00:00. Metadata-only postdiagnostic applied frozen V1 materiality and LOOCV predictor gates. No simulation/training/validation64/test access. Physical oracle pass=True; total oracle pass=True; predictor gate pass=False. Decision: after verified backup, run a bounded controlled continuation/value-and-transition diagnostic on representative V1 states because oracle opportunity is material but metadata predictability is not yet robust enough for broad retraining/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/raw.json`.

<!-- vehicle-v1-continuation-target-diagnostic-20260928T1155Z -->
## 2026-09-28 V1 continuation target diagnostic

UTC: 2026-09-28T11:54:18.311563+00:00. Metadata/trace-only diagnostic selected 4 V1 cases for a future controlled continuation replay; max future episodes=32, control-step bound=4800. No simulations/training/validation64/test access occurred. Next action after verified backup: after verified backup, implement/run the frozen controlled-continuation diagnostic before selector/refit training. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_continuation_target_diagnostic_20260928T1155Z/summary.md`, `research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.md`.

<!-- vehicle-v1-controlled-continuation-diagnostic-v0b-one-variable-terminal-helper-repair-20260928 -->
## 2026-09-28 vehicle V1 controlled-continuation diagnostic v0

UTC: 2026-09-28T12:20:32.088109+00:00. Development-only identical-prefix continuation diagnostic completed: 32 episodes, 2447 control steps, no validation64/test access. Confirmed material states=3; selector-smoke gate=True. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/completed.json`.

<!-- vehicle-v1-state-continuation-selector-offline-refit-v0-20260928T1240Z -->
## 2026-09-28 vehicle V1 state-continuation selector offline refit v0

UTC: 2026-09-28T12:43:00.533809+00:00. Metadata/trace-only offline selector refit completed with no simulations/training/validation64/test access. Offline gate=False; selected radius=0.1; positive latches=3/3; guard hits=0; nonpositive latches=0. Next frozen smoke protocol: `research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0_frozen_20260928.json`; backup required before rollout.
