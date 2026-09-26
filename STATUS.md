# 接管状态（2026-09-26）

当前结论：尚不能宣称复现成功。部署及证据迁移进行中，服务存活以服务器 state/heartbeat.json、systemctl 和实际 PID 为准。

已直接核查：D 盘目标分支 codex/change-aware-probabilistic-mppi，HEAD a9aea0c；有两处既有受跟踪修改和大量未跟踪历史文件。D 盘 Bøhn 原型只有 vehicle/cartpole 环境与 cartpole MPC，无可见正式训练结果。原始 git status/diff/branch/log/remote 和25份相关文件清单保存在 docs/bohn2021_takeover。

WSL 工作区 HEAD cad9f76，独立脏工作树。具有作者固定来源、原方法/改进方法、多 seed 模型、固定H网格以及约32GB原始证据。旧报告结论只作线索，迁移后以哈希复核结果为准。

最新 latency_tree_2026-09-26：文件显示 vehicle seeds 0/1/2、pendulum seed0 四个完成标记；pendulum seed1 仅 started/threshold_reference，seed2 未启动。旧训练 PID1694525不存在，Windows 接续 PID26188、7108不存在；状态 active:true 已失真。保留全部状态与失败/中断证据。

下一步：完整迁移及哈希审计；原 Python3.7/TF1 环境移植；两任务一步工程 smoke；登记跨主机恢复规则；恢复有信息价值的下一项工作。由于浅树选择使用实测时间，禁止将 WSL/AWS 计时当同质样本直接混用。

API smoke 已通过 gpt-5.5/xhigh，未切换模型。密钥在仓库外受限文件，本文不含凭据。GitHub 现有远程可写，分块外部恢复备份由独立程序校验 SHA256。

<!-- latency-tree-recovery-migration-amendment-20260926 -->
## 2026-09-26 recovery amendment status

Frozen recovery/migration amendment now governs further latency-tree work. Backup status from supervisor context is verified (`remaining_changed_files=0`, release `bohn-aws-evidence-20260926`, commit `1e93d44d9c0b10c1a1452eb12a88459c382b9e55`). Current scientific status remains: no reproduction success claim; final test unauthorized/sealed. Next action: exact metadata-only pendulum inventory, then decide an AWS-only paired remeasurement/validation or AWS-only pendulum recovery experiment under the amendment.

<!-- pendulum-inventory-interpretation-20260926 -->
## 2026-09-26 pendulum inventory finalized

UTC: 2026-09-26T12:23:05.106778+00:00. Metadata-only inventory `experiments/bohn2021_aws/pendulum_inventory_metadata.py` completed with no simulations, no validation reads, and no sealed-test reads. pendulum_s0 is a completed training artifact (selected `g1_c10`, 66 completed markers). pendulum_s1 is a stale/interrupted threshold-reference run (progress {'episodes': 1, 'expected': 12, 'pid': 1694525, 'steps': 13}, dead PIDs [1694525], tmp files 1). pendulum_s2 is absent/unstarted at train root. This is not control-performance evidence. The partial WSL pendulum_s1 timing-sensitive work must be counted as interrupted budget and must not be spliced into AWS timing objectives. Final test remains sealed/unauthorized; validation64 remains unopened for post-amendment model selection.

Current next action: use the post-inventory preflight output to freeze the next bounded experiment. Do not rerun completed vehicle diagnostics or the pendulum inventory unless artifact hashes are missing.

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
