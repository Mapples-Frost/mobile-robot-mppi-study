# 复现协议索引

当前为迁移/审计协议v1，非新正式效果实验。工程 smoke 不构成复现或速度结论。主任务仅Bøhn 2021 vehicle和inverted pendulum。

继承的正式浅树协议：docs/protocols/bohn2021_latency_tree_2026-09-26.md；机器冻结与哈希：research_artifacts/bohn2021_reproduction_2026-09-17/results/latency_tree_2026-09-26/registration.json。原方法和历次改进协议/结果均保留。不得覆盖或倒改既有验收条件。

当前三训练seed=0/1/2；H=5,10,...50；每seed fit12、训练内select16；每任务validation64与sealed test128。独立固定终端、共用终端固定H完整搜索和所有失败必须报告。具体奖励、安全和2%非劣/3%成本/10%时间门槛按继承协议逐字执行，不因结果修改。浅树属IMPROVED，不是ORIGINAL SAC。

任何恢复涉及计时排名改变必须先登记迁移amendment，记录硬件、保留模型、重测区块与新增预算。不打开封存test结果，不将已经暴露test当作新独立证据。独立test前冻结代码、配置、模型与选择规则；必须先通过validation和独立审计。

在正式开始新的学习路线前，由长期worker另存版本化完整协议，列明train/validation/test生成、种子、所有训练/调参/仿真预算、固定/自适应选择规则、checkpoint、指标和失败标准。不得把本索引当作已经完成新协议冻结。

<!-- latency-tree-recovery-migration-amendment-20260926 -->
## 2026-09-26 amendment index

Additional governing document for migrated latency-tree work: `docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md`. It preserves the inherited preregistration unchanged but adds recovery rules for host-specific timing, incomplete pendulum runs, behaviorally fixed trees, validation access, and final-test gating. The amendment is stricter than the historical protocol where necessary and does not weaken any original success gate.

<!-- vehicle-validation-gate-freeze-20260926 -->
## 2026-09-26 vehicle validation gate freeze

UTC: 2026-09-26T13:05:41.673579+00:00. Metadata-only no-validation gate frozen at `research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json`. Validation bank content opened=false; sealed test content opened=false; simulations=0. Gate froze 42 unique rollout arms, 2688 planned validation episodes over case indices only, and 12 bounded shards. Vehicle learned s0 and s1 are preclassified as fixed/nonadaptive by structure; only s2 is structurally switching. External backup of this new gate is required before formal validation64 rollout; final test remains unauthorized.

<!-- vehicle-validation64-shard-runner-dryrun-20260926 -->
## 2026-09-26 vehicle validation64 shard runner dry-run

UTC: 2026-09-26T13:16:00.900755+00:00. New runner `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py` dry-run completed with validation_accessed=false, test_accessed=false, simulations=0. It verified the frozen gate, source/model/policy hashes, bank stat metadata without opening validation/test content, and formal-run backup-proof requirements. Artifacts: `research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/dry_run.json`, `research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/summary.md`. Runner and dry-run outputs now require external backup before any validation64 content access.

<!-- post-dryrun-backup-blocker-audit-20260926 -->
## 2026-09-26 post-dry-run backup blocker audit

UTC: 2026-09-26T13:21:03+00:00. Metadata-only blocker audit completed with no validation/test bank content opened and no simulations. The dry-run/gate/runner hashes remain consistent, but no adequate post-dry-run external backup proof is present in the repository. A new backup request was written at `research_artifacts/aws_backup_proofs/REQUEST_POST_DRYRUN_BACKUP_20260926T132103.json`. Formal vehicle validation64 shard0 remains blocked until the supervisor provides a verified external backup proof with `backup_verified=true`, `remaining_changed_files=0`, commit, GitHub release asset/download SHA256, runner/gate hashes, and dry-run artifact hashes. Sealed test remains closed.

<!-- vehicle-validation64-shard00-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard00 audit

UTC: 2026-09-26T14:40:37+00:00. Post-run audit of formal shard00 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard00 has 224 episodes and 19832 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard00: s0 fixed H25 (408 steps), s1 fixed H25 (390 steps), s2 used H25/H35 ({'25': 287, '35': 60}) with 18 switches. This is only 1/12 validation evidence and not final model selection. New formal evidence requires external backup before shard01; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_20260926T144050.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard00-audit-v2-schema-repair-20260926 -->
## 2026-09-26 vehicle validation64 shard00 audit-v2 schema repair

UTC: 2026-09-26T14:45:06+00:00. Corrective audit-v2 classified audit-v1's nonzero exit as a registry schema false negative: the shard run registry records `commit_sha=523ec69d0986ebde3f10f24fd6dd8ad7df30e1f9` while audit-v1 required a `git_commit` field. Substantive v1 checks were complete and passed: completed-hash audit=True, episode trace/hash audit=True, aggregate replay=True, schedule mismatches=[]. validation_accessed=true because already-created shard00 results were read; sealed test accessed=false; simulations=0; training steps=0. Shard00 remains only 1/12 validation evidence, not model selection. External backup covering shard00, audit-v1, audit-v2, docs, registries and backup requests is required before shard01.

<!-- vehicle-validation64-shard01-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard01 audit

UTC: 2026-09-26T16:04:04+00:00. Post-run audit of formal shard01 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard01 has 224 episodes and 20036 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard01: s0 fixed H25 (457 steps), s1 fixed H25 (420 steps), s2 used horizons {'25': 526, '35': 18} with 16 switches. This is only 2/12 validation evidence and not final model selection. New formal evidence requires external backup before shard02; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD01_AUDIT_20260926T160417.json`. Sealed test remains closed.

<!-- vehicle-validation64-shard02-modern-runtime-failure-recovery-20260926 -->
## 2026-09-26 vehicle validation64 shard02 modern-runtime failure recovery

UTC: 2026-09-26T16:17:32+00:00. The planned shard02 run `20260926T161356_fb71c8d7` failed before any episode/control step because it was accidentally launched with the modern interpreter, where TensorFlow is unavailable (`ModuleNotFoundError: No module named 'tensorflow'`). The failed run had already opened the validation64 bank and wrote only `run_started.json` plus `schedule.json`; no `episodes/`, `progress.json`, `raw.json`, `summary.md`, or `completed.json` existed. Recovery archived the empty partial directory to `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/failed_shard02_20260926T161356_modern_tf_missing` so the frozen runner can later create `shard02` cleanly. Recovery itself reopened no validation bank content, opened no sealed test content, and ran 0 simulations / 0 control steps / 0 gradient steps. Legacy runtime import under `/home/mapples/.local/share/bohn2021-python37/bin/python` passed with TensorFlow `None`. A verified external backup covering the failure archive, recovery artifacts, docs, registry and backup request is required before retrying shard02 with the legacy interpreter.
