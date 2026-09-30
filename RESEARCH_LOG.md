# 研究日志

2026-09-26：启动接管，发现D/WSL源工作区实质分叉。服务器挂载正常，无既有科研服务；API最高标准xhigh smoke成功。WSL训练与Windows续接均中断，保留状态。开始原始资料迁移、环境复制、持久化调度器与外部备份部署。后续每轮自动追加，原始证据优先。


## 2026-09-26T11:09:33.974241+00:00



## 2026-09-26T11:17:18.045214+00:00



## 2026-09-26T11:19:38.695208+00:00



## 2026-09-26T11:27:51.496738+00:00



# AWS短训练可行性实测（2026-09-26）

范围：工程探针，不是正式复现结论；没有读取validation/test。t3a.medium，EPYC7571，2vCPU/4GiB。固定H25(vehicle)/H30(pendulum)，seed2609268801，各200步，前100步学习预热，batch64，buffer2000。与正式50000容量回放缓存不同，不能把本次内存当作完整训练上限。

vehicle：38.8967秒，101次梯度更新，200步均有限，参数实际变化且全部有限，0个求解失败步，两次goal终止。
pendulum：21.7766秒，101次梯度更新，200步均有限，参数实际变化且全部有限，9/200求解失败步；两个已结束回合分别steps与constraint。失败全部保留，不能把工程跑通当作控制成功。

控制step（含环境工作，非纯solver计时）：vehicle mean/median/P95为176.2/169.3/215.6ms；pendulum为93.4/80.5/116.7ms。整个连续进程峰值RSS427664KiB，约417.6MiB。两任务训练时间均不含初始化（各约5秒）。

仅按这次200步平均速度线性折算15000步：vehicle约48.6分钟，pendulum约27.2分钟。不是正式训练工期：完整训练更新比例更高、后期求解难度/策略会变、未含完整评估，也未验证CPU积分耗尽后的速度或费用。不能据此承诺全天持续吞吐。

探针运行期间bohn-research让出资源，由独立systemd单元运行，ExecStopPost恢复主服务；探针退出0，主服务已active。运行命令为legacy Python调用experiments/bohn2021_aws/training_feasibility_probe.py；脚本、配置、checkpoint、逐步记录与registry均保留。


## 2026-09-26T11:42:52.251639+00:00



## 2026-09-26T11:50:07.741850+00:00


<!-- latency-tree-recovery-migration-amendment-20260926 -->
## 2026-09-26 latency-tree recovery/migration amendment

Wrote `docs/bohn2021_takeover/LATENCY_TREE_RECOVERY_MIGRATION_AMENDMENT_20260926.md`. No simulations were run. No validation or sealed-test outcomes were read. The amendment records the vehicle selection-noise diagnosis, separates WSL and AWS timing evidence, classifies behaviorally fixed trees as fixed-H comparators, keeps final test sealed, and sets the next queue to a metadata-only pendulum inventory followed by AWS-only paired timing/validation or AWS-only pendulum recovery as appropriate. Supervisor context reports the GitHub release backup as verified with zero remaining changed files before new formal evidence is accumulated.


## 2026-09-26T12:15:53.643215+00:00



## 2026-09-26T12:22:32.764231+00:00

<!-- pendulum-inventory-interpretation-20260926 -->
## 2026-09-26 pendulum inventory finalized

UTC: 2026-09-26T12:23:05.106778+00:00. Metadata-only inventory `experiments/bohn2021_aws/pendulum_inventory_metadata.py` completed with no simulations, no validation reads, and no sealed-test reads. pendulum_s0 is a completed training artifact (selected `g1_c10`, 66 completed markers). pendulum_s1 is a stale/interrupted threshold-reference run (progress {'episodes': 1, 'expected': 12, 'pid': 1694525, 'steps': 13}, dead PIDs [1694525], tmp files 1). pendulum_s2 is absent/unstarted at train root. This is not control-performance evidence. The partial WSL pendulum_s1 timing-sensitive work must be counted as interrupted budget and must not be spliced into AWS timing objectives. Final test remains sealed/unauthorized; validation64 remains unopened for post-amendment model selection.

Next research step: run an AWS-only recovery/candidate preflight and then choose either a vehicle-only paired development remeasurement/freeze or a fresh AWS pendulum recovery block. No validation/test outcomes were opened by this inventory.


## 2026-09-26T12:27:47.781793+00:00



## 2026-09-26T12:33:39.360386+00:00

<!-- post-amendment-vehicle-freeze-diagnostic-20260926 -->
## 2026-09-26 post-amendment vehicle freeze diagnostic

UTC: 2026-09-26T12:34:18.073579+00:00. Metadata-only diagnostic completed with no simulations, no validation64 content/outcome read, and no sealed-test read. Vehicle learned candidates s0/s1/s2 and fixed-H comparator inventory were hashed. Next actual experiment is frozen as a non-formal AWS-only vehicle smoke paired timing/control block versus fixed H25; formal validation remains unopened and whole two-task validation remains blocked by pendulum_s1/s2 recovery. Artifacts: `research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/raw.json`, `research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/summary.md`, `research_artifacts/aws_diagnostics/post_amendment_vehicle_freeze_diagnostic/vehicle_development_timing_freeze.json`. New artifacts require backup before unique formal evidence accumulates.


## 2026-09-26T12:37:36.097092+00:00

<!-- vehicle-development-smoke-pairing-20260926 -->
## 2026-09-26 vehicle development smoke pairing

UTC: 2026-09-26T12:51:06.985559+00:00. Non-formal AWS-only vehicle smoke completed on `vehicle_smoke_bank` with 24 episodes and 1764 control steps. Replay passed=True; validation_accessed=false; test_accessed=false. Use only for engineering readiness/timing-boundary checks, not validation/model selection. Artifacts: `research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing/raw.json`, `research_artifacts/aws_diagnostics/vehicle_development_smoke_pairing/summary.md`. New artifacts require backup before formal evidence.


## 2026-09-26T12:52:06.762954+00:00

<!-- vehicle-smoke-artifact-digest-20260926 -->
## 2026-09-26 vehicle smoke artifact digest

UTC: 2026-09-26T12:56:46.885426+00:00. Metadata-only audit of the non-formal AWS vehicle smoke outputs completed. Top-level completed hash check passed=True; episode completed hash check passed=True; independent replay passed=True with 12 pairs. Seed2 H35 trigger count=8 and intervals=[{'episode_index': 2, 'repeat': 0, 'case': 1, 'start_step': 14, 'end_step': 16, 'length': 3}, {'episode_index': 2, 'repeat': 0, 'case': 1, 'start_step': 31, 'end_step': 31, 'length': 1}, {'episode_index': 13, 'repeat': 1, 'case': 1, 'start_step': 14, 'end_step': 16, 'length': 3}, {'episode_index': 13, 'repeat': 1, 'case': 1, 'start_step': 31, 'end_step': 31, 'length': 1}]. No simulations were run; validation_accessed=false; test_accessed=false. This remains engineering/development evidence only, not model selection or reproduction evidence. Artifacts: `research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest/raw.json`, `research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest/summary.md`, `research_artifacts/aws_diagnostics/vehicle_smoke_artifact_digest/completed.json`. New digest artifacts require backup before formal evidence.


## 2026-09-26T12:57:41.142328+00:00



## 2026-09-26T13:05:00.691092+00:00

<!-- vehicle-validation-gate-freeze-20260926 -->
## 2026-09-26 vehicle validation gate freeze

UTC: 2026-09-26T13:05:41.673579+00:00. Metadata-only no-validation gate frozen at `research_artifacts/aws_diagnostics/vehicle_validation_gate_20260926/vehicle_validation_gate_20260926.json`. Validation bank content opened=false; sealed test content opened=false; simulations=0. Gate froze 42 unique rollout arms, 2688 planned validation episodes over case indices only, and 12 bounded shards. Vehicle learned s0 and s1 are preclassified as fixed/nonadaptive by structure; only s2 is structurally switching. External backup of this new gate is required before formal validation64 rollout; final test remains unauthorized.


## 2026-09-26T13:09:07.097404+00:00



## 2026-09-26T13:14:06.893422+00:00

<!-- vehicle-validation64-shard-runner-dryrun-20260926 -->
## 2026-09-26 vehicle validation64 shard runner dry-run

UTC: 2026-09-26T13:16:00.900755+00:00. New runner `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py` dry-run completed with validation_accessed=false, test_accessed=false, simulations=0. It verified the frozen gate, source/model/policy hashes, bank stat metadata without opening validation/test content, and formal-run backup-proof requirements. Artifacts: `research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/dry_run.json`, `research_artifacts/aws_diagnostics/vehicle_validation64_shard_runner_dryrun_20260926/summary.md`. Runner and dry-run outputs now require external backup before any validation64 content access.


## 2026-09-26T13:18:32.806978+00:00

<!-- post-dryrun-backup-blocker-audit-20260926 -->
## 2026-09-26 post-dry-run backup blocker audit

UTC: 2026-09-26T13:21:03+00:00. Metadata-only blocker audit completed with no validation/test bank content opened and no simulations. The dry-run/gate/runner hashes remain consistent, but no adequate post-dry-run external backup proof is present in the repository. A new backup request was written at `research_artifacts/aws_backup_proofs/REQUEST_POST_DRYRUN_BACKUP_20260926T132103.json`. Formal vehicle validation64 shard0 remains blocked until the supervisor provides a verified external backup proof with `backup_verified=true`, `remaining_changed_files=0`, commit, GitHub release asset/download SHA256, runner/gate hashes, and dry-run artifact hashes. Sealed test remains closed.


## 2026-09-26T13:22:18.001852+00:00
Continued with concrete, bounded actions while respecting the validation/test access gates.

Actions completed:
- Created `experiments/bohn2021_aws/post_dryrun_backup_blocker_audit.py`
  - SHA256: `2a2604e6e6a04fc80525c8e70a48d3273180c4cb7b03d4d12fe70245d989651c`
- Ran metadata-only backup gate audit:
  - Experiment ID: `20260926T132103_214d3cd1`
  - Exit status: `0`
  - Runtime: `2.0048s`
  - Validation accessed: `false`
  - Sealed test accessed: `false`
  - New simulations: `0`
  - New gradient steps: `0`
- Verified:
  - Runner SHA intact: `cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e`
  - Gate SHA intact: `5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b`
  - Dry-run artifacts still match expected hashes.
  - No formal validation files currently exist.
  - Adequate post-dryrun external backup proofs found: `0`

Artifacts written:
- `research_artifacts/aws_diagnostics/post_dryrun_backup_blocker_audit_20260926/raw.json`
  - SHA256: `3d9cb90010f6b7e2d96c38c980a4d2290ea0006ded65f42f964cdb68e1a30e39`
- `research_artifacts/aws_diagnostics/post_dryrun_backup_blocker_audit_20260926/summary.md`
  - SHA256: `057e3fcb10206d4eaacbe16883b6525a269b951f5312fc3bbdfeafb26c0da62d`
- `research_artifacts/aws_diagnostics/post_dryrun_backup_blocker_audit_20260926/completed.json`
  - SHA256: `2e270f5eb8d2b3741991d65cb8481fc19df13f6588f037b9a628c916659ced48`
- `research_artifacts/aws_backup_proofs/REQUEST_POST_DRYRUN_BACKUP_20260926T132103.json`
  - SHA256: `446599d571b48c8dcfdba5d624464424ab7eefcdea1682663c8cf7394936a7ed`

State/documents updated:
- `STATUS.md`
- `RESEARCH_LOG.md`
- `RESULTS_AUDIT.md`
- `DECISIONS.md`
- `REPRODUCTION_PROTOCOL.md`
- `EXPERIMENT_REGISTRY.csv`
- Durable supervisor state via `update_state`

Current blocker:
- Formal vehicle validation64 shard0 remains blocked because there is still no verified external backup after the dry-run and after the new backup request/audit.
- I did **not** open `vehicle_validation_bank.json`.
- I did **not** open sealed test.
- I did **not** run formal validation.

Next exact action:
1. Wait for or detect supervisor-provided verified external backup after `2026-09-26T13:21:03Z` covering the dry-run artifacts/docs/registry/proofs and the new request/audit.
2. Once available, create repository-local post-dryrun proof JSON.
3. Then run only:
   - `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 0 --backup-proof <proof> --i-accept-validation-access`
   - legacy interpreter
   - validation budget: exactly 224 episodes, ≤33600 control steps
   - `validation_accessed=true`, `test_accessed=false`

<!-- vehicle-validation64-shard-complete-20260926-shard00 -->
## 2026-09-26 vehicle validation64 shard 00

UTC: 2026-09-26T14:35:01.431574+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19832. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard00/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-26T14:39:09.767574+00:00

<!-- vehicle-validation64-shard00-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard00 audit

UTC: 2026-09-26T14:40:37+00:00. Post-run audit of formal shard00 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard00 has 224 episodes and 19832 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard00: s0 fixed H25 (408 steps), s1 fixed H25 (390 steps), s2 used H25/H35 ({'25': 287, '35': 60}) with 18 switches. This is only 1/12 validation evidence and not final model selection. New formal evidence requires external backup before shard01; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_20260926T144050.json`. Sealed test remains closed.


## 2026-09-26T14:44:47.963963+00:00

<!-- vehicle-validation64-shard00-audit-v2-schema-repair-20260926 -->
## 2026-09-26 vehicle validation64 shard00 audit-v2 schema repair

UTC: 2026-09-26T14:45:06+00:00. Corrective audit-v2 classified audit-v1's nonzero exit as a registry schema false negative: the shard run registry records `commit_sha=523ec69d0986ebde3f10f24fd6dd8ad7df30e1f9` while audit-v1 required a `git_commit` field. Substantive v1 checks were complete and passed: completed-hash audit=True, episode trace/hash audit=True, aggregate replay=True, schedule mismatches=[]. validation_accessed=true because already-created shard00 results were read; sealed test accessed=false; simulations=0; training steps=0. Shard00 remains only 1/12 validation evidence, not model selection. External backup covering shard00, audit-v1, audit-v2, docs, registries and backup requests is required before shard01.


## 2026-09-26T14:46:08.254667+00:00
Concrete research action completed and state preserved.

## This iteration

Executed the planned single experiment:

- Script: `experiments/bohn2021_aws/vehicle_validation64_shard00_audit_v2_schema_repair.py`
- Experiment ID: `20260926T144506_2c69a9d3`
- Exit status: `0`
- Runtime: `2.0048 s`
- Validation accessed: `true` — existing shard00 validation outputs read
- Validation bank reopened: `false`
- Test accessed: `false`
- Sealed test content opened/hashed: `false`
- New simulations/control steps/training: `0 / 0 / 0`

## Audit-v2 result

Audit-v2 passed and repaired the audit-v1 false negative:

- `passed: true`
- `v1_false_negative_repaired: true`
- `episodes_audited: 224`
- `control_steps_audited: 19832`
- Root cause confirmed: audit-v1 expected `git_commit`, while shard00 run registry records:
  - `commit_sha = 523ec69d0986ebde3f10f24fd6dd8ad7df30e1f9`
  - `git_commit = null`

Substantive v1 checks were confirmed as passing:

- completed hash audit: passed
- episode trace/hash audit: passed
- aggregate replay audit: passed
- schedule mismatches: none
- sealed final test remained closed

Key artifact hashes from `completed.json`:

- v2 raw: `7accb91d8b46e6556b2ca3bb039323256919fc797636e4a62fc7d853437989ee`
- v2 summary: `3ca54b54f61871fb67b12d4da6d1ff90694be76f72010050a1995ea616a86c8a`
- v2 backup request: `71cbfc3b5d646c53ff9536888f0e33353e745880ec325f253bd934a067a5cb53`

## Files inspected after run

Inspected:

- `research_artifacts/aws_diagnostics/vehicle_validation64_shard00_audit_v2_schema_repair_20260926/completed.json`
- `raw.json`
- `summary.md`
- `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_AUDIT_V2_20260926T144507.json`
- `research_artifacts/aws_runs/20260926T144506_2c69a9d3/registry.json`
- stdout/stderr logs
- `STATUS.md`
- `EXPERIMENT_REGISTRY.csv`
- backup proof listing

Docs were updated by the script:

- `STATUS.md`
- `RESEARCH_LOG.md`
- `RESULTS_AUDIT.md`
- `DECISIONS.md`
- `REPRODUCTION_PROTOCOL.md`
- `EXPERIMENT_REGISTRY.csv`

## Current gate/blocker

No post-v2 verified external backup proof is present yet. Existing backup proof listing contains only older proofs and backup requests, including the new request:

`research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD00_AUDIT_V2_20260926T144507.json`

Therefore shard1 must not be run yet.

## Scientific status

- Vehicle validation64: `1/12` shards complete; shard00 audit-v2 passed.
- No model selection or reproduction claim is allowed.
- Learned candidates in shard00 remain:
  - s0: fixed H25 only
  - s1: fixed H25 only
  - s2: H25/H35 switching observed
- Method remains `IMPROVED latency-tree`, not ORIGINAL SAC.
- Sealed final test remains closed and unauthorized.
- Pendulum remains partial: s0 complete, s1 interrupted, s2 unstarted.

## Persisted next action

State was updated via `update_state`.

Next iteration should:

1. Check for a verified post-v2 external backup proof covering shard00, audit-v1, audit-v2, registries, docs, and backup requests.
2. If present and adequate, run exactly one formal experiment: vehicle validation64 shard1 with legacy interpreter and sealed test closed.
3. If absent, do not run new formal simulations; preserve backup gate and continue checking/requesting supervisor backup.

<!-- vehicle-validation64-shard-complete-20260926-shard01 -->
## 2026-09-26 vehicle validation64 shard 01

UTC: 2026-09-26T16:00:03.047684+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=20036. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard01/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard01/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-26T16:02:48.292386+00:00

<!-- vehicle-validation64-shard01-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard01 audit

UTC: 2026-09-26T16:04:04+00:00. Post-run audit of formal shard01 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard01 has 224 episodes and 20036 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard01: s0 fixed H25 (457 steps), s1 fixed H25 (420 steps), s2 used horizons {'25': 526, '35': 18} with 16 switches. This is only 2/12 validation evidence and not final model selection. New formal evidence requires external backup before shard02; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD01_AUDIT_20260926T160417.json`. Sealed test remains closed.


## 2026-09-26T16:08:00.273425+00:00



## 2026-09-26T16:12:14.507427+00:00



## 2026-09-26T16:17:04.041826+00:00

<!-- vehicle-validation64-shard02-modern-runtime-failure-recovery-20260926 -->
## 2026-09-26 vehicle validation64 shard02 modern-runtime failure recovery

UTC: 2026-09-26T16:17:32+00:00. The planned shard02 run `20260926T161356_fb71c8d7` failed before any episode/control step because it was accidentally launched with the modern interpreter, where TensorFlow is unavailable (`ModuleNotFoundError: No module named 'tensorflow'`). The failed run had already opened the validation64 bank and wrote only `run_started.json` plus `schedule.json`; no `episodes/`, `progress.json`, `raw.json`, `summary.md`, or `completed.json` existed. Recovery archived the empty partial directory to `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/failed_shard02_20260926T161356_modern_tf_missing` so the frozen runner can later create `shard02` cleanly. Recovery itself reopened no validation bank content, opened no sealed test content, and ran 0 simulations / 0 control steps / 0 gradient steps. Legacy runtime import under `/home/mapples/.local/share/bohn2021-python37/bin/python` passed with TensorFlow `None`. A verified external backup covering the failure archive, recovery artifacts, docs, registry and backup request is required before retrying shard02 with the legacy interpreter.


## 2026-09-26T16:20:14.257420+00:00
Concrete action completed and state preserved.

- Ran metadata-only recovery diagnostic:
  - Experiment: `20260926T161732_f10358dc`
  - Script: `experiments/bohn2021_aws/vehicle_validation64_shard02_modern_failure_recovery.py`
  - Interpreter: `legacy`
  - Exit: `0`
  - Runtime: `6.01s`
  - No simulations / control steps / gradient steps.
  - No sealed test access.
  - Recovery passed.

Key verified facts:
- Failed shard02 attempt `20260926T161356_fb71c8d7` was due to wrong interpreter: modern Python lacks TensorFlow.
- Failed attempt opened validation64 bank metadata and wrote only:
  - `run_started.json`
  - `schedule.json`
- No episode outputs, no `raw.json`, no `summary.md`, no `completed.json`, no control steps.
- Recovery archived partial shard02 to:
  - `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/failed_shard02_20260926T161356_modern_tf_missing`
- Original `shard02` path is now free for a clean retry.
- Legacy TF1 runtime import check passed under:
  - `/home/mapples/.local/share/bohn2021-python37/bin/python`

Artifacts created/verified:
- `research_artifacts/aws_diagnostics/vehicle_validation64_shard02_modern_failure_recovery_20260926/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_validation64_shard02_modern_failure_recovery_20260926/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_validation64_shard02_modern_failure_recovery_20260926/completed.json`
- Backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_SHARD02_MODERN_FAILURE_RECOVERY_20260926T161736.json`
- Final backup addendum:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_SHARD02_MODERN_FAILURE_RECOVERY_FINAL_ADDENDUM_20260926T161800.json`
- Backup blocker note:
  - `research_artifacts/aws_diagnostics/post_shard02_recovery_backup_blocker_check_20260926.md`

Current gate:
- Formal validation remains blocked until an external verified backup proof is available after the recovery/addendum.
- Latest existing proof is still only valid for the pre-failure/pre-recovery state.
- Do not run more formal validation until backup covers recovery artifacts and docs.

Next action after verified backup:
- Run shard02 with the frozen runner using `interpreter=legacy`, not modern:
  - `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py`
  - shard `2`
  - post-recovery backup proof
  - no sealed test access.

State was persisted via `update_state`. Current conclusion unchanged: not reproduced; vehicle validation is partial only, ORIGINAL SAC not established, pendulum still partial, sealed final test closed.

<!-- vehicle-validation64-shard-complete-20260926-shard02 -->
## 2026-09-26 vehicle validation64 shard 02

UTC: 2026-09-26T17:32:27.856583+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19542. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard02/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard02/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-26T17:36:25.011351+00:00

<!-- vehicle-validation64-shard02-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard02 audit

UTC: 2026-09-26T17:37:53+00:00. Post-run audit of formal shard02 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard02 has 224 episodes and 19542 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard02: s0 fixed H25 (497 steps), s1 fixed H25 (403 steps), s2 used horizons {'25': 346, '35': 46} with 15 switches. Vehicle validation64 progress is now 3/12 completed shards with 672 completed episodes and 59410 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard03; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD02_AUDIT_20260926T173807.json`. Sealed test remains closed.


## 2026-09-26T17:42:06.313122+00:00



## 2026-09-26T17:45:31.317440+00:00
已继续执行本轮允许的具体动作；未启动新的正式实验。

已核查证据：

- `research_artifacts/aws_backup_proofs/*.json`：目前没有 `2026-09-26T17:38:45Z` 之后的 verified external backup proof。
- 最新仓库内 proof 仍是：
  - `backup_proof_20260926T162038_after_shard02_recovery_and_addendum.json`
  - 仅覆盖 shard02 失败恢复后的、正式 shard02 retry 之前状态；不能覆盖 shard02 正式输出、17:37:53 audit、17:38:30 final addendum。
- 已检查 shard02 audit 完成证据：
  - `research_artifacts/aws_diagnostics/vehicle_validation64_shard02_audit_20260926/completed.json`
  - audit passed=true
  - validation_accessed=true（读取既有 shard02 输出）
  - validation bank 未重开
  - sealed test 未打开/未 hash
  - 新仿真 / 控制步 / 梯度步：`0 / 0 / 0`
- 已检查 final backup addendum：
  - `REQUEST_BACKUP_AFTER_VALIDATION64_SHARD02_AUDIT_FINAL_ADDENDUM_20260926T173830.json`
  - 明确要求下一 proof 覆盖 shard02 正式输出、audit 输出、audit run registry/stdout/stderr、docs/registry 更新、backup requests/final addendum。
- 已检查 `STATUS.md`、`EXPERIMENT_REGISTRY.csv`、`post_shard02_audit_backup_blocker_check_20260926.md`，三者一致：shard03 之前必须先有新的 verified external backup proof。
- 已通过 `update_state` 持久化当前阶段、blocker、下一步和禁止事项。

当前状态：

- Vehicle validation64：`3/12` shards 已完成并审计。
- 累计正式 validation：`672` episodes，`59410` control steps。
- 另有一次 shard02 modern-interpreter 失败尝试：validation bank 曾打开，但 `0` episodes / `0` control steps / `0` gradient steps，已归档。
- sealed final test 仍关闭且未授权。
- 不能声明复现成功；当前只是 IMPROVED latency-tree 的部分 validation evidence，不是 ORIGINAL SAC。

当前硬阻塞：

- **不得运行 shard03**，因为缺少 shard02 audit/final addendum 之后的 verified external backup proof。
- 需要下一 proof 满足：
  - `backup_verified=true`
  - `remaining_changed_files=0`
  - commit/verified state 晚于 `2026-09-26T17:38:45Z`
  - GitHub release asset/download SHA256 verified
  - runner SHA `cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e`
  - gate SHA `5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b`
  - 覆盖 shard02 formal outputs、audit artifacts、run logs、docs/registry、backup requests/addendum、blocker note。

下一步一旦 proof 出现：

```text
experiments/bohn2021_aws/vehicle_validation64_shard_runner.py
--shard 3
--backup-proof <post-shard02-audit-proof.json>
--i-accept-validation-access
```

解释器必须为 `legacy`；sealed final test 继续关闭。随后先审计 shard03，再决定 shard04。

<!-- vehicle-validation64-shard-complete-20260926-shard03 -->
## 2026-09-26 vehicle validation64 shard 03

UTC: 2026-09-26T18:59:14.269869+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19887. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard03/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-26T19:03:26.934183+00:00

<!-- vehicle-validation64-shard03-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard03 audit

UTC: 2026-09-26T19:04:50+00:00. Post-run audit of formal shard03 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard03 has 224 episodes and 19887 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard03: s0 fixed H25 (460 steps), s1 fixed H25 (550 steps), s2 used horizons {'25': 685, '35': 72} with 34 switches. Vehicle validation64 progress is now 4/12 completed shards with 896 completed episodes and 79297 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard04; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD03_AUDIT_20260926T190503.json`. Sealed test remains closed.


## 2026-09-26T19:08:39.281477+00:00


<!-- post-shard03-audit-backup-gate-state-20260926 -->
## 2026-09-26 post-shard03 audit backup gate state

UTC: 2026-09-26T19:13:04+00:00. Metadata-only state preservation after shard03 audit and user continuation. No simulations, no control steps, no gradient steps, no validation-bank reopen, and no sealed-test access/open/hash occurred in this action. Existing shard03 formal/audit metadata may be hashed for provenance only; this is not model selection, not final-test evidence, and not an ORIGINAL SAC result.

Backup gate result: shard04 formal validation is BLOCKED at this state check. Required proof time remains after `2026-09-26T19:05:30Z` and must cover shard03 formal outputs, shard03 audit outputs, audit run registry/stdout/stderr, docs/registry updates, backup requests, final addendum, and the blocker note. Latest local proof observed: `research_artifacts/aws_backup_proofs/backup_proof_20260926T190416_after_shard03_formal_before_audit.json` at `2026-09-26T19:04:16.474763+00:00`; adequate proofs found: 0.

Vehicle validation64 progress remains 4/12 completed shards, 896 formal validation episodes, 79297 formal validation control steps, plus the archived modern-interpreter shard02 failed attempt with 0 episodes/control steps. Sealed final test remains closed and unauthorized.

Next action if and only if a verified external backup proof satisfying the gate appears: run exactly one formal experiment, `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py --shard 4 --backup-proof <post-shard03-audit-proof> --i-accept-validation-access`, with the legacy interpreter; then audit shard04 before any further shard. Do not use the modern interpreter for the TF1 runner.



## 2026-09-26T19:14:25.964112+00:00

<!-- vehicle-validation64-shard-complete-20260926-shard04 -->
## 2026-09-26 vehicle validation64 shard 04

UTC: 2026-09-26T20:27:57.550241+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=20101. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard04/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard04/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-26T20:31:26.581895+00:00

<!-- vehicle-validation64-shard04-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard04 audit

UTC: 2026-09-26T20:32:40+00:00. Post-run audit of formal shard04 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard04 has 224 episodes and 20101 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard04: s0 fixed H25 (440 steps), s1 fixed H25 (321 steps), s2 used horizons {'25': 220, '35': 32} with 14 switches. Vehicle validation64 progress is now 5/12 completed shards with 1120 completed episodes and 99398 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard05; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD04_AUDIT_20260926T203240.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD04_AUDIT_FINAL_ADDENDUM_20260926T203240.json`. Sealed test remains closed.


## 2026-09-26T20:34:22.718955+00:00
Continued authorized research with one concrete experiment and preserved state.

## Action completed

Ran the prepared shard04 post-run audit:

- Experiment ID: `20260926T203240_92122dd6`
- Script: `experiments/bohn2021_aws/vehicle_validation64_shard04_audit.py`
- Script SHA256: `dfc37ca495eab9668ce61d254356b8228f995ac371959cdf52b926106f80147c`
- Interpreter: `modern`
- Exit status: `0`
- Runtime: `14.0116 s`
- Validation access: yes, only existing shard04 output artifacts
- Validation bank reopened: no
- Sealed final test opened/hashed: no
- New simulations/control steps/gradient steps: `0 / 0 / 0`

## Shard04 audit result

Audit passed.

Key artifacts:

- Audit raw:  
  `research_artifacts/aws_diagnostics/vehicle_validation64_shard04_audit_20260926/raw.json`  
  SHA256 `bd80da559cc5dd1358401a4480b9a92926d55efeaea9c9b9eb951c9002553d58`

- Audit summary:  
  `research_artifacts/aws_diagnostics/vehicle_validation64_shard04_audit_20260926/summary.md`  
  SHA256 `179c6d78edbe0b4eeeb66a56a0c572a9e0c3b692f3934fe3f180759e61e3242c`

- Audit completed marker:  
  `research_artifacts/aws_diagnostics/vehicle_validation64_shard04_audit_20260926/completed.json`  
  SHA256 `07a735d4fc470550c3b2202f2330e1191ac908748d230905c320076f9e9f2ca1`

Audit checks:

- Completed hash audit: passed, `1574` records
- Episode dirs: `224`
- Episode trace/hash audit: passed
- Aggregate replay: passed
- Shard04 budget: `224` episodes, `20101` control steps, upper bound `33600`

Learned candidates in shard04:

- `learned_s0`: fixed H25, `5` episodes, `440` steps, `5/5` success
- `learned_s1`: fixed H25, `4` episodes, `321` steps, `4/4` success
- `learned_s2`: adaptive in shard, H25/H35, `3` episodes, `252` steps, `3/3` success, `14` switches

## Updated cumulative status

Vehicle validation64 is now:

- `5/12` shards completed and audited
- `1120` formal validation episodes
- `99398` formal validation control steps
- plus one preserved failed shard02 modern-interpreter attempt with `0` episodes/control steps/gradient steps

This remains partial vehicle validation evidence only for the IMPROVED latency-tree method. It is not ORIGINAL SAC, not final-test evidence, and not a reproduction-success claim.

## Backup gate

Created backup request artifacts:

- `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD04_AUDIT_20260926T203240.json`  
  SHA256 `df3551e971b2c70df752547607d190f84d57bb04bda6f9f69c1d44f72aa1cab7`

- `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD04_AUDIT_FINAL_ADDENDUM_20260926T203240.json`  
  SHA256 `f51b7cc2b6d020920d3382de6e0b3bde1f5f94aaeb9a475b3b417fa72f913060`

- Blocker note:  
  `research_artifacts/aws_diagnostics/post_shard04_audit_backup_blocker_check_20260926.md`  
  SHA256 `5a263e209b394ffe81c8316f3cabc8ccfccb27997746c1239f85696d4f517fbb`

Checked for local post-shard04 backup proofs: none found matching `backup_proof_20260926T20*.json`.

Important: the supervisor backup in context at `2026-09-26T20:32:14.890193+00:00` predates the shard04 audit at `20:32:40Z`, so it is not sufficient for shard05.

## Documents/state preserved

The audit updated and hashed:

- `STATUS.md`
- `RESEARCH_LOG.md`
- `RESULTS_AUDIT.md`
- `DECISIONS.md`
- `REPRODUCTION_PROTOCOL.md`
- `EXPERIMENT_REGISTRY.csv` includes audit run `20260926T203240_92122dd6`

I also persisted the handoff via `update_state`.

## Next iteration

Do not rerun shard04 or its audit.

First check for a verified external backup proof after the shard04 audit/final-addendum/blocker state, covering shard04 formal outputs, shard04 audit artifacts, audit run registry/stdout/stderr, docs/registry updates, backup request, final addendum, and blocker note.

If adequate proof is present, run exactly one formal experiment:

```text
experiments/bohn2021_aws/vehicle_validation64_shard_runner.py
--shard 5
--backup-proof <post-shard04-audit-proof.json>
--i-accept-validation-access
```

Use `legacy` interpreter only. Sealed final test remains closed and unauthorized.

<!-- vehicle-validation64-shard-complete-20260926-shard05 -->
## 2026-09-26 vehicle validation64 shard 05

UTC: 2026-09-26T21:55:13.414681+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19660. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard05/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard05/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-26T21:57:55.402991+00:00

<!-- vehicle-validation64-shard05-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard05 audit

UTC: 2026-09-26T21:59:16+00:00. Post-run audit of formal shard05 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard05 has 224 episodes and 19660 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=False. Learned candidates in shard05: s0 {'episodes': 4, 'steps': 289, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 289}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 3, 'steps': 211, 'success_count': 3, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 211}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 3, 'steps': 195, 'success_count': 3, 'episode_failure_count': 0, 'switches': 4, 'horizon_counts': {'25': 193, '35': 2}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 6/12 completed shards with 1344 completed episodes and 119058 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard06; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_20260926T215916.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_FINAL_ADDENDUM_20260926T215916.json`. Sealed test remains closed.


## 2026-09-26T22:04:18.082999+00:00

<!-- vehicle-validation64-shard05-audit-v2-schema-repair-20260926 -->
## 2026-09-26 vehicle validation64 shard05 audit v2 schema repair

UTC: 2026-09-26T22:05:23+00:00. Repaired post-run audit of formal shard05 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. V2 preserves failed v1 audit `research_artifacts/aws_diagnostics/vehicle_validation64_shard05_audit_20260926/completed.json` and fixes only audit-schema false positives: absent derived gross-decision mean is auxiliary, and shard05 registry's closed-test budget is accepted because episodes/control_steps are zero, sealed_test_bank_content_opened=false, test_authorization=false, and raw/completed/run_started all show test_accessed=false. Shard05 has 224 episodes and 19660 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard05: s0 {'episodes': 4, 'steps': 289, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 289}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 3, 'steps': 211, 'success_count': 3, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 211}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 3, 'steps': 195, 'success_count': 3, 'episode_failure_count': 0, 'switches': 4, 'horizon_counts': {'25': 193, '35': 2}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 6/12 completed formal shards with 1344 episodes and 119058 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and the failed v1 audit with 0 simulations. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard06; v2 request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_20260926T220523.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD05_AUDIT_V2_FINAL_ADDENDUM_20260926T220523.json`. Sealed test remains closed.


## 2026-09-26T22:08:13.656210+00:00


<!-- post-shard05-v2-backup-gate-recheck-20260926T221104 -->
## 2026-09-26 post-shard05 v2 backup gate recheck

UTC: 2026-09-26T22:11:04+00:00. Metadata-only backup inventory recheck before shard06: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. No adequate repository-local verified external backup proof after shard05 v2 audit finalization was found (`backup_proof_20260926T22*.json` count 0; adequate proofs 0). Latest supervisor backup in context remains `2026-09-26T22:04:47.633765+00:00`, which predates shard05 v2 audit/blocker finalization and is insufficient. Shard06 remains backup-gated. New recheck artifacts: `research_artifacts/aws_diagnostics/post_shard05_v2_backup_gate_recheck_20260926T221104/raw.json`, `research_artifacts/aws_diagnostics/post_shard05_v2_backup_gate_recheck_20260926T221104/summary.md`, `research_artifacts/aws_diagnostics/post_shard05_v2_backup_gate_recheck_20260926T221104/completed.json`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_SHARD05_V2_GATE_RECHECK_20260926T221104.json`. Next proof must be after this recheck and after the metadata run registry/stdout/stderr finalize, and must cover shard05 formal outputs, failed v1 audit, passed v2 audit, docs/registry, all shard05-v2 backup requests/addenda/blocker notes, this recheck/request, runner SHA `cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e`, and gate SHA `5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b`. No reproduction or final-test conclusion is permitted; method remains IMPROVED, not ORIGINAL SAC.

<!-- vehicle-validation64-shard-complete-20260926-shard06 -->
## 2026-09-26 vehicle validation64 shard 06

UTC: 2026-09-26T23:24:17.125157+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19137. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard06/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-26T23:28:03.565474+00:00

<!-- vehicle-validation64-shard06-audit-20260926 -->
## 2026-09-26 vehicle validation64 shard06 audit

UTC: 2026-09-26T23:29:08+00:00. Post-run audit of formal shard06 completed with validation_accessed=true (reading shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard06 has 224 episodes and 19137 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard06: s0 359 steps, s1 634 steps, s2 horizons {'25': 270, '35': 33} with 7 switches. Vehicle validation64 progress is now 7/12 completed formal shards with 1568 completed episodes and 138195 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard07; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_20260926T232908.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_FINAL_ADDENDUM_20260926T232908.json`. Sealed test remains closed.


## 2026-09-26T23:31:38.432755+00:00
Concrete action completed.

- Ran `experiments/bohn2021_aws/vehicle_validation64_shard06_audit.py` with the modern interpreter.
- Experiment ID: `20260926T232908_163b4f98`
- Exit status: `0`
- Runtime: `14.01 s`
- Scope: metadata/hash/trace/budget audit only.
- No simulations, no training, no new validation episodes, no control steps, no gradient steps.
- Validation bank was not reopened by the audit.
- Sealed final test remained closed and was not hashed/opened.

Audit passed.

Key artifacts:
- `research_artifacts/aws_diagnostics/vehicle_validation64_shard06_audit_20260926/raw.json`
  - SHA256: `8322dc5a4f2ebf1eacfa772f8038ac0bbcc5284f0da9dbff416ef096d96ebcc0`
- `research_artifacts/aws_diagnostics/vehicle_validation64_shard06_audit_20260926/summary.md`
  - SHA256: `7037c85a7594047953e6877bdc5592da01f80a4c3a138ba6e07f389da0d87275`
- `research_artifacts/aws_diagnostics/vehicle_validation64_shard06_audit_20260926/completed.json`
  - SHA256: `8ecd91ef4c79a8ef252fe9f1ae98403c21a8d515b5373417e39044040b54c830`
- Backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_20260926T232908.json`
  - SHA256: `f5eb58c8a8046913667daf2db54c81833e8ccd4bfc3acf355363c15c77361df4`
- Final addendum:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD06_AUDIT_FINAL_ADDENDUM_20260926T232908.json`
  - SHA256: `eab9e63350ec78146350a7681b2e1709c6c7fe2e5fecd6fdbc243a27123f4d56`
- Backup blocker note:
  - `research_artifacts/aws_diagnostics/post_shard06_audit_backup_blocker_check_20260926.md`
  - SHA256: `ff23003a1b2b5d71047b33abfc2813f916f99f0b61078ea16a7c10c43dbcac3e`

Shard06 audit evidence:
- Episodes audited: `224`
- Control steps audited: `19137`
- Declared upper bound: `33600`
- Completed hash audit: passed over `1574` records
- Episode trace/hash audit: passed
- Aggregate replay: passed
- Learned candidate snapshot:
  - `learned_s0`: 5/5 successes, 359 steps, H25 only
  - `learned_s1`: 8/8 successes, 634 steps, H25 only
  - `learned_s2`: 4/4 successes, 303 steps, H25/H35, 7 switches

Cumulative vehicle validation64 status:
- Formal shards completed: `0–6` / `12`
- Audited shards passed: `0–6`
- Formal episodes completed: `1568`
- Formal control steps completed: `138195`
- Failed formal attempts preserved: one shard02 modern-interpreter TensorFlow-missing attempt, 0 episodes/control steps.
- Failed audit attempts preserved: shard00 v1 schema false negative, shard05 v1 schema false positive.

Current gate:
- Shard07 is blocked until a verified external backup proof after the shard06 audit run finalization exists.
- The latest known verified backup from supervisor context was at `2026-09-26T23:28:51Z`, which predates the shard06 audit at `23:29:08Z`; therefore it is insufficient for shard07.
- State was persisted via `update_state` with the exact next action and backup-gate requirements.

Next action once backup proof is available:
- Reconstruct/verify local post-shard06 backup proof if needed.
- Run `experiments/bohn2021_aws/vehicle_validation64_shard_runner.py` for shard07 with legacy interpreter, sealed test closed.
- Then audit shard07 before any further shard.

<!-- post-shard06-audit-backup-gate-recheck-20260926T233430+0000 -->
## 2026-09-26 post-shard06 audit backup gate recheck

UTC: 2026-09-26T23:34:30+00:00. Metadata-only backup inventory recheck before shard07: validation_accessed=true only for existing shard06 output hashing/counting, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. No adequate repository-local verified external backup proof after shard06 audit finalization and persisted state was found (`backup_proof_20260926T23*.json` count 0; adequate proofs 0). Latest supervisor backup in context `2026-09-26T23:28:51.819071+00:00` predates the shard06 audit and is insufficient. Key shard06 formal/audit/runner/gate hashes matched expected values: True. Shard07 remains backup-gated. New recheck artifacts: `research_artifacts/aws_diagnostics/post_shard06_audit_backup_gate_recheck_20260926T233430+0000/raw.json`, `research_artifacts/aws_diagnostics/post_shard06_audit_backup_gate_recheck_20260926T233430+0000/summary.md`, `completed.json` in the same directory. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_SHARD06_AUDIT_GATE_RECHECK_20260926T233430+0000.json`. Next proof must be after this recheck and after the metadata run registry/stdout/stderr finalize, and must cover shard06 formal outputs, shard06 audit outputs, formal/audit/recheck run logs, docs/registry, shard06 backup request/final addendum/blocker note, this recheck/request, runner SHA `cb3c775808de3213fd1ef6cef5727aec9f7b473ac5d0b1270dca4cb37b44dd0e`, and gate SHA `5797821873cc689129a16818ef80b2260ee5cb1998b270ac5588e77b61bc382b`. Current status remains partial validation evidence only for an IMPROVED latency-tree method, not ORIGINAL SAC, not final-test evidence, and not a reproduction-success claim.


## 2026-09-26T23:36:27.009070+00:00

<!-- vehicle-validation64-shard-complete-20260926-shard07 -->
## 2026-09-26 vehicle validation64 shard 07

UTC: 2026-09-27T00:49:15.611490+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19532. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard07/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard07/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-27T00:51:58.306983+00:00

<!-- vehicle-validation64-shard07-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard07 audit

UTC: 2026-09-27T00:53:19+00:00. Post-run audit of formal shard07 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard07 has 224 episodes and 19532 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard07: s0 {'episodes': 2, 'steps': 170, 'success_count': 2, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 170}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 9, 'steps': 658, 'success_count': 9, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 658}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 6, 'steps': 551, 'success_count': 5, 'episode_failure_count': 1, 'switches': 14, 'horizon_counts': {'25': 532, '35': 19}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 8/12 completed formal shards with 1792 episodes and 157727 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard08; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_20260927T005319.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_FINAL_ADDENDUM_20260927T005319.json`. Sealed test remains closed.


## 2026-09-27T00:56:03.955228+00:00

<!-- vehicle-validation64-shard07-audit-v2-schema-repair-20260927 -->
## 2026-09-27 vehicle validation64 shard07 audit v2 schema repair

UTC: 2026-09-27T00:56:35+00:00. Repaired post-run audit of formal shard07 completed with validation_accessed=true (reading existing shard outputs and failed v1 audit artifacts), sealed test accessed=false, simulations=0, training steps=0. V2 preserves failed v1 audit `research_artifacts/aws_diagnostics/vehicle_validation64_shard07_audit_20260927/completed.json` and fixes only a registry-schema false positive: shard07's run registry records validation budget as `episodes_exact=224` rather than `episodes`/`validation_episodes`. Shard07 has 224 episodes and 19532 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard07: s0 {'episodes': 2, 'steps': 170, 'success_count': 2, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 170}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 9, 'steps': 658, 'success_count': 9, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 658}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 6, 'steps': 551, 'success_count': 5, 'episode_failure_count': 1, 'switches': 14, 'horizon_counts': {'25': 532, '35': 19}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 8/12 completed formal shards with 1792 episodes and 157727 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard08; v2 request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_V2_20260927T005635.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_V2_FINAL_ADDENDUM_20260927T005635.json`. Sealed test remains closed.


## 2026-09-27T00:58:54.184203+00:00
Concrete actions completed.

- Ran the prepared shard07 audit-v2 schema repair as the one bounded experiment:
  - Experiment: `20260927T005634_eb449ed8`
  - Script: `experiments/bohn2021_aws/vehicle_validation64_shard07_audit_v2_schema_repair.py`
  - Script SHA256: `da24d26c5fe85d43b05d3113ec80c40f837022536c0ab9f02a14da0015d62e05`
  - Interpreter: modern
  - Exit: `0`
  - Runtime: `14.0125s`
  - New simulations/control steps/gradient steps: `0 / 0 / 0`
  - Validation access: existing shard07 artifacts only; validation bank not reopened
  - Sealed final test: not accessed/opened/hashed

- Audit-v2 result:
  - Passed: `true`
  - Preserved failed audit-v1 `20260927T005319_5d1112ca`
  - Confirmed v1 failure was schema-only: `registry validation budget episode mismatch`
  - Repair variable only: accept `validation_budget.episodes_exact=224` as equivalent shard episode-budget field
  - Completed hash audit: passed
  - Episode trace/hash audit: passed
  - Aggregate replay: passed
  - Episodes audited: `224`
  - Control steps audited: `19532` / upper bound `33600`

- Key artifacts:
  - Raw: `research_artifacts/aws_diagnostics/vehicle_validation64_shard07_audit_v2_schema_repair_20260927/raw.json`
    - SHA256: `0f4893f37e8a4d6244bdef5970cb34059982668fa5527f83a9c171781c2f0419`
  - Summary: `research_artifacts/aws_diagnostics/vehicle_validation64_shard07_audit_v2_schema_repair_20260927/summary.md`
    - SHA256: `b76c72b2a043fe9596b1a89fdd28f1c596b22d273b758fb6fb7fc1d2b2c665cd`
  - Completed: `research_artifacts/aws_diagnostics/vehicle_validation64_shard07_audit_v2_schema_repair_20260927/completed.json`
    - SHA256: `4764dabfc11f18ed33b29386509a48ef68e7bfe54bc4a92c8e61ca2cf3cccb6f`
  - Run registry: `research_artifacts/aws_runs/20260927T005634_eb449ed8/registry.json`
  - Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_V2_20260927T005635.json`
    - SHA256: `2b5c9534ac8429951cb6fc7fee1e61e11258066b5d3cc3fbe81ab719138143a2`
  - Final addendum: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_V2_FINAL_ADDENDUM_20260927T005635.json`
    - SHA256: `cb76dc76ae62fab34f8ef301964e3b3ec57b28bf8b5fa5d9e3282d6324b38cf6`
  - Post-run-finalized backup addendum written:
    - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD07_AUDIT_V2_RUN_FINALIZED_20260927T005634.json`
    - SHA256: `2539abb366416dfe9a1b069909f65182db375f76ec3ee05cb12831590058d0e7`

- Cumulative vehicle validation64 state:
  - Formal shards completed: `0-7`
  - Accepted audited shards: `0-7`
  - Formal validation episodes completed: `1792 / 2688`
  - Formal validation control steps completed: `157727`
  - Remaining shards: `8-11`
  - Preserved failures:
    - shard02 modern-interpreter TensorFlow-missing formal attempt, 0 episodes/control steps
    - shard00 audit-v1 schema false negative, repaired by v2
    - shard05 audit-v1 schema false positive, repaired by v2
    - shard07 audit-v1 schema false positive, repaired by v2

- Current gate:
  - Shard08 is blocked until a verified external backup proof after shard07 audit-v2 run finalization and the run-finalized addendum exists.
  - Latest supervisor backup context predates shard07 audit-v2, so it is not adequate for shard08.
  - Required next proof must cover shard07 formal outputs, failed v1 audit, passed v2 audit, finalized v2 run logs, docs/registry updates, backup requests/addenda/blocker, and the new run-finalized addendum.

- State preserved:
  - Durable state updated via `update_state`.
  - Existing docs/registry were updated by the audit script.
  - Final-test gate remains closed and unauthorized.

Next action if backup proof appears: run exactly one formal experiment, shard08, with legacy interpreter and sealed test closed.

<!-- vehicle-validation64-shard-complete-20260926-shard08 -->
## 2026-09-26 vehicle validation64 shard 08

UTC: 2026-09-27T02:11:46.710508+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19703. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-27T02:13:58.359309+00:00

<!-- vehicle-validation64-shard08-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard08 audit

UTC: 2026-09-27T02:15:20+00:00. Post-run audit of formal shard08 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard08 has 224 episodes and 19703 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard08: s0 {'episodes': 4, 'steps': 290, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 290}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 3, 'steps': 208, 'success_count': 3, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 208}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 4, 'steps': 314, 'success_count': 4, 'episode_failure_count': 0, 'switches': 8, 'horizon_counts': {'25': 307, '35': 7}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 9/12 completed formal shards with 2016 episodes and 177430 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard09; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_20260927T021520.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_FINAL_ADDENDUM_20260927T021520.json`. Sealed test remains closed.


## 2026-09-27T02:17:44.619381+00:00
Concrete actions completed.

### Evidence inspected
- `STATUS.md`
- `EXPERIMENT_REGISTRY.csv` tail
- shard08 formal artifacts:
  - `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard08/completed.json`
  - `summary.md`
  - formal run registry `research_artifacts/aws_runs/20260927T010021_f39456a9/registry.json`
- prepared audit source:
  - `experiments/bohn2021_aws/vehicle_validation64_shard08_audit.py`
- pre-shard08 backup proof:
  - `research_artifacts/aws_backup_proofs/backup_proof_20260927T005920_after_shard07_audit_v2_run_finalized.json`

No validation bank content was reopened during inspection. Sealed final test remained closed and was not hashed.

### Experiment run
Ran exactly one bounded metadata audit:

- Script: `experiments/bohn2021_aws/vehicle_validation64_shard08_audit.py`
- Interpreter: modern
- Experiment ID: `20260927T021520_6b245ecf`
- Exit status: `0`
- Runtime: `14.0126 s`
- New simulations/control steps/gradient steps: `0 / 0 / 0`
- Validation access: existing shard08 outputs only
- Validation bank reopened: `false`
- Sealed final test accessed/opened/hashed: `false / false / false`

### Shard08 audit result
Audit passed.

- Episodes audited: `224`
- Control steps audited: `19703`
- Control-step upper bound: `33600`
- Completed hash audit: `true`
- Episode trace/hash audit: `true`
- Aggregate replay: `true`
- Registry validation-budget schema repair accepted `episodes_exact=224`; no scientific criterion, controller behavior, model-selection rule, or final-test gate changed.

Learned-horizon audit:

| candidate | episodes | steps | successes | horizons | switches | adaptive in shard08 |
|---|---:|---:|---:|---|---:|---|
| learned_s0 | 4 | 290 | 4 | H25 only | 0 | false |
| learned_s1 | 3 | 208 | 3 | H25 only | 0 | false |
| learned_s2 | 4 | 314 | 4 | H25/H35 | 8 | true |

### Key artifacts
- Audit raw: `research_artifacts/aws_diagnostics/vehicle_validation64_shard08_audit_20260927/raw.json`
  - SHA256 `2ef86bb9e2e68b673205cbd96811f9cb39711235cce8ce57d93126ae58ffe9fa`
- Audit summary: `research_artifacts/aws_diagnostics/vehicle_validation64_shard08_audit_20260927/summary.md`
  - SHA256 `ab9b319eb4e104d96740f39d44a4533c48eb48654720d546ab1fbdb6f9fa3f9c`
- Audit completed marker: `research_artifacts/aws_diagnostics/vehicle_validation64_shard08_audit_20260927/completed.json`
  - SHA256 `4c8cfbe5a6f68c1e4540c66dfd7e328e6222be0bde71162eec6e82c7aa4eb5a9`
- Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_20260927T021520.json`
  - SHA256 `39942fb3f51edf9c970b54ed59371f9eadb4d57eddb3b89670f83b38b154375e`
- Final addendum: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_FINAL_ADDENDUM_20260927T021520.json`
  - SHA256 `eb0277557c49a20c83a91023063ef965225ea625e438e1de4439dbe4c567e525`
- Backup blocker note: `research_artifacts/aws_diagnostics/post_shard08_audit_backup_blocker_check_20260927.md`
  - SHA256 `2cc7e666f2d291bfb77ae45d1b22f953f91e7d4571edece15feda9e10f4ac159`
- Run-finalized backup request addendum written:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD08_AUDIT_RUN_FINALIZED_20260927T021520.json`
  - SHA256 `286e5821b8e55e69295a8f2660f2ebc35f3d55703a4465afc184e6ce1d553ec6`

### Cumulative vehicle validation64 status
- Formal shards completed: `0–8`
- Accepted audited shards: `0–8`
- Formal validation episodes completed: `2016 / 2688`
- Formal validation control steps completed: `177430`
- Remaining shards: `9, 10, 11`
- Preserved failed formal attempt:
  - shard02 modern-interpreter TensorFlow-missing attempt, 0 episodes/control steps/gradient steps
- Preserved failed audit attempts:
  - shard00 audit-v1 schema false negative, repaired by v2
  - shard05 audit-v1 schema false positive, repaired by v2
  - shard07 audit-v1 `episodes_exact` schema false positive, repaired by v2

### Current blocker / next iteration
Shard09 is blocked until a verified external backup proof exists after the shard08 audit run-finalized addendum.

I checked for `research_artifacts/aws_backup_proofs/backup_proof_20260927T02*.json`; none exists yet.

Next iteration should:
1. Inspect only for an adequate post-shard08-audit external backup proof.
2. If adequate, reconstruct/verify local pre-shard09 proof.
3. Then run exactly one formal shard09 validation with legacy interpreter and sealed final test closed.
4. If no proof exists, do not create new formal validation evidence.

State was persisted via `update_state`. No reproduction-success claim is supported yet; this remains partial validation evidence for the IMPROVED latency-tree vehicle method, not ORIGINAL SAC, and not final-test evidence.

<!-- post-shard08-audit-backup-gate-recheck-20260927T022136+0000 -->
## 2026-09-27 post-shard08 audit backup gate recheck

UTC: 2026-09-27T02:21:36+00:00. Metadata-only backup proof inventory check before shard09: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. Required proof was after `2026-09-27T02:15:35+00:00`. Repository-local `backup_proof_20260927T02*.json` count was 0; all `backup_proof_20260927T*.json` files were ['research_artifacts/aws_backup_proofs/backup_proof_20260927T005920_after_shard07_audit_v2_run_finalized.json']; adequate post-shard08 proofs found: 0. The supervisor-context backup at `2026-09-27T02:14:50.580286+00:00` is insufficient because it predates the shard08 audit run-finalized threshold and no local proof file exists. Shard09 remains backup-gated. No formal validation evidence was created; vehicle validation64 remains 9/12 shards, 2016/2688 episodes, 177430 control steps, final test closed. New artifacts: `research_artifacts/aws_diagnostics/post_shard08_audit_backup_gate_recheck_20260927T022136+0000/raw.json`, `research_artifacts/aws_diagnostics/post_shard08_audit_backup_gate_recheck_20260927T022136+0000/summary.md`, `research_artifacts/aws_diagnostics/post_shard08_audit_backup_gate_recheck_20260927T022136+0000/completed.json`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_POST_SHARD08_AUDIT_GATE_RECHECK_20260927T022136+0000.json`. Next proof must be after this recheck and this metadata run's finalized registry/stdout/stderr/cloudwatch snapshot before shard09 can run. Method remains IMPROVED latency-tree, not ORIGINAL SAC, and no reproduction-success claim is supported.


## 2026-09-27T02:22:50.418918+00:00

<!-- vehicle-validation64-shard-complete-20260926-shard09 -->
## 2026-09-26 vehicle validation64 shard 09

UTC: 2026-09-27T03:37:07.147485+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19876. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard09/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard09/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-27T03:39:01.031229+00:00

<!-- vehicle-validation64-shard09-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard09 audit

UTC: 2026-09-27T03:40:23+00:00. Post-run audit of formal shard09 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard09 has 224 episodes and 19876 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard09: s0 {'episodes': 10, 'steps': 742, 'success_count': 10, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 742}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 4, 'steps': 293, 'success_count': 4, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 293}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 6, 'steps': 474, 'success_count': 6, 'episode_failure_count': 0, 'switches': 17, 'horizon_counts': {'25': 387, '35': 87}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 10/12 completed formal shards with 2240 episodes and 197306 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard10; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD09_AUDIT_20260927T034023.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD09_AUDIT_FINAL_ADDENDUM_20260927T034023.json`. Sealed test remains closed.


## 2026-09-27T03:42:51.464430+00:00
Concrete actions completed this iteration:

- Inspected current shard09 evidence and backup context without reopening the validation bank and without sealed-test access.
- Ran exactly one metadata-only audit experiment:
  - Script: `experiments/bohn2021_aws/vehicle_validation64_shard09_audit.py`
  - Experiment: `20260927T034023_8a42639a`
  - Interpreter: modern
  - Simulations/control steps/gradient steps: `0 / 0 / 0`
  - Validation bank reopened: `false`
  - Sealed test opened/hashed/accessed: `false / false / false`
- Shard09 audit passed:
  - Completed-hash audit: `true`
  - Episode trace/hash audit: `true`
  - Aggregate replay: `true`
  - Episodes audited: `224`
  - Control steps audited: `19876`
- Shard09 learned-horizon audit:
  - `learned_s0`: 10 episodes, 742 steps, H25 only, 10/10 success
  - `learned_s1`: 4 episodes, 293 steps, H25 only, 4/4 success
  - `learned_s2`: 6 episodes, 474 steps, H25/H35, 17 switches, 6/6 success
- Updated/preserved evidence:
  - Audit completed: `research_artifacts/aws_diagnostics/vehicle_validation64_shard09_audit_20260927/completed.json`
  - SHA256: `4c1c57478ef99351f471a83ff8e3687cda639ca66bd3792eea5b647f35cd35fc`
  - Audit raw: `research_artifacts/aws_diagnostics/vehicle_validation64_shard09_audit_20260927/raw.json`
  - SHA256: `e9f9fa9d6e3dd037990ac820f6b23ba763007fe428720e389908507574f0a943`
  - Audit summary: `research_artifacts/aws_diagnostics/vehicle_validation64_shard09_audit_20260927/summary.md`
  - SHA256: `eef30d2b73007c44d69a7425a2a74fb9252e6d71b81b03018d678d511671470b`
- Wrote run-finalized backup request addendum:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD09_AUDIT_RUN_FINALIZED_20260927T034023.json`
  - SHA256: `5b6f78fa7fdabd96bff775fc615f4524bb6abbb66165b5f398c89561069cfb87`
- Persisted durable state via `update_state`.

Current cumulative vehicle validation64 status:

- Formal shards completed: `0–9` = `10/12`
- Accepted audited shards: `0–9` = `10/12`
- Formal validation episodes completed: `2240 / 2688`
- Formal validation control steps completed: `197306`
- Remaining shards: `10, 11`
- Final sealed test: still closed, unauthorized, not opened, not hashed.
- Scientific status: partial validation evidence for the IMPROVED latency-tree method only; no ORIGINAL SAC reproduction claim and no final success claim.

Backup gate:

- Shard10 is blocked.
- No adequate repository-local `backup_proof_20260927T034*.json` or `backup_proof_20260927T03*.json` was found after the shard09 audit/run-finalized addendum.
- Required next proof must be after `2026-09-27T03:40:38Z` and cover shard09 formal outputs, shard09 audit outputs, finalized audit run registry/stdout/stderr/cloudwatch snapshot, docs/registry updates, backup requests/addenda/blocker note, and the new run-finalized addendum.

Next autonomous action:

- First inspect only for an adequate post-shard09-audit-run-finalized external backup proof.
- If adequate proof exists: reconstruct/verify local pre-shard10 proof, then run exactly one formal shard10 validation with the legacy interpreter using the frozen runner; sealed test remains closed.
- If no adequate proof exists: do not create new formal validation evidence; preserve the backup gate and state.

<!-- vehicle-case43-shard07-trajectory-diagnostic-20260927 -->
## 2026-09-27 vehicle shard07 case43 trajectory diagnostic

UTC: 2026-09-27T03:47:33.055258+00:00. Metadata-only diagnostic of already-created validation outputs completed: learned_s2 shard07 case43 versus same-seed terminal25 fixed H25 case43. No validation bank reopen, no sealed-test access/hash, no simulations/control steps/gradient steps.

Key diagnostic facts: learned_s2 case43 failed at 150 steps with physical+constraint cost 39125.9 and horizons {'25': 149, '35': 1}; fixed seed2 terminal25 H25 succeeded in 96 steps with physical+constraint cost 90.4598. H35 steps were [3]. Prefix max previous-state/input diffs before the singleton H35 were 0 / 0; first input/post-state divergence steps were 3 / 4. Learned solver calls were all successful/accepted with retry_rows=0. Objective/terminal-value components and warm-start vectors were not persisted, so an instrumented non-formal replay is required for that part of the diagnosis.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_case43_shard07_trajectory_diagnostic_20260927/raw.json`, `research_artifacts/aws_diagnostics/vehicle_case43_shard07_trajectory_diagnostic_20260927/summary.md`, `research_artifacts/aws_diagnostics/vehicle_case43_shard07_trajectory_diagnostic_20260927/completed.json`. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_CASE43_SHARD07_DIAGNOSTIC_20260927T034733.json`; shard10 remains blocked until an adequate verified external backup proof covers shard09 audit run-finalized evidence and subsequent diagnostic artifacts.


## 2026-09-27T03:48:01.506727+00:00

<!-- vehicle-validation64-shard-complete-20260926-shard10 -->
## 2026-09-26 vehicle validation64 shard 10

UTC: 2026-09-27T05:01:42.259853+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19831. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard10/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-27T05:04:18.568880+00:00

<!-- vehicle-validation64-shard10-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard10 audit

UTC: 2026-09-27T05:05:50+00:00. Post-run audit of formal shard10 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard10 has 224 episodes and 19831 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard10: s0 {'episodes': 2, 'steps': 148, 'success_count': 2, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 148}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 5, 'steps': 425, 'success_count': 5, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 425}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 5, 'steps': 344, 'success_count': 5, 'episode_failure_count': 0, 'switches': 13, 'horizon_counts': {'25': 327, '35': 17}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 progress is now 11/12 completed formal shards with 2464 episodes and 217137 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. New formal evidence requires external backup before shard11; request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_20260927T050550.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD10_AUDIT_FINAL_ADDENDUM_20260927T050550.json`. Sealed test remains closed.


## 2026-09-27T05:08:56.256469+00:00


<!-- post-shard10-audit-backup-gate-recheck-20260927T051156+0000 -->
## 2026-09-27 post-shard10 audit backup gate recheck

UTC: 2026-09-27T05:11:56+00:00. Metadata-only backup proof inventory check before shard11: validation_accessed=false, validation_bank_reopened=false, sealed test accessed/opened/hashed=false, simulations/control_steps/gradient_steps=0/0/0. Required proof was after `2026-09-27T05:06:10+00:00`. Repository-local `backup_proof_20260927T05*.json` count was 0; all `backup_proof_20260927T*.json` files were ['research_artifacts/aws_backup_proofs/backup_proof_20260927T005920_after_shard07_audit_v2_run_finalized.json', 'research_artifacts/aws_backup_proofs/backup_proof_20260927T022317_after_post_shard08_gate_recheck_run_finalized.json', 'research_artifacts/aws_backup_proofs/backup_proof_20260927T034828_after_shard09_audit_and_case43_diagnostic_run_finalized.json']; adequate post-shard10 proofs found: 0. The supervisor-context backup at `2026-09-27T05:05:16.430103+00:00` is insufficient because it predates shard10 audit completion/finalization. Shard11 remains backup-gated. No formal validation evidence was created; vehicle validation64 remains 11/12 shards, 2464/2688 episodes, 217137 control steps, final test closed. New artifacts: `research_artifacts/aws_diagnostics/post_shard10_audit_backup_gate_recheck_20260927T051156+0000/raw.json`, `research_artifacts/aws_diagnostics/post_shard10_audit_backup_gate_recheck_20260927T051156+0000/summary.md`, `research_artifacts/aws_diagnostics/post_shard10_audit_backup_gate_recheck_20260927T051156+0000/completed.json`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_POST_SHARD10_AUDIT_GATE_RECHECK_20260927T051156+0000.json`. Next proof must be after this recheck and this metadata run's finalized registry/stdout/stderr/cloudwatch snapshot before shard11 can run. Method remains IMPROVED latency-tree, not ORIGINAL SAC, and no reproduction-success claim is supported.


## 2026-09-27T05:14:56.501257+00:00

<!-- vehicle-validation64-shard-complete-20260926-shard11 -->
## 2026-09-26 vehicle validation64 shard 11

UTC: 2026-09-27T06:26:48.030739+00:00. Formal vehicle validation shard completed with validation_accessed=true, test_accessed=false, episodes=224, control_steps=19211. Artifacts: `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard11/raw.json`, `research_artifacts/aws_formal_validation/vehicle_validation64_20260926/shard11/summary.md`. This is shard-level validation evidence only; final sealed test remains unauthorized.


## 2026-09-27T06:28:20.663960+00:00

<!-- vehicle-validation64-shard11-audit-20260927 -->
## 2026-09-27 vehicle validation64 shard11 audit

UTC: 2026-09-27T06:31:22+00:00. Post-run audit of formal shard11 completed with validation_accessed=true (reading existing shard outputs), sealed test accessed=false, simulations=0, training steps=0. Shard11 has 224 episodes and 19211 control steps, within the declared 224/33600 budget. Completed hash audit passed=True; episode trace/hash audit passed=True; aggregate replay checks passed=True. Learned candidates in shard11: s0 {'episodes': 9, 'steps': 707, 'success_count': 9, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 707}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s1 {'episodes': 6, 'steps': 447, 'success_count': 6, 'episode_failure_count': 0, 'switches': 0, 'horizon_counts': {'25': 447}, 'unique_horizons': [25], 'adaptive_in_this_shard': False}, s2 {'episodes': 7, 'steps': 541, 'success_count': 7, 'episode_failure_count': 0, 'switches': 18, 'horizon_counts': {'25': 525, '35': 16}, 'unique_horizons': [25, 35], 'adaptive_in_this_shard': True}. Vehicle validation64 formal shard execution is now 12/12 complete with 2688 episodes and 236348 control steps, plus one counted failed validation-access attempt with 0 episodes/control steps and preserved audit-schema false positives. This is not final model selection or a reproduction claim. Full paired comparison/model-selection analysis is backup-gated until this final shard audit and finalized audit run logs are externally recoverable. Request written at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD11_AUDIT_20260927T063122.json` and final addendum at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VALIDATION64_SHARD11_AUDIT_FINAL_ADDENDUM_20260927T063122.json`. Sealed test remains closed.


## 2026-09-27T06:33:26.380485+00:00

<!-- vehicle-learned-policy-collapse-diagnostic-v2-full-validation64-20260927 -->
## 2026-09-27 vehicle learned-policy collapse diagnostic v2 (full validation64)

UTC: 2026-09-27T06:36:13+00:00. Existing-output diagnostic for learned_s0/s1/s2 across validation64 shards 00-11 completed with validation_accessed=true only because saved validation output traces/summaries were read; validation bank reopened=false; sealed final test accessed/opened/hashed=false; simulations/control_steps/gradient_steps=0/0/0. Learned trace episodes read: 192; scanned episode directories: 2688.

Key classifications: {"learned_s0": {"actual_non25_steps": 0, "classification": "nonH25_leaves_exist_but_validation_states_never_reach_them", "falsifiable_next_step": "Compare feature distributions with thresholds and training states; non-H25 opportunity may be outside visited validation distribution.", "policy_trace_mismatch_count": 0, "predicted_non25_steps": 0, "structurally_constant_policy": false, "unique_leaf_horizons": []}, "learned_s1": {"actual_non25_steps": 0, "classification": "nonH25_leaves_exist_but_validation_states_never_reach_them", "falsifiable_next_step": "Compare feature distributions with thresholds and training states; non-H25 opportunity may be outside visited validation distribution.", "policy_trace_mismatch_count": 0, "predicted_non25_steps": 0, "structurally_constant_policy": false, "unique_leaf_horizons": []}, "learned_s2": {"actual_non25_steps": 409, "classification": "adaptive_horizon_used_on_existing_validation_traces", "falsifiable_next_step": "Use paired aggregate validation analysis and targeted replays to decide whether non-H25 decisions improve cost/time tradeoff or induce failures.", "policy_trace_mismatch_count": 0, "predicted_non25_steps": 0, "structurally_constant_policy": false, "unique_leaf_horizons": []}}

Artifacts: `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v2_full_validation64/raw.json`, `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v2_full_validation64/summary.md`, `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v2_full_validation64/completed.json` (SHA256 `d45a4e09bd616489751d163c2b43be89bee72194891b36854c556d63ff81203c`). Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_LEARNED_POLICY_COLLAPSE_DIAGNOSTIC_V2_FULL_VALIDATION64_20260927T063613+0000.json`. Method remains IMPROVED latency-tree, not ORIGINAL SAC; no final-test or reproduction-success claim is supported. Next scientific step is full paired validation aggregation/model selection and then targeted case43/instrumentation diagnostics or versioned method revision as indicated by the aggregate evidence.


## 2026-09-27T06:37:42.045380+00:00

<!-- vehicle-learned-policy-collapse-diagnostic-v3-full-validation64-20260927-research-log -->
### Vehicle learned-policy collapse diagnostic v3 full validation64 (2026-09-27T06:40:45+00:00)

- Ran metadata/trace diagnostic only: no simulations, no control steps, no gradient steps, no validation-bank reopen, no sealed-test access/hash.
- Repaired failed v2 policy lookup by hashing exact frozen latency-tree vehicle policy files recorded in the 20260926 validation gate summary.
- Result: `passed`; artifacts: `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v3_full_validation64/raw.json`, `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v3_full_validation64/summary.md`, `research_artifacts/aws_diagnostics/vehicle_learned_policy_collapse_diagnostic_20260927_v3_full_validation64/completed.json`.
- learned_s0: `{'classification': 'extracted_policy_structurally_constant_H25', 'structurally_constant_policy': True, 'unique_leaf_horizons': [25], 'actual_non25_steps': 0, 'predicted_non25_steps': 0, 'policy_trace_mismatch_count': 0, 'falsifiable_next_step': 'Inspect training/selection objective and candidate extraction logs; rollout timing noise is not needed to explain H25-only behavior.'}`.
- learned_s1: `{'classification': 'extracted_policy_structurally_constant_H25', 'structurally_constant_policy': True, 'unique_leaf_horizons': [25], 'actual_non25_steps': 0, 'predicted_non25_steps': 0, 'policy_trace_mismatch_count': 0, 'falsifiable_next_step': 'Inspect training/selection objective and candidate extraction logs; rollout timing noise is not needed to explain H25-only behavior.'}`.
- learned_s2: `{'classification': 'adaptive_horizon_used_and_trace_matches_policy', 'structurally_constant_policy': False, 'unique_leaf_horizons': [25, 35], 'actual_non25_steps': 409, 'predicted_non25_steps': 409, 'policy_trace_mismatch_count': 0, 'falsifiable_next_step': 'Use full paired aggregate analysis and targeted deterministic replays to test cost/time benefit and failure causality.'}`.
- Next action: run full paired validation64 aggregate/model-selection analysis against the strong fixed-H grid before any final-test gate or method revision.


## 2026-09-27T06:42:05.757776+00:00



## 2026-09-27T06:47:05.332885+00:00

<!-- vehicle-validation64-full-aggregate-model-selection-20260927-RESEARCH_LOG.md -->
### Vehicle validation64 full aggregate/model-selection diagnostic (2026-09-27T06:48:01+00:00)

- Metadata-only validation aggregation over existing shard outputs: 2688 episode summaries and 236348 represented control steps; no new simulations/control/training, no validation-bank reopen, no sealed-test access/hash.
- Artifacts: `research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/raw.json`, `research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/summary.md`, completed marker `research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/completed.json`; tables `{'episode_table_csv': 'research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/episode_table.csv', 'arm_summary_csv': 'research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/arm_summary.csv', 'paired_deltas_summary_csv': 'research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/paired_deltas_summary.csv', 'case43_table_csv': 'research_artifacts/aws_diagnostics/vehicle_validation64_full_aggregate_model_selection_20260927/case43_table.csv'}`.
- Matched-terminal all-seed fixed-H nominations: performance `matched_terminal_allseeds_H25`, speed-within-3%-cost `matched_terminal_allseeds_H25`.
- Independent-terminal seed0 nominations: performance `fixed_seed0_terminal30_controllerH30`, speed-within-3%-cost `fixed_seed0_terminal30_controllerH30`.
- Learned candidates: `learned_s0`: episodes=64, success=64, failures=0, phys=1563.71, total=1687.89, decision_mean_s=0.16884143365196969, horizons={'25': 4967}; `learned_s1`: episodes=64, success=64, failures=0, phys=1198.54, total=1322.54, decision_mean_s=0.1727579607954373, horizons={'25': 4960}; `learned_s2`: episodes=64, success=63, failures=1, phys=40228.6, total=40358.1, decision_mean_s=0.17713809508124634, horizons={'25': 4605, '35': 409}.
- Adaptive diagnostic: `{'learned_s0': {'same_seed_speed_nomination': 'fixed_seed0_terminal25_controllerH25', 'adaptive_horizons_used': False, 'unique_horizons': [25], 'success_noninferior_proxy': True, 'cost_within_3pct_proxy': True, 'decision_at_least_10pct_faster_proxy': False, 'passes_against_same_seed_speed_nomination': False, 'reasons': ['not adaptive on validation: only one horizon used', 'decision mean does not show >=10% reduction vs same-seed speed nomination']}, 'learned_s1': {'same_seed_speed_nomination': 'fixed_seed1_terminal25_controllerH25', 'adaptive_horizons_used': False, 'unique_horizons': [25], 'success_noninferior_proxy': True, 'cost_within_3pct_proxy': True, 'decision_at_least_10pct_faster_proxy': False, 'passes_against_same_seed_speed_nomination': False, 'reasons': ['not adaptive on validation: only one horizon used', 'decision mean does not show >=10% reduction vs same-seed speed nomination']}, 'learned_s2': {'same_seed_speed_nomination': 'fixed_seed2_terminal25_controllerH25', 'adaptive_horizons_used': True, 'unique_horizons': [25, 35], 'success_noninferior_proxy': True, 'cost_within_3pct_proxy': False, 'decision_at_least_10pct_faster_proxy': False, 'passes_against_same_seed_speed_nomination': False, 'reasons': ['physical/control cost exceeds +3% proxy vs same-seed speed nomination', 'decision mean does not show >=10% reduction vs same-seed speed nomination']}}`.
- Interpretation: The frozen validation campaign is complete and shows that the current IMPROVED latency-tree candidate is not a robust 3-seed adaptive-horizon success: learned_s0 and learned_s1 are H25-only, while learned_s2 is adaptive but has one catastrophic validation failure. These findings are validation/development model-selection evidence only; the sealed final test remains unauthorized. Paired fixed-H comparisons and timing here should guide diagnosis/revision, not a reproduction-success claim.
- Next action after backup: prioritize targeted diagnosis of selection/training collapse (s0/s1 constant H25) and learned_s2 case43 deterministic replay/one-variable ablations before any final-test gate; likely prepare a versioned IMPROVED method revision.


## 2026-09-27T06:51:09.485750+00:00



## 2026-09-27T06:54:43.444444+00:00

<!-- vehicle-training-selection-collapse-diagnostic-v2-20260927-RESEARCH_LOG.md -->
### Vehicle training/selection collapse diagnostic v2 (2026-09-27T06:55:45+00:00)

- Repair of failed v1 `research_artifacts/aws_runs/20260927T065214_05cbaa7a/registry.json`: v1 expected `horizon` for constant policies but vehicle_s1 uses `h=25`; v2 accepts both schemas and writes new artifacts.
- Metadata-only development diagnostic over existing vehicle_s0/s1/s2 training/selection artifacts; no validation access, no validation-bank reopen, no sealed-test access/hash, no simulations/control steps/gradient steps.
- Artifacts: `research_artifacts/aws_diagnostics/vehicle_training_selection_collapse_diagnostic_20260927_v2/raw.json`, `research_artifacts/aws_diagnostics/vehicle_training_selection_collapse_diagnostic_20260927_v2/summary.md`, candidate table `research_artifacts/aws_diagnostics/vehicle_training_selection_collapse_diagnostic_20260927_v2/candidate_table.csv`, completed marker `research_artifacts/aws_diagnostics/vehicle_training_selection_collapse_diagnostic_20260927_v2/completed.json`. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRAINING_SELECTION_COLLAPSE_DIAGNOSTIC_V2_20260927T065545+0000.json`.
- Headline: `{'cross_seed_classifications': {'0': 'selection_chose_structurally_constant_H25_tree', '1': 'selection_chose_fixed_H25_baseline', '2': 'selection_chose_adaptive_tree'}, 'selected': {'0': 'g3_c09', '1': 'fixed', '2': 'g3_c08'}, 'final_horizons': {'0': [25], '1': [25], '2': [25, 35]}, 'eligible_adaptive_counts': {'0': 15, '1': 17, '2': 18}, 'best_eligible_adaptive': {'0': 'g1_c00', '1': 'g2_c05', '2': 'g3_c08'}, 'interpretation_short': 'selection/objective-collapse rather than runtime-dispatch bug; current frozen candidate insufficient for final test'}`.
- Interpretation: The completed v2 metadata diagnostic supports a selection/objective-collapse diagnosis for the current frozen IMPROVED latency-tree candidate. vehicle_s0 had eligible adaptive candidates and the best in-fit candidate was adaptive, but the final stored policy is a structurally constant H25 tree, implying collapse occurred during later final selection/generalization/noisy paired selection rather than at runtime. vehicle_s1 explicitly selected the fixed H25 baseline; its stored policy is constant with schema key h=25, explaining all H25 validation behavior. vehicle_s2 selected an adaptive tree, consistent with validation H25/H35 usage, but validation aggregate already showed no speed advantage and one catastrophic case43 failure. Across seeds, the old objective is a linear cost_change + 0.5*(time_ratio-1) with hard eligibility gates, which is not producing robust adaptive-horizon policies under the current setup. Next revision should be versioned IMPROVED work on selection/extraction and risk/robustness handling, not a runtime dispatch repair, and must use fresh non-test validation after any code/config change.
- Next action: after external backup proof for aggregate+v1 failure+v2 diagnostic, freeze a versioned IMPROVED revision plan targeting selection objective/risk handling and decide a bounded non-formal case43 replay if needed; do not open final test.


## 2026-09-27T07:07:01.247846+00:00



## 2026-09-27T07:11:37.909842+00:00

<!-- vehicle-case43-instrumented-replay-v1-20260927 -->
## 2026-09-27 vehicle case43 instrumented replay/ablation v1

UTC: 2026-09-27T07:14:24.502108+00:00. Development diagnostic intentionally reopened the already-used vehicle validation bank case43 and ran 5 deterministic replay/ablation episodes (588 new control steps, 0 gradient steps, sealed test closed). This is not fresh independent validation/model-selection evidence.

Outcome: forcing H25 at the singleton H35 rescues the learned replay, and a single H35 perturbation on the constant-H25 path fails; evidence supports the step3 horizon intervention as a direct cause of this case43 failure under this terminal/source setup.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/raw.json`, `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/summary.md`, `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/episode_summary.csv`, completed marker `research_artifacts/aws_diagnostics/vehicle_case43_instrumented_replay_20260927_v1/completed.json`. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_CASE43_INSTRUMENTED_REPLAY_V1_20260927T071424+0000.json`.


## 2026-09-27T07:18:44.431640+00:00



## 2026-09-27T07:22:31.802391+00:00



## 2026-09-27T07:25:21.548537+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): ~20.5 hours. Cumulative server API total_tokens from `research.sqlite`: unknown — repository search found no `research.sqlite`; this excludes desktop conversation usage.

Concrete actions completed this iteration:

1. **Ran the queued v2 H35-leaf diagnostic**
   - Script: `experiments/bohn2021_aws/vehicle_h35_leaf_validation_diagnostic_v2.py`
   - Experiment id: `20260927T072246_e2505cea`
   - Result: **failed immediately**
   - Cause: engineering guard error — `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v2/` already contained `PREPARED_NOT_RUN_STATE.md`, so the script aborted with:
     - `DiagnosticError('partial output exists; preserve before retry: ...')`
   - Scientific budget:
     - new simulations: `0`
     - new control steps: `0`
     - gradient steps: `0`
     - validation bank reopened: `false`
     - sealed test accessed: `false`

2. **Inspected and preserved the failure evidence**
   - Failure marker:
     - `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v2/failure.json`
   - Run registry:
     - `research_artifacts/aws_runs/20260927T072246_e2505cea/registry.json`
   - Diagnosis: this was not a scientific/model failure; it was an output-directory guard collision caused by putting the prepared-state marker inside the intended v2 output directory.

3. **Wrote a fresh corrected diagnostic script**
   - New script:
     - `experiments/bohn2021_aws/vehicle_h35_leaf_validation_diagnostic_v3.py`
   - SHA256:
     - `43dd410cc9f4e5e8d008c0f043b3ef52df6b28cf8dfb3e5aafe792643c8bfa1c`
   - v3 fixes:
     - preserves v1/v2 failures;
     - uses rollout trace index when `step` is absent;
     - retains full `first_h35` object in `per_case_by_case`;
     - writes to fresh output dir `vehicle_h35_leaf_validation_diagnostic_20260927_v3`;
     - avoids the v2 prepared-marker collision;
     - writes `completed.json` only after raw/summary/CSV/docs/backup request succeed.

4. **Attempted to run v3, but tool policy blocked it**
   - The experiment tool enforces **one experiment per iteration**.
   - v3 was therefore **not executed** and is queued for the next bounded cycle.

5. **Persisted durable next-state**
   - Wrote:
     - `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v3_PREPARED_NOT_RUN_STATE.md`
   - SHA256:
     - `b7689710982c4d76a984c5105641d7b3bfe023382540d1bd99661e9b98ce5601`
   - Updated supervisor state with phase, failure diagnosis, hypothesis, and next experiment.

Current scientific status remains unchanged:

- Vehicle validation64 is complete.
- Sealed final test remains closed/unopened/unauthorized.
- Current learned latency-tree vehicle candidate is not sufficient:
  - `learned_s0` and `learned_s1` collapsed to H25.
  - `learned_s2` uses H25/H35 but has catastrophic validation case43 failure.
  - No robust timing advantage over same-seed H25 has been established.
- Existing evidence supports diagnosing/removing unsafe longer-horizon H35 behavior before any improved adaptive-horizon claim.

Next precise action:

Run exactly this next cycle:

- Script: `experiments/bohn2021_aws/vehicle_h35_leaf_validation_diagnostic_v3.py`
- Interpreter: modern
- Timeout: 300s
- Method: `IMPROVED_latency_tree_vehicle_h35_leaf_validation_diagnostic_v3_existing_outputs_not_original_SAC`
- Split: `development_existing_outputs_only_validation64_already_opened_no_sealed_test`
- Budgets:
  - simulations: `0`
  - control steps: `0`
  - gradient steps: `0`
  - validation existing outputs read: `true`
  - validation bank reopened: `false`
  - sealed test accessed: `false`

If v3 passes, inspect its `summary.md` and `completed.json`, then freeze the next versioned IMPROVED safe-shortening/risk-sensitive vehicle revision protocol and source before any smoke or retraining.


## 2026-09-27T07:29:42.254663+00:00

<!-- vehicle-h35-leaf-validation-diagnostic-v4-20260927 -->
## 2026-09-27 vehicle H35 leaf validation diagnostic v4

UTC: 2026-09-27T07:30:08.460730+00:00. Existing validation64 rollout outputs only; no new simulation/control steps/gradient steps, no validation bank reopen, sealed test closed. v4 preserves failed v1/v2/v3 and fixes compact-tree parsing plus first-H35 step-index handling.

Finding: learned_s2's H35 branch follows rule `heading_error_5 <= 0.0957597175326 and abs_yaw_input <= 0.250382459863`. It appeared in 52/64 validation cases (409 total H35 steps). First-H35 rollout-step stats: `{'count': 52, 'sum': 1497.0, 'mean': 28.78846153846154, 'median': 31.0, 'p95': 63.80000000000001, 'min': 0.0, 'max': 67.0}`. Logged-vs-replayed policy horizon mismatches: 0. Case43 remains the only learned_s2 validation failure, with H35 steps `[3]`; prior deterministic replay showed that forcing H25 at the singleton H35 rescues it while forcing H35 into the constant-H25 path reproduces the failure. Excluding case43, learned_s2 vs same-seed fixed H25 physical deltas are summarized by `{'count': 63, 'sum': -17.59428504099685, 'mean': -0.2792743657301088, 'median': 0.0, 'p95': 0.0005317749462619757, 'min': -12.132815113962833, 'max': 0.0031670370110390422}`.

Decision: next IMPROVED vehicle revision should test safe-shortening/risk-sensitive extraction rather than allowing H>25 cost-seeking branches to compete as adaptive-horizon acceleration. Artifacts: `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v4/summary.md`, `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v4/raw.json`, `research_artifacts/aws_diagnostics/vehicle_h35_leaf_validation_diagnostic_20260927_v4/per_case_h35.csv`.


## 2026-09-27T07:38:17.666616+00:00



## 2026-09-27T07:44:46.760799+00:00



## 2026-09-27T07:49:55.812180+00:00



## 2026-09-27T07:54:31.705075+00:00

<!-- vehicle-safe-shortening-v1-smoke-20260927-v3 -->
## 2026-09-27 vehicle safe-shortening v1 smoke

UTC: 2026-09-27T08:02:25.089325+00:00. Engineering smoke for IMPROVED safe-shortening wrapper completed on vehicle smoke bank only: 24 episodes, 1764 control steps, replay passed=True. Validation64 and sealed test remained closed. This is not model-selection/final evidence. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v3/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v3/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_smoke_20260927_v3/completed.json`.


## 2026-09-27T08:06:08.251033+00:00



## 2026-09-27T08:13:45.975089+00:00



## 2026-09-27T08:21:07.474866+00:00

<!-- vehicle-safe-shortening-v1-case43-replay-v3-collate-20260927 -->
## 2026-09-27 vehicle safe-shortening v1 case43 replay v3 collation

UTC: 2026-09-27T08:22:03.413286+00:00. Collated preserved v2 episode outputs after v2 post-processing failure. New v3 simulations/control steps/gradient steps: 0/0/0; preserved v2 diagnostic budget collated: 6 episodes, 576 control steps. All six same-seed case43 episodes succeeded; no adaptive H>25 dispatch; adaptive horizons by seed: {0: {'fixed_success': 1, 'adaptive_success': 1, 'adaptive_horizons': {'20': 31, '25': 65}, 'phys_delta': 1.116521520572178e-06, 'decision_ratio': 1.0423525549844879, 'used_above_25': False}, 1: {'fixed_success': 1, 'adaptive_success': 1, 'adaptive_horizons': {'15': 35, '25': 61}, 'phys_delta': 0.8297200493651786, 'decision_ratio': 0.9936114746938678, 'used_above_25': False}, 2: {'fixed_success': 1, 'adaptive_success': 1, 'adaptive_horizons': {'10': 1, '25': 95}, 'phys_delta': 0.0392611901705493, 'decision_ratio': 0.9923426646971993, 'used_above_25': False}}. This reuses already-opened validation case43 and is contaminated development evidence only, not formal validation or final test. Fresh validation bank remains unopened/uncreated; sealed test remains closed. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v3_collate/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v3_collate/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v3_collate/completed.json`; backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE43_REPLAY_V2_FAILURE_AND_V3_COLLATION_20260927T000000Z.json`.


## 2026-09-27T08:23:21.396947+00:00
Elapsed service lifetime since 2026-09-26T10:55:29.419331Z: at least 21h 26m 34s / 21.44 hours as of the v3 artifact timestamp 2026-09-27T08:22:03Z.  
Cumulative server API total_tokens from research.sqlite: unknown; `**/research.sqlite` is still not present in the repository context, so this excludes desktop conversation usage.

Concrete actions completed:

1. Inspected the prepared v3 collation source and v2 failure evidence:
   - `experiments/bohn2021_aws/vehicle_safe_shortening_v1_case43_replay_v3_collate.py`
   - v2 failure: `KeyError('decision_timing_s')` after all six case43 episodes had already completed.
   - v2 progress confirmed: 6 episodes, 576 control steps, no fresh validation bank, no sealed test access.

2. Ran the queued v3 collation-only recovery:
   - Run ID: `20260927T082203_909c622d`
   - Script: `vehicle_safe_shortening_v1_case43_replay_v3_collate.py`
   - Interpreter: legacy
   - Exit status: 0
   - Runtime: ~2.0s
   - New simulations/control steps/gradient steps: `0 / 0 / 0`
   - Collated preserved v2 diagnostic budget: 6 episodes, 576 control steps.
   - Sealed test: not accessed.
   - Fresh validation bank: not generated/opened.

3. Key v3 result:
   - All six same-seed contaminated case43 development replay episodes succeeded.
   - Adaptive safe-shortening arms used no `H>25`.
   - Adaptive horizons:
     - seed0: `H20` for 31 steps, `H25` for 65 steps.
     - seed1: `H15` for 35 steps, `H25` for 61 steps.
     - seed2: `H10` for 1 step, `H25` for 95 steps.
   - Fixed and adaptive success/failure counts matched for all seeds.
   - This supports only the narrow engineering hypothesis that safe-shortening avoids the known unsafe H35 mode on already-opened case43. It is not fresh validation, not model selection, and not final-test evidence.

4. Artifacts inspected/preserved:
   - `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case43_replay_20260927_v3_collate/summary.md`
   - `.../completed.json`
   - `.../raw.json`
   - Backup request:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE43_REPLAY_V2_FAILURE_AND_V3_COLLATION_20260927T000000Z.json`
   - Documentation tails and registry now include the v3 collation entry.

5. State preserved via `update_state`.

Current scientific status:

- Vehicle latency-tree validation64 remains negative/inadequate for adaptive-horizon success:
  - learned_s0/s1 collapsed to H25-only.
  - learned_s2 adapts H25/H35 but failed validation case43 and lacks robust timing advantage.
- Safe-shortening v1 has passed engineering smoke plus contaminated case43 recovery, but has no independent development-validation result yet.
- No ORIGINAL SAC reproduction claim.
- No IMPROVED adaptive-horizon success claim.
- Sealed final test remains closed and unauthorized.

Current blocker / next gate:

- Need verified external backup after the v3 collation artifacts. Latest known verified backup in the provided supervisor context was before the v3 run.
- Do not generate/open the fresh safe-shortening development-validation bank until backup covers v2 failure outputs, v3 source/artifacts/docs/registry, and backup request.

Next queued action once backup is verified:

- Run the frozen vehicle safe-shortening v1 fresh development-validation paired comparison:
  - adaptive seeds 0/1/2,
  - strong fixed-H grid comparators,
  - equal declared budgets,
  - full cost/success/solver/timing metrics,
  - no sealed test access.


## 2026-09-27T08:31:13.539498+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard00 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 00

UTC: 2026-09-27T09:24:18.053445+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13784 control steps, cases [14, 29, 42, 63], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/completed.json`.


## 2026-09-27T09:25:52.532012+00:00
Elapsed service lifetime since 2026-09-26T10:55:29.419331Z: **at least 22h28m49s / 22.48h** as of shard00 artifact timestamp 2026-09-27T09:24:18.053445Z. Cumulative server API total_tokens from research.sqlite: **unknown**; repository search for `**/research.sqlite` and `**/research.sqlite3` returned none this cycle, excluding desktop conversation usage.

Continued with concrete actions:

1. **Ran the next informative experiment**
   - Script: `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py --shard 0`
   - Interpreter: legacy TF1 runtime
   - Method: `IMPROVED_vehicle_safe_shortening_v1_devval64_shard0_not_original_SAC`
   - Registry: `research_artifacts/aws_runs/20260927T083224_83809524/registry.json`
   - Exit: complete, status 0
   - Runtime: ~3115.9 s

2. **Generated/froze fresh vehicle devval bank and gate**
   - Fresh devval bank RNG: `2609273000`
   - Bank resets: 64
   - Bank artifact: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/bank/vehicle_safe_shortening_v1_devval64_bank.json`
   - Bank sha256: `731766c8113b527d0e5cda2caea8119842649e629ee3ba182919e7e9235f8b36`
   - Historical validation64 bank reopened: false
   - Sealed test accessed: false

3. **Completed fresh devval shard00**
   - Cases: `[14, 29, 42, 63]`
   - Arms: all 43 frozen arms per case
   - Episodes: 172
   - Control steps: 13,784 / 25,800 upper bound
   - New gradient steps: 0
   - Sealed test episodes: 0
   - Artifacts:
     - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/summary.md`
     - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/raw.json`
     - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard00/completed.json`

4. **Shard00 partial scientific findings**
   - This is **development-validation/model-selection evidence only**, not final-test evidence and not ORIGINAL SAC reproduction.
   - Same-seed adaptive vs matched-terminal fixed H25:
     - Seed0: success 4/4 vs 4/4; adaptive horizons `H20/H25`; no H>25; physical+constraint delta `+0.00988585`; total-cost delta `-0.0701142`; mean decision-time ratio `0.95745`.
     - Seed1: success 4/4 vs 4/4; adaptive horizons `H15/H25`; no H>25; physical+constraint delta `+0.227574`; total-cost delta `-0.232426`; mean decision-time ratio `0.99553`.
     - Seed2: success 4/4 vs 4/4; adaptive stayed H25 on these four cases; no H>25; physical+constraint delta `0`; total-cost delta `0`; mean decision-time ratio `0.98833`.
   - Interpretation: safety guard behaved correctly on shard00; seed0/seed1 adapted below H25; seed2 remained constant H25 in this shard. No success claim yet.

5. **Preserved state**
   - Wrote post-run state:
     - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD00_POSTRUN_STATE_20260927T092418Z.md`
     - sha256 `08f946c8d15c6da76f5e5b7b74a9f0169c01de4b1910f916afa7719c515de264`
   - Updated durable supervisor state via `update_state`.

6. **Current blocker / next action**
   - Backup request written:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD00_20260927T092418Z.json`
   - **Do not run shard01 yet** until a verified external backup proof appears after shard00 completion and covers shard00 artifacts, bank/gate, runner, protocol, registry, docs, and the new post-run state file.
   - Once backup is verified, next exact experiment is:
     - `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py --shard 1`
     - legacy interpreter
     - unchanged frozen controller/checkpoints/schedule/selection rules
     - no sealed test.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard01 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 01

UTC: 2026-09-27T10:24:23.361302+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 16086 control steps, cases [21, 26, 33, 52], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard01/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard01/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard01/completed.json`.


## 2026-09-27T10:27:41.240057+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard02 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 02

UTC: 2026-09-27T11:25:22.996476+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15754 control steps, cases [8, 15, 30, 61], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard02/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard02/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard02/completed.json`.


## 2026-09-27T11:28:54.770839+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard03 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 03

UTC: 2026-09-27T12:20:50.233484+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13603 control steps, cases [5, 24, 32, 56], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard03/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard03/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard03/completed.json`.


## 2026-09-27T12:22:52.228459+00:00
Elapsed service lifetime since first supervisor event: **≥25h25m20.814s / 25.4224 h** as of latest shard03 artifact timestamp `2026-09-27T12:20:50.233484Z`.  
Cumulative server API `total_tokens` from `research.sqlite`: **unknown**; repository tools still do not expose `research.sqlite` / `research.sqlite3`. Desktop conversation usage excluded.

Concrete actions completed this iteration:

1. **Inspected current frozen campaign evidence**
   - Verified campaign directory inventory includes shards `00–02`, gate, bank.
   - Read/checked:
     - `shard02/completed.json`
     - frozen runner `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py`
     - `gate_completed.json`
     - current status/audit/registry tails after shard03.
   - No historical validation64 bank or sealed test was opened.

2. **Executed next frozen vehicle devval shard**
   - Ran:
     - `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py --shard 3`
     - legacy interpreter
   - Exit status: **0**
   - Runtime: **3051.896 s**
   - Registry:
     - `research_artifacts/aws_runs/20260927T113002_ce43de29/registry.json`
   - Shard03 cases:
     - `[5, 24, 32, 56]`
   - Budget consumed:
     - **172 development-validation episodes**
     - **13,603 control steps**
     - **0 training episodes**
     - **0 gradient steps**
     - **0 sealed-test episodes**

3. **Shard03 main findings — development-validation only**
   - Safe-shortening guard preserved: **no adaptive H > 25**.
   - Same-seed H25 primary pairings:
     - Seed0: adaptive vs fixed H25 success **4/4 vs 4/4**, horizons `{20: 27, 25: 246}`, physical delta `+0.0001136`, total delta `-0.134886`, decision ratio mean `1.00234`.
     - Seed1: **4/4 vs 4/4**, horizons `{15: 27, 25: 246}`, physical delta `+0.0000374`, total delta `-0.269963`, decision ratio mean `1.01224`.
     - Seed2: **4/4 vs 4/4**, horizons `{25: 273}`, physical delta `0`, total delta `0`, decision ratio mean `1.05534`.
   - Interpretation unchanged: seed0/seed1 show limited shortening with mixed timing; seed2 remains effectively constant H25.

4. **Cumulative current campaign after shards 00–03**
   - Completed shards: **4 / 16**
   - Fresh devval cases scored: **16 / 64**
   - Episodes: **688**
   - Control steps: **59,227**
   - Fresh bank generation resets counted: **64**
   - Preliminary primary same-seed H25 aggregate, not for selection:
     - Seed0: success **16/16 vs 16/16**, adaptive horizons `{20: 127, 25: 1089}`, physical delta `+0.231487`, total delta `-0.403513`, approximate shard-mean decision ratio `0.98644`.
     - Seed1: success **16/16 vs 16/16**, adaptive horizons `{15: 213, 25: 1003}`, physical delta `+1.171264`, total delta `-0.958736`, ratio `1.00570`.
     - Seed2: success **16/16 vs 16/16**, adaptive horizons `{10: 2, 25: 1214}`, physical delta `+0.039709`, total delta `+0.009709`, ratio `1.01094`.

5. **State and backup preservation**
   - Wrote durable postrun state:
     - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD03_POSTRUN_STATE_20260927T122050Z.md`
     - sha256 `f39fb292df226816acdc1de91b17a6d7e5ef07b8034378b699790af9f461aa00`
   - Wrote backup addendum request:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD03_STATE_ADDENDUM_20260927T122050Z.json`
     - sha256 `77b8a8a1f8c710ab71ef93e5b02c5f6159a0f566ce9033eebc1944615863c68e`
   - Updated persistent research state via `update_state`.

6. **Access control**
   - Historical validation64 bank reopened: **false**
   - Sealed final test accessed: **false**
   - Final-test authorization requested: **false**
   - No controller/checkpoint/selection-rule changes.

Next queued action: **verify/observe backup for shard03 artifacts if available, then run shard04 with the same frozen runner and protocol**:

```text
experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py --shard 4
```

Expected shard04 budget: **172 episodes**, control-step upper bound **25,800**, no training, no sealed test.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard04 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 04

UTC: 2026-09-27T13:13:53.294129+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 12912 control steps, cases [6, 11, 49, 53], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard04/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard04/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard04/completed.json`.


## 2026-09-27T13:17:05.921801+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard05 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 05

UTC: 2026-09-27T14:13:22.323523+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15126 control steps, cases [22, 25, 44, 51], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard05/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard05/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard05/completed.json`.


## 2026-09-27T14:16:19.724987+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard06 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 06

UTC: 2026-09-27T15:11:23.792340+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14178 control steps, cases [7, 17, 37, 41], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/completed.json`.


## 2026-09-27T15:13:20.134497+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): **28h15m54.373s / 28.2651 h** to latest shard06 artifact timestamp **2026-09-27T15:11:23.792340Z**. Cumulative server API `total_tokens` from `research.sqlite` calls: **unknown** because repository tools expose no `*sqlite*` files; desktop conversation usage excluded.

Concrete action completed: continued the frozen vehicle safe-shortening v1 development-validation campaign by running **shard06**.

Key evidence:
- Run registry: `research_artifacts/aws_runs/20260927T141730_ae6afa30/registry.json`
- Runner: `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py`
- Runner SHA256: `21db4b319901243bc512a356ceac71c2280c2db0fc9e473136586300678435d5`
- Shard artifacts:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/summary.md`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/raw.json`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard06/completed.json`
- Postrun state written:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD06_POSTRUN_STATE_20260927T151123Z.md`
  - SHA256 `6dc88555aea042b86e11859c1dc6558a6c5c3b670ca99a648db7820cad681f15`
- Backup request written:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD06_STATE_ADDENDUM_20260927T151123Z.json`
  - SHA256 `1250eb61bc4099a28a86ea1c39a44cbae5816c0ac68aa9950fe8a1058d5bf664`

Shard06 budget/access:
- Episodes: **172**
- Control steps: **14,178**
- New training episodes / gradient steps: **0 / 0**
- Sealed test episodes: **0**
- Historical validation64 bank opened: **False**
- Sealed final test opened: **False**
- Final-test authorization requested: **False**

Shard06 scientific findings:
- Cases: `[7, 17, 37, 41]`
- Seed0: adaptive vs same-seed H25 fixed success **3/4 vs 3/4**; case37 failed for both. Horizons `{20: 48, 25: 324}`, no H>25. Total-cost delta `-0.23999`, physical+constraint delta approximately `+0.000009`.
- Seed1: success **4/4 vs 4/4**. Horizons `{15: 37, 25: 251}`, no H>25. Total-cost delta `-0.33200`, physical+constraint delta `+0.037997`.
- Seed2: success **4/4 vs 4/4**. Horizons `{10: 1, 25: 287}`, no H>25. Total-cost delta `-0.01500`, physical+constraint delta approximately `-0.000003`.

Cumulative campaign status after shard06:
- Completed shards: **7/16** (`00–06`)
- Cases scored: **28/64**
- Episodes completed: **1,204**
- Control steps completed: **101,443**
- New gradient training remains **0**; this campaign reuses historical gated policies and is not newly trained.

Current interpretation:
- Safe-shortening v1 continues to avoid unsafe **H>25** choices and preserves same-seed H25 success/failure parity on completed fresh devval cases.
- Adaptation remains sparse and one-sided. Seed2 is effectively constant H25; seed0/seed1 shorten occasionally.
- Physical+constraint performance is not convincingly improved; total-cost changes are mostly from the explicit horizon penalty/shortening term.
- Actual AWS decision-time ratios remain noisy/near 1, so no timing-speedup claim is supported yet.
- This is development-validation/model-selection evidence only, not final-test evidence and not a reproduction success claim.

State has been persisted with `update_state`. Next concrete action queued: run **shard07** of the same frozen campaign, with no controller/checkpoint/selection-rule changes and no sealed-test access.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard07 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 07

UTC: 2026-09-27T16:08:45.227251+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14624 control steps, cases [23, 31, 46, 62], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/completed.json`.


## 2026-09-27T16:11:06.000185+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): **29h13m15.808s / 29.2211 h** to latest shard07 artifact timestamp **2026-09-27T16:08:45.227251Z**. Cumulative server API `total_tokens` from `research.sqlite` calls: **unknown** because repository tools expose no `*sqlite*` files; desktop conversation usage excluded.

Concrete action completed: continued the frozen vehicle safe-shortening v1 development-validation campaign by running **shard07**.

Key evidence:
- Run registry: `research_artifacts/aws_runs/20260927T151430_7f7e5026/registry.json`
- Runner: `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py`
- Runner SHA256: `21db4b319901243bc512a356ceac71c2280c2db0fc9e473136586300678435d5`
- Shard artifacts:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/summary.md`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/raw.json`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard07/completed.json`
- Postrun state written:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD07_POSTRUN_STATE_20260927T160845Z.md`
  - SHA256 `c8a65453dbabb415fd40d308d963974c5c050ffd898f7446e5553796c0e935b0`
- Backup request written:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD07_STATE_ADDENDUM_20260927T160845Z.json`
  - SHA256 `d3b89d999ba1e1f723f96bbbd388ba206882ce9de944d78b20bb1ff99b000762`

Shard07 budget/access:
- Episodes: **172**
- Control steps: **14,624**
- New training episodes / gradient steps: **0 / 0**
- Sealed test episodes/control steps: **0 / 0**
- Historical validation64 bank opened: **False**
- Sealed final test opened: **False**
- Final-test authorization requested: **False**

Shard07 scientific findings:
- Cases: `[23, 31, 46, 62]`
- Seed0: adaptive vs same-seed H25 fixed success **4/4 vs 4/4**; horizons `{20: 42, 25: 263}`, no H>25. Physical+constraint delta `+0.000699`, total-cost delta `-0.209301`, decision-time ratio mean `1.00283`.
- Seed1: success **4/4 vs 4/4**; horizons `{15: 82, 25: 223}`, no H>25. Physical+constraint delta `+0.028683`, total-cost delta `-0.791317`, decision-time ratio mean `0.96308`.
- Seed2: success **4/4 vs 4/4**; horizons `{10: 7, 25: 298}`, no H>25. Physical+constraint delta `-0.000050`, total-cost delta `-0.105050`, decision-time ratio mean `0.94577`.

Cumulative campaign status after shard07:
- Completed shards: **8/16** (`00–07`)
- Cases scored: **32/64**
- Episodes completed: **1,376**
- Control steps completed: **116,067**
- New gradient training remains **0**; this campaign reuses historical gated policies and is not newly trained.

Current interpretation:
- Safe-shortening v1 remains safety-preserving and one-sided: **no H>25** selections on shards00–07.
- Same-seed H25 success/failure parity is preserved so far: cumulative seed0 **31/32 vs 31/32**, seed1 **32/32 vs 32/32**, seed2 **32/32 vs 32/32**.
- Adaptation remains sparse; seed2 is still effectively constant H25.
- Physical+constraint deltas remain slightly adverse overall; total-cost gains are mostly explained by explicit horizon-shortening penalties.
- Actual AWS timing remains noisy/near 1, so no acceleration claim is supported yet.
- This is development-validation/model-selection evidence only, not final-test evidence and not a reproduction success claim.

State has been persisted with `update_state`. Next concrete action queued: verify shard07 backup, then run **shard08** of the same frozen campaign with no controller/checkpoint/selection-rule changes and no sealed-test access.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard08 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 08

UTC: 2026-09-27T17:02:28.089742+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13165 control steps, cases [13, 35, 38, 43], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/completed.json`.


## 2026-09-27T17:03:59.761475+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): **30h06m58.670s / 30.1163 h** to latest shard08 artifact timestamp **2026-09-27T17:02:28.089742Z**.  
Cumulative server API `total_tokens` from `research.sqlite` calls: **unknown**; repository tools expose no `*.sqlite` files, and desktop conversation usage is excluded.

Concrete action completed: continued the frozen vehicle safe-shortening v1 development-validation campaign by running **shard08**.

Key evidence:
- Run registry: `research_artifacts/aws_runs/20260927T161218_f87fa0e0/registry.json`
- Runner: `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py`
- Runner SHA256: `21db4b319901243bc512a356ceac71c2280c2db0fc9e473136586300678435d5`
- Shard artifacts:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/summary.md`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/raw.json`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard08/completed.json`
- Postrun state written:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD08_POSTRUN_STATE_20260927T170228Z.md`
  - SHA256 `80ba181093eb37bdc577219cec6535c33919d1d173e1f17a33d5468c31c59cef`
- Backup requests:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD08_20260927T170228Z.json`
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD08_STATE_ADDENDUM_20260927T170228Z.json`
  - State backup request SHA256 `b24fd135cacfcb2de931f11f836a9aea59d896833da13226118b8f79d149e243`

Shard08 budget/access:
- Episodes: **172**
- Control steps: **13,165**
- New training episodes / gradient steps: **0 / 0**
- Sealed test episodes/control steps: **0 / 0**
- Historical validation64 bank opened: **False**
- Sealed final test opened: **False**
- Final-test authorization requested: **False**

Shard08 scientific findings:
- Cases: `[13, 35, 38, 43]`
- Seed0: adaptive vs same-seed H25 fixed success **4/4 vs 4/4**; horizons `{20: 36, 25: 248}`, no H>25. Physical+constraint delta `+0.000075`, total-cost delta `-0.179925`, decision-time ratio mean `1.00279`.
- Seed1: success **4/4 vs 4/4**; horizons `{15: 39, 25: 245}`, no H>25. Physical+constraint delta `+0.429949`, total-cost delta `+0.039949`, decision-time ratio mean `1.01316`.
- Seed2: success **4/4 vs 4/4**; horizons `{25: 284}`, no H>25. Physical+constraint delta `0`, total-cost delta `0`, decision-time ratio mean `1.01197`.

Cumulative campaign status after shard08:
- Completed shards: **9/16** (`00–08`)
- Cases scored: **36/64**
- Episodes completed: **1,548**
- Control steps completed: **129,232**
- New gradient training remains **0**; this campaign reuses historical gated policies and is not newly trained.

Current interpretation:
- Safe-shortening v1 remains one-sided/safety-preserving: **no H>25** selections on shards00–08.
- Same-seed H25 success/failure parity is preserved so far: cumulative seed0 **35/36 vs 35/36**, seed1 **36/36 vs 36/36**, seed2 **36/36 vs 36/36**.
- Adaptation remains sparse; seed2 is effectively constant H25.
- Physical+constraint deltas are still not improved overall; total-cost gains mostly reflect explicit horizon-shortening penalties.
- Actual AWS timing remains noisy/near 1, so no acceleration claim is supported.
- This is development-validation/model-selection evidence only, not final-test evidence and not a reproduction success claim.

State has been persisted with `update_state`. Next queued action: verify/await backup for shard08 artifacts and state addendum, then run **shard09** of the same frozen campaign with no controller/checkpoint/selection-rule changes and no sealed-test access.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard09 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 09

UTC: 2026-09-27T17:56:53.522498+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14176 control steps, cases [4, 12, 18, 57], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/completed.json`.


## 2026-09-27T17:59:04.841975+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): **31h01m24.103s / 31.0234 h** to latest shard09 artifact timestamp **2026-09-27T17:56:53.522498Z**; this is service wall-clock, not summed experiment runtime. Shard09 experiment runtime was **3109.944 s**.  
Cumulative server API `total_tokens` from `research.sqlite` calls: **unknown**; repository tools expose no `*.sqlite` files, and desktop conversation usage is excluded.

Concrete action completed: continued the frozen vehicle safe-shortening v1 development-validation campaign by running **shard09**.

Key evidence:
- Run registry: `research_artifacts/aws_runs/20260927T170508_c4dd5ff6/registry.json`
- Runner: `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py`
- Runner SHA256: `21db4b319901243bc512a356ceac71c2280c2db0fc9e473136586300678435d5`
- Shard artifacts:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/summary.md`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/raw.json`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard09/completed.json`
- Postrun state written:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD09_POSTRUN_STATE_20260927T175653Z.md`
  - SHA256 `50619e4c1f9100b6c568aef1bf19b17e0077c30e9bbfbbed23bef81f5e374741`
- Backup requests:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD09_20260927T175653Z.json`
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD09_STATE_ADDENDUM_20260927T175653Z.json`
  - State backup request SHA256 `96980e46419cd352f578445a73f4d9683ba3071595b024cdaf7deae42fc3d7fd`

Shard09 budget/access:
- Cases: `[4, 12, 18, 57]`
- Episodes: **172**
- Control steps: **14,176**
- New training episodes / gradient steps: **0 / 0**
- Sealed test episodes/control steps: **0 / 0**
- Historical validation64 bank opened: **False**
- Sealed final test opened: **False**
- Final-test authorization requested: **False**

Shard09 scientific findings:
- Seed0: adaptive vs same-seed H25 fixed success **4/4 vs 4/4**; horizons `{20: 43, 25: 232}`, no H>25. Physical+constraint delta `+0.105059`, total-cost delta `-0.109941`, decision-time ratio mean `1.01413`.
- Seed1: success **4/4 vs 4/4**; horizons `{15: 42, 25: 233}`, no H>25. Physical+constraint delta `+0.0000858`, total-cost delta `-0.419914`, decision-time ratio mean `0.99928`.
- Seed2: success **4/4 vs 4/4**; horizons `{10: 1, 25: 274}`, no H>25. Physical+constraint delta `+0.0000579`, total-cost delta `-0.014942`, decision-time ratio mean `1.04458`.

Cumulative campaign status after shard09:
- Completed shards: **10/16** (`00–09`)
- Cases scored: **40/64**
- Episodes completed: **1,720**
- Control steps completed: **143,408**
- New gradient training remains **0**; this campaign reuses historical gated policies and is not newly trained.

Current interpretation:
- Safe-shortening v1 remains one-sided/safety-preserving: **no H>25** selections on shards00–09.
- Same-seed H25 success/failure parity is preserved so far: cumulative seed0 **39/40 vs 39/40**, seed1 **40/40 vs 40/40**, seed2 **40/40 vs 40/40**.
- Adaptation remains sparse, especially seed2, which is still effectively constant H25.
- Physical+constraint deltas remain not improved overall; total-cost gains mostly reflect explicit horizon-shortening penalties.
- Actual AWS timing ratios remain noisy/near 1, so no acceleration claim is supported.
- This remains development-validation/model-selection evidence only, not final-test evidence and not a reproduction success claim.

State has been persisted with `update_state`. Next queued action: verify/await backup for shard09 artifacts and state addendum, then run **shard10** of the same frozen campaign with no controller/checkpoint/selection-rule changes and no sealed-test access.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard10 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 10

UTC: 2026-09-27T18:54:43.290131+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14237 control steps, cases [10, 28, 47, 54], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/completed.json`.


## 2026-09-27T18:57:05.274101+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): **31h59m13.871s / 31.9872 h** to latest shard10 artifact timestamp **2026-09-27T18:54:43.290131Z**; this is service wall-clock, not summed experiment runtime. Shard10 experiment runtime was **3137.973 s**.  
Cumulative server API `total_tokens` from `research.sqlite` calls: **unknown**; repository tools expose no `*.sqlite` files, and desktop conversation usage is excluded.

Concrete action completed: continued the frozen vehicle safe-shortening v1 development-validation campaign by running **shard10**.

Key evidence:
- Run registry: `research_artifacts/aws_runs/20260927T180230_b2159a78/registry.json`
- Runner: `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py`
- Runner SHA256: `21db4b319901243bc512a356ceac71c2280c2db0fc9e473136586300678435d5`
- Shard artifacts:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/summary.md`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/raw.json`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard10/completed.json`
- Artifact hashes from registry:
  - summary SHA256 `ae84dcffaa5fc7cfe8d2a80e2fe3a12ea7fef46a0d70adde3ff86644377474f1`
  - raw SHA256 `efc5aef51d23b7e3b2726c452cc097477dbf9e68fa642f42099eb15c9f05a0dd`
  - completed SHA256 `9e11288f3acf7b1fefc6352ea24a9f33994101d5b77fc76b3963d5b5a2125722`
- Postrun state written:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD10_POSTRUN_STATE_20260927T185443Z.md`
  - SHA256 `6a7baa5ddc8f36bbc209d89e5bde5951bf0fc502bbc7b9e2ef41ebe63983a5d7`
- Backup requests:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD10_20260927T185443Z.json`
  - SHA256 `e9ebe8c573b8c798705b9ea2ca3f346e3969903486e3e4307719fb1d25f126d2`
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD10_STATE_ADDENDUM_20260927T185443Z.json`
  - SHA256 `e1c8ac6f9410a2e95dab4cfdaf0d859243c5b4681478e54235bde8ba9aef7f30`

Shard10 budget/access:
- Cases: `[10, 28, 47, 54]`
- Episodes: **172**
- Control steps: **14,237**
- New training episodes / gradient steps: **0 / 0**
- Sealed test episodes/control steps: **0 / 0**
- Historical validation64 bank opened: **False**
- Sealed final test opened/accessed: **False**
- Final-test authorization requested: **False**
- CloudWatch: **unavailable**; missing CloudWatch values are unknown, not zero.

Shard10 scientific findings:
- Seed0: adaptive vs same-seed H25 fixed success **4/4 vs 4/4**; horizons `{20: 65, 25: 231}`, no H>25. Physical+constraint delta `+0.019961`, total-cost delta `-0.305039`, decision-time ratio mean `0.98861`.
- Seed1: success **4/4 vs 4/4**; horizons `{15: 90, 25: 206}`, no H>25. Physical+constraint delta `+2.849900`, total-cost delta `+1.949900`, decision-time ratio mean `0.98434`.
- Seed2: success **4/4 vs 4/4**; horizons `{25: 296}`, no H>25. Physical+constraint delta `0.0`, total-cost delta `0.0`, decision-time ratio mean `1.00610`.

Cumulative frozen campaign status after shard10:
- Completed shards: **11/16** (`00–10`)
- Cases scored: **44/64**
- Episodes completed: **1,892**
- Control steps completed: **157,645**
- New gradient training remains **0**; this campaign reuses historical gated policies and is not newly trained.

Current interpretation:
- Safe-shortening v1 remains one-sided/safety-preserving: **no H>25** selections on shards00–10.
- Same-seed H25 success/failure parity is preserved so far.
- Adaptation remains sparse: seed0/seed1 shorten somewhat, but seed2 is still effectively constant H25 and was exactly H25-only on shard10.
- Physical+constraint cost is not improved overall versus same-seed H25; total-cost changes still mostly reflect the explicit horizon-shortening penalty.
- Actual AWS timing ratios remain noisy/near 1, so no acceleration claim is supported.
- This remains development-validation/model-selection evidence only, not final-test evidence and not a reproduction-success claim.

State has been persisted with `update_state`. Next queued action: verify/await external backup covering shard10 artifacts, run logs, docs, and the state addendum; then run **shard11** of the same frozen campaign with no controller/checkpoint/selection-rule changes and no sealed-test access.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard11 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 11

UTC: 2026-09-27T19:50:07.888027+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 13756 control steps, cases [1, 34, 48, 58], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/completed.json`.


## 2026-09-27T19:51:46.626314+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): **32h54m43.605s / 32.9121 h** to shard11 registry end time **2026-09-27T19:50:13.024525Z**; this is service wall-clock, not summed experiment runtime. Shard11 experiment runtime was **3111.938 s**.  
Cumulative server API `total_tokens` from `research.sqlite` calls: **unknown**; repository tools expose no `*.sqlite` files, and desktop conversation usage is excluded.

Concrete action completed: continued the frozen vehicle safe-shortening v1 development-validation campaign by running **shard11**.

Key evidence:
- Run registry: `research_artifacts/aws_runs/20260927T185820_013de85f/registry.json`
- Runner: `experiments/bohn2021_aws/vehicle_safe_shortening_v1_devval_shard_runner.py`
- Runner SHA256: `21db4b319901243bc512a356ceac71c2280c2db0fc9e473136586300678435d5`
- Shard artifacts:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/summary.md`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/raw.json`
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard11/completed.json`
- Artifact hashes:
  - summary SHA256 `669203fb62b783e81e92f8f960f717e171ecfe923dae40fc08f44c9d8be22ea6`
  - raw SHA256 `7d9e356552e5909939be4a65ed1e0c30f29c987f6b119b6a3fc32288f7f54ea0`
  - completed SHA256 `f2ba19773e0f9569f85ba9994c94c5be4be3d6ec1686aa51bf06b05905014da5`
- Postrun state written:
  - `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/SHARD11_POSTRUN_STATE_20260927T195013Z.md`
  - SHA256 `775fb41e0d3eda093123bb5e2daae25a668c2e9ed61829812616b5801dd0fc83`
- Backup requests:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD11_20260927T195008Z.json`
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_DEVVAL64_SHARD11_STATE_ADDENDUM_20260927T195013Z.json`
  - state-addendum request SHA256 `abebec86c2d0a4982b83f40436cebf926e6c7c8e27427911d97ce45c04120f37`

Shard11 budget/access:
- Cases: `[1, 34, 48, 58]`
- Episodes: **172**
- Control steps: **13,756**
- New training episodes / gradient steps: **0 / 0**
- Sealed test episodes/control steps: **0 / 0**
- Historical validation64 bank opened: **False**
- Sealed final test opened/accessed: **False**
- Final-test authorization requested: **False**
- CloudWatch: **unavailable**; missing CloudWatch values are unknown, not zero.

Shard11 primary same-seed H25 findings:
- Seed0: adaptive vs same-seed H25 fixed success **4/4 vs 4/4**; horizons `{20: 24, 25: 265}`, no H>25. Physical+constraint delta `+0.0000503`, total-cost delta `-0.119950`, decision-time ratio mean `0.97517`.
- Seed1: success **4/4 vs 4/4**; horizons `{15: 57, 25: 232}`, no H>25. Physical+constraint delta `+0.230412`, total-cost delta `-0.339588`, decision-time ratio mean `0.98009`.
- Seed2: success **4/4 vs 4/4**; horizons `{10: 10, 25: 279}`, no H>25. Physical+constraint delta `+2.191780`, total-cost delta `+2.041780`, decision-time ratio mean `1.02046`.

Cumulative frozen campaign status after shard11:
- Completed shards: **12/16** (`00–11`)
- Cases scored: **48/64**
- Episodes completed: **2,064**
- Control steps completed: **171,401**
- New gradient training remains **0**; this campaign reuses historical gated policies and is not newly trained.

Current interpretation:
- Safe-shortening v1 remains one-sided/safety-preserving: **no H>25** selections on shards00–11.
- Same-seed H25 success/failure parity is still preserved so far.
- Adaptation remains sparse; seed2 remains effectively constant H25.
- Physical+constraint cost is not improved overall; shard11 worsened physical+constraint for all three seeds, especially seed2.
- Total-cost improvements where present are still largely attributable to the explicit shortening penalty, not better control.
- AWS timing ratios remain noisy/near 1; no acceleration claim is supported.
- This remains development-validation/model-selection evidence only, not final-test evidence and not a reproduction-success claim.

State has been persisted with `update_state`. Next queued action: verify/await external backup covering shard11 artifacts, run logs, docs, and the state addendum; then run **shard12** under the same frozen controller/checkpoint/selection rules with no sealed-test access.

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard12 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 12

UTC: 2026-09-27T20:47:34.091485+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15129 control steps, cases [0, 2, 55, 59], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard12/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard12/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard12/completed.json`.


## 2026-09-27T20:49:46.244355+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard13 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 13

UTC: 2026-09-27T21:44:10.683619+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14479 control steps, cases [9, 36, 40, 50], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard13/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard13/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard13/completed.json`.


## 2026-09-27T21:48:55.333047+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard14 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 14

UTC: 2026-09-27T22:47:15.250406+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 15632 control steps, cases [3, 19, 20, 39], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard14/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard14/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard14/completed.json`.


## 2026-09-27T22:50:00.755263+00:00

<!-- vehicle-safe-shortening-v1-devval64-shard-20260927-v1-shard15 -->
## 2026-09-27 vehicle safe-shortening v1 fresh development-validation shard 15

UTC: 2026-09-27T23:44:53.534841+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v1: 172 episodes, 14802 control steps, cases [16, 27, 45, 60], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard15/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard15/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v1_devval64_20260927_v1/shard15/completed.json`.


## 2026-09-27T23:48:53.550820+00:00

<!-- vehicle-safe-shortening-v1-devval-progress-digest-v2-00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15 -->
## 2026-09-27 vehicle safe-shortening v1 devval64 progress digest v2

UTC: 2026-09-27T23:51:01.751104+00:00. Metadata-only digest over completed fresh devval shards [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]: 2752 episodes, 231443 control steps, 64/64 cases scored. No rollout/training, no historical validation64 bank reopen, and no sealed-test access. Primary same-seed H25 preliminary deltas: {'0': {'paired_case_count': 64, 'adaptive_success_count': 63, 'fixed_success_count': 63, 'physical_delta_sum': 8.701375885893775, 'total_delta_sum': 4.9963758858937695, 'decision_ratio_overall_total_decision_s': 0.9904014853310217, 'adaptive_horizon_counts': {'20': 741, '25': 4131}, 'used_adaptive_h_above_25': False}, '1': {'paired_case_count': 64, 'adaptive_success_count': 64, 'fixed_success_count': 64, 'physical_delta_sum': 9.592265011229458, 'total_delta_sum': -1.357734988770634, 'decision_ratio_overall_total_decision_s': 0.9867955839764387, 'adaptive_horizon_counts': {'15': 1095, '25': 3693}, 'used_adaptive_h_above_25': False}, '2': {'paired_case_count': 64, 'adaptive_success_count': 63, 'fixed_success_count': 64, 'physical_delta_sum': 753.7911792487655, 'total_delta_sum': 754.8811792487656, 'decision_ratio_overall_total_decision_s': 1.017903663842976, 'adaptive_horizon_counts': {'25': 4812, '10': 49}, 'used_adaptive_h_above_25': False}}. Continue frozen shards before model selection/final-test gate. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_devval_progress_digest_20260927_v2_shards00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_devval_progress_digest_20260927_v2_shards00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_devval_progress_digest_20260927_v2_shards00-01-02-03-04-05-06-07-08-09-10-11-12-13-14-15/completed.json`.


## 2026-09-27T23:53:59.163740+00:00

<!-- vehicle-safe-shortening-v1-shard13-case9-seed2-trace-diagnostic-20260927T235655Z -->
## 2026-09-27 vehicle safe-shortening v1 shard13 case9 seed2 trace diagnostic

UTC: 2026-09-27T23:56:55.994529+00:00. Ran a metadata-only diagnostic over already-created shard13 raw evidence; no rollout/training, no historical validation64 bank reopen, and no sealed-test access. Adaptive seed2 case9 failed with summary horizons {'10': 1, '25': 149} and raw horizons {'10': 1, '25': 149}, clamped=0, solver_fallback=0, solver_fail_steps=0, retries=0; same-seed H25 succeeded. Adaptive-minus-H25 deltas: steps 73, physical 751.34848, total 753.15848. Conclusions: ['The episode-level fields show no clamp, solver-failure fallback, solver-failure steps, or retries for the adaptive failure; an execution fallback/clamp explanation is not supported by these aggregate raw fields.', 'The raw artifact did not expose a recoverable step-level horizon sequence despite aggregate horizon_counts showing H10 once; this prevents exact localization from raw metadata alone.', 'The adaptive case9 failure is a real paired degradation relative to same-seed H25 in this development shard: adaptive exhausted the 150-step cap, while H25 reached the goal earlier.', 'Same-case fixed-grid evidence suggests very short horizons are risky on this scenario while H25+ succeeds, consistent with but not proving that the rare H10 decision could be harmful.', 'Existing raw fields appear insufficient for terminal-value/objective/state-normalization causality; a bounded instrumented deterministic replay is the appropriate next diagnostic after the frozen campaign/backup gate.']. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_shard13_case9_seed2_trace_diagnostic_20260927T235655Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_shard13_case9_seed2_trace_diagnostic_20260927T235655Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_shard13_case9_seed2_trace_diagnostic_20260927T235655Z/completed.json`.


## 2026-09-27T23:57:27.814686+00:00



## 2026-09-28T00:03:01.176484+00:00

<!-- vehicle-safe-shortening-v1-trace-policy-diagnostic-20260928 -->
## 2026-09-28 vehicle safe-shortening v1 trace/policy diagnostic

UTC: 2026-09-28T00:04:02.901867+00:00. Parsed existing safe-shortening v1 devval traces for all adaptive seeds plus matched fixed-grid summaries; no new rollout/control steps/training and no sealed-test access. Main result: no raw/executed horizon inconsistency, no H>25 requests, and near-constant H25 is explained by the frozen gate thresholds/policies rather than shard dispatch. Seed non25 shares: {0: 0.1520935960591133, 1: 0.2286967418546366, 2: 0.010080230405266406}. Case9 seed2 local H10/deviation evidence and fixed-grid short-H opportunity summaries are in `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_trace_policy_diagnostic_20260928T000351Z/summary.md`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_TRACE_POLICY_DIAGNOSTIC_20260928T000402Z.json`. Next: bounded deterministic state-level counterfactual replay/ablation before freezing any v2 retraining/selection change.


## 2026-09-28T00:06:25.058859+00:00



## 2026-09-28T00:08:56.405005+00:00

<!-- vehicle-safe-shortening-v1-case9-counterfactual-v1-20260928 -->
## 2026-09-28 vehicle safe-shortening v1 case9 seed2 counterfactual v1

UTC: 2026-09-28T00:10:40.253366+00:00. Ran two deterministic one-variable counterfactual episodes on already-opened fresh devval shard13/case9 seed2: force H25 at the saved adaptive H10 step, and inject H10 at the same step into the matched-H25 path. Budget: 2 episodes, 227 control steps, 0 training/gradient steps; sealed test closed. Outcome: The one-step horizon choice is causally implicated: forcing H25 at the saved H10 decision rescues the case, while injecting H10 into the matched-H25 path reproduces the adaptive failure pattern. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_counterfactual_v1_20260928T000944Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_counterfactual_v1_20260928T000944Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_counterfactual_v1_20260928T000944Z/completed.json`; backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE9_COUNTERFACTUAL_V1_20260928T001040Z.json`.


## 2026-09-28T00:15:00.575027+00:00

<!-- vehicle-safe-shortening-v1-case9-switch-sensitivity-v1-20260928 -->
## 2026-09-28 vehicle safe-shortening v1 case9 seed2 switch-sensitivity v1

UTC: 2026-09-28T00:16:50.954291+00:00. Ran three deterministic case9 seed2 switch-sensitivity counterfactual episodes on the already-opened fresh devval case: one-step H15, one-step H20, and H10 from step57 onward. Budget: 3 episodes, 231 control steps, 0 training/gradient steps; sealed test closed. Conclusion: On this diagnosed case, H15/H20 one-step switches and continuing H10 from step57 all avoid the prior single-H10-then-H25 failure; the v1 catastrophe is most consistent with an unsafe abrupt return to H25 after an H10 step rather than horizon switching in general. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_switch_sensitivity_v1_20260928T001549Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_switch_sensitivity_v1_20260928T001549Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_switch_sensitivity_v1_20260928T001549Z/completed.json`; backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE9_SWITCH_SENSITIVITY_V1_20260928T001650Z.json`.


## 2026-09-28T00:19:54.187929+00:00

<!-- vehicle-safe-shortening-v1-case9-ramp-sensitivity-v1-20260928 -->
## 2026-09-28 vehicle safe-shortening v1 case9 seed2 H10-return/ramp sensitivity v1

UTC: 2026-09-28T00:21:46.705508+00:00. Ran three deterministic transition/ramp counterfactual episodes on already-opened fresh devval shard13/case9 seed2. Budget: 3 episodes, 450 control steps, 0 training/gradient steps; sealed test closed. Conclusion: The H10-return/ramp schedules still failed on this case; v2 should avoid H10 in comparable states unless a stronger continuation/risk model justifies it. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1_20260928T002008Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1_20260928T002008Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_ramp_sensitivity_v1_20260928T002008Z/completed.json`; backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V1_CASE9_RAMP_SENSITIVITY_V1_20260928T002146Z.json`.


## 2026-09-28T00:25:09.500168+00:00

<!-- vehicle-safe-shortening-v1-case9-h10-hold-sensitivity-v1-20260928T002551Z -->
## 2026-09-28 vehicle safe-shortening v1 case9 H10 hold diagnostic

Development-only case9 seed2 H10 hold-length diagnostic completed: 4 new episodes, 308 new control steps, no training and no sealed-test access. Result: At least one finite H10 hold repaired the case9 transition; shortest successful hold among tested lengths is 3 steps. This supports considering transition dwell/hysteresis in v2, but only as development evidence from one already-opened case. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_20260928T002551Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_20260928T002551Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v1_case9_h10_hold_sensitivity_v1_20260928T002551Z/completed.json`.


## 2026-09-28T00:31:35.917800+00:00

<!-- vehicle-safe-shortening-v2-transition-hold-smoke-20260928 -->
## 2026-09-28 vehicle safe-shortening v2 transition-hold smoke

UTC: 2026-09-28T00:39:40.931469+00:00. IMPROVED v2 transition-hold protocol frozen and engineering smoke completed on a fresh v2 smoke bank: 24 episodes, 1800 control steps, replay passed=True. No validation64 bank or sealed test was opened. This is not model-selection/final evidence. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_smoke_20260928/completed.json`.


## 2026-09-28T00:45:02.486179+00:00

<!-- vehicle-safe-shortening-v2-transition-hold-devval-preflight-20260928T004500Z -->
## 2026-09-28 vehicle safe-shortening v2 transition-hold devval preflight

UTC: 2026-09-28T00:46:17.422154+00:00. Ran metadata/runtime preflight for the frozen IMPROVED v2 transition-hold development-validation campaign. No devval bank generation, no rollout/control steps, no historical validation64 bank reopen, and no sealed-test access. Schedule/arm dimensions verified: 2752 episodes, 16 shards, 43 arms; terminal-source metadata verified for 12 unique sources. Decision: Preflight passed. Do not run v2 devval shard until supervisor reports verified external backup for smoke, protocols, runner and this preflight. After backup, next concrete action is shard00 with legacy interpreter. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/completed.json`.


## 2026-09-28T00:47:58.804161+00:00
Elapsed service lifetime since first supervisor event: ~37h52m from 2026-09-26T10:55:29.419331Z to ~2026-09-28T00:47Z; this is service wall-clock, not summed experiment runtime.  
Cumulative server API total_tokens from research.sqlite calls: unavailable to repository tools; `*sqlite*` search found no usable database entries, and desktop conversation usage is excluded.

Concrete actions completed this iteration:

1. Ran the frozen v2 development-validation preflight:
   - Script: `experiments/bohn2021_aws/vehicle_safe_shortening_v2_devval_preflight.py`
   - Interpreter: legacy
   - Registry: `research_artifacts/aws_runs/20260928T004617_70cc8a85/registry.json`
   - Exit status: `0`
   - Runtime: `4.01s`
   - No rollout episodes, no control steps, no bank generation, no training, no sealed-test access.

2. Verified evidence from the preflight:
   - Legacy TF runtime passed: TensorFlow `1.15.5`.
   - V2 smoke completed marker/hash/access flags passed.
   - Smoke budget confirmed: `24` episodes / `1800` control steps.
   - Smoke had `validation64_bank_opened=false`, `sealed_test_accessed=false`.
   - Fresh v2 devval bank is absent and not partially created.
   - Frozen v2 devval schedule verified:
     - `2752` planned episodes
     - `16` shards
     - `43` arms
     - `172` episodes per shard
     - `64` cases, each paired across all arms
     - `3` adaptive v2 arms
     - `30` same-seed matched-terminal fixed-H grid arms
     - `10` independent-terminal seed0 fixed-H grid arms
   - `12` terminal sources verified by metadata/hash.

3. Key artifacts created/verified:
   - `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/summary.md`
   - `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/raw.json`
   - `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_transition_hold_devval_preflight_20260928T004500Z/completed.json`
   - Backup request:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_DEVVAL_PREFLIGHT_20260928T004617.json`

4. Important hashes:
   - Preflight script: `05b9d530ad4347674f9e21d086d9ac3c54bce4c3c25448593255647041d4dcce`
   - Devval shard runner: `0f35aa0432f4d0d7f57df74c8dc5aee5ae33d0a848f191ea3696e196291da229`
   - Devval protocol: `57556418b6caee02fd28a949aa92ceeba705361c4a80c183e12b1d67e80c293a`
   - Transition protocol: `cba016c6ca3615de2a1c2b770e24475d4f06784b6060d97ac30e50f509bae65e`

5. State preserved via `update_state`.
   - Current phase: v2 transition-hold preflight complete, backup gate active.
   - No sealed final test opened.
   - No historical validation64 bank reopened.
   - No fresh v2 devval bank generated yet.
   - No formal final-test evidence claimed.

Current gate:

- Last verified external backup in supervisor context was at `2026-09-28T00:45:40Z`, before this preflight.
- New unbacked files now include the v2 preflight outputs, backup request, run registry, and doc/registry appends.
- Per frozen protocol, do **not** start v2 devval shard00 until a verified external backup after `2026-09-28T00:46:17Z` covers the v2 smoke, protocols, runner, preflight outputs, registry/docs, and backup request.

Next concrete action after backup verification:

Run exactly one v2 fresh development-validation shard:

```text
experiments/bohn2021_aws/vehicle_safe_shortening_v2_transition_hold_devval_shard_runner.py --shard 0
```

with legacy interpreter only.

Declared shard00 budget:
- 172 rollout episodes
- Max 25,800 control steps
- 64 fresh devval bank reset snapshots if bank is absent
- 0 training episodes
- 0 gradient steps
- 0 sealed-test episodes

Purpose: first paired frozen v2 devval block across all 43 arms for 4 fresh cases, including all 3 adaptive seeds and full strong fixed-H comparators.

<!-- vehicle-safe-shortening-v2-devval-backup-gate-recheck-20260928T005101+0000 -->
## 2026-09-28 vehicle safe-shortening v2 devval backup gate recheck

UTC: 2026-09-28T00:51:01+00:00. Metadata-only gate check before frozen IMPROVED v2 transition-hold devval shard00. No rollout/control steps, no training, no fresh devval bank generation, no historical validation64 bank reopen, and no sealed-test access/hash occurred. Required proof time was after `2026-09-28T00:46:20+00:00` and after preflight/run-log finalization. Repository-local post-preflight candidate proofs found: 0; adequate proofs found: 0. Latest supervisor-context verified backup remains `2026-09-28T00:45:40.181961+00:00`, which predates the v2 preflight and is therefore inadequate. Fresh v2 devval root exists=False, shard00 exists=False, fresh bank exists=False. Shard00 blocked=True. Artifacts: `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_devval_backup_gate_recheck_20260928T005101+0000/summary.md`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_devval_backup_gate_recheck_20260928T005101+0000/raw.json`, `research_artifacts/aws_diagnostics/vehicle_safe_shortening_v2_devval_backup_gate_recheck_20260928T005101+0000/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_SAFE_SHORTENING_V2_DEVVAL_BACKUP_GATE_RECHECK_20260928T005101+0000.json`. Next after verified backup: run exactly one legacy shard00 (`experiments/bohn2021_aws/vehicle_safe_shortening_v2_transition_hold_devval_shard_runner.py --shard 0`) with 172 episodes/max 25,800 control steps and sealed test closed.


## 2026-09-28T00:53:30.850716+00:00

<!-- vehicle-safe-shortening-v2-transition-hold-devval64-shard-20260928-v1-shard00 -->
## 2026-09-28 vehicle safe-shortening v2 transition-hold fresh development-validation shard 00

UTC: 2026-09-28T01:48:44.635740+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED safe-shortening v2 transition-hold: 172 episodes, 14487 control steps, cases [3, 44, 52, 56], no historical validation64 bank reopen and no sealed-test access. The shard includes all 43 arms (3 adaptive plus matched fixed-H grid and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval64_20260928_v1/shard00/summary.md`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval64_20260928_v1/shard00/raw.json`, `research_artifacts/aws_development_validation/vehicle_safe_shortening_v2_transition_hold_devval64_20260928_v1/shard00/completed.json`.


## 2026-09-28T01:50:34.473008+00:00



## 2026-09-28T01:55:48.534100+00:00


<!-- vehicle-current-gated-horizon-training-audit-v1-20260928 -->
## 2026-09-28 current gated-horizon training/search audit v1

UTC: 2026-09-28T01:56:06+00:00. Metadata-only audit of the CURRENT reused gated-horizon policies used by AWS safe-shortening v1/v2, not merely the older latency-tree policy. No rollouts/control steps, no training/gradient steps, no historical validation64 reopen, and no sealed-test access/hash occurred. Audited finite search/reselection over 36 structured candidates plus fixed H25 per seed and selected/fixed training traces; already-opened v2 shard00 adaptive traces were used only for development coverage comparison. Key result: current policies came from finite candidate search with gradient_updates=0; objective is mean raw total_cost among hard-gated admissible candidates, with no measured runtime term and no transition/dwell-risk model. Candidate tables and coverage summaries are in `research_artifacts/aws_diagnostics/vehicle_current_gated_horizon_training_audit_v1_20260928T015606+0000/summary.md` / `research_artifacts/aws_diagnostics/vehicle_current_gated_horizon_training_audit_v1_20260928T015606+0000/raw.json`. Next: freeze and run a bounded IMPROVED re-selection/refit diagnostic before any further unchanged validation shard.


## 2026-09-28T02:01:06.753011+00:00


<!-- vehicle-gated-horizon-risk-reselection-v1-20260928T020201+0000 -->
### Vehicle gated-horizon risk-first re-selection v1 (2026-09-28T02:02:01+00:00)
Metadata-only re-selection over existing current gated-search training traces completed: research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_20260928T020201+0000/completed.json. No rollout/control/training/test access. Frozen decision `freeze_and_run_small_smoke_after_backup`; adaptive nominated seeds=3, changed-from-current adaptive seeds=3. Summary: research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_20260928T020201+0000/summary.md. Backup request: research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_20260928T020201+0000.json.


## 2026-09-28T02:07:47.540574+00:00

<!-- vehicle-gated-horizon-risk-reselection-v1-smoke-20260928 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 smoke

UTC: 2026-09-28T02:21:04.988578+00:00. IMPROVED risk-first gated-horizon re-selection smoke completed on a fresh engineering bank: 36 episodes, 3258 control steps, hard_pass=True, risk_reselected_below_H25_seed_count=3. No validation64 bank or sealed test was opened. This is not model-selection/final evidence and does not modify the frozen v2 shard campaign. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_smoke_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_smoke_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_risk_reselection_v1_smoke_20260928/completed.json`.


## 2026-09-28T02:26:09.885706+00:00

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard00 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 00

UTC: 2026-09-28T03:27:13.683697+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16091 control steps, cases [18, 42, 43, 50], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard00/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard00/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard00/completed.json`.


## 2026-09-28T03:28:29.587496+00:00
Elapsed service lifetime: ~40.6 h since first supervisor event 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable to repository tools; no usable `research.sqlite`/`research.sqlite3` found, excluding desktop conversation usage.

Concrete actions completed this iteration:

1. Inspected the frozen risk-reselection devval protocol/runner and confirmed the pre-run backup from supervisor context was verified:
   - Backup status: `verified`
   - Remaining changed files: `0`
   - Commit: `e7a85738b7209bc1bbc85e97d9880675be6aee7f`

2. Ran exactly one legacy TF1 development-validation experiment:
   - Script: `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 0`
   - Registry: `research_artifacts/aws_runs/20260928T022816_6029a456/registry.json`
   - Exit status: `0`
   - Runtime: `3544.26 s`
   - Episodes: `184`
   - Control steps: `16091`
   - Fresh bank generation resets counted: `64`
   - New training/gradient steps: `0 / 0`
   - Historical validation64 bank opened: `False`
   - Sealed test opened: `False`

3. Key shard00 partial devval results on cases `[18, 42, 43, 50]`:
   - Seed0 risk-reselected vs fixed H25: success `4/4 vs 4/4`, physical delta `+0.0850605`, decision ratio mean `0.9852`, H20 used `13/307` steps.
   - Seed1 risk-reselected vs fixed H25: success `4/4 vs 4/4`, physical delta `+0.0000122`, decision ratio mean `0.9757`, H20 used `41/307` steps. It also improved physical cost vs current stored gated seed1 by about `-0.400464` on this shard.
   - Seed2 risk-reselected vs fixed H25: both failed case18; reported paired success `3/3 vs 3/3`, physical delta `-0.000000592`, decision ratio mean `0.9999`, H15 used `4/365` steps.
   - No constraints, initial solver failures, or final solver failures were reported for adaptive arms in shard00.
   - No H > 25 was executed under the common safe wrapper.

4. Preserved state and backup request:
   - Shard summary: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard00/summary.md`
   - Completed marker: `.../shard00/completed.json`
   - Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_DEVVAL64_SHARD00_20260928T032713Z.json`
   - Handoff state note: `research_artifacts/aws_state/vehicle_gated_horizon_risk_reselection_v1_devval_shard00_state_20260928T0328Z.md`
   - Durable supervisor state updated via `update_state`.

Interpretation: shard00 supports implementation readiness and shows no obvious safety regression, but it is only partial fresh development-validation evidence. It is not final-test evidence and not an ORIGINAL SAC reproduction claim. The full frozen 64-case campaign criteria remain in force.

Next bounded action: wait for/verify external backup covering shard00 artifacts and docs. If backup is verified, run exactly one next shard: `vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 1` with the legacy interpreter, no sealed test, no historical validation64 bank access.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard01 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 01

UTC: 2026-09-28T04:29:51.388015+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16541 control steps, cases [19, 20, 28, 49], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard01/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard01/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard01/completed.json`.


## 2026-09-28T04:31:36.243299+00:00
Elapsed service lifetime: ~41.7 h since first supervisor event `2026-09-26T10:55:29.419331Z` (service wall-clock, not summed experiment runtime; desktop conversation excluded). Cumulative server API `total_tokens`: unavailable to repository tools; no usable `research.sqlite`/`research.sqlite3` has been found, desktop usage excluded.

Concrete actions completed:

1. Ran exactly one bounded legacy TF1 development-validation experiment:
   - Script: `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 1`
   - Registry: `research_artifacts/aws_runs/20260928T032939_2e7c58c1/registry.json`
   - Exit status: `0`
   - Runtime: `3618.29 s`
   - Episodes: `184`
   - Control steps: `16541`
   - Fresh devval bank reset calls counted: `64`
   - New training / gradient steps: `0 / 0`
   - Historical validation64 bank opened: `False`
   - Sealed final test opened: `False`

2. Inspected shard01 outputs:
   - Summary: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard01/summary.md`
   - Completed marker: `.../shard01/completed.json`
   - Cases: `[19, 20, 28, 49]`

3. Key shard01 partial devval findings:
   - Seed0 risk-reselected vs matched fixed H25:
     - success `4/4 vs 4/4`
     - physical delta `+3.45203e-05`
     - total delta `-0.109965`
     - mean decision-time ratio `1.00605`
     - horizons `H25:298, H20:22`
   - Seed1 risk-reselected vs matched fixed H25:
     - success `4/4 vs 4/4`
     - physical delta `-1.2633e-05`
     - total delta `-0.0950126`
     - mean decision-time ratio `1.03222`
     - horizons `H25:301, H20:19`
   - Seed2 risk-reselected vs matched fixed H25:
     - success `4/4 vs 4/4`
     - physical delta `-10.0139`
     - total delta `-10.0539`
     - mean decision-time ratio `1.01697`
     - horizons `H25:316, H15:4`
   - No adaptive-arm constraints, initial solver failures, or final solver failures were reported in shard01.
   - No `H > 25` was executed under the common safe wrapper.

4. Current risk-reselection devval progress:
   - Completed shards: `2/16` (`shard00`, `shard01`)
   - Completed development-validation episodes: `368`
   - Completed control steps: `32632`
   - This is still development-validation/model-selection evidence only, not final-test evidence and not an ORIGINAL SAC reproduction claim.

5. Preserved state and requested backup:
   - State note written:
     - `research_artifacts/aws_state/vehicle_gated_horizon_risk_reselection_v1_devval_shard01_state_20260928T0435Z.md`
   - Backup addendum request written:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_DEVVAL64_SHARD01_STATE_ADDENDUM_20260928T0435Z.json`
   - Durable supervisor state updated via `update_state`.

Scientific interpretation:
- Shard01 continues to support implementation readiness and shows no obvious adaptive-arm safety regression.
- Risk-first re-selection remains a plausible selection-objective bottleneck repair, but the evidence is still partial.
- Actual decision-time ratios are mixed/slower in shard01 despite shorter horizons, so no compute-time advantage claim is supported yet.
- Seed2 still uses short horizon very sparsely on these early devval shards.

Current blocker / next action:
- A verified external backup covering shard01 artifacts, registry/logs, docs, state note, and backup requests is required before running more simulations.
- After backup verification, next exact action is one experiment only:
  - `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 2`
  - legacy interpreter
  - no historical validation64 bank access
  - no sealed final-test access.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard02 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 02

UTC: 2026-09-28T05:27:19.235465+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 15023 control steps, cases [2, 5, 14, 56], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard02/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard02/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard02/completed.json`.


## 2026-09-28T05:29:45.385219+00:00
Elapsed service lifetime: ~42.7 h since first supervisor event `2026-09-26T10:55:29.419331Z` (service wall-clock; excludes summed experiment runtime and desktop conversation). Cumulative server API `total_tokens`: unavailable to repository tools; no usable `research.sqlite`/`research.sqlite3` has been found; desktop conversation usage excluded.

Concrete actions completed this iteration:

1. Ran exactly one bounded legacy TF1 development-validation experiment:
   - Script: `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 2`
   - Method: `IMPROVED_vehicle_gated_horizon_risk_reselection_v1_devval64_shard02_not_original_SAC`
   - Registry: `research_artifacts/aws_runs/20260928T043243_dd7fca89/registry.json`
   - Exit status: `0`
   - Runtime: `3282.04 s`
   - Episodes: `184`
   - Control steps: `15023`
   - New training / gradient steps: `0 / 0`
   - Historical validation64 bank opened: `False`
   - Sealed final test opened: `False`

2. Inspected shard02 outputs:
   - Summary: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard02/summary.md`
   - Completed marker: `.../shard02/completed.json`
   - Cases: `[2, 5, 14, 56]`

3. Key shard02 development-validation findings:
   - Seed0 risk-reselected vs matched fixed H25:
     - Success `4/4 vs 4/4`
     - Physical delta `+0.067169`
     - Total delta `-0.112831`
     - Mean decision-time ratio `1.00959`
     - Horizons `H25:248, H20:36`
   - Seed1 risk-reselected vs matched fixed H25:
     - Success `4/4 vs 4/4`
     - Physical delta `+2.31745e-05`
     - Total delta `-0.274977`
     - Mean decision-time ratio `0.997687`
     - Horizons `H25:229, H20:55`
   - Seed2 risk-reselected vs matched fixed H25:
     - Success `4/4 vs 4/4`
     - Physical delta `+5.19315e-07`
     - Total delta `-0.0499995`
     - Mean decision-time ratio `1.02585`
     - Horizons `H25:279, H15:5`

4. Safety / execution evidence:
   - No adaptive-arm constraints reported in shard02.
   - No adaptive-arm initial solver failures reported.
   - No adaptive-arm final solver failures reported.
   - Common safe wrapper executed no `H > 25`.

5. Campaign progress after shard02:
   - Completed risk-reselection devval shards: `3/16` (`shard00`, `shard01`, `shard02`)
   - Completed episodes: `552`
   - Completed control steps: `47655`
   - Completed fresh devval case blocks: `12/64`
   - Still development-validation/model-selection evidence only.
   - Not final-test evidence.
   - Not an ORIGINAL SAC/neural-training reproduction claim.

Scientific interpretation:
- Risk-first re-selection remains implementation-ready and has not shown obvious safety regression in completed fresh devval shards.
- Total cost vs matched fixed H25 remains favorable in shard02 because the synthetic horizon/work term rewards shorter horizons, but physical-control deltas are small and slightly positive for all three seeds in shard02.
- Actual measured decision-time ratios remain mixed; seed2 is slower despite rare short-H use. No compute-time advantage claim is supported yet.
- Seed2 adaptation is still sparse, so candidate sufficiency / policy-class limitations remain active hypotheses.
- The frozen v2 transition-hold campaign remains preserved and unchanged.

State preserved:
- Wrote state note:
  - `research_artifacts/aws_state/vehicle_gated_horizon_risk_reselection_v1_devval_shard02_state_20260928T0535Z.md`
- Wrote backup addendum request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_DEVVAL64_SHARD02_STATE_ADDENDUM_20260928T0535Z.json`
- Durable supervisor state updated with shard02 evidence, campaign progress, backup gate, and next action.

Current blocker / next action:
- A verified external backup is required before running more simulations.
- After backup verification, next exact action is one experiment only:
  - `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 3`
  - legacy interpreter
  - frozen risk-reselection devval64 protocol
  - no historical validation64 bank access
  - no sealed final-test access.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard03 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 03

UTC: 2026-09-28T06:29:54.353947+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16619 control steps, cases [26, 31, 36, 39], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard03/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard03/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard03/completed.json`.


## 2026-09-28T06:32:16.626883+00:00
Elapsed service lifetime: ~43.7 h since first supervisor event `2026-09-26T10:55:29.419331Z` (service wall-clock; excludes summed experiment runtime and desktop conversation). Cumulative server API `total_tokens`: unavailable to repository tools; no usable `research.sqlite` / `research.sqlite3` found in repository; desktop conversation usage excluded.

Concrete actions completed this iteration:

1. Ran exactly one bounded legacy TF1 development-validation experiment:
   - Script: `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 3`
   - Method: `IMPROVED_vehicle_gated_horizon_risk_reselection_v1_devval64_shard03_not_original_SAC`
   - Registry: `research_artifacts/aws_runs/20260928T053054_7b7a221f/registry.json`
   - Exit status: `0`
   - Runtime: `3546.23 s`
   - Episodes: `184`
   - Control steps: `16619`
   - New training / gradient steps: `0 / 0`
   - Historical validation64 bank opened: `False`
   - Sealed final test opened: `False`

2. Inspected shard03 evidence:
   - Summary: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard03/summary.md`
   - Completed marker: `.../shard03/completed.json`
   - Cases: `[26, 31, 36, 39]`

3. Key shard03 development-validation findings, risk-reselected vs matched fixed H25:
   - Seed0:
     - Success `4/4 vs 4/4`
     - Physical delta `+0.128395`
     - Total delta `+0.068395`
     - Mean decision-time ratio `1.01134`
     - Horizons `H25:312, H20:12`
   - Seed1:
     - Success `4/4 vs 4/4`
     - Physical delta `+4.3744e-05`
     - Total delta `-0.224956`
     - Mean decision-time ratio `1.03403`
     - Horizons `H25:279, H20:45`
   - Seed2:
     - Success `4/4 vs 4/4`
     - Physical delta `0`
     - Total delta `0`
     - Mean decision-time ratio `1.00141`
     - Horizons `H25:324`; no short-horizon use in this shard.

4. Safety / execution evidence:
   - No adaptive-arm constraints reported.
   - No adaptive-arm initial solver failures reported.
   - No adaptive-arm final solver failures reported.
   - Common safe wrapper executed no `H > 25`.

5. Campaign progress after shard03:
   - Completed risk-reselection devval shards: `4/16`
   - Completed fresh case blocks: `16/64`
   - Completed development-validation episodes: `736`
   - Completed development-validation control steps: `64274`
   - Frozen v2 transition-hold campaign remains preserved and unchanged at shard00 only.

Scientific interpretation:
- Shard03 continues implementation-readiness and safety evidence for the IMPROVED risk-reselection policy.
- The scientific signal remains mixed:
  - Seed0 is worse than matched H25 on both physical and total cost in this shard.
  - Seed1 has essentially unchanged physical cost and lower total cost due to the horizon/work term.
  - Seed2 did not adapt at all in this shard.
  - Measured decision-time ratios are still mixed/slower, so compute-time benefit is not established.
- This remains development-validation/model-selection evidence only, not final-test evidence and not an ORIGINAL SAC reproduction claim.

State preserved:
- Wrote state note:
  - `research_artifacts/aws_state/vehicle_gated_horizon_risk_reselection_v1_devval_shard03_state_20260928T0638Z.md`
- Wrote backup addendum request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_DEVVAL64_SHARD03_STATE_ADDENDUM_20260928T0638Z.json`
- Existing shard03 backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_DEVVAL64_SHARD03_20260928T062954Z.json`
- Durable supervisor state updated with shard03 evidence, campaign progress, backup gate, and next action.

Current blocker / next action:
- A verified external backup covering shard03 outputs, run registry/logs, docs, state note, and backup requests is required before running more simulations.
- After backup verification, next exact action is one experiment only:
  - `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 4`
  - legacy interpreter
  - frozen risk-reselection devval64 protocol
  - no historical validation64 bank access
  - no sealed final-test access.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard04 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 04

UTC: 2026-09-28T07:34:31.779002+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 17494 control steps, cases [22, 37, 52, 62], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard04/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard04/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard04/completed.json`.


## 2026-09-28T07:36:56.341965+00:00
Elapsed service lifetime: ~44.8 h since `2026-09-26T10:55:29.419331Z` (service wall-clock; excludes summed experiment runtime and desktop conversation). Cumulative server API `total_tokens`: unavailable to repository tools; no repository-accessible `research.sqlite` / `research.sqlite3` found.

Concrete actions completed:

1. Ran exactly one bounded legacy TF1 development-validation experiment:
   - Script: `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 4`
   - Method: `IMPROVED_vehicle_gated_horizon_risk_reselection_v1_devval64_shard04_not_original_SAC`
   - Registry: `research_artifacts/aws_runs/20260928T063323_819f578e/registry.json`
   - Exit status: `0`
   - Runtime: `3674.28 s`
   - Episodes/control steps: `184 / 17494`
   - Cases: `[22, 37, 52, 62]`
   - New training / gradient steps: `0 / 0`
   - Historical validation64 bank opened: `False`
   - Sealed final test opened: `False`

2. Inspected shard04 evidence:
   - Summary: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard04/summary.md`
   - Completed marker: `.../shard04/completed.json`
   - Run registry: `research_artifacts/aws_runs/20260928T063323_819f578e/registry.json`

3. Shard04 key results, risk-reselected vs matched fixed H25:
   - Seed0:
     - Success `4/4 vs 4/4`
     - Physical delta `+0.298031`
     - Total delta `+0.088031`
     - Decision-time ratio mean `1.000526`
     - Horizons `H25:310, H20:42`
   - Seed1:
     - Success `4/4 vs 4/4`
     - Physical delta `+3.59685e-05`
     - Total delta `-0.534964`
     - Decision-time ratio mean `1.019261`
     - Horizons `H25:245, H20:107`
   - Seed2:
     - Success `4/4 vs 4/4`
     - Physical delta `+1.14043e-05`
     - Total delta `-0.549989`
     - Decision-time ratio mean `1.011188`
     - Horizons `H25:297, H15:55`

4. Shard04 adaptive-arm safety:
   - No adaptive-arm constraints reported.
   - No adaptive-arm initial solver failures reported.
   - No adaptive-arm final solver failures reported.
   - Safe wrapper executed no `H > 25`.

5. Campaign progress after shard04:
   - Risk-reselection devval64 completed shards: `5/16`
   - Completed case blocks: `20/64`
   - Completed development-validation episodes/control steps: `920 / 81768`
   - No sealed test access.
   - No historical validation64 bank access.
   - Frozen v2 transition-hold campaign remains preserved and unchanged at shard00 only.

Scientific interpretation:
- This remains development-validation/model-selection evidence only, not final-test evidence and not ORIGINAL SAC reproduction.
- Shard04 preserves implementation/safety readiness but weakens the overall signal:
  - Seed0 is worse than H25 on both physical and total cost in shard04.
  - Seed1/seed2 total-cost gains are largely from the synthetic horizon/work term, with negligible physical deltas.
  - Measured decision-time ratios remain mixed/slower; actual compute-time superiority is still unproven.
- Approximate partial cumulative risk-vs-H25 over shards00–04:
  - Seed0: physical `+0.57869`, total `-0.04631`, H20 `125/1587`, mean decision ratio ~`1.00254`
  - Seed1: physical `+0.000102`, total `-1.33490`, H20 `267/1587`, mean decision ratio ~`1.01177`
  - Seed2: physical `-10.01389`, total `-10.69389`, H15 `68/1645`, mean decision ratio ~`1.01107`
- Risk-first reselection is still a plausible selection-objective repair, but current partial evidence is mixed and not enough for success, futility, or final-test gating.

State preserved:
- Wrote state note:
  - `research_artifacts/aws_state/vehicle_gated_horizon_risk_reselection_v1_devval_shard04_state_20260928T0742Z.md`
- Wrote backup addendum request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_DEVVAL64_SHARD04_STATE_ADDENDUM_20260928T0742Z.json`
- Existing shard04 backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_GATED_HORIZON_RISK_RESELECTION_V1_DEVVAL64_SHARD04_20260928T073431Z.json`
- Durable supervisor state updated with shard04 evidence, campaign progress, blockers, and next action.

Current blocker / next action:
- A verified external backup covering shard04 outputs, run registry/logs, docs, state note, and backup requests is required before any more simulations.
- After backup verification, next exact action is one experiment only:
  - `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_devval_shard_runner.py --shard 5`
  - legacy interpreter
  - frozen risk-reselection devval64 protocol
  - no historical validation64 bank access
  - no sealed final-test access.

<!-- vehicle-gated-horizon-risk-reselection-v1-devval64-shard-20260928-v1-shard05 -->
## 2026-09-28 vehicle gated-horizon risk-reselection v1 fresh development-validation shard 05

UTC: 2026-09-28T08:35:14.280065+00:00. Ran a frozen fresh development-validation paired case-block shard for IMPROVED risk-reselected gated-horizon v1: 184 episodes, 16079 control steps, cases [0, 34, 54, 55], no historical validation64 bank reopen and no sealed-test access. The shard includes all 46 arms (risk-reselected, current stored gated policies, matched fixed-H grid, and seed0 independent-terminal grid) for each included case. This is development-validation/model-selection evidence only, not final-test evidence. Artifacts: `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard05/summary.md`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard05/raw.json`, `research_artifacts/aws_development_validation/vehicle_gated_horizon_risk_reselection_v1_devval64_20260928_v1/shard05/completed.json`.


## 2026-09-28T08:41:27.362343+00:00



## 2026-09-28T08:48:06.178376+00:00


<!-- vehicle-risk-reselection-partial-opportunity-diagnostic-20260928 -->
## 2026-09-28 risk-reselection partial opportunity/runtime diagnostic

UTC: 2026-09-28T08:50:05.424948+00:00. Read-only diagnostic over fresh devval shards [0, 1, 2, 3, 4, 5]; no simulations, no training, no sealed-test access, and no historical validation64 bank access. Parsed 1104 per-episode summaries and shard-reported 1104 episodes / 97847 control steps.

Key result: Do not allocate another unchanged long risk-reselection shard by default. Preserve the partial campaign and prioritize a versioned training/selection or scenario-opportunity diagnostic that can create a stronger candidate or explain absent adaptive opportunity.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_20260928T0845Z/completed.json`. New diagnostic artifacts and doc updates require external backup before further simulations.


## 2026-09-28T08:54:39.683564+00:00


<!-- vehicle-risk-reselection-partial-opportunity-diagnostic-v2-20260928 -->
## 2026-09-28 risk-reselection partial opportunity/runtime diagnostic V2

UTC: 2026-09-28T08:55:56.583581+00:00. Corrected read-only diagnostic over fresh devval shards [0, 1, 2, 3, 4, 5]; no simulations, no training, no sealed-test access, and no historical validation64 bank access. V2 supersedes v1 fixed-H timing-opportunity statements because v1 did not parse nested per-episode timing. Parsed 1104 per-episode summaries and shard-reported 1104 episodes / 97847 control steps.

Decision: Pause additional unchanged long risk-reselection devval shards after backup; freeze an outcome-informed scheduling amendment and design a smaller versioned training/selection or scenario-opportunity experiment with actual measured-time objectives.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/completed.json`. New diagnostic artifacts and doc updates require external backup before further simulations.


## 2026-09-28T09:00:00.718212+00:00


<!-- vehicle-gated-horizon-actual-time-reselection-v2-20260928 -->
## 2026-09-28 vehicle actual-time-aware gated-horizon re-selection V2

UTC: 2026-09-28T09:00:46+00:00. Metadata-only IMPROVED finite re-selection using existing training candidate metrics and corrected V2 timing/opportunity diagnostics; no simulations, no training, no historical validation64 bank reopen, and no sealed-test access. Adaptive nominations: 2/3; acceptance_for_smoke_met=True. Nominations: {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}. Next action after backup: `after_backup_freeze_and_run_small_actual_time_reselection_v2_smoke`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2_20260928T0905Z/completed.json`.


## 2026-09-28T09:04:18.513108+00:00
Cumulative elapsed since first supervisor event (2026-09-26T10:55:29.419331Z): ~1d 22h 10m as of 2026-09-28T09:06Z.  
Cumulative server API `total_tokens` from `research.sqlite`: unavailable from accessible repository tools this iteration; no `research.sqlite`/token-usage file was visible under the repository, so I am not inventing a number.

## Concrete actions completed

1. **Inspected current evidence/state**
   - Read `STATUS.md`.
   - Read V2 opportunity/runtime diagnostic outputs:
     - `research_artifacts/aws_diagnostics/vehicle_risk_reselection_devval_partial_opportunity_diagnostic_v2_20260928T0855Z/summary.md`
     - `.../completed.json`
     - `.../raw.json` partially.
   - Read the outcome-informed scheduling amendment:
     - `research_artifacts/aws_protocols/vehicle_risk_reselection_v1_scheduling_amendment_after_v2_20260928.md`
   - Inspected actual-time reselection script source.

2. **Executed the frozen metadata-only actual-time-aware reselection**
   - Run: `20260928T090045_278ad959`
   - Script: `experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2.py`
   - Exit status: `0`
   - Runtime: ~2.01 s
   - Budget: **0 rollouts, 0 control steps, 0 training episodes, 0 gradient steps**
   - Access: **no sealed test, no historical validation64 bank reopen**

3. **Main V2 result**
   - The existing finite candidate class is **not fully exhausted** under measured-time-aware selection.
   - Correct V2 nominal nominations:
     - seed0: fixed H25 fallback
     - seed1: `h15_p1_g5`
     - seed2: `h15_p1_g5`
   - Adaptive nominations: `2/3`
   - Frozen metadata acceptance for smoke: `True`
   - This is still **IMPROVED finite search/reselection**, not ORIGINAL SAC and not gradient training.

4. **Important correction found**
   - Post-run source inspection found a seed0 bookkeeping bug in V2:
     - code used `overhead_by_seed.get(seed or -1)`
     - for seed0, `seed or -1` becomes `-1`, so seed0 adaptive candidates were scored with zero selection overhead.
   - Seeds 1 and 2 and the overall V2 acceptance appear unaffected, but **V2 seed0 candidate timing fields are unreliable** until repaired.

5. **Prepared repair source**
   - Wrote:
     - `experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair.py`
   - SHA256:
     - `529436ff04ab288474a4ad3f0a6579a4793b0de512d8ba1523933f6b7db0996e`
   - Not executed yet, preserving one-experiment-per-iteration discipline.

6. **Preserved state**
   - Wrote:
     - `research_artifacts/aws_state/vehicle_actual_time_reselection_v2_and_v2b_repair_pending_20260928T0906Z.md`
   - Updated durable supervisor state via `update_state`.

## Current blockers / next action

- **Backup required before any further simulations** covering:
  - V2 outputs,
  - run registry/logs,
  - docs/state updates,
  - V2b repair source.
- Before any smoke rollout, run the metadata-only V2b repair:
  - `experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2b_seed0_overhead_repair.py`
- If V2b still confirms ≥2 adaptive nominations, then after verified backup freeze and run a small actual-time V2b smoke/confirmation block.
- Do **not** resume unchanged risk-reselection devval shards06–15 by default.
- Do **not** open/request sealed final test.

<!-- vehicle-gated-horizon-actual-time-reselection-v2b-seed0-overhead-repair-20260928 -->
## 2026-09-28 vehicle actual-time-aware re-selection V2b seed0-overhead repair

UTC: 2026-09-28T09:05:35+00:00. Metadata-only repair of V2 seed0 selection-overhead bookkeeping; no simulations, no training, no validation64 bank reopen, and no sealed-test access. Corrected nominations: {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}; adaptive_nominated_seed_count=2/3; acceptance_for_smoke_met=True; changes_vs_V2={'seed0_overhead_fraction_applied_now': 0.0016665805221004085, 'previous_v2_acceptance_for_smoke_met': True, 'corrected_acceptance_for_smoke_met': True, 'acceptance_changed': False, 'previous_v2_nomination_ids': {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}, 'corrected_nomination_ids': {'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}, 'nomination_ids_changed': False}. Next action after backup: `after_backup_freeze_and_run_small_actual_time_reselection_v2b_smoke`.


## 2026-09-28T09:10:25.791864+00:00

<!-- vehicle-gated-horizon-actual-time-reselection-v2b-smoke-preflight-20260928 -->
## 2026-09-28 vehicle actual-time-aware V2b smoke preflight

UTC: 2026-09-28T09:12:46+00:00. Metadata-only preflight for the frozen IMPROVED V2b smoke completed: hard_pass=True, nominations={'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}. No simulations, training, validation64 bank access, or sealed-test access. Next action after verified backup: `after_verified_backup_run_legacy_actual_time_v2b_smoke`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_preflight_20260928T0910Z/completed.json`.



## 2026-09-28T09:15:59.323723+00:00

<!-- vehicle-actual-time-v2b-candidate-failure-surface-diagnostic-20260928 -->
## 2026-09-28 vehicle actual-time V2b candidate failure-surface diagnostic

UTC: 2026-09-28T09:17:06+00:00. Metadata-only diagnostic completed over V2b candidate metrics and preflight artifacts: rows=111, nominations={'0': 'fixed', '1': 'h15_p1_g5', '2': 'h15_p1_g5'}, adaptive_nominated_seed_count=2/3. No simulations, no training, no validation64 bank access, and no sealed-test access. Verified seed0 has no eligible adaptive candidate under the current finite V2b gates; seeds1/2 both nominate h15_p1_g5. Next action after verified backup: run the already-frozen 36-episode legacy V2b smoke; do not resume the unchanged long risk-reselection devval shards by default. Artifacts: `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/completed.json`.



## 2026-09-28T09:19:14.111493+00:00
~1d 22h 25m elapsed since first supervisor event (2026-09-26T10:55:29.419331Z). Cumulative server API `total_tokens`: unavailable from repository tools — no `research.sqlite` / `*.sqlite` found; this excludes desktop conversation usage.

Concrete actions completed:

1. Inspected current V2b smoke/protocol context:
   - `STATUS.md`
   - `experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2b_smoke.py`
   - `experiments/bohn2021_aws/vehicle_gated_horizon_risk_reselection_v1_smoke.py`
   - V2b smoke protocol and preflight artifacts
   - V2b nomination artifacts

2. Executed the prepared metadata-only V2b candidate failure-surface diagnostic:
   - Run: `research_artifacts/aws_runs/20260928T091706_df2c9e7e/registry.json`
   - Script: `experiments/bohn2021_aws/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic.py`
   - Exit status: 0
   - Runtime: ~2.0 s
   - New rollouts/control steps/training/gradient steps: 0
   - Historical validation64 bank opened: false
   - Sealed test accessed: false

3. Key findings:
   - Parsed 111 candidate rows, 37 per seed.
   - V2b remains IMPROVED finite metadata re-selection over existing gated candidates; zero new gradient training; not ORIGINAL SAC.
   - Corrected nominations remain:
     - seed0: fixed H25 fallback
     - seed1: `h15_p1_g5`
     - seed2: `h15_p1_g5`
   - Adaptive nominated seed count: 2/3.
   - Seed0 has zero eligible adaptive candidate under current actual-time/physical-risk gates.
   - Seeds1/2 each have exactly one eligible adaptive candidate; both nominate the same finite candidate.
   - Smoke preflight remains hard-pass, but no V2b rollout/smoke has yet tested actual executability, replay determinism, or realized timing.

4. Artifacts written:
   - `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/summary.md`
   - `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/raw.json`
   - `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z/completed.json`
   - `research_artifacts/aws_state/vehicle_actual_time_v2b_candidate_failure_surface_diagnostic_20260928T0915Z.md`
   - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_ACTUAL_TIME_V2B_CANDIDATE_FAILURE_SURFACE_DIAGNOSTIC_20260928T0915Z.json`
   - Additional preserved iteration state:
     - `research_artifacts/aws_state/vehicle_actual_time_v2b_iteration_state_20260928T0920Z.md`

5. State preserved via `update_state`.

Decision preserved for next iteration:

- Do not resume unchanged long risk-reselection devval shards by default.
- Backup is required before any further simulations because new diagnostic/state artifacts now exist.
- After verified backup, run the already-frozen 36-episode legacy V2b smoke with:
  - `experiments/bohn2021_aws/vehicle_gated_horizon_actual_time_reselection_v2b_smoke.py`
  - legacy interpreter
  - engineering-smoke classification only
  - no validation64/test access.

<!-- vehicle-gated-horizon-actual-time-reselection-v2b-smoke-20260928 -->
## 2026-09-28 vehicle actual-time-aware gated-horizon V2b smoke

UTC: 2026-09-28T09:31:58.719179+00:00. IMPROVED actual-time-aware gated-horizon V2b smoke completed on a fresh engineering bank: 36 episodes, 2826 control steps, hard_pass=True, selected_below_H25_seed_count=2. No validation64 bank or sealed test was opened. This is not model-selection/final evidence and does not modify any frozen devval campaign. Artifacts: `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_gated_horizon_actual_time_reselection_v2b_smoke_20260928/completed.json`.


## 2026-09-28T09:35:01.486270+00:00

<!-- vehicle-actual-time-v2b-smoke-postdiagnostic-20260928 -->
## 2026-09-28 vehicle actual-time V2b smoke postdiagnostic

UTC: 2026-09-28T09:36:20.558519+00:00. Metadata-only postdiagnostic of the V2b smoke completed; no simulations/training, no validation64 bank access, and no sealed-test access. Smoke hard_pass=True; episodes=36; control_steps=2826. Weighted selected/fixed decision-time ratio=0.975365; adaptive seed ratios={'1': 0.923164173083399, '2': 1.016109314669204}. Decision: V2b is engineering-ready but not strong enough to justify a long validation campaign by default. Next protocol draft: `research_artifacts/aws_protocols/vehicle_scenario_opportunity_capability_diagnostic_v0_after_v2b_smoke_20260928.md`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_actual_time_v2b_smoke_postdiagnostic_20260928T0935Z/completed.json`.


## 2026-09-28T09:49:12.027747+00:00

<!-- vehicle-scenario-opportunity-capability-diagnostic-v0-20260928T0950Z -->
## Vehicle scenario-opportunity/capability diagnostic V0

UTC: 2026-09-28T09:50:19.198779+00:00. Metadata/source-only audit; no simulations, training, validation64 bank access, or sealed-test access. Finding: the vehicle environment supports straight-line goal/path length and heading variation plus three obstacle constraints/noisy forecasts, but the current registered vehicle config fixes initial x/y/theta, lacks plant process noise/model randomization, has no direct initial-speed state, and has no native curved-path generator. Decision: Run a bounded fixed-H opportunity probe only after backup; do not redesign scenarios or resume long adaptive validation until fixed-H Pareto opportunity is measured. Artifacts: `research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_scenario_opportunity_capability_diagnostic_v0_20260928T0950Z/completed.json`.


## 2026-09-28T09:54:21.293994+00:00

<!-- vehicle-fixed-h-opportunity-probe-v0-preflight-20260928T1000Z -->
## Vehicle fixed-H opportunity probe V0 preflight

UTC: 2026-09-28T09:55:04.355094+00:00. Metadata-only preflight froze the next bounded fixed-H opportunity-probe design: 80 planned fixed-H episodes, <=12000 control steps, 24 fresh candidate-bank resets, full H grid 5..50, independent seed0 terminal per H. Terminal grid ready=True. No simulations/training/validation64/test access occurred. External backup is required before the rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_preflight_20260928T1000Z/completed.json`, `research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v0_frozen_20260928.md`.


## 2026-09-28T10:00:17.636007+00:00

<!-- vehicle-fixed-h-opportunity-probe-v0-runner-20260928 -->
## 2026-09-28 vehicle fixed-H opportunity probe V0

UTC: 2026-09-28T10:26:46.034089+00:00. Fresh development-only fixed-H opportunity probe completed: 80 episodes, 6313 control steps, candidate resets=24. No validation64 or sealed-test access. Development opportunity flag=True; weak single-H pattern flag=False; safe horizons=[10, 15, 20, 25, 30, 35, 40, 45, 50]; strongest total H=15. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_20260928/completed.json`.


## 2026-09-28T10:30:34.115197+00:00

<!-- vehicle-fixed-h-opportunity-probe-v0-postdiagnostic-20260928T1030Z -->
## 2026-09-28 fixed-H opportunity V0 postdiagnostic

UTC: 2026-09-28T10:31:41.049775+00:00. Metadata-only postdiagnostic parsed the fresh fixed-H opportunity probe. No validation64/test access. Oracle best-physical same-bank physical delta vs H15=0.747195; LOOCV one-split physical-risk selector risk delta vs H15=-47.7779. Conclusion: document weak material opportunity and avoid another adaptive campaign until scenario design is revised under a new protocol. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v0_postdiagnostic_20260928T1030Z/raw.json`.


## 2026-09-28T10:36:56.196054+00:00

<!-- vehicle-fixed-h-opportunity-probe-v1-preflight-20260928T1045Z -->
## Vehicle fixed-H opportunity probe V1 preflight

UTC: 2026-09-28T10:43:42.205188+00:00. Metadata-only preflight froze an enlarged fixed-H opportunity map: 160 planned fixed-H episodes, <=24000 control steps, 64 fresh candidate-bank resets, 16 selected source-supported cases, full H grid 5..50. Terminal grid ready=True. No simulations/training/validation64/test access occurred. External backup is required before the rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_preflight_20260928T1045Z/completed.json`, `research_artifacts/aws_protocols/vehicle_fixed_h_opportunity_probe_v1_frozen_20260928.md`.


## 2026-09-28T10:46:43.791632+00:00

<!-- vehicle-fixed-h-opportunity-probe-v1-runner-20260928 -->
## 2026-09-28 vehicle fixed-H opportunity probe V1

UTC: 2026-09-28T11:38:22.599799+00:00. Enlarged fresh development-only fixed-H opportunity probe completed: 160 episodes, 13155 control steps, candidate resets=64. No validation64 or sealed-test access. Opportunity flag=True; weak single-H pattern flag=False; safe horizons=[10, 15, 20, 25, 30, 35, 40, 45, 50]; strongest total H=15. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_20260928/completed.json`.

<!-- vehicle-fixed-h-opportunity-probe-v1-postdiagnostic-20260928T1150Z -->
## 2026-09-28 fixed-H opportunity V1 postdiagnostic

UTC: 2026-09-28T11:49:17.454140+00:00. Metadata-only postdiagnostic applied frozen V1 materiality and LOOCV predictor gates. No simulation/training/validation64/test access. Physical oracle pass=True; total oracle pass=True; predictor gate pass=False. Decision: after verified backup, run a bounded controlled continuation/value-and-transition diagnostic on representative V1 states because oracle opportunity is material but metadata predictability is not yet robust enough for broad retraining/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_fixed_h_opportunity_probe_v1_postdiagnostic_20260928T1150Z/raw.json`.


## 2026-09-28T11:52:51.644256+00:00

<!-- vehicle-v1-continuation-target-diagnostic-20260928T1155Z -->
## 2026-09-28 V1 continuation target diagnostic

UTC: 2026-09-28T11:54:18.311563+00:00. Metadata/trace-only diagnostic selected 4 V1 cases for a future controlled continuation replay; max future episodes=32, control-step bound=4800. No simulations/training/validation64/test access occurred. Next action after verified backup: after verified backup, implement/run the frozen controlled-continuation diagnostic before selector/refit training. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_continuation_target_diagnostic_20260928T1155Z/summary.md`, `research_artifacts/aws_protocols/vehicle_v1_controlled_continuation_diagnostic_v0_frozen_20260928.md`.


## 2026-09-28T11:58:57.747763+00:00



## 2026-09-28T12:08:10.590846+00:00

<!-- vehicle-v1-controlled-continuation-diagnostic-v0b-one-variable-terminal-helper-repair-20260928 -->
## 2026-09-28 vehicle V1 controlled-continuation diagnostic v0

UTC: 2026-09-28T12:20:32.088109+00:00. Development-only identical-prefix continuation diagnostic completed: 32 episodes, 2447 control steps, no validation64/test access. Confirmed material states=3; selector-smoke gate=True. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_controlled_continuation_diagnostic_v0b_20260928/completed.json`.


## 2026-09-28T12:23:25.780435+00:00



## 2026-09-28T12:40:50.968495+00:00

<!-- vehicle-v1-state-continuation-selector-offline-refit-v0-20260928T1240Z -->
## 2026-09-28 vehicle V1 state-continuation selector offline refit v0

UTC: 2026-09-28T12:43:00.533809+00:00. Metadata/trace-only offline selector refit completed with no simulations/training/validation64/test access. Offline gate=False; selected radius=0.1; positive latches=3/3; guard hits=0; nonpositive latches=0. Next frozen smoke protocol: `research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0_frozen_20260928.json`; backup required before rollout.


## 2026-09-28T12:49:33.202120+00:00

<!-- vehicle-v1-state-continuation-selector-caselevel-audit-v0-20260928T1250Z -->
## 2026-09-28 vehicle V1 state-continuation selector case-level latch audit v0

UTC: 2026-09-28T12:50:58.024985+00:00. No-simulation case-level audit completed after offline refit. Strict offline gate=False, case-level latch gate=True; positive cases latched=[7, 10]; guard hits=0; nonpositive-case latches=0; bad premature latches=0. Frozen v0b selector-smoke protocol: `research_artifacts/aws_protocols/vehicle_v1_state_continuation_selector_smoke_v0b_caselevel_gate_frozen_20260928.json`. Backup required before any rollout.


## 2026-09-28T12:56:26.527691+00:00

<!-- vehicle-v1-state-continuation-selector-smoke-v0b-20260928 -->
## 2026-09-28 vehicle V1 state-continuation selector smoke v0b

UTC: 2026-09-28T13:03:28.592333+00:00. Development-only selector smoke completed: 16 episodes, 1224 control steps, no validation64/test access. Expansion gate=True; safety=True; opportunity cases improved=[7, 10]. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_smoke_v0b_20260928/completed.json`.


## 2026-09-28T13:09:28.413414+00:00

<!-- vehicle-v1-state-selector-shadow-scan-v0-20260928 -->
## 2026-09-28 vehicle V1 state-selector shadow scan v0

UTC: 2026-09-28T13:10:35.706528+00:00. No-simulation scan over existing V1 H15 traces: triggered_cases=[7, 10], extra_trigger_cases=[], shadow_gate=True. No validation64/test access. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_state_selector_shadow_scan_v0_20260928T1310Z/raw.json`. Backup required before further simulation.


## 2026-09-28T13:16:35.196216+00:00

<!-- vehicle-v1-state-continuation-selector-broad-dev-confirmation-v0-20260928 -->
## 2026-09-28 vehicle V1 state-continuation selector broad development confirmation v0

UTC: 2026-09-28T13:38:03.170039+00:00. Development-only broad selector confirmation completed: 64 episodes, 4880 control steps, no validation64/test access. Protocol acceptance=True; safety=True; benefit_vs_H15=(physical 38.7155, total 36.9005); triggered_cases=[7, 10]. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_state_continuation_selector_broad_dev_confirmation_v0_20260928/completed.json`.


## 2026-09-28T13:42:10.128618+00:00

<!-- vehicle-v1-state-selector-broad-postdiagnostic-v0-20260928 -->
## 2026-09-28 vehicle V1 state-selector broad postdiagnostic v0

UTC: 2026-09-28T13:43:04.216531+00:00. No-simulation postdiagnostic completed after broad selector confirmation. Broad selector acceptance=True; benefit_vs_H15=(physical 38.7155, total 36.9005); triggered_cases=[7, 10]. Next frozen protocol: `research_artifacts/aws_protocols/vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928.md` / `research_artifacts/aws_protocols/vehicle_v1_fresh_continuation_label_probe_v0_frozen_20260928.json`. No validation64/test access and no new rollout/training. Backup required before further simulation.


## 2026-09-28T14:21:30.892859+00:00

<!-- vehicle-v1-fresh-continuation-label-probe-v0-20260928 -->
## 2026-09-28 vehicle V1 fresh continuation-label probe v0

UTC: 2026-09-28T14:53:19.951434+00:00. Development-only fresh continuation-label probe completed: 96 episodes, 7632 control steps, fresh candidate resets=48. No validation64/test access. Positive states=0 across cases=[]; negative/neutral states=24; refit/training gate=False. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_probe_v0_20260928/completed.json`.


## 2026-09-28T14:56:43.337384+00:00



## 2026-09-28T15:01:41.444392+00:00
Elapsed service lifetime: about 2 days 4h13m since 2026-09-26T10:55:29.419331Z. Server API total_tokens from research.sqlite: unknown/unavailable from repository tools; not reported as zero.

Concrete actions completed this iteration:

1. Executed the pending no-simulation postdiagnostic:
   - Script: `experiments/bohn2021_aws/vehicle_v1_fresh_continuation_label_postdiagnostic_v0.py`
   - Registry: `research_artifacts/aws_runs/20260928T145806_b791973a/registry.json`
   - Exit status: `1`
   - New simulations/control steps/training/gradient: `0/0/0/0`
   - Historical validation64 opened: `False`
   - Sealed test accessed: `False`

2. Failure diagnosis:
   - The script wrote the substantive raw/summary/protocol outputs, then failed only during documentation append.
   - Cause: stale key name in `append_docs`:
     - used: `best_non_H15_total_gain_candidates`
     - actual raw key: `max_non_H15_total_gain_candidates`
   - This is a documentation/completion schema error, not a simulation/scientific result failure.

3. Verified partial postdiagnostic evidence:
   - `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_20260928T1500Z/summary.md`
   - `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_20260928T1500Z/raw.json`
   - Frozen next protocol:
     - `research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md`
     - `research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.json`

4. Key scientific state preserved:
   - Fresh continuation-label probe remains `0/24` material-positive non-H15 identical-H15-prefix states.
   - Fresh refit/training gate remains `False`.
   - Best non-H15 total gain was only about `0.560688`, far below the frozen `+3` material threshold.
   - v0 summary reports `35` large non-H15 harms.
   - This weakens the previously mined selector result and argues against immediate refit/retraining or unchanged validation.

5. Wrote repair source:
   - `experiments/bohn2021_aws/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair.py`
   - SHA256: `cbf4f6f3b95aae89ba6d1f2542937cc2a78c4fadf15aae910bc46a248da03db4`
   - Purpose: validate partial v0 outputs, preserve failed-run evidence, append docs with corrected key, write completed marker and backup request.
   - It was not executed because the experiment runner enforces one experiment per iteration.

6. Persisted state for continuation:
   - `research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_postdiagnostic_v0_failed_and_repair_pending_20260928T1508Z.md`
   - Updated durable supervisor state via `update_state`.

Next action:
- Run `vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair.py` as the next no-simulation diagnostic.
- Then require verified external backup covering failed v0, v0b repair, repair source, and frozen transient-state protocol before any new rollout.
- After backup, execute the frozen `vehicle_v1_transient_state_continuation_probe_v0` to test whether generic branch states missed high-transient opportunity.


## 2026-09-28T15:08:38.210396+00:00
Elapsed service lifetime: about 2 days 4h17m since 2026-09-26T10:55:29.419331Z. Server API total_tokens from research.sqlite: unknown/unavailable from repository tools; not reported as zero.

Concrete actions completed:

1. Inspected the pending repair and evidence:
   - `experiments/bohn2021_aws/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair.py`
   - v0 failed postdiagnostic summary/raw/registry
   - frozen transient-state continuation protocol

2. Executed the pending no-simulation v0b schema repair:
   - Script: `experiments/bohn2021_aws/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair.py`
   - Registry: `research_artifacts/aws_runs/20260928T150458_c0c16c13/registry.json`
   - Exit status: `1`
   - Runtime: about `2.0s`
   - New rollouts/control steps/training/gradient: `0 / 0 / 0 / 0`
   - Historical validation64 opened: `False`
   - Sealed test accessed: `False`

3. Diagnosed the v0b failure:
   - v0b validated the scientific inputs and wrote:
     - `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_schema_repair_20260928T1505Z/raw.json`
   - It then failed before summary/completed marker because `write_summary(out)` referenced `out['backup_request']` before that field was populated.
   - This is another documentation/completion ordering bug only; it consumed no scientific simulation/training budget and does not change the fresh-probe finding.

4. Preserved scientific findings:
   - Fresh continuation-label probe remains `0/24` material-positive non-H15 states.
   - Fresh refit/training gate remains `False`.
   - Best non-H15 total gain remains about `0.560688`, below the frozen `+3` material threshold.
   - Large non-H15 harms remain `35`.
   - Interpretation unchanged: the earlier nearest-state selector gain is development-only and concentrated in mined cases/states; immediate refit/retraining or unchanged validation is not justified yet.

5. Wrote robust v0c repair source:
   - `experiments/bohn2021_aws/vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair.py`
   - SHA256: `de8851666f09e16c8559219f7fef6878c64845d0254bd4d1c0b3772f56571355`
   - Purpose: account for both failed no-simulation completion attempts, validate preserved artifacts, write summary/raw/completed/state/backup request, and leave the frozen transient-state protocol unchanged.

6. Persisted state:
   - `research_artifacts/aws_state/vehicle_v1_fresh_continuation_label_postdiagnostic_v0b_failed_v0c_pending_20260928T1512Z.md`
   - Updated durable supervisor state via `update_state`.

Next action:
- Run `vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair.py` as the next no-simulation diagnostic.
- Then require verified external backup covering v0/v0b/v0c artifacts, repair source, and frozen transient-state protocol before any new rollout.
- After backup, execute the frozen `vehicle_v1_transient_state_continuation_probe_v0`; if it finds fewer than 2 material positives, move to versioned stress-scenario opportunity design rather than retraining on sparse labels.

<!-- vehicle-v1-fresh-continuation-label-postdiagnostic-v0c-schema-repair-20260928T1510Z -->
## 2026-09-28 vehicle V1 fresh continuation-label postdiagnostic v0c schema repair

UTC: 2026-09-28T15:09:21.816207+00:00. No simulations, no training/refit, no historical validation64, no sealed test. The preceding v0 and v0b postdiagnostic repair attempts failed only in documentation/completion code after validating/writing partial artifacts. Fresh continuation labels remain 0/24 material positives; immediate selector refit/retraining remains deferred. Next frozen diagnostic: `research_artifacts/aws_protocols/vehicle_v1_transient_state_continuation_probe_v0_frozen_20260928.md`; execute only after a verified external backup covers v0/v0b/v0c outputs and the protocol. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair_20260928T1510Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair_20260928T1510Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_fresh_continuation_label_postdiagnostic_v0c_schema_repair_20260928T1510Z/completed.json`.


## 2026-09-28T15:13:32.401639+00:00

<!-- vehicle-v1-transient-state-shadow-selection-v0-20260928T1515Z -->
## 2026-09-28 vehicle V1 transient-state shadow selection v0

UTC: 2026-09-28T15:16:08.577389+00:00. No-simulation target-selection diagnostic selected 12 H15-only transient/control branch states for the frozen continuation probe; future rollout budget 48 episodes / 7200 steps. No validation64/test/training access. Backup remains required before simulations. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0_20260928T1515Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0_20260928T1515Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0_20260928T1515Z/completed.json`.


## 2026-09-28T15:20:33.308679+00:00

<!-- vehicle-v1-transient-state-shadow-selection-v0b-unique-repair-20260928T1520Z -->
## 2026-09-28 vehicle V1 transient-state shadow selection v0b unique repair

UTC: 2026-09-28T15:21:49.063213+00:00. No-simulation repair fixed the v0 duplicated transient target schedule by collapsing to one H15 reference trace per case before target selection. Selected 12 unique case/kind targets; future rollout budget 48 episodes / 7200 steps. No validation64/test/training access. Verified backup is required before any continuation rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/completed.json`.


## 2026-09-28T15:24:13.516184+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): about **2 days 4h28m**.  
Cumulative server API `total_tokens` from `research.sqlite` calls, excluding this desktop conversation: **unavailable through repository tools this cycle; not reported as zero**.

Concrete action completed:

- Ran the pending no-simulation repair:
  - Script: `experiments/bohn2021_aws/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair.py`
  - Registry: `research_artifacts/aws_runs/20260928T152148_c0e1440e/registry.json`
  - Exit status: `0`
  - Runtime: `2.0056s`
  - New rollouts/control steps/training/gradient: `0 / 0 / 0 / 0`
  - Historical validation64 opened: `False`
  - Sealed test accessed: `False`

Evidence inspected/preserved:

- v0 target selection had a no-simulation duplicate-schedule defect: 12 selected rows but only 6 unique case/kind/step targets.
- v0b repaired this by collapsing to one H15 reference trace per case before target selection.
- v0b selected **12 unique case/kind targets**:
  - High-transient: cases `0,2,3,6,7,9,10,11`
  - Low/control: cases `1,4,5,8`
  - Branch horizons remain `[10, 15, 30, 35]`
  - Future rollout budget remains `48 episodes / 7200 control steps`
- This is still only target-selection evidence, not adaptive opportunity evidence.
- Fresh-label negative result remains unchanged: `0/24` material-positive non-H15 identical-H15-prefix states; immediate selector refit/retraining remains premature.

Key artifacts:

- `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z/completed.json`
- `research_artifacts/aws_state/vehicle_v1_transient_state_shadow_selection_v0b_unique_repair_20260928T1520Z.md`
- Backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_SHADOW_SELECTION_V0B_UNIQUE_REPAIR_20260928T1520Z.json`

Current scientific status:

- The mined nearest-state selector gain remains provisional and concentrated in cases 7/10.
- The leading hypothesis remains sparse/case-specific canonical V1 adaptive opportunity, possibly concentrated in high-transient or near-obstacle states.
- The next discriminating experiment is the frozen transient-state continuation rollout, but it is **backup-gated**.
- No validation64 or sealed test access occurred; no reproduction or success claim is allowed.

State preserved via `update_state`.

Next iteration queue:

1. Verify external backup after `2026-09-28T15:21:49Z` covering v0/v0b outputs, source, protocol, registries/docs/state.
2. After backup, write/freeze rollout runner consuming the v0b unique target list without changing selection.
3. Smoke runner with no simulation; request/verify backup for runner source.
4. Execute frozen transient-state continuation rollout under legacy interpreter:
   - max `48 episodes / 7200 steps`
   - prefix `H15`
   - branch horizons `[10,15,30,35]`
   - no training, no validation64, no sealed test.
5. If `<2` material positives and no solver/safety artifact explains absence, move to versioned stress-scenario opportunity design before retraining.

<!-- vehicle-v1-transient-state-backup-gate-recheck-v0-20260928T1530Z -->
## 2026-09-28 vehicle V1 transient-state backup gate recheck v0

UTC: 2026-09-28T15:31:09+00:00. Metadata-only pre-rollout backup gate recheck completed with no simulations/training, no historical validation64 access, and no sealed-test access. Adequate verified external backup proofs after `2026-09-28T15:21:49.063213+00:00` found: `0`; post-required candidates: `0`. The supervisor-context backup at `2026-09-28T15:21:12.658722+00:00` predates v0b completion and is inadequate for this gate. Rollout-runner source work and continuation rollout remain blocked pending external backup covering v0/v0b artifacts/source, frozen transient protocol, this recheck/source/outputs, docs/registry/state and run logs. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1_TRANSIENT_STATE_GATE_RECHECK_V0_20260928T1530Z.json`.


## 2026-09-28T15:32:54.539325+00:00



## 2026-09-28T15:56:19.601407+00:00

<!-- vehicle-v1-transient-state-continuation-probe-v0-runner-20260928 -->
## 2026-09-28 vehicle V1 transient-state continuation probe v0

UTC: 2026-09-28T16:13:21.894935+00:00. Development-only H15-prefix transient/control continuation probe completed: 48 episodes, 3900 control steps. No validation64/test/training access. Positive states=0/12; positive cases outside mined case7/case10=[]; gate_pass=False; clean_scenario_scarcity_fail=False; next=repair/diagnose rollout artifacts before interpreting opportunity scarcity. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_continuation_probe_v0_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_continuation_probe_v0_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_continuation_probe_v0_20260928/completed.json`.


## 2026-09-28T16:15:51.831061+00:00

<!-- vehicle-v1-transient-state-prefix-artifact-postdiagnostic-v0-20260928T1620Z -->
## 2026-09-28 vehicle V1 transient prefix-artifact postdiagnostic

UTC: 2026-09-28T16:17:02.727726+00:00. No-simulation recomputation repaired the transient probe artifact: original prefix mismatches 36/36 were due to retained `decision.branch_horizon` metadata; repaired mismatches 0/36. Positive states remain 0/12; best non-H15 total gain 0.4132901295410196; clean scenario-scarcity gate fail False. Next after backup: freeze a versioned stress-scenario opportunity protocol before retraining. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_prefix_artifact_postdiagnostic_v0_20260928T1620Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1_transient_state_prefix_artifact_postdiagnostic_v0_20260928T1620Z/raw.json`.


## 2026-09-28T16:21:12.257506+00:00

<!-- vehicle-stress-scenario-opportunity-protocol-v0-20260928T1625Z -->
## 2026-09-28 vehicle stress-scenario opportunity protocol v0

UTC: 2026-09-28T16:21:58.495258+00:00. No-simulation protocol freeze completed after canonical fresh/transient continuation produced no material non-H15 label density. Frozen a source-supported natural-stress Vehicle V1 diagnostic: 128 candidate resets, 12 selected cases, Stage1 fixed-H grid H5..H50 with 120 episodes/18,000-step cap, and conditional identical-state continuation before any selector/refit. No validation64 or sealed-test access; no training. Next requires external backup before runner source/smoke or simulation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v0_20260928T1625Z/summary.md`, `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.md`, `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json`.


## 2026-09-28T16:25:15.901620+00:00
Elapsed service lifetime: ~53h27m since 2026-09-26T10:55:29.419331Z. Cumulative server API `total_tokens` from `research.sqlite`: unavailable to repository tools this cycle (`research.sqlite` was not exposed/found in the repo); desktop conversation usage is excluded.

Concrete actions completed:

1. Executed the no-simulation Vehicle V1 stress-scenario protocol freezer:
   - Script: `experiments/bohn2021_aws/vehicle_stress_scenario_opportunity_protocol_v0.py`
   - Run record: `research_artifacts/aws_runs/20260928T162158_f90130c6/registry.json`
   - Exit status: `0`
   - Runtime: ~2.0 s
   - No simulations, no training, no validation64 access, no sealed-test access.

2. Frozen new development-only protocol:
   - Markdown: `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.md`
   - JSON: `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v0_frozen_20260928.json`
   - Completed marker: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v0_20260928T1625Z/completed.json`
   - Protocol JSON SHA256: `6e66f314dc1e3070c934d5b56c5994273ad3f3c0d17dddb3b133f9124d7ab1c6`

3. Preserved decision rationale:
   - Canonical fresh continuation: `0/24` material-positive non-H15 states.
   - Repaired transient continuation: `0/12` material-positive states; prefix mismatch artifact repaired to `0/36` mismatches.
   - Therefore immediate retraining/refit on canonical sparse labels remains unjustified.
   - Next informative action is source-supported stress-scenario opportunity mapping.

4. Frozen Stage 1 stress opportunity design:
   - Candidate resets: `128`
   - Selected cases: `12` by pre-outcome metadata selection
   - Horizons: `[5,10,15,20,25,30,35,40,45,50]`
   - Budget: `120` episodes, upper bound `18,000` control steps
   - Zero training / zero gradient updates
   - Development-only; not model selection; not final test.

5. Backup gate status:
   - New backup request written:
     `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_OPPORTUNITY_PROTOCOL_V0_20260928T1625Z.json`
   - Checked for post-16:21 backup proof matching `backup_proof_20260928T16*.json`; none present.
   - Therefore no runner source or rollout was started.

State preserved via `update_state`.  

Next iteration should first verify a new external backup proof postdating `2026-09-28T16:21:58Z`. If present, write/import-smoke the Stage 1 stress fixed-H runner with zero simulation, request/verify backup for that source, then run the 120-episode legacy-interpreter stress map.


## 2026-09-28T16:30:46.533870+00:00

<!-- vehicle-stress-scenario-opportunity-stage1-runner-dryrun-20260928T1645Z -->
## 2026-09-28 vehicle stress-scenario Stage1 runner dry-run

UTC: 2026-09-28T16:42:12.906760+00:00. Wrote/froze and dry-ran `experiments/bohn2021_aws/vehicle_stress_scenario_opportunity_probe_v0_runner.py` with no simulations, no candidate resets, no training, no validation64-bank access and no sealed-test access. The dry-run verified the frozen stress protocol, post-protocol backup proof, legacy runtime import and H5..H50 terminal metadata. Stage1 rollout remains blocked until external backup covers the runner source, dry-run outputs, docs/state and backup request. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_runner_dryrun_20260928T1645Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_runner_dryrun_20260928T1645Z/completed.json`.


## 2026-09-28T16:43:10.417411+00:00

<!-- vehicle-stress-scenario-opportunity-stage1-runner-20260928 -->
## 2026-09-28 vehicle stress-scenario opportunity Stage1

UTC: 2026-09-28T17:23:53.013870+00:00. Development-only stress fixed-H map completed: 120 episodes, 9930 control steps, candidate resets=128. No training, no validation64-bank access and no sealed-test access. Stage2 candidate trigger=True; total oracle gain=25.1128; physical oracle gain=26.6419. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/completed.json`.


## 2026-09-28T17:26:52.098049+00:00

<!-- vehicle-stress-scenario-stage1-postdiagnostic-20260928T1730Z -->
## 2026-09-28 vehicle stress Stage1 postdiagnostic

UTC: 2026-09-28T17:28:00.487210+00:00. No new simulations/training and no validation64/test access. Postdiagnostic of the 120-episode stress fixed-H map found branch horizons [10, 15, 25, 30, 35, 45] strict-safe, material non-H15 branch-positive cases [5] (stress positives [5]), negative/nonmaterial cases 11, and aggregate Stage1 oracle gains total=25.112830251670744 physical=26.64187681523896. Decision: freeze Stage2 identical-state continuation after verified backup; do not retrain/refit yet because Stage1 is episode-level. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/completed.json`.


## 2026-09-28T17:29:07.617640+00:00
Elapsed service lifetime: ~54h32m41s since 2026-09-26T10:55:29.419331Z, measured to latest recorded run at 2026-09-28T17:28:10Z. Cumulative server API total_tokens: unavailable/not recomputed because `research.sqlite` was not present in the repository-accessible path; desktop conversation usage excluded.

Concrete actions completed:

1. Ran the Stage1 postdiagnostic, no simulation/training/test access:
   - Script: `experiments/bohn2021_aws/vehicle_stress_scenario_stage1_postdiagnostic_v0.py`
   - Run registry: `research_artifacts/aws_runs/20260928T172800_15070a65/registry.json`
   - Exit status: 0
   - New rollouts/control steps/training/test: all 0
   - Validation64 and sealed final test remained closed.

2. Inspected and preserved Stage1 stress-scenario evidence:
   - Output summary: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/summary.md`
   - Output raw: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/raw.json`
   - Completed marker: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_postdiagnostic_20260928T1730Z/completed.json`

3. Main evidence from postdiagnostic:
   - Branch horizons for Stage2 `[10,15,25,30,35,45]` were strict-safe across Stage1:
     - 72/72 branch rows successful
     - 0 constraints
     - 0 initial failed steps
     - 0 solver failed steps
   - H5 was excluded as unsafe/ineffective:
     - 10/12 H5 rows failed or were constrained.
   - Stage1 aggregate oracle gains remain material:
     - total gain: `25.112830251670744`
     - physical gain: `26.64187681523896`
   - But case-level material opportunity is narrow:
     - exactly one material non-H15 branch-positive case: case 5, a stress case
     - best total and physical branch horizon for that case: H30
     - 11/12 cases were nonmaterial.
   - Aggregate cost/time frontier still spans multiple horizons `[10,15,25,30,35]`, so Stage2 matched-state continuation remains scientifically justified, but broad retraining/refit is still premature.

4. Decision preserved:
   - Stage2 identical-state continuation should be frozen after backup.
   - Rationale: Stage1 shows safe episode-level opportunity, but only matched continuation can determine whether there is useful within-episode adaptive-horizon opportunity.
   - Do not retrain/refit yet; wait for Stage2 labels.

5. Backup gate:
   - New backup request created:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_POSTDIAGNOSTIC_20260928T1730Z.json`
   - I checked for a post-17:28 backup proof and none was present in the repository listing.
   - Therefore I did not run Stage2 or any additional simulation.

6. State preserved via `update_state`.
   - Current phase: `vehicle_stress_scenario_stage1_postdiagnostic_complete_backup_blocked_before_stage2`
   - Next iteration should first verify an external backup proof covering the postdiagnostic artifacts and run registry. Only then freeze Stage2 protocol/runner and proceed.


## 2026-09-28T17:32:33.625564+00:00
Elapsed service lifetime: ~54h33m+ since 2026-09-26T10:55:29Z. Cumulative server API total_tokens: unavailable/not recomputed because `research.sqlite` is not present in the repository-accessible path; desktop conversation usage excluded.

Concrete actions completed this iteration:

1. Checked the backup gate before doing any further scientific work.
   - Searched `research_artifacts/aws_backup_proofs` for a post-17:28 verified backup proof.
   - No such backup proof was present.
   - Latest visible verified proof remains pre-postdiagnostic: `backup_proof_20260928T164347_from_supervisor_context_after_stage1_runner_dryrun.json`.

2. Reaffirmed the backup requirement by writing:
   - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE1_POSTDIAGNOSTIC_GATE_RECHECK_20260928T1731Z.json`
   - SHA256: `bcf04cf072d5396c571bc37182686360d90af3f3e4a4a8b41490176effa5814f`

3. Preserved state with `update_state`.
   - Current phase: `vehicle_stress_scenario_stage1_postdiagnostic_complete_backup_blocked_before_stage2_reaffirmed`

4. Did not run any simulations or training.
   - New rollouts: 0
   - New control steps: 0
   - New training episodes: 0
   - New gradient steps: 0
   - historical_validation64_bank_opened: false
   - sealed_test_accessed: false

Scientific status remains unchanged:

- Stage1 stress fixed-H map/postdiagnostic is complete but not yet externally backed up after the postdiagnostic.
- Stage1 showed:
  - branch horizons `[10,15,25,30,35,45]` were strict-safe,
  - H5 was unsafe/ineffective,
  - aggregate oracle gains were material,
  - only one material non-H15 branch-positive case, case 5.
- Therefore Stage2 matched identical-state continuation is still the next discriminating experiment, but it is blocked until a verified backup proof appears.

Next action:

- On the next cycle, first check for a verified backup proof postdating the Stage1 postdiagnostic/gate-recheck request.
- If present, verify proof status and coverage, then freeze the Stage2 identical-state continuation protocol/runner.
- If absent, continue to preserve the blocker state and avoid new simulations.


## 2026-09-28T17:40:52.600561+00:00



## 2026-09-28T17:45:26.106153+00:00

<!-- vehicle-stress-scenario-stage2-continuation-v0b-prepare-20260928T1748Z -->
## 2026-09-28 vehicle stress-scenario Stage2 continuation prepare v0

UTC: 2026-09-28T17:47:05.449308+00:00. No-simulation Stage2 matched-continuation target/protocol freeze completed after a verified Stage1-postdiagnostic backup proof. Targets=12, planned branch horizons=[10, 15, 25, 30, 35, 45], planned episodes=72, control-step cap=10800. No validation64-bank or sealed-test access and no training/refit. Rollout is blocked until external backup covers `experiments/bohn2021_aws/vehicle_stress_scenario_stage2_continuation_v0_runner.py`, `research_artifacts/aws_protocols/vehicle_stress_scenario_stage2_continuation_v0b_frozen_20260928T1748Z.json`, `research_artifacts/aws_protocols/vehicle_stress_scenario_stage2_continuation_v0b_frozen_20260928T1748Z.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_prepare_20260928T1748Z` and `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_SCENARIO_STAGE2_CONTINUATION_V0B_ROLLOUT_20260928T1748Z.json`.


## 2026-09-28T17:47:56.095831+00:00

<!-- vehicle-stress-scenario-stage2-continuation-v0b-run-20260928T1748Z -->
## 2026-09-28 vehicle stress-scenario Stage2 continuation v0

UTC: 2026-09-28T18:13:19.087643+00:00. Development-only matched-continuation rollout completed: 72 episodes, 5892 control steps. Positive states=0, negative/neutral=12, label gate=False, blocking artifacts=60. No training, no validation64-bank access, no sealed-test access. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_continuation_v0b_20260928T1748Z/completed.json`.


## 2026-09-28T18:15:33.917833+00:00

<!-- vehicle-stress-scenario-stage2-postdiagnostic-v0-20260928T1815Z -->
## 2026-09-28 vehicle stress-scenario Stage2 postdiagnostic v0

UTC: 2026-09-28T18:17:00.039473+00:00. Analysis-only Stage2 postdiagnostic completed after the 72-episode matched-continuation rollout. Frozen Stage2 gate remains failed: positive states=0. Diagnostic semantic prefix comparison indicates 60/60 non-H15 prefixes match on state/control/cost semantics despite frozen hash mismatches. Under a diagnostic state-matched relaxation (not changing the frozen gate), positive states=1 and large total harms=11/60. Decision: do not retrain/refit now; the label signal is too sparse. Next high-information action is an analysis-only terminal/reward/modeling audit or a separately versioned stronger source-supported scenario design after backup. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_stage2_postdiagnostic_v0_20260928T1815Z/raw.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_STAGE2_POSTDIAGNOSTIC_V0_20260928T1815Z.json`.


## 2026-09-28T18:20:13.811969+00:00

<!-- vehicle-stress-reward-timing-terminal-audit-v0-20260928T1820Z -->
## 2026-09-28 vehicle stress reward/timing/terminal audit v0

UTC: 2026-09-28T18:20:44.138685+00:00. Analysis-only, no new rollouts/training and no validation64/test access. The audit separated physical cost, synthetic horizon penalty and measured decision time. Synthetic h_penalty is effectively 0.001 per H-step, so total_cost is not measured runtime. The only Stage2 material positive is physical-driven, but labels remain too sparse for selector/refit training; large non-H15 harms persist. Decision: do not retrain/refit now; after backup run a bounded source/config/checkpoint terminal-value and objective audit before any new simulation or IMPROVED refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_reward_timing_terminal_audit_v0_20260928T1820Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_reward_timing_terminal_audit_v0_20260928T1820Z/raw.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_REWARD_TIMING_TERMINAL_AUDIT_V0_20260928T1820Z.json`.


## 2026-09-28T18:23:54.552095+00:00



## 2026-09-28T18:28:37.380924+00:00

<!-- vehicle-terminal-objective-source-audit-v0-20260928T1830Z -->
## 2026-09-28 vehicle terminal/objective source audit v0

UTC: 2026-09-28T18:29:34.156351+00:00. Analysis-only; no rollouts/training/refit and no validation64/test access. The audit verified that vehicle reward/search uses a synthetic horizon penalty separate from measured wall time, stress fixed-H mapping has H-specific terminal sources without detected hash/config mismatches, and current gated policies are short-only finite search (`gradient_updates=0`) using H25 terminal lineage. The current policy class cannot choose longer-H positives such as the parsed Stage2 H30 direction, so immediate refit of the current class is not justified. Terminal-value accuracy remains missing because traces did not record numeric value errors; source supports instrumentation via `mpc_value_fn`. Decision: do not retrain/refit now; after backup, run a bounded terminal-value instrumentation smoke or freeze a stronger source-supported stress-v1 scenario protocol before any broader retraining. Artifacts: `research_artifacts/aws_diagnostics/vehicle_terminal_objective_source_audit_v0_20260928T1830Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_terminal_objective_source_audit_v0_20260928T1830Z/raw.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TERMINAL_OBJECTIVE_SOURCE_AUDIT_V0_20260928T1830Z.json`.


## 2026-09-28T18:32:12.261165+00:00

<!-- vehicle-stage2-positive-branch-schema-diagnostic-v0-20260928T1840Z -->
## 2026-09-28 Stage2 positive-branch schema diagnostic v0

UTC: 2026-09-28T18:33:51.465676+00:00. Analysis-only; no rollouts, no training/refit, no validation64/test access. Findings: parsed Stage2 positive horizon union `[1, 3, 10, 25, 30]`; case5 step18 matching raw dictionaries `25`; direct saved vector fields `0`. Frozen Stage2 gate remains failed and retrain/refit remains false. Next after verified backup: After verified backup, freeze a tiny deterministic replay-to-branch terminal-value smoke for case5 step18 plus neutral/harm controls; raw artifacts do not expose enough direct state-vector fields for state injection. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stage2_positive_branch_schema_diagnostic_v0_20260928T1840Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stage2_positive_branch_schema_diagnostic_v0_20260928T1840Z/raw.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STAGE2_POSITIVE_BRANCH_SCHEMA_DIAGNOSTIC_V0_20260928T1840Z.json`.


## 2026-09-28T18:36:06.590129+00:00



## 2026-09-28T18:38:47.904198+00:00



## 2026-09-28T18:42:27.554304+00:00



## 2026-09-28T18:46:34.077374+00:00

<!-- vehicle-stage2-prefix-dynamics-hash-diagnostic-v0-20260928T1845Z -->
## 2026-09-28 vehicle Stage2 prefix dynamics-hash diagnostic v0

UTC: 2026-09-28T18:47:55.310618+00:00. Analysis-only over existing Stage2 v0b traces. Original saved-prefix mismatches among non-H15 pairs: 60/60; dynamics-only mismatches: 0/60; metadata-artifact confirmed=True. Corrected material-positive states=1 with horizons {'10': 1, '25': 1, '30': 1}; corrected refit gate=False. No new rollouts/control steps/training; no validation64 or sealed-test access. Next: after backup, run tiny terminal/objective instrumentation smoke on case5 step18 plus neutral/harm controls, or pivot if instrumentation is blocked.


## 2026-09-28T18:53:12.968013+00:00



## 2026-09-28T18:58:03.862326+00:00



## 2026-09-28T19:03:56.236875+00:00

<!-- vehicle-stage2-terminal-objective-smoke-v0-20260928T1900Z -->
## 2026-09-28 vehicle Stage2 terminal/objective instrumentation smoke v0

UTC: 2026-09-28T19:14:57.185496+00:00. Development-only terminal/objective smoke completed: 25 episodes, 2339 control steps. No validation64/test access and no training/refit. Per-H positive counts by terminal mode: {'h15_terminal': 2, 'h25_terminal': 2, 'per_h': 3, 'zero_terminal': 3}; terminal-mode material-label flips: 5. Case5-step18 H30 per-H branch value/objective recorded in `research_artifacts/aws_diagnostics/vehicle_stage2_terminal_objective_smoke_v0_20260928T1900Z/summary.md` and raw artifacts. Decision remains no retraining/refit from this smoke alone; use results to choose terminal refit vs stress-v1 scenario opportunity mapping. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STAGE2_TERMINAL_OBJECTIVE_SMOKE_V0_20260928T191457.185496+0000.json`.


## 2026-09-28T19:18:52.462389+00:00



## 2026-09-28T19:21:04.011370+00:00

<!-- vehicle-stage2-terminal-mode-broad-scan-v0-20260928T1915Z -->
## 2026-09-28 vehicle Stage2 terminal-mode broad scan v0

UTC: 2026-09-28T20:13:04.601788+00:00. Development-only broad terminal-mode scan completed: 156 episodes, 12825 control steps. No validation64/test access and no training/refit. Material positives rows/states/cases=10/1/1; terminal label flips rows/states/cases=3/1/1. Predeclared gate=False; next=do_not_refit_selector; freeze_source_supported_stress_v1_opportunity_protocol. Summary: `research_artifacts/aws_diagnostics/vehicle_stage2_terminal_mode_broad_scan_v0_20260928T1915Z/summary.md`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STAGE2_TERMINAL_MODE_BROAD_SCAN_V0_20260928T201304.601788+0000.json`.


## 2026-09-28T20:15:51.013431+00:00



## 2026-09-28T20:23:06.248021+00:00

<!-- vehicle-stress-scenario-opportunity-protocol-v1-20260928T2025Z -->
## 2026-09-28 vehicle stress-scenario opportunity protocol v1

UTC: 2026-09-28T20:24:37.134009+00:00. Metadata-only diagnostic/protocol freeze completed after broad terminal-mode scan showed positives/flips local to case5 step18. No rollouts, control steps, training, validation64-bank access or sealed-test access. Frozen stress-v1 uses 256 metadata-only candidate resets, 20 selected source-supported cases, full fixed-H grid H5..H50 (200 episodes / 30,000-step cap) and a separate matched-continuation gate before any selector/refit. Immediate decision: no training/refit on current sparse labels; require backup before runner source/smoke or simulation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_20260928T2025Z/summary.md`, `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.md`, `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.json`.


## 2026-09-28T20:25:20.089072+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): ~57h29m.  
Cumulative server API `total_tokens`: not refreshed/available in this desktop/tool context; I did not infer or fabricate it.

Concrete actions completed:

1. Persisted the supervisor-provided verified backup proof locally:
   - `research_artifacts/aws_backup_proofs/backup_proof_20260928T202344_from_supervisor_context_after_broad_scan_and_v1_source.json`
   - SHA256: `4d9401375c1b05a927b7445edafbfe545fca79a6b8dbdcc6a43820284a90afd3`

2. Executed the zero-rollout stress-v1 protocol-freeze diagnostic:
   - Script: `experiments/bohn2021_aws/vehicle_stress_scenario_opportunity_protocol_v1.py`
   - Registry: `research_artifacts/aws_runs/20260928T202436_5f20690d/registry.json`
   - Exit status: `0`
   - Runtime: ~2.0s
   - New rollouts/control steps/training/gradient steps: `0 / 0 / 0 / 0`
   - Historical validation64 opened: `false`
   - Sealed test accessed: `false`

3. Frozen stress-v1 protocol artifacts:
   - Protocol JSON: `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.json`
   - Protocol MD: `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.md`
   - Summary: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_20260928T2025Z/summary.md`
   - Completed marker: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_20260928T2025Z/completed.json`
   - State note: `research_artifacts/aws_state/vehicle_stress_scenario_opportunity_protocol_v1_20260928T2025Z.md`

Scientific decision preserved:

- Broad terminal-mode scan showed positives/flips only in case5 step18, so immediate selector refit/training is not warranted; it would overfit one mined development state.
- Stress-v1 is now frozen as a source-supported scenario-opportunity diagnostic:
  - 256 metadata-only candidate resets.
  - 20 selected cases fixed before horizon scoring.
  - Stage1 fixed-H grid: H5, H10, H15, H20, H25, H30, H35, H40, H45, H50.
  - Stage1 budget: exactly 200 episodes, cap 30,000 control steps.
  - No selector/refit/training unless Stage1 and separately frozen Stage2 matched-continuation gates produce multicase robust labels.

Current blocker / next queued action:

- Backup gate is active. New protocol artifacts require verified external backup before writing runner source, running a dry-run, or starting Stage1 simulations.
- Backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_SCENARIO_OPPORTUNITY_PROTOCOL_V1_20260928T2025Z.json`

State has been durably updated via `update_state`. Research is continuing; no sealed test was accessed.


## 2026-09-28T20:28:17.872643+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): ~57h34m31s.  
Cumulative server API `total_tokens`: not refreshed/available in this desktop/tool context; I did not infer or fabricate it.

Concrete actions completed:

1. Inspected the frozen stress-v1 protocol and supporting evidence:
   - `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.json`
   - `research_artifacts/aws_protocols/vehicle_stress_scenario_opportunity_probe_v1_frozen_20260928T2025Z.md`
   - `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_20260928T2025Z/completed.json`
   - `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_protocol_v1_20260928T2025Z/summary.md`
   - Stress-v0 runner source for planned v1 derivation.

2. Rechecked the backup gate:
   - Available local proof: `research_artifacts/aws_backup_proofs/backup_proof_20260928T202344_from_supervisor_context_after_broad_scan_and_v1_source.json`
   - Proof time: `2026-09-28T20:23:44.869486+00:00`
   - Stress-v1 protocol freeze time: `2026-09-28T20:24:37.134009+00:00`
   - Decision: the proof predates the frozen protocol, so it does **not** satisfy the protocol’s explicit requirement for verified backup before runner source, dry-run, or rollout.

3. Preserved state:
   - Wrote: `research_artifacts/aws_state/vehicle_stress_v1_backup_gate_recheck_20260928T2030Z.md`
   - SHA256: `8df648b78f84bb8e30140d9357169b114ca0f7ad7cdecf7f4969cfafaf86c96b`

4. Requested backup:
   - Wrote: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1_BACKUP_GATE_RECHECK_20260928T2030Z.json`
   - SHA256: `0a51e68fc3e0904138edeb9e6cbda940b8853d53a2caabf6d65d9d4da8727341`

Budget/access this iteration:

- New experiments: `0`
- New rollouts/control steps: `0 / 0`
- New training/gradient steps: `0 / 0`
- Historical validation64 bank opened: `false`
- Sealed final test accessed: `false`

Scientific state is unchanged:

- No reproduction success or stable adaptive acceleration claim.
- Leading hypothesis remains scenario/opportunity design bottleneck.
- Stress-v0 and terminal broad scan evidence remains too sparse/local for selector refit or training.
- Stress-v1 Stage1 remains the next informative experiment, but only after backup.

Current next queued action after verified backup appears:

1. Write/import-smoke the stress-v1 Stage1 runner derived from the audited v0 runner.
2. Run zero-rollout dry-run under legacy interpreter.
3. Request/verify backup of runner source and dry-run outputs.
4. Then run the frozen 200-episode fixed-H H5..H50 Stage1 opportunity map if dry-run passes.


## 2026-09-28T20:31:14.105901+00:00

<!-- vehicle-stress-scenario-opportunity-stage1-v1-runner-dryrun-20260928T2045Z -->
## 2026-09-28 vehicle stress-scenario Stage1 v1 runner dry-run

UTC: 2026-09-28T20:36:29.506408+00:00. Wrote and dry-ran `experiments/bohn2021_aws/vehicle_stress_scenario_opportunity_probe_v1_runner.py` with no simulations, no candidate resets, no training, no validation64-bank access and no sealed-test access. The dry-run verified the frozen stress-v1 protocol, post-protocol backup proof, legacy runtime import and H5..H50 terminal metadata inherited from the verified fixed-H grid. Stage1 rollout remains blocked until external backup covers the runner source, dry-run outputs, docs/state and backup request. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_runner_dryrun_20260928T2045Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_runner_dryrun_20260928T2045Z/completed.json`.


## 2026-09-28T20:38:42.209107+00:00

<!-- vehicle-stress-scenario-opportunity-stage1-v1-runner-20260928T2045Z -->
## 2026-09-28 vehicle stress-scenario opportunity Stage1 v1

UTC: 2026-09-28T21:46:28.826778+00:00. Development-only stress-v1 fixed-H map completed: 200 episodes, 16912 control steps, candidate resets=256. No training, no validation64-bank access and no sealed-test access. Stage2 candidate trigger=False; total oracle gain=176.655; physical oracle gain=179.657; material cases=[1, 4, 5, 6]. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/completed.json`.


## 2026-09-28T21:49:07.358304+00:00

<!-- vehicle-stress-v1-stage1-postdiagnostic-20260928T2155Z -->
## 2026-09-28 vehicle stress-v1 Stage1 postdiagnostic

UTC: 2026-09-28T21:50:13.083133+00:00. Analysis-only postdiagnostic over stress-v1 Stage1 completed with no simulations/control steps/training, no validation64-bank access and no sealed-test access. The predeclared Stage1 gate remains failed (aggregate=True, case-diversity=False, Stage2 trigger=False); material cases are [1, 4, 5, 6] and remain concentrated in the high_heading_long_or_medium stratum. No selector/refit/training is warranted yet. Next after verified backup: freeze and dry-run a v1b matched-continuation postdiagnostic to test whether the high-heading episode-level signal corresponds to real within-episode adaptive opportunity. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/completed.json`.


## 2026-09-28T21:51:11.355750+00:00
Elapsed service lifetime: ~58h55m since first supervisor event (2026-09-26T10:55:29.419331Z), using latest artifact time 2026-09-28T21:50:13Z. Cumulative server API `total_tokens`: not refreshed/available in this desktop-tool context; this excludes the current desktop conversation usage.

Concrete action completed:
- Ran analysis-only postdiagnostic:
  - Script: `experiments/bohn2021_aws/vehicle_stress_v1_stage1_postdiagnostic.py`
  - Registry: `research_artifacts/aws_runs/20260928T215012_a9311541/registry.json`
  - Runtime: ~2.0s
  - New simulations/control steps/training: 0
  - Validation64 opened: false
  - Sealed test accessed: false

Key preserved outputs:
- `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_postdiagnostic_20260928T2155Z/completed.json`
- `research_artifacts/aws_state/vehicle_stress_v1_stage1_postdiagnostic_and_next_20260928T2155Z.md`
- Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1_STAGE1_POSTDIAGNOSTIC_20260928T2155Z.json`

Scientific outcome:
- The predeclared stress-v1 Stage1 gate remains failed and was not retrospectively changed:
  - aggregate guard: true
  - case-diversity gate: false
  - Stage2 continuation trigger: false
- Material H15-counterfactual cases are `[1, 4, 5, 6]`, all in `high_heading_long_or_medium`.
- Stress-v1 shows stronger episode-level fixed-H opportunity than earlier probes, but the opportunity is still concentrated rather than broadly distributed.
- H5 remains unusable as a positive label source: 17/20 failures.
- Strict safe aggregate horizons are `[10, 15, 20, 25]`.
- Best strict aggregate physical fixed horizon: H15.
- Fastest strict aggregate horizon: H10.
- Longer horizons can help physically in a few selected cases, but measured timing remains nonmonotone and scalar total cost includes synthetic horizon penalty, so there is no valid acceleration claim.

Current diagnosis:
- Leading bottleneck remains scenario/opportunity structure, not immediate training.
- Evidence is not yet sufficient for selector refit/training: labels are one-stratum and episode-level only.
- The next informative experiment is matched continuation from identical H15-prefix states, not another unchanged validation batch and not training on sparse labels.

Blocker before further simulation:
- A verified external backup is required before more simulations or unique development evidence. The required request file has been written.

Exact next queued action after backup:
1. Consume verified backup proof covering Stage1 rollout + this postdiagnostic.
2. Freeze a `vehicle_stress_v1b_matched_continuation_postdiagnostic` protocol.
3. Dry-run it with zero simulations.
4. If dry-run passes, run bounded matched-prefix continuation:
   - candidate positives: `[1, 4, 5, 6]`
   - same-stratum controls: `[0, 2, 3, 7]`
   - lower-stress controls: `[16, 17]`
   - approximate upper bound: 140 branch continuations / 21k control steps.
5. Decide from those matched-state labels whether scenario redesign, terminal/objective correction, or an IMPROVED longer-H selector/refit is warranted.

State has been durably preserved with `update_state`. Final test remains sealed and unauthorized.


## 2026-09-28T21:55:44.869134+00:00



## 2026-09-28T21:59:45.650867+00:00

<!-- vehicle-stress-v1b-matched-continuation-prepare-20260928T2210Z-schema-repair -->
## 2026-09-28 vehicle stress-v1b matched-continuation prepare

UTC: 2026-09-28T22:01:19.844699+00:00. Analysis-only prepare/dry-run froze a matched-continuation postdiagnostic after stress-v1 Stage1 failed its diversity gate. Targets=20 (positive cases [1, 4, 5, 6], same-stratum controls [0, 2, 3, 7], lower-stress controls [16, 17]); branch horizons=[10, 15, 25, 30, 35, 45, 50]; planned continuations=140; control-step cap=21000. No simulations, no training/refit, no validation64-bank access and no sealed-test access. Rollout is backup-blocked until external backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_MATCHED_CONTINUATION_ROLLOUT_20260928T2210Z_SCHEMA_REPAIR.json`, `research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_20260928T2210Z_schema_repair.json`, `research_artifacts/aws_protocols/vehicle_stress_v1b_matched_continuation_frozen_20260928T2210Z_schema_repair.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_prepare_20260928T2210Z_schema_repair` and this runner source.


## 2026-09-28T22:02:50.615479+00:00



## 2026-09-28T22:09:47.262474+00:00



## 2026-09-28T22:13:51.557556+00:00

<!-- vehicle-stress-v1b-matched-continuation-rollout-v0b-dryrun-20260928T2220Z-schema-repair -->
## 2026-09-28 vehicle stress-v1b matched-continuation rollout v0 dry-run

UTC: 2026-09-28T22:15:55.826627+00:00. No-simulation runner dry-run passed after the verified post-prepare backup proof. It verified v1b prepare/protocol hashes, legacy runtime import, H10/H15/H25/H30/H35/H45/H50 terminal metadata and the 140-episode/21000-step budget. No validation64-bank or sealed-test access and no training/refit. Rollout is blocked until external backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_MATCHED_CONTINUATION_ROLLOUT_V0B_20260928T2220Z_SCHEMA_REPAIR.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_dryrun_20260928T2220Z_schema_repair/completed.json`, state/docs and this runner source.


## 2026-09-28T22:16:31.974508+00:00



## 2026-09-28T22:20:09.548447+00:00

<!-- vehicle-stress-v1b-matched-continuation-rollout-v0b-run-20260928T2220Z-schema-repair -->
## 2026-09-28 vehicle stress-v1b matched-continuation rollout v0

UTC: 2026-09-28T23:05:38.893157+00:00. Development-only matched-continuation rollout completed: 140 episodes, 11779 control steps. Positive states=0 across cases=[], negative/neutral=20, label gate=False, blocking artifacts=120. No training, no validation64-bank access, no sealed-test access. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_20260928T2220Z_schema_repair/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_20260928T2220Z_schema_repair/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_matched_continuation_rollout_v0b_20260928T2220Z_schema_repair/completed.json`.


## 2026-09-28T23:09:29.525464+00:00

<!-- vehicle-stress-v1b-prefix-artifact-postdiagnostic-v0-20260928T2315Z -->
## 2026-09-28 vehicle stress-v1b prefix-artifact postdiagnostic v0

UTC: 2026-09-28T23:10:29.568387+00:00. No-simulation reanalysis of existing v1b traces found the reported `prefix_mismatch_non_H15=120` was a bookkeeping artifact: all 120/120 non-H15 comparisons match the H15 physical prefix after excluding non-causal prefix-step `decision.branch_horizon`/timing metadata. Corrected labels: positive states=4 across cases=[4, 5], negative/neutral=16, corrected label gate=False. This repairs the zero-positive headline but still does not justify selector/refit because positives are sparse and development-mined. Next after verified backup: bounded terminal/reward-source ablation on corrected-positive states plus controls. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_prefix_artifact_postdiagnostic_v0_20260928T2315Z/completed.json`.


## 2026-09-28T23:15:22.890171+00:00

<!-- vehicle-stress-v1b-terminal-reward-ablation-v0-dryrun-20260928T2320Z -->
## 2026-09-28 vehicle stress-v1b terminal/reward-source ablation v0 dry-run

UTC: 2026-09-28T23:16:27.654790+00:00. No-simulation protocol freeze completed after prefix-artifact postdiagnostic. Corrected v1b positives remain 4 states across cases=[4, 5], label gate=False. Froze terminal/reward-source ablation over all four corrected positives plus deterministic controls [8,9,16,18]; smoke plan=27 episodes, full plan=59 episodes. No validation64/test access, no training/refit. Simulation remains blocked until verified external backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0_SMOKE_20260928T2320Z.json`, `research_artifacts/aws_protocols/vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_20260928T2320Z.json`, dry-run outputs and source. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z/completed.json`.


## 2026-09-28T23:18:04.748301+00:00
Elapsed service lifetime since first supervisor event: ~60h22m31s (2026-09-26T10:55:29Z → latest artifact/state around 2026-09-28T23:18Z). Cumulative server API total_tokens: latest available supervisor value was ~89.84M tokens; not refreshed by this desktop/tool context, and this conversation’s usage is excluded.

Concrete actions completed:

1. Inspected the prior v1b prefix-artifact evidence and handoff.
   - Confirmed from `summary.md` / `completed.json` that the previous strict `prefix_mismatch_non_H15=120` was a non-causal bookkeeping/hash artifact.
   - Corrected dynamic physical prefixes match `120/120`.
   - Corrected label set remains sparse: `4` positive states, cases `[4, 5]`; corrected label/refit gate remains false.
   - No validation64-bank access and no sealed-test access.

2. Ran the prepared no-simulation terminal/reward-source ablation dry-run:
   - Script: `experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0.py`
   - Command mode: `--dry-run --i-accept-development-terminal-reward-ablation`
   - Registry: `research_artifacts/aws_runs/20260928T231626_050f5c38/registry.json`
   - Exit status: `0`
   - Runtime: ~2.0 s
   - New rollouts/control steps/training/gradient steps: all `0`
   - validation64 bank: closed
   - sealed final test: closed

3. Frozen outputs produced:
   - Dry-run summary: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z/summary.md`
   - Dry-run raw: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z/raw.json`
   - Dry-run completed marker: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0_dryrun_20260928T2320Z/completed.json`
   - Frozen protocol JSON: `research_artifacts/aws_protocols/vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_20260928T2320Z.json`
   - Frozen protocol MD: `research_artifacts/aws_protocols/vehicle_stress_v1b_terminal_reward_ablation_v0_frozen_20260928T2320Z.md`
   - Backup request before smoke: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0_SMOKE_20260928T2320Z.json`
   - Durable handoff: `research_artifacts/aws_state/continue_state_20260928T2318Z_after_terminal_reward_ablation_dryrun.md`

Scientific status:

- Research is continuing; no reproduction success or final-test claim.
- The leading hypothesis remains terminal/reward-source or objective mismatch: sparse corrected positives may depend on H-specific terminal values or synthetic cost terms.
- Scenario opportunity sparsity remains a strong competing explanation.
- Training/refit is still deferred because the corrected label gate is false and evidence is not yet dense or robust enough.
- The next simulation is blocked only by the backup policy, not by a scientific blocker.

Next concrete action:

1. Obtain/verify external backup postdating `2026-09-28T23:16:27.654790Z`, covering the dry-run outputs, source, protocol files, and backup request.
2. Then run only the bounded smoke:

```text
modern python experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0.py \
  --run-smoke \
  --input-backup-proof <verified_backup_proof_path> \
  --i-accept-development-terminal-reward-ablation
```

Planned smoke budget: `27` development-only episodes, cap `4050` control steps, `0` training episodes, `0` gradient steps, no validation64, no sealed test.

State has been preserved via `update_state` and the handoff file.


## 2026-09-28T23:24:04.574888+00:00


<!-- vehicle-stress-v1b-terminal-reward-ablation-presmoke-readiness-v0-20260928T2325Z -->
## 2026-09-28 vehicle stress-v1b terminal/reward ablation pre-smoke readiness v0

UTC: 2026-09-28T23:26:36.817792+00:00. No-simulation readiness diagnostic completed; no validation64/test access and no training/refit. It did not repeat the prefix audit. The frozen smoke covers 27 episodes: positive targets [2, 4] from cases [4, 5] plus controls [8, 16]; full protocol remains 59 episodes. A-priori risks are ['sparse_development_mined_positive_labels', 'physical_vs_total_best_horizon_disagreement', 'multi_horizon_material_labels_possible_terminal_or_path_sensitivity']. Adequate local backup proofs after `2026-09-28T23:23:38.071390+00:00`: 1; therefore the next smoke remains backup-gated unless a newer external proof is supplied. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_presmoke_readiness_v0_20260928T2325Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_presmoke_readiness_v0_20260928T2325Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_presmoke_readiness_v0_20260928T2325Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_PRESMOKE_READINESS_V0_20260928T2325Z.json`.


## 2026-09-28T23:28:10.048905+00:00



## 2026-09-28T23:32:09.478100+00:00



## 2026-09-28T23:35:36.409905+00:00

<!-- vehicle-stress-v1b-terminal-reward-ablation-v0c-backup-gate-handoff-20260928T2348Z -->
## 2026-09-28 v0c terminal/reward ablation backup-gate handoff

UTC: 2026-09-28T23:38:12.743052+00:00. No-simulation backup-gate audit after writing `experiments/bohn2021_aws/vehicle_stress_v1b_terminal_reward_ablation_v0c_legacy_schema_repair.py`. Existing latest adequate backup proof is `research_artifacts/aws_backup_proofs/backup_proof_20260928T233249_from_supervisor_context_after_v0b_legacy_retry_wrapper.json` at `2026-09-28T23:32:49.855291+00:00`, which does not postdate the v0c source gate `2026-09-28T23:35:06.156002+00:00`. Smoke remains blocked pending backup; no rollouts/training/validation/test occurred. Preserved v0 modern-TF failure and v0b schema KeyError. Next exact action after backup: run the v0c legacy schema-repair 27-episode smoke, then inspect terminal-label flips, physical-vs-total gains, zero-terminal robustness, safety/solver rows and timing before any training/refit/scenario revision. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_backup_gate_handoff_20260928T2348Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_backup_gate_handoff_20260928T2348Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_backup_gate_handoff_20260928T2348Z/completed.json`, request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0C_SMOKE_20260928T2348Z.json`.


## 2026-09-28T23:39:08.756709+00:00

<!-- vehicle-stress-v1b-terminal-reward-ablation-v0c-legacy-schema-repair-run-20260928T2340Z_legacy_schema_repair-smoke -->
## 2026-09-28 vehicle stress-v1b terminal/reward-source ablation v0 smoke

UTC: 2026-09-28T23:49:55.683264+00:00. Development-only ablation completed: 27 episodes, 2402 control steps. No validation64/test access and no training/refit. Terminal-artifact strength=strong; material-label flips=6; synthetic-only material rows=0. Decision remains no selector/refit from this diagnostic alone. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_smoke_20260928T2340Z_legacy_schema_repair/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_smoke_20260928T2340Z_legacy_schema_repair/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_smoke_20260928T2340Z_legacy_schema_repair/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0_SMOKE_20260928T234955.683264+0000.json`.


## 2026-09-28T23:53:27.836279+00:00

<!-- vehicle-stress-v1b-terminal-reward-ablation-v0c-legacy-schema-repair-run-20260928T2340Z_legacy_schema_repair-full -->
## 2026-09-28 vehicle stress-v1b terminal/reward-source ablation v0 full

UTC: 2026-09-29T00:15:30.098963+00:00. Development-only ablation completed: 59 episodes, 5185 control steps. No validation64/test access and no training/refit. Terminal-artifact strength=strong; material-label flips=14; synthetic-only material rows=0. Decision remains no selector/refit from this diagnostic alone. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_v0c_full_20260928T2340Z_legacy_schema_repair/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_V0_FULL_20260929T001530.098963+0000.json`.


## 2026-09-29T00:17:13.953474+00:00

<!-- vehicle-stress-v1b-terminal-reward-ablation-full-postdiagnostic-v0-20260929T0018Z -->
## 2026-09-29 vehicle stress-v1b terminal/reward-source ablation full postdiagnostic

UTC: 2026-09-29T00:18:17.418695+00:00. No-simulation postdiagnostic of the completed full v0c ablation. Full input budget was 59 episodes / 5185 control steps; validation64 and sealed test stayed closed; no training/refit. Terminal flips=14, synthetic-only material rows=0, robust same-H zero-terminal positives=4/4, controls with material labels=0/4. Decision: no selector/refit from current per-H labels; next after backup is an artifact-resistant matched/zero-terminal label-density probe or terminal-value-refit diagnostic. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_full_postdiagnostic_v0_20260929T0018Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_reward_ablation_full_postdiagnostic_v0_20260929T0018Z/raw.json`, state `research_artifacts/aws_state/continue_state_20260929T0018Z_after_full_terminal_reward_ablation_postdiagnostic.md`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1B_TERMINAL_REWARD_ABLATION_FULL_POSTDIAGNOSTIC_V0_20260929T0018Z.json`.


## 2026-09-29T00:22:02.859285+00:00

<!-- vehicle-stress-v1b-terminal-objective-alignment-postdiagnostic-v0-20260929T0025Z -->
## 2026-09-29 vehicle stress-v1b terminal/objective alignment postdiagnostic

UTC: 2026-09-29T00:22:25.208521+00:00. No-simulation postdiagnostic of the completed full terminal/reward-source ablation. No validation64/sealed-test access and no training/refit. Terminal flips=14, synthetic-only rows=0, robust same-H zero positives=4, control material states=0, strict min-objective alignment failures=0, min-objective selected non-success rows=0. Decision: do not train/refit now; next after backup is a fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0_20260929T0025Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0_20260929T0025Z/raw.json`, state `research_artifacts/aws_state/continue_state_20260929T0025Z_after_terminal_objective_alignment_postdiagnostic.md`.


## 2026-09-29T00:25:44.942478+00:00

<!-- vehicle-stress-v1b-terminal-objective-alignment-v0b-schema-repair-20260929T0035Z -->
## 2026-09-29 vehicle stress-v1b terminal/objective alignment v0b schema repair

UTC: 2026-09-29T00:26:51.732123+00:00. No-simulation schema-repair postdiagnostic. v0 objective-alignment counts are superseded because v0 looked for `objective` while full rows use `branch_objective_opt_f_num`; v0 missing selected objectives 28/28. Corrected coverage {'rows': 59, 'objective_field': 0, 'branch_objective_opt_f_num': 59, 'value_fn_field': 0, 'branch_mpc_value_fn': 59}. Terminal flips=14, synthetic-only rows=0, material state-mode rows=10, min-objective selected material in 8, strict min-objective failures=1, min-objective selected non-success rows=2. Decision: do not train/refit now; after backup freeze/run a fresh terminal-stable label-density probe with realized continuation labels and objective-alignment metrics. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/raw.json`, state `research_artifacts/aws_state/continue_state_20260929T0035Z_after_terminal_objective_alignment_v0b_schema_repair.md`.


## 2026-09-29T00:30:03.422258+00:00

<!-- vehicle-stress-v1c-terminal-stable-label-density-probe-v0-dryrun-20260929T0040Z -->
## 2026-09-29 vehicle stress-v1c terminal-stable label-density probe v0 dry-run

UTC: 2026-09-29T00:33:47.519199+00:00. No-simulation dry-run completed for the fresh non-mined terminal-stable label-density probe. Verified frozen protocol `research_artifacts/aws_protocols/vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z.json` and corrected objective diagnostic v0b `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/completed.json`; no candidate resets, rollouts, training/refit, validation64-bank access or sealed-test access. Smoke plan remains blocked until external backup covers the new runner, dry-run outputs, v0b artifacts and request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0_SMOKE_20260929T0040Z.json`. Planned smoke upper bound is 168 episodes/25200 control steps; full upper bound is 560 episodes/84000 control steps.


## 2026-09-29T00:36:50.973534+00:00

<!-- vehicle-stress-v1c-terminal-stable-label-density-probe-v0b-dryrun-20260929T0045Z_schema_repair -->
## 2026-09-29 vehicle stress-v1c terminal-stable label-density probe v0 dry-run

UTC: 2026-09-29T00:38:54.331272+00:00. No-simulation dry-run completed for the fresh non-mined terminal-stable label-density probe. Verified frozen protocol `research_artifacts/aws_protocols/vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z.json` and corrected objective diagnostic v0b `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/completed.json`; no candidate resets, rollouts, training/refit, validation64-bank access or sealed-test access. Smoke plan remains blocked until external backup covers the new runner, dry-run outputs, v0b artifacts and request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0B_SMOKE_20260929T0045Z_schema_repair.json`. Planned smoke upper bound is 168 episodes/25200 control steps; full upper bound is 560 episodes/84000 control steps.


## 2026-09-29T00:39:59.069845+00:00



## 2026-09-29T00:42:50.450025+00:00

<!-- vehicle-stress-v1c-terminal-stable-label-density-probe-v0c-dryrun-20260929T0055Z_hash_repair -->
## 2026-09-29 vehicle stress-v1c terminal-stable label-density probe v0 dry-run

UTC: 2026-09-29T00:45:25.842292+00:00. No-simulation dry-run completed for the fresh non-mined terminal-stable label-density probe. Verified frozen protocol `research_artifacts/aws_protocols/vehicle_stress_v1c_terminal_stable_label_density_v0_frozen_20260929T0040Z.json` and corrected objective diagnostic v0b `research_artifacts/aws_diagnostics/vehicle_stress_v1b_terminal_objective_alignment_postdiagnostic_v0b_schema_repair_20260929T0035Z/completed.json`; no candidate resets, rollouts, training/refit, validation64-bank access or sealed-test access. Smoke plan remains blocked until external backup covers the new runner, dry-run outputs, v0b artifacts and request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1C_TERMINAL_STABLE_LABEL_DENSITY_PROBE_V0C_SMOKE_20260929T0055Z_hash_repair.json`. Planned smoke upper bound is 168 episodes/25200 control steps; full upper bound is 560 episodes/84000 control steps.


## 2026-09-29T00:47:09.586747+00:00

<!-- vehicle-stress-v1c-v0c-smoke-backup-gate-recheck-20260929T0105Z -->
## 2026-09-29 vehicle stress-v1c v0c smoke backup gate recheck

UTC: 2026-09-29T00:49:36.253292+00:00. Metadata-only backup gate recheck completed with no simulations, no candidate resets, no training/refit, no validation64-bank access and no sealed-test access. v0c dry-run hash audit passed=True. Required backup time for smoke is >= `2026-09-29T00:45:25.888895+00:00`. Adequate local proof count=0; latest proof `research_artifacts/aws_backup_proofs/backup_proof_20260929T004329_from_user_context_v1c_v0c_gate_recheck_not_adequate_for_smoke.json` at `2026-09-29T00:43:29.230337+00:00` is adequate=False. The supervisor-context backup at 2026-09-29T00:43:29Z was transcribed for audit but rejected for v1c v0c smoke because it predates the dry-run/request at 00:45:25Z. Smoke remains blocked until a verified external backup appears after the required time. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1c_v0c_smoke_backup_gate_recheck_20260929T0105Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1c_v0c_smoke_backup_gate_recheck_20260929T0105Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1c_v0c_smoke_backup_gate_recheck_20260929T0105Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1C_V0C_DRYRUN_GATE_RECHECK_20260929T0105Z.json`.


## 2026-09-29T00:50:52.103668+00:00

<!-- vehicle-stress-v1c-terminal-stable-label-density-probe-v0c-smoke-20260929T0055Z_hash_repair -->
## 2026-09-29 vehicle stress-v1c terminal-stable label-density probe v0 smoke

UTC: 2026-09-29T01:51:16.916158+00:00. Development-only terminal-stable label-density smoke completed: 168 episodes, 13328 control steps, candidate resets=256. Robust-positive states=0 across cases=[]; negative/neutral states=12; smoke gate=False; full label-density gate=False; blocking artifacts=144. No validation64-bank or sealed-test access, no training/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_smoke_20260929T0055Z_hash_repair/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_smoke_20260929T0055Z_hash_repair/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_smoke_20260929T0055Z_hash_repair/completed.json`.


## 2026-09-29T01:52:31.883015+00:00



## 2026-09-29T01:57:18.562580+00:00

<!-- vehicle-stress-v1c-terminal-stable-smoke-reanalysis-v0-20260929T0205Z -->
## 2026-09-29 vehicle stress-v1c terminal-stable smoke reanalysis v0

UTC: 2026-09-29T01:57:43.455155+00:00. No-simulation development reanalysis completed. Legacy prefix-hash mismatches were diagnosed as bookkeeping over-blocking: legacy mismatches=144, repaired physical-prefix mismatches=0, max physical-prefix deviation=0.0. Repaired robust-positive states remain 0 across cases []; repaired smoke gate=False. Therefore do not train/refit from v1c labels and do not run full v1c unchanged; next action is a versioned scenario/terminal-opportunity diagnostic on fresh source-supported states. No validation64/test access, no new rollouts/control steps/training. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_20260929T0205Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_20260929T0205Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_20260929T0205Z/completed.json`.


## 2026-09-29T02:00:41.449651+00:00

<!-- vehicle-stress-v1d-trace-selected-terminal-stable-opportunity-v0-dryrun-20260929T0210Z -->
## 2026-09-29 vehicle stress-v1d trace-selected terminal-stable opportunity dry-run

UTC: 2026-09-29T02:06:02.076034+00:00. No-simulation dry-run completed for the trace-selected terminal-stable opportunity probe. Verified frozen protocol `research_artifacts/aws_protocols/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_frozen_20260929T0210Z.json`, v1c repaired smoke reanalysis `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_smoke_reanalysis_v0_20260929T0205Z/completed.json`, and v1c bank marker `research_artifacts/aws_diagnostics/vehicle_stress_v1c_terminal_stable_label_density_probe_v0c_bank_20260929T0055Z_hash_repair/completed.json`. No candidate resets, rollouts, training/refit, validation64-bank access or sealed-test access. Planned smoke upper bound is 188 episodes/28200 control steps (20 H15 scan episodes plus 168 common-prefix branch episodes). Smoke is blocked until verified external backup covers the new runner/dry-run/protocol and request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1D_TRACE_SELECTED_TERMINAL_STABLE_OPPORTUNITY_V0_SMOKE_20260929T0210Z.json`.


## 2026-09-29T02:07:45.613230+00:00

<!-- vehicle-stress-v1d-trace-selected-terminal-stable-opportunity-v0-smoke-20260929T0210Z -->
## 2026-09-29 vehicle stress-v1d trace-selected terminal-stable opportunity smoke

UTC: 2026-09-29T03:12:03.807872+00:00. Development-only trace-selected terminal-stable opportunity smoke completed: 188 episodes, 14913 control steps, candidate resets=0. Robust-positive states=0 across cases=[]; negative/neutral states=12; smoke gate=False; blocking artifacts=0. No validation64-bank or sealed-test access, no training/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1d_trace_selected_terminal_stable_opportunity_v0_smoke_20260929T0210Z/completed.json`.


## 2026-09-29T03:15:05.201981+00:00



## 2026-09-29T03:20:28.953342+00:00



## 2026-09-29T03:24:31.313387+00:00



## 2026-09-29T03:29:10.194687+00:00

<!-- vehicle-stress-v1d-solver-timing-schema-postdiagnostic-v0-20260929T0340Z -->
## 2026-09-29 vehicle stress-v1d solver/decision timing schema postdiagnostic

UTC: 2026-09-29T03:29:37.352912+00:00. No-simulation timing-schema repair over existing v1d smoke outputs. The earlier Pareto script reported solver-time gains as null because v1d rows store `solver_attempt_sum_s`; this diagnostic re-extracted solver-attempt timing while keeping whole-decision timing separate. Both decision+solver strict compute-safe states=3/12, both relaxed=6/12; decision-only strict/relaxed states=0/1. Classification `solver_confirmed_compute_opportunity_sparse_or_constantH_absorbable`; train/refit now remains false. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1d_solver_timing_schema_postdiagnostic_v0_20260929T0340Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1D_SOLVER_TIMING_SCHEMA_POSTDIAGNOSTIC_V0_20260929T0340Z.json`. Validation64 and sealed test stayed closed.


## 2026-09-29T03:34:01.537997+00:00

<!-- vehicle-v1d-h10-constant-baseline-absorption-postdiagnostic-v0-20260929T0350Z -->
## 2026-09-29 vehicle v1d H10 constant-baseline absorption postdiagnostic

UTC: 2026-09-29T03:35:06.371049+00:00. No-simulation diagnostic over existing v1d timing and validation aggregate outputs. v1d best both-relaxed compute-safe states=6/12 and all best relaxed labels were H10; best strict states=3/12 with H10 best in 2. Existing validation matched-terminal all-seed H10 was faster in mean decision time but not a control/safety-noninferior fixed-H solution: H10 success=84/192 with failures=108 vs H25 success=192/192 and physical-cost ratio=24.57793104123237. Classification `local_H10_compute_labels_do_not_make_constant_H10_a_strong_global_baseline`; train/refit remains false. Next: after backup, freeze blocked repeated-timing confirmation on H10-labelled states before any selector/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_20260929T0350Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_20260929T0350Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_v1d_h10_constant_baseline_absorption_postdiagnostic_v0_20260929T0350Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1D_H10_CONSTANT_BASELINE_ABSORPTION_POSTDIAGNOSTIC_V0_20260929T0350Z.json`. Sealed test stayed closed; validation bank was not reopened.


## 2026-09-29T03:39:27.017300+00:00

<!-- vehicle-v1d-compute-tradeoff-repeated-timing-confirmation-v0-dryrun-20260929T0410Z -->
## 2026-09-29 vehicle v1d compute-tradeoff repeated timing confirmation dry-run

UTC: 2026-09-29T03:40:02.024862+00:00. No-simulation dry-run built a frozen 186-episode repeated-timing confirmation schedule over 10 v1d development states: 6 H10-labelled compute states plus 4 no-label negative controls, 3 repeat blocks, terminal modes ['h15_common_terminal', 'zero_terminal'], horizon counts {'10': 60, '25': 60, '15': 60, '30': 6}. No validation64-bank or sealed-test access and no training/refit. Existing H10 absorption context remains negative for constant fixed-H10 as a global control/safety baseline. Actual repeated timing rollouts are blocked until external backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_V1D_COMPUTE_TRADEOFF_REPEATED_TIMING_CONFIRMATION_V0_RUN_20260929T0410Z.json`, the protocol/source/dry-run artifacts, and the recent v1d/H10 diagnostics.


## 2026-09-29T03:43:08.117831+00:00

<!-- vehicle-v1d-compute-tradeoff-single-run-gate-postdiagnostic-v0-20260929T0415Z -->
## 2026-09-29 vehicle v1d compute-tradeoff single-run gate postdiagnostic

UTC: 2026-09-29T03:44:19.713265+00:00. No-simulation diagnostic applied the frozen repeated-timing confirmation gate to existing v1d single-run branch rows for the 6 H10-labelled states plus 4 negative controls. Strict confirmed H10-labelled states=1/6, relaxed=6/6, strict negative controls=0/4, missing required arms=0. Decision: do not spend the 186-episode confirmation yet; freeze negative/absorbed compute-opportunity diagnosis or design a versioned scenario/value diagnostic. No validation64/sealed-test access and no training/refit. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1D_COMPUTE_TRADEOFF_SINGLE_RUN_GATE_POSTDIAGNOSTIC_V0_20260929T0415Z.json` before further simulation.


## 2026-09-29T03:48:38.105900+00:00

<!-- vehicle-v1d-compute-tradeoff-oracle-upper-bound-postdiagnostic-v0-20260929T0355Z -->
## 2026-09-29 vehicle v1d compute-tradeoff oracle upper-bound postdiagnostic

UTC: 2026-09-29T03:49:04.748176+00:00. No-simulation upper-bound analysis over existing v1d branch rows: strict state-level H10 oracle selects H10 in 1/10 states and saves 1.08% decision time vs fixed H25 on this branch bank; relaxed labelled oracle saves 4.76% but is epsilon/terminal sensitive. Classification: `strict_compute_selector_upper_bound_too_sparse`. Decision: do not run the 186-episode repeated timing confirmation or selector/refit now; freeze a negative current-scenario compute-opportunity diagnosis, then design a versioned source-supported scenario/value diagnostic. No validation64/sealed-test access and no training/refit. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_V1D_COMPUTE_TRADEOFF_ORACLE_UPPER_BOUND_POSTDIAGNOSTIC_V0_20260929T0355Z.json` before further simulation.


## 2026-09-29T03:51:48.151754+00:00



## 2026-09-29T03:58:38.476259+00:00

<!-- vehicle-terminal-value-rank-predictivity-postdiagnostic-v0-20260929T0425Z -->
## 2026-09-29 terminal-value/objective rank-predictivity postdiagnostic

UTC: 2026-09-29T04:00:06.808140+00:00. No-simulation diagnostic over v1d fresh trace-selected and v1b mined terminal-ablation branch rows. v1d compute-safe non-H15 state-mode groups=20; objective-min matched best measured-compute horizon in 5 groups (rate=0.25). v1b severe objective-rank harm groups=7. Classification `objective_terminal_rank_misaligned_for_compute_safe_choices`. Decision: freeze an IMPROVED objective/terminal repair feasibility diagnostic using realised continuation and measured-time labels; do not train a selector to imitate raw branch objective minima. No validation64/sealed-test access and no training/refit. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TERMINAL_VALUE_RANK_PREDICTIVITY_POSTDIAGNOSTIC_V0_20260929T0425Z.json`.


## 2026-09-29T04:05:55.615076+00:00

<!-- vehicle-objective-terminal-repair-feasibility-v0-20260929T0415Z -->
## 2026-09-29 objective/terminal repair feasibility v0

UTC: 2026-09-29T04:06:49.123415+00:00. Ran frozen no-simulation leave-state-out closed-form diagnostic over existing v1d/v1b development branch rows. Classification `simple_objective_terminal_repair_not_supported_from_current_labels`. Ridge objective/value/horizon severe harms=11 mean loss=6.94513; raw objective severe harms=7 mean loss=4.5757; horizon-only severe harms=9 mean loss=2.55875. v1d compute-safe match rates ridge/raw/horizon=0.0/0.25/0.0. Decision: do not train/refit a selector from current v1d/v1b labels; pivot to versioned scenario/opportunity redesign or richer terminal-value representation diagnostic. No validation64/sealed-test access, no rollout/training/persistent refit. Backup requested at `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_OBJECTIVE_TERMINAL_REPAIR_FEASIBILITY_V0_20260929T0415Z.json`.


## 2026-09-29T04:10:30.402495+00:00



## 2026-09-29T04:15:43.859627+00:00

## vehicle_stress_v1d_state_coverage_opportunity_postdiagnostic_v0-20260929T0435Z

# Vehicle stress-v1d state-coverage/opportunity postdiagnostic v0

UTC: `2026-09-29T04:17:05.010358+00:00`. Development-only no-simulation diagnostic; validation64 and sealed test stayed closed.

## Headline

- Classification: `target_state_coverage_failure_with_prefix_blocked_high_leverage_candidates`.
- Stage1 runner-material episode cases: `[1, 4, 5, 6]`; v1d selected coverage: `1/4`.
- Stage1 physical-gain>=3 episode cases: `[4, 5]`; v1d selected coverage: `0/2`.
- v1d terminal-stable robust positives remain: `0` / `12`.
- Stage2 prefix-blocked but state-close material candidates: `3` across cases `[5]`; accepted prefix-clean material candidates: `0`.
- Train/refit now: `False`.
- Recommended next action: `freeze targeted v1e common-prefix continuation smoke before scenario redesign or selector/value refit`.

## Case-level coverage table

| case | group | runner material | phys>=3 case | strict material H | v1d targets | v1d branch steps | v1d gains | stage2 blocked material branches/H/maxgain |
|---:|---|---:|---:|---|---:|---|---|---|
| 0 | `high_heading_long_or_medium` | 0 | 0 | `[]` | 2 | `[14, 36]` | `[0.0, 0.0]` | `[]` / `[]` / 0 |
| 1 | `high_heading_long_or_medium` | 1 | 0 | `[30, 45, 50]` | 0 | `[]` | `[]` | `[]` / `[]` / 0 |
| 2 | `high_heading_long_or_medium` | 0 | 0 | `[]` | 1 | `[32]` | `[0.000366]` | `[]` / `[]` / 0 |
| 3 | `high_heading_long_or_medium` | 0 | 0 | `[]` | 1 | `[41]` | `[0.000162]` | `[]` / `[]` / 0 |
| 4 | `high_heading_long_or_medium` | 1 | 1 | `[45]` | 0 | `[]` | `[]` | `[]` / `[]` / 0 |
| 5 | `high_heading_long_or_medium` | 1 | 1 | `[10, 20, 25, 30, 35, 40, 45, 50]` | 0 | `[]` | `[]` | `[18]` / `[10, 25, 30]` / 26.18 |
| 6 | `high_heading_long_or_medium` | 1 | 0 | `[10]` | 1 | `[22]` | `[0.0]` | `[]` / `[]` / 0 |
| 9 | `high_heading_long_or_medium` | 0 | 0 | `[]` | 1 | `[54]` | `[0.600712]` | `[]` / `[]` / 0 |
| 11 | `high_heading_short` | 0 | 0 | `[]` | 1 | `[66]` | `[1.2e-05]` | `[]` / `[]` / 0 |
| 12 | `high_heading_short` | 0 | 0 | `[]` | 1 | `[44]` | `[5.1e-05]` | `[]` / `[]` / 0 |
| 15 | `low_heading_low_clearance` | 0 | 0 | `[]` | 1 | `[29]` | `[0.001805]` | `[]` / `[]` / 0 |
| 16 | `lower_stress_control` | 0 | 0 | `[]` | 2 | `[33, 20]` | `[0.0, 0.0]` | `[]` / `[]` / 0 |
| 18 | `lower_stress_control` | 0 | 0 | `[]` | 1 | `[63]` | `[0.027609]` | `[]` / `[]` / 0 |

## Interpretation

- SCENARIOS: opportunity in the current source-supported vehicle family is still sparse and concentrated, but absence of adaptive opportunity is not established because v1d selected no states from the two Stage1 cases with episode-level physical gain >= 3.
- REWARD/TERMINAL/VALUE: objective/terminal repair and raw objective imitation remain unsupported; terminal-stable labels are still required before any selector target is trusted.
- TRAINING: the failed v1d label gate should not trigger selector/value training; it currently diagnoses state-selection coverage rather than a trainable label bank.
- COMPARISONS: shorter H or oracle branch labels do not prove speed; any follow-up must keep physical cost, success/safety and repeated measured decision time separate against fixed-H/Pareto baselines.

## Next discriminating intervention

Freeze a compact IMPROVED v1e targeted common-prefix continuation smoke (development only, no validation64/test): rerun common H15-prefix branch comparisons for the missed episode-positive cases, especially Stage2's prefix-blocked case5 branch-step 18 candidate, plus case1/case4 high-trace states and negative/control states.  Acceptance should require >=2 terminal-stable material positive states from at least two cases, retained negatives/controls, and no prefix/state/safety artifacts before any selector/value refit.  If it fails, pivot toward versioned scenario/opportunity redesign or richer dynamics/terminal-value modeling rather than another unchanged label-density sweep.

Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1D_STATE_COVERAGE_OPPORTUNITY_POSTDIAGNOSTIC_V0_20260929T0435Z.json`.


## 2026-09-29T04:20:50.900149+00:00

## vehicle_stress_v1d_stage2_crossbank_identity_audit_v0-20260929T0445Z

# Vehicle stress-v1d / Stage2 cross-bank identity audit v0

UTC: `2026-09-29T04:21:11.547953+00:00`. Development-only no-simulation audit; validation64 and sealed test stayed closed.

## Headline

- Classification: `stage2_prefix_blocked_candidates_are_cross_bank_not_valid_v1d_targets`.
- Stage2 v0b source Stage1 raw: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v0_stage1_20260928/raw.json`.
- Stress-v1 Stage1 raw: `research_artifacts/aws_diagnostics/vehicle_stress_scenario_opportunity_probe_v1_stage1_20260928T2045Z/raw.json`.
- Stage2 v0b rollout time precedes stress-v1 Stage1/postdiagnostic: `True`.
- Prefix-blocked Stage2 candidate cases in prior coverage diagnostic: `[5]`.
- Same-case identity supported for those cases: `[]`; mismatched cases: `[1, 4, 5, 6]`.
- Corrective decision: `downgrade prior prefix-blocked case5 evidence to legacy/provenance diagnostic only; do not use it as stress-v1/v1d target evidence`.
- Train/refit now: `False`.
- Next action: `freeze stress-v1-only v1e targeted common-prefix target-preparation/protocol before any rollout`.

## Case identity comparisons

| case | v1 source idx/theta/traj/group | stage2-v0b source idx/theta/traj/group | same identity? |
|---:|---|---|---:|
| 1 | `101` / `-0.5773938991142985` / `87` / `high_heading_long_or_medium` | `None` / `None` / `None` / `None` | 0 |
| 4 | `127` / `-0.5854361683372875` / `86` / `high_heading_long_or_medium` | `11` / `-0.6458841144466452` / `63` / `stress` | 0 |
| 5 | `148` / `0.6077726581324596` / `98` / `high_heading_long_or_medium` | `34` / `-0.613104347801943` / `94` / `stress` | 0 |
| 6 | `49` / `0.6453497915281274` / `99` / `high_heading_long_or_medium` | `None` / `None` / `None` / `None` | 0 |

## Interpretation

- SCENARIOS/IMPLEMENTATION: the previous state-coverage result correctly shows that v1d missed stress-v1 episode-positive cases 4 and 5, but the Stage2 prefix-blocked case5 candidates came from an older stress-v0 Stage1 bank and must not be treated as same-case evidence for stress-v1/v1d case5.
- REWARD/TERMINAL/VALUE: objective/terminal repair remains unsupported; this audit only corrects candidate provenance and does not create selector labels.
- TRAINING: do not train/refit now. The valid next rollout, if backed up, must target stress-v1 H15 traces for missed cases 4/5 (and controls) directly rather than replaying older Stage2-v0b case ids as positives.
- COMPARISONS: no timing or adaptive-performance claim is made; all future speed claims still require blocked repeated measured timing against fixed-H/Pareto baselines.

## Revised next discriminating intervention

Freeze a v1e stress-v1-only targeted common-prefix smoke: choose states from stress-v1 Stage1 H15 traces for cases 4 and 5 (plus runner-material case1/6 and negative/control cases), before any non-H15 branch outcomes; include horizons [10,15,20,25,30,45,50] and terminal-stable modes [zero_terminal,H15_common_terminal]. Acceptance before any refit remains >=2 terminal-stable material positive states across >=2 stress-v1 cases, retained controls, and zero prefix/state/safety artifacts. This replaces using the older Stage2-v0b prefix-blocked candidates as v1d targets.

Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1D_STAGE2_CROSSBANK_IDENTITY_AUDIT_V0_20260929T0445Z.json`.


## 2026-09-29T04:26:18.612966+00:00

## vehicle_stress_v1e_targeted_common_prefix_prepare_v0-20260929T0505Z

# Vehicle stress-v1e targeted common-prefix prepare v0

UTC: `2026-09-29T04:27:30.547378+00:00`. No simulations, no candidate resets, no training/refit, no validation64 bank, no sealed test.

## Headline

- Cross-bank audit confirmed Stage2-v0b prefix-blocked candidates are legacy stress-v0 evidence, not valid stress-v1/v1d targets.
- This prepare step freezes stress-v1-only targets from existing stress-v1 H15 traces for missed episode-positive cases plus controls.
- Targets: `12`; planned v1e smoke after backup: `168` episodes / `25200` control-step cap.
- Train/refit now: `False`.

## Frozen targets

| target | case | role | group | source cand | branch step | window | score | v1d target count | stage1 material H |
|---:|---:|---|---|---:|---:|---|---:|---:|---|
| 0 | 4 | `missed_physical_gain_ge3` | `high_heading_long_or_medium` | 127 | 20 | `early` | 1.45093 | 0 | `[45]` |
| 1 | 4 | `missed_physical_gain_ge3` | `high_heading_long_or_medium` | 127 | 36 | `middle` | 8.18294 | 0 | `[45]` |
| 2 | 4 | `missed_physical_gain_ge3` | `high_heading_long_or_medium` | 127 | 46 | `late` | 2.19811 | 0 | `[45]` |
| 3 | 5 | `missed_physical_gain_ge3` | `high_heading_long_or_medium` | 148 | 27 | `early` | 4.10003 | 0 | `[10, 20, 25, 30, 35, 40, 45, 50]` |
| 4 | 5 | `missed_physical_gain_ge3` | `high_heading_long_or_medium` | 148 | 53 | `middle` | 37.5759 | 0 | `[10, 20, 25, 30, 35, 40, 45, 50]` |
| 5 | 5 | `missed_physical_gain_ge3` | `high_heading_long_or_medium` | 148 | 54 | `late` | 40.566 | 0 | `[10, 20, 25, 30, 35, 40, 45, 50]` |
| 6 | 1 | `runner_material_not_physical_ge3_context` | `high_heading_long_or_medium` | 101 | 42 | `mid_high_trace` | 10.4074 | 0 | `[30, 45, 50]` |
| 7 | 6 | `runner_material_not_physical_ge3_context` | `high_heading_long_or_medium` | 49 | 28 | `mid_high_trace` | 10.2702 | 1 | `[10]` |
| 8 | 0 | `same_stratum_negative_control` | `high_heading_long_or_medium` | 95 | 42 | `mid_high_trace` | 15.0081 | 2 | `[]` |
| 9 | 2 | `same_stratum_negative_control` | `high_heading_long_or_medium` | 114 | 52 | `mid_high_trace` | 6.59007 | 1 | `[]` |
| 10 | 16 | `lower_stress_control` | `lower_stress_control` | 130 | 49 | `mid_high_trace` | 11.806 | 2 | `[]` |
| 11 | 17 | `lower_stress_control` | `lower_stress_control` | 187 | 13 | `mid_high_trace` | 10.4667 | 0 | `[]` |

## Four-axis decision update

- SCENARIOS: v1d missed stress-v1 physical-gain cases 4/5. v1e now targets those exact stress-v1 cases from H15 traces, plus runner-material context and negative/lower-stress controls.
- REWARD/TERMINAL: labels remain terminal-stable realised physical continuation under zero and H15-common terminal; raw objective/per-H/H25 labels remain excluded as selector targets.
- TRAINING: no selector/value refit is justified until the v1e smoke gate passes; this prepare is zero training/gradient/refit.
- COMPARISONS: no adaptive or timing claim. A later method, if any, must face strong same-distribution fixed-H/Pareto baselines and measured timing.

## Next action after backup

Implement/run the bounded v1e stress-v1-only common-prefix smoke using the frozen schedule in the protocol. Do not run validation64 or sealed test. Do not reuse Stage2-v0b candidates as stress-v1 targets.

Frozen protocol JSON: `research_artifacts/aws_protocols/vehicle_stress_v1e_targeted_common_prefix_prepare_v0_frozen_20260929T0505Z.json`.
Backup request before v1e smoke: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_20260929T0505Z.json`.


## 2026-09-29T04:31:00.759066+00:00

<!-- vehicle-stress-v1e-targeted-common-prefix-smoke-v0-dryrun-20260929T0515Z -->
## 2026-09-29 vehicle stress-v1e targeted common-prefix smoke v0 dry-run

UTC: 2026-09-29T04:31:34.690006+00:00. No-simulation readiness diagnostic completed for the stress-v1-only v1e common-prefix smoke. Verified the frozen prepare protocol, 12 targets, 168 scheduled episodes, stress-v1 bank target/candidate consistency and closed validation/test access. Rollout remains blocked until verified external backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_V0_RUN_20260929T0515Z.json` plus this runner/dry-run/protocol/prepare artifacts.


## 2026-09-29T04:33:44.657758+00:00



## 2026-09-29T04:37:19.025967+00:00

<!-- vehicle-stress-v1e-targeted-common-prefix-smoke-v0b-schema-repair-dryrun-20260929T0545Z -->
## 2026-09-29 vehicle stress-v1e targeted common-prefix smoke v0b schema-repair dry-run

UTC: 2026-09-29T04:42:19.715892+00:00. No-simulation schema repair readiness completed. Parent v0 failure is confirmed as `KeyError('role')`; v0b changes only the schedule-item role alias (`role = case_role`) before calling the inherited branch runner. Frozen v1e targets/schedule/gate remain unchanged: 168 planned episodes, control-step cap 25200, no training/refit, no validation64 or sealed-test access. Run is blocked until verified external backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_TARGETED_COMMON_PREFIX_SMOKE_V0B_SCHEMA_REPAIR_RUN_20260929T0545Z.json` and v0b source/amendment/dry-run artifacts.


## 2026-09-29T04:44:01.296806+00:00

<!-- vehicle-stress-v1e-targeted-common-prefix-smoke-v0b-schema-repair-run-20260929T0545Z -->
## 2026-09-29 vehicle stress-v1e targeted common-prefix smoke v0b schema-repair run

UTC: 2026-09-29T05:43:14.124635+00:00. Development-only v1e targeted common-prefix smoke completed after one-variable role-alias schema repair: 168 episodes, 14308 control steps. Robust-positive states=2 across cases=[5]; gate=False; blocking artifacts=0. No validation64-bank or sealed-test access, no training/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_run_20260929T0545Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_run_20260929T0545Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1e_targeted_common_prefix_smoke_v0b_schema_repair_run_20260929T0545Z/completed.json`.


## 2026-09-29T05:45:53.242487+00:00



## 2026-09-29T05:51:24.303194+00:00

<!-- vehicle-stress-v1e-smoke-postdiagnostic-v0b-schema-repair-20260929T0600Z -->
## 2026-09-29 vehicle stress-v1e smoke postdiagnostic v0b schema repair

UTC: 2026-09-29T05:52:23.686516+00:00. No-simulation repaired postdiagnostic of the completed v1e v0b schema-repair smoke. Failed postdiagnostic v0 is preserved as a reporting/parser failure (missing `backup_request`; Markdown table columns misparsed) and is superseded. Authoritative parent raw analysis shows 2 robust-positive states across cases [5], 10 negative/neutral states, control positives 0/4, blocking artifacts 0, and gate=False. Positives are concentrated in case 5, so do not train/refit a selector from this bank. Next bounded action after backup: freeze and, after verified backup, run a small repeated positive-state stability/timing diagnostic for the two case-5 robust-positive H15-prefix states using paired H10 versus H15 under both terminal modes. No validation64 bank or sealed-test access; no training/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair_20260929T0600Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair_20260929T0600Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair_20260929T0600Z/completed.json`.


## 2026-09-29T05:54:33.370734+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): ~66.95 h / ~2 d 18 h 57 m as of the completed run.  
Cumulative server API `total_tokens` from repository-accessible `research.sqlite`: not reliably available; the script found no usable token-usage table (`sources=[]`) and reported `total_tokens=0`, so I am treating the cumulative token count as unavailable rather than as a measured zero.

Concrete action completed this iteration:

- Ran exactly one bounded no-simulation postdiagnostic:
  - Script: `experiments/bohn2021_aws/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair.py`
  - SHA256: `53f4d570669961e8a9c5403c7e32763c52c78b10946f8ab4c92a64223ec083e1`
  - Registry: `research_artifacts/aws_runs/20260929T055223_b0e5ed69/registry.json`
  - Exit status: `0`
  - Runtime: ~2.01 s
  - New rollouts/control steps/training/gradient/refit: `0/0/0/0/0`
  - Validation64 opened: `False`
  - Sealed test accessed: `False`

Key evidence inspected/produced:

- Repaired the failed v1e smoke postdiagnostic:
  - Failed v0 is preserved as a reporting/parser failure: missing `backup_request` and Markdown-table column misparse.
  - v0b uses authoritative parent `raw.json -> analysis.state_rows`.
- Parent v1e v0b smoke remains unchanged:
  - `168` development episodes
  - `14308` control steps
  - no validation64/test access
  - no training/refit
- Authoritative v1e label result:
  - Robust-positive states: `2/12`
  - Robust-positive cases: `[5]` only
  - Negative/neutral states: `10`
  - Control positives: `0/4`
  - Blocking artifacts: `0`
  - Frozen selector/value-refit gate: `False`

Scientific decision:

- The result partially supports the target-state coverage hypothesis: v1d missed some relevant stress-v1 states, and targeted v1e sampling found two positives.
- But the positives are too concentrated: both are adjacent states in case 5.
- Therefore: do **not** train/refit a selector from the v1e labels.
- The next informative action is not another unchanged label-density sweep. It is a bounded repeated paired diagnostic to test whether the two case-5 H10-vs-H15 positives are stable local opportunities or one-run solver/terminal/timing artifacts.

Artifacts created:

- `research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair_20260929T0600Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair_20260929T0600Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair_20260929T0600Z/completed.json`
- `research_artifacts/aws_state/vehicle_stress_v1e_smoke_postdiagnostic_v0b_schema_repair_20260929T0600Z.md`
- Backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_SMOKE_POSTDIAGNOSTIC_V0B_SCHEMA_REPAIR_20260929T0600Z.json`

Next preserved action:

1. Obtain/verify external backup covering the new postdiagnostic artifacts, docs/state updates, run registry, backup request, and the already-written case5 timing runner source.
2. Then run the no-simulation dry-run:
   - `experiments/bohn2021_aws/vehicle_stress_v1e_case5_positive_stability_timing_v0_runner.py --dry-run --i-accept-development-v1e-case5-stability-timing-v0`
3. After that dry-run and its backup, run the 24-episode legacy repeated paired diagnostic:
   - two case5 positive states
   - H10 vs H15
   - terminal modes `zero_terminal` and `h15_common_terminal`
   - 3 repeats
   - development only; no validation64; no sealed test.

State was persisted via `update_state` with the active hypothesis, completed run, locked decisions, backup gate, and next experiment queue.

<!-- vehicle-stress-v1e-case5-positive-stability-timing-v0-dryrun-20260929T0605Z -->
## 2026-09-29 vehicle stress-v1e case5 positive stability/timing v0 dry-run

UTC: 2026-09-29T05:56:00.632000+00:00. No-simulation readiness for a repeated paired diagnostic of the two v1e case-5 robust-positive states. Frozen design: 24 development episodes, cap 3600 control steps, H10 vs H15, terminal modes ['zero_terminal', 'h15_common_terminal'], repeats 3; no validation64 or sealed-test access; no training/refit. Run is blocked until verified backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_CASE5_POSITIVE_STABILITY_TIMING_V0_RUN_20260929T0605Z.json`.


## 2026-09-29T05:56:57.849543+00:00
Elapsed service lifetime since first supervisor event (2026-09-26T10:55:29.419331Z): ~67.01 h / 2 d 19 h 0 m as of the completed dry-run.  
Cumulative server API `total_tokens` from repository-accessible `research.sqlite`: unavailable; prior check found no usable token-usage table, so `0` is not treated as a measured cumulative count.

Concrete action completed this iteration:

- Ran exactly one bounded no-simulation dry-run:
  - Script: `experiments/bohn2021_aws/vehicle_stress_v1e_case5_positive_stability_timing_v0_runner.py`
  - Mode: `--dry-run`
  - Interpreter: modern
  - Registry: `research_artifacts/aws_runs/20260929T055600_a5c85728/registry.json`
  - Exit status: `0`
  - Runtime: ~2.01 s
  - New rollouts/control steps/training/gradient/refit: `0/0/0/0/0`
  - Validation64 opened: `False`
  - Sealed test accessed: `False`

What was frozen:

- Diagnostic: repeated paired stability/timing check for the two v1e case-5 positive states:
  - `v1e_t04_case05_cand148_b053_middle`
  - `v1e_t05_case05_cand148_b054_late`
- Design:
  - H10 vs H15
  - terminal modes: `zero_terminal`, `h15_common_terminal`
  - 3 repeats
  - planned episodes: `24`
  - control-step cap: `3600`
- Frozen protocol:
  - `research_artifacts/aws_protocols/vehicle_stress_v1e_case5_positive_stability_timing_v0_frozen_20260929T0605Z.json`
  - protocol SHA256: `2cf2775e8c0a4f220d4bae00101577533dc9df278f681805f3fdf0bc4a39feb9`

Artifacts created/verified:

- `research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_dryrun_20260929T0605Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_dryrun_20260929T0605Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_dryrun_20260929T0605Z/completed.json`
- `research_artifacts/aws_state/vehicle_stress_v1e_case5_positive_stability_timing_v0_dryrun_20260929T0605Z.md`
- Backup request before any more simulation:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_STRESS_V1E_CASE5_POSITIVE_STABILITY_TIMING_V0_RUN_20260929T0605Z.json`

Scientific decision preserved:

- Do **not** train/refit a selector from v1e labels: positives are only `2/12` states and confined to case 5.
- This dry-run is not validation evidence, not final-test evidence, and not a speed claim.
- The next simulation should be the frozen 24-episode paired case5 diagnostic, not another unchanged label-density sweep.

Current blocker:

- More simulations are blocked until a verified external backup covers the new dry-run/protocol/source/docs/request artifacts.
- I attempted to schedule `scripts/research_service/backup.py`, but the tool rejected a second experiment in this iteration with: “One experiment per iteration.” Backup itself did not fail or run.

Next preserved action:

1. Run `scripts/research_service/backup.py` as the next bounded experiment.
2. Verify `status=verified` and `remaining_changed_files=0`.
3. Then run, with legacy interpreter:

```bash
experiments/bohn2021_aws/vehicle_stress_v1e_case5_positive_stability_timing_v0_runner.py \
  --run-repeats \
  --backup-proof <verified_post_dryrun_backup_proof> \
  --i-accept-development-v1e-case5-stability-timing-v0
```

State was persisted with the active hypothesis, evidence, locked decisions, backup gate, and next-experiment queue.


## 2026-09-29T06:02:19.742677+00:00

<!-- backup-failure-status-capture-after-case5-dryrun-v0-20260929T0600Z -->
## 2026-09-29 backup status capture after case5 dry-run backup failure

UTC: 2026-09-29T06:03:36.445026+00:00. Metadata-only capture of supervisor backup state completed with no simulations, no control steps, no training/refit, no validation64-bank access and no sealed-test access.
The preceding backup attempt registry is `research_artifacts/aws_runs/20260929T055736_7b5b40b7/registry.json` with exit status `1` and runtime seconds `34.02472666601534`.
Backup status classification: adequate post-case5-dryrun backup=`False`; reasons=`['status_file_missing_or_unreadable']`.
Minimum required backup time for the case5 run is `None`.
Next action: `retry scripts/research_service/backup.py once; if it fails again, continue infrastructure diagnosis before simulations`. Artifacts: `research_artifacts/aws_diagnostics/backup_failure_status_capture_after_case5_dryrun_v0_20260929T0600Z/summary.md`, `research_artifacts/aws_diagnostics/backup_failure_status_capture_after_case5_dryrun_v0_20260929T0600Z/completed.json`.


## 2026-09-29T06:06:26.936696+00:00

<!-- vehicle-stress-v1e-case5-positive-stability-timing-v0-run-20260929T0605Z -->
## 2026-09-29 vehicle stress-v1e case5 positive stability/timing v0 run

UTC: 2026-09-29T06:15:47.686870+00:00. Development-only repeated H10-vs-H15 diagnostic completed on the two v1e case-5 positive states: 24 episodes, 2280 control steps. Material pairs=12/12, all_group_stable_2_of_3=True, overall median relative H10 decision-time saving=-0.024332120133337094, pass_to_next_design_consideration=False. No validation64-bank or sealed-test access; no training/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_stress_v1e_case5_positive_stability_timing_v0_run_20260929T0605Z/completed.json`.


## 2026-09-29T06:18:01.032303+00:00

<!-- vehicle-stress-v1e-case5-timing-decomposition-postdiagnostic-v0-20260929T0625Z -->
## 2026-09-29 vehicle stress-v1e case5 timing decomposition postdiagnostic v0

UTC: 2026-09-29T06:21:53.357872+00:00. Metadata/no-rollout decomposition of the 24-episode v1e case5 repeated diagnostic. No new simulations, no training/refit, no validation64 bank, and no sealed-test access. Key result: stable local H10 physical gains do not translate into a robust measured compute tradeoff under the current implementation. Executed H10 and H15 steps have the same recorded optimizer-vector size `{'10': [862], '15': [862]}`, measured decision differences track solver-attempt wall time, and no state/terminal group reaches the predeclared >=5% median whole-episode decision-time saving. Classification: `stable_local_physical_opportunity_but_current_fixed_size_mpc_no_measured_compute_tradeoff`. Next action after backup: Next run should be metadata/source audit plus, if supported, a one-variable IMPROVED solver experiment that actually changes MPC problem dimension or controller construction for H10/H15 on development states. If variable-dimensional repair is not feasible on this codebase, pivot to value/terminal/modeling/scenario-opportunity work and treat shorter-H-as-speed as invalid for current implementation. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_STRESS_V1E_CASE5_TIMING_DECOMPOSITION_POSTDIAGNOSTIC_V0_20260929T0625Z.json`.


## 2026-09-29T06:23:54.185225+00:00

<!-- vehicle-mpc-horizon-source-audit-v0-20260929T0630Z -->
## 2026-09-29 vehicle MPC horizon source audit v0

UTC: 2026-09-29T06:24:54.793907+00:00. No-simulation source audit after the v1e case5 timing decomposition. No rollouts, no training/refit, no validation64 bank, no sealed-test access. Classification: `trace_fixed_size_but_source_has_possible_rebuild_path_needs_targeted_instrumentation`. The audit supports treating current shorter-H timing as a fixed-size masked-horizon implementation issue rather than a selector label-density problem alone. Next action after backup: After backup, freeze a no/one-rollout instrumentation smoke around the possible rebuild path before any selector/refit. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_MPC_HORIZON_SOURCE_AUDIT_V0_20260929T0630Z.json`.


## 2026-09-29T06:28:02.711281+00:00

<!-- vehicle-true-variable-horizon-case5-smoke-v0-dryrun-20260929T0640Z -->
## 2026-09-29 vehicle true variable-horizon case5 smoke v0 dry-run

UTC: 2026-09-29T06:34:58.895215+00:00. Metadata-only dry-run froze a four-episode development smoke to test true variable-dimension MPC controllers on the two v1e case5 H15-common-terminal positive states. Planned smoke: 4 direct branch episodes, cap 600 control steps, true `mpc.params.n_horizon` in [10, 15], no candidate resets, no training/refit, no validation64/sealed-test access. It is motivated by the fixed-size H50 AHMPC timing bottleneck and replaces another unchanged v1c/v1d/v1e label-density sweep. Smoke is blocked until verified backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_CASE5_SMOKE_V0_RUN_20260929T0640Z.json` plus source/protocol/dry-run artifacts.


## 2026-09-29T06:36:11.271451+00:00



## 2026-09-29T06:41:55.726340+00:00



## 2026-09-29T06:46:52.549446+00:00

<!-- vehicle-true-variable-horizon-case5-smoke-v0b-schema-repair-run-20260929T0645Z -->
## 2026-09-29 vehicle true variable-horizon case5 smoke v0 run

UTC: 2026-09-29T06:49:27.363915+00:00. Development-only true variable-dimension MPC smoke completed: 4 episodes, 170 control steps. Dimension reduced for all pairs=True; safety ok for all pairs=True; median solver relative H10 saving=0.3497927735098441; pass to broader variable-H block=True. No validation64-bank or sealed-test access, no training/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair_run_20260929T0645Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair_run_20260929T0645Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_case5_smoke_v0b_schema_repair_run_20260929T0645Z/completed.json`.


## 2026-09-29T06:52:32.665896+00:00

<!-- vehicle-true-variable-horizon-broader-block-freeze-v0-20260929T0650Z -->
## 2026-09-29 vehicle true variable-H broader block freeze v0

UTC: 2026-09-29T06:53:40.714474+00:00. Metadata-only/no-simulation protocol freeze completed after v0b. v0b proved true-H construction speed feasibility but physical tradeoff remains unresolved (H10-vs-H15 physical gains [-450.49814372103845, -0.046977699789351846]). Frozen next development protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_broader_block_v0_frozen_20260929T0650Z.json` has 48 planned branch episodes, cap 7200 control steps, horizons [10, 15, 25], terminal profiles ['matched_terminal', 'shared_h15_terminal'], repeats 2, validation64 closed and sealed test closed. More simulation is blocked until backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_BROADER_BLOCK_V0_RUN_20260929T0650Z.json` and all v0b/freeze artifacts.


## 2026-09-29T06:58:45.125102+00:00

<!-- vehicle-true-variable-horizon-broader-block-v0-run-20260929T0650Z -->
## 2026-09-29 vehicle true variable-H broader block v0 run

UTC: 2026-09-29T07:06:37.183945+00:00. Development-only broader true variable-H block completed: 48 branch episodes, 2952 control steps, validation64 closed, sealed test closed, no training/refit. Gates: implementation=True, safety=True, compute=True, physical=True, pass_to_selector_or_value_learning_design=True. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_v0_run_20260929T0650Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_v0_run_20260929T0650Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_broader_block_v0_run_20260929T0650Z/completed.json`.


## 2026-09-29T07:09:47.963944+00:00

<!-- vehicle-true-variable-horizon-broader-block-postdiagnostic-v0-20260929T0710Z -->
## 2026-09-29 vehicle true variable-H broader block postdiagnostic v0

UTC: 2026-09-29T07:10:49.936167+00:00. Metadata-only postdiagnostic completed with no simulations/training/refit, validation64 closed and sealed test closed. H10 is not a robust global physical improvement (strict non-control H10-vs-H15 improvements 0/6; global non-control H10 physical is much worse than H15/H25), but a near-best physical oracle over H10/H15/H25 is mixed ({'15': 2, '10': 4, '25': 2}) and has non-control decision-time saving 0.367 vs fixed H25 with physical delta -1.60051. Decision: larger source-supported true-H oracle-label/value-modeling diagnostic before any selector training; do not repeat sparse label sweeps.


## 2026-09-29T07:14:08.745914+00:00



## 2026-09-29T07:18:50.489791+00:00

<!-- vehicle-true-variable-horizon-oracle-bank-freeze-v0-20260929T0725Z -->
## 2026-09-29 vehicle true variable-H oracle bank freeze v0

UTC: 2026-09-29T07:20:14.573384+00:00. Metadata-only/no-simulation protocol freeze completed for a larger source-supported true-H oracle-label/value-modeling bank. Planned development branch budget is 192 episodes / 28800 control-step cap, horizons [10, 15, 25], terminal profiles ['matched_terminal', 'shared_h15_terminal'], repeats 2, selected states 16 (11 non-control for primary gate). Validation64 and sealed test remain closed; training/refit/candidate resets are zero. More simulation/training/refit is blocked until an external verified backup covers `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_ORACLE_BANK_V0_RUN_20260929T0725Z.json` and this freezer output. Protocol: `research_artifacts/aws_protocols/vehicle_true_variable_horizon_oracle_bank_v0_frozen_20260929T0725Z.json`.


## 2026-09-29T07:23:18.062424+00:00

<!-- vehicle-true-variable-horizon-oracle-bank-v0-run-20260929T0725Z -->
## 2026-09-29 vehicle true variable-H oracle bank v0 run

UTC: 2026-09-29T07:46:20.416993+00:00. Development-only oracle-label bank completed: 192 branch episodes, 9072 control steps, validation64 closed, sealed test closed, no training/refit. Gates: {'physical_gate_vs_H25_primary_noncontrol': True, 'aggregate_decision_saving_gate_vs_H25_primary_noncontrol': True, 'median_decision_saving_gate_vs_H25_primary_noncontrol': True, 'safety_gate': True, 'label_non_degeneracy_gate': True, 'pass_to_selector_or_value_modeling_design': True, 'train_or_refit_now': False}. Label counts: {'15': 5, '10': 25, '25': 2}. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_oracle_bank_v0_run_20260929T0725Z/completed.json`.


## 2026-09-29T07:49:13.436217+00:00



## 2026-09-29T07:57:16.435368+00:00

<!-- vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0-20260929T0755Z -->
## 2026-09-29 vehicle true-variable-H selector feasibility postdiagnostic v0

UTC: 2026-09-29T07:59:39.665914+00:00. Offline development diagnostic completed with no simulation/training/refit, validation64 closed, sealed test closed. Accepted for fresh rollout: False. Primary label counts: {'10': 17, '15': 3, '25': 2}. H25 unique primary states: 1. Decision: Do not launch selector training/refit yet from this oracle bank: the offline deployable leave-one-state-out gate did not pass. Next highest-information action is a versioned fresh development risk-anchor/terminal-consistency bank (true H10/H15/H25, matched/shared terminal, blocked timing) or terminal-value/objective repair, not scaling a selector on sparse anchor-dependent labels. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0_20260929T0755Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_feasibility_postdiagnostic_v0_20260929T0755Z/raw.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_SELECTOR_FEASIBILITY_POSTDIAGNOSTIC_V0_20260929T075939.665914+0000.json`.


## 2026-09-29T08:03:54.253484+00:00

<!-- vehicle-true-variable-H-risk-anchor-acquisition-freeze-v0-20260929T0810Z -->
## 2026-09-29 vehicle true-variable-H risk-anchor acquisition freeze v0

UTC: 2026-09-29T08:05:06.624876+00:00. Metadata-only/no-simulation protocol freeze completed after selector feasibility failed. Selector accepted for fresh rollout: False; primary labels {'10': 17, '15': 3, '25': 2}; H25 unique primary states 1; terminal-profile disagreements 5. Frozen source-independent protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_frozen_20260929T0810Z.json` selects 10 stress-v1 H15-trace targets outside oracle-bank cases, with 120 planned true-H branch episodes and cap 18000 control steps. No validation64/test/training/refit. Further simulation/training/refit requires verified backup covering `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_RISK_ANCHOR_ACQUISITION_FREEZE_V0_20260929T0810Z.json` and this new source/protocol.


## 2026-09-29T08:07:30.181582+00:00

<!-- vehicle-true-variable-H-risk-anchor-acquisition-v0-dryrun-20260929T0825Z -->
## 2026-09-29 vehicle true-variable-H risk-anchor acquisition v0 dry-run

UTC: 2026-09-29T08:12:36.696857+00:00. Wrote and dry-ran `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_runner.py` with no simulations, no candidate resets, no training/refit, no validation64-bank access and no sealed-test access. It verifies the frozen source-independent risk-anchor protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_anchor_acquisition_freeze_v0_frozen_20260929T0810Z.json` (120 planned true-H branch episodes, cap 18000 control steps) after the selector-feasibility failure. Further simulation is blocked until verified external backup covers this runner/dry-run/freeze artifacts and `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_BEFORE_VEHICLE_TRUE_VARIABLE_HORIZON_RISK_ANCHOR_ACQUISITION_V0_RUN_20260929T0825Z.json`.


## 2026-09-29T08:13:07.939324+00:00

<!-- vehicle-true-variable-H-risk-anchor-acquisition-v0-run-20260929T0825Z -->
## 2026-09-29 vehicle true-variable-H risk-anchor acquisition v0 run

UTC: 2026-09-29T08:25:34.518865+00:00. Development-only source-independent risk-anchor acquisition completed: 120 branch episodes, 4414 control steps, validation64 closed, sealed test closed, no training/refit. Gates: {'fresh_H15_H25_anchor_cases_or_all_H10_gate': True, 'stable_H15_H25_anchor_cases_or_all_H10_gate': False, 'terminal_consistency_not_dominated_gate': False, 'physical_gate_vs_H25_risk_anchor_primary': True, 'aggregate_decision_saving_gate_vs_H25_risk_anchor_primary': True, 'median_decision_saving_gate_vs_H25_risk_anchor_primary': True, 'safety_gate': True, 'fixed_short_not_absorbed_gate': False, 'pass_to_offline_deployable_selector_cv': False, 'train_or_refit_now': False}. Risk labels: {'10': 6, '15': 3, '25': 1}. Decision: block selector refit; prioritize terminal-value/objective/modeling repair because labels are terminal-profile dependent or too concentrated. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/completed.json`.


## 2026-09-29T08:28:32.026640+00:00

<!-- vehicle-true-variable-H-terminal-profile-effect-postdiagnostic-v0-20260929T0835Z -->
## 2026-09-29 vehicle true-variable-H terminal-profile effect postdiagnostic v0

UTC: 2026-09-29T08:29:37.325397+00:00. No simulations/training/refit; validation64 and sealed test stayed closed. Parsed `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_anchor_acquisition_v0_run_20260929T0825Z/raw.json` after the source-independent risk-anchor acquisition. Label flips across terminal profiles were 7/10 states (4/5 risk states), material non-H15 terminal physical effects occurred in 7 states, and fixed H15 absorbed the risk-group H25 tradeoff (`fixed_H15_absorbs_vs_H25=True`). Selector/refit remains blocked. Next action after backup: await external backup, then freeze a bounded terminal-value residual/calibration audit plus H10/H15 risk-aware offline CV; do not run closed-loop selector or gradient/value refit yet. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_profile_effect_postdiagnostic_v0_20260929T0835Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_profile_effect_postdiagnostic_v0_20260929T0835Z/raw.json`.


## 2026-09-29T08:34:43.588815+00:00



## 2026-09-29T08:39:57.341943+00:00

<!-- vehicle-true-variable-H-h10-h15-residual-cv-diagnostic-v0b-20260929T0845Z -->
## 2026-09-29 vehicle true-variable-H H10/H15 residual/CV diagnostic v0b

UTC: 2026-09-29T08:40:47.765150+00:00. Offline/no-simulation syntax-repaired diagnostic parsed existing oracle-bank and risk-anchor branch outputs. Validation64 and sealed test stayed closed; no training/refit. Samples=52, states=26, terminal H10/H15 label flips=11. Deployable-proxy CV passes vs fixed H15: [{'dataset': 'all_profiles', 'policy': 'cat:selection_group', 'deployable_proxy': True, 'strong_10pct': True, 'leave_state': {'horizon_counts': {'15': 24, '10': 28}, 'physical_delta_vs_H15': 38.965320252177094, 'physical_tolerance_vs_H15': 104.0, 'decision_relative_saving_vs_H15': 0.15118612787452596, 'solver_relative_saving_vs_H15': 0.16652804416676212, 'unsafe_chosen_groups': 0}, 'leave_case': {'horizon_counts': {'15': 24, '10': 28}, 'physical_delta_vs_H15': 38.965320252177094, 'physical_tolerance_vs_H15': 104.0, 'decision_relative_saving_vs_H15': 0.15118612787452596, 'solver_relative_saving_vs_H15': 0.16652804416676212, 'unsafe_chosen_groups': 0}}, {'dataset': 'risk_anchor_source_all_profiles', 'policy': 'cat:selection_group', 'deployable_proxy': True, 'strong_10pct': False, 'leave_state': {'horizon_counts': {'15': 16, '10': 4}, 'physical_delta_vs_H15': 3.7196055961351817, 'physical_tolerance_vs_H15': 40.0, 'decision_relative_saving_vs_H15': 0.0543353990326759, 'solver_relative_saving_vs_H15': 0.059198089140973374, 'unsafe_chosen_groups': 0}, 'leave_case': {'horizon_counts': {'15': 16, '10': 4}, 'physical_delta_vs_H15': 3.7196055961351817, 'physical_tolerance_vs_H15': 40.0, 'decision_relative_saving_vs_H15': 0.0543353990326759, 'solver_relative_saving_vs_H15': 0.059198089140973374, 'unsafe_chosen_groups': 0}}]. Decision: after verified backup, freeze a tiny development rollout protocol with selector overhead and blocked timing against fixed true H10/H15/H25; do not use validation64/test. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair_20260929T0845Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_residual_cv_diagnostic_v0b_schema_repair_20260929T0845Z/raw.json`.


## 2026-09-29T08:44:24.550084+00:00

<!-- vehicle-true-variable-H-h10-h15-feature-deployability-audit-v0-20260929T0850Z -->
## 2026-09-29 vehicle true-variable-H H10/H15 feature-deployability audit v0

UTC: 2026-09-29T08:45:28.701100+00:00. Offline/no-simulation audit of the v0b H10/H15 residual-CV result. Validation64 and sealed test stayed closed; no training/refit. Original v0b deployable-proxy passes=2, but state/online-observable passes after feature audit=0; metadata-feature passes=3. Leakage blocks selector rollout=True. Decision: block the proposed selector-overhead rollout based on cat:selection_group; the only v0b passes use non-online scenario/protocol metadata.  Next bounded action should be a state-observable H10/H15 model-CV/refit or terminal-value calibration diagnostic, not a rollout of a metadata oracle.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_feature_deployability_audit_v0_20260929T0850Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h10_h15_feature_deployability_audit_v0_20260929T0850Z/raw.json`.


## 2026-09-29T08:51:41.023734+00:00
Elapsed since first supervisor event: ~2d 21h 50m (`251,399.281769` s recorded by the completed audit). Cumulative server API `total_tokens`: unavailable from repository tools this iteration; I did not invent a value.

Concrete progress completed:

1. **Ran the H10/H15 feature-deployability/leakage audit**
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_h10_h15_feature_deployability_audit_v0.py`
   - Run registry: `research_artifacts/aws_runs/20260929T084528_3ce13e0a/registry.json`
   - Status: complete, exit `0`
   - Budget: `0` simulations, `0` control steps, `0` training/refit/gradient steps
   - Validation64: not opened
   - Sealed test: not accessed

2. **Key scientific finding**
   - v0b had reported `2` apparent H10/H15 CV passes against fixed true H15.
   - This audit found **0 valid state/online-observable deployable passes**.
   - The passes were driven by metadata:
     - `metadata_feature_passes = 3`
     - `selection_group_passes = 2`
   - Decision: **block selector-overhead rollout based on `cat:selection_group`**. It is protocol/source metadata leakage, not a deployable adaptive MPC-horizon policy.

3. **Interpretation**
   - The true variable-H mechanism still has development evidence of measured compute savings.
   - But the current H10/H15 selector evidence does **not** yet establish a deployable learned rule.
   - Fixed true H15 remains the main comparator.
   - Terminal-profile/value mismatch remains a major bottleneck.

4. **Prepared next diagnostic source**
   - Wrote:
     - `experiments/bohn2021_aws/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0.py`
       - SHA256: `3bb498fd47784df5a960e85ef5cd66d1c25d738dfa54aec1136275dc1b3cde30`
     - `experiments/bohn2021_aws/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded.py`
       - SHA256: `aef308708bcf63f4f060fed9eb0e108c38039d174c0651fa526188f6854d8e9b`
   - Preferred next script: the bounded v0b version.

5. **Attempted next execution, blocked by contract guard**
   - Attempted to run bounded state-observable CV script.
   - Tool returned: `ValueError: One experiment per iteration`.
   - No script execution occurred; this is only a scheduler/contract guard, not a scientific result.

6. **State preserved**
   - Wrote handoff:
     - `research_artifacts/aws_state/vehicle_true_variable_horizon_feature_deployability_and_state_observable_cv_handoff_20260929T0852Z.md`
   - Wrote backup request:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_FEATURE_DEPLOYABILITY_AND_STATE_OBSERVABLE_CV_SOURCE_PREP_20260929T0852Z.json`
   - Updated durable supervisor state via `update_state`.

Current blocker / next action:

- Latest verified backup predates the feature audit and new CV source files, so **unbacked artifacts exist**.
- Do not run new simulations/training/refit until backup is verified.
- Next exact research action after backup, or if supervisor allows offline diagnostics before backup:
  - Run `experiments/bohn2021_aws/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded.py`
  - Modern interpreter
  - Offline only: `0` simulations, `0` control steps, `0` validation/test episodes
  - Purpose: determine whether an online/state-observable H10/H15 rule can beat fixed true H15 without metadata leakage.
- If it passes only terminal-fixed deployable gates, freeze a tiny development rollout with selector overhead and blocked/randomized timing.
- If it fails or only pooled/terminal-dependent subsets pass, pivot to terminal-value/objective calibration or scenario/modeling intervention rather than rolling out a metadata selector.

<!-- vehicle-true-variable-H-state-observable-h10-h15-model-cv-v0b-bounded-20260929T0905Z -->
## 2026-09-29 vehicle true-variable-H state-observable H10/H15 model-CV v0b bounded

UTC: 2026-09-29T08:56:19.227244+00:00. Offline/no-simulation diagnostic using only initial branch observations and online-derived transforms; validation64 and sealed test stayed closed. Candidate configs per outer fold=36. Passing datasets=['risk_anchor_source_all_profiles']; strong passes=[]; terminal-fixed passes=[]. Proceed to selector rollout=False. Decision: state-observable CV passes only in pooled/diagnostic subsets; do not rollout yet because terminal-profile dependence remains unresolved. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded_20260929T0905Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_state_observable_h10_h15_model_cv_v0b_bounded_20260929T0905Z/raw.json`.


## 2026-09-29T09:00:16.908812+00:00

<!-- vehicle-true-variable-H-terminal-consistency-audit-v0-20260929T0915Z -->
## 2026-09-29 vehicle true-variable-H terminal-consistency audit v0

UTC: 2026-09-29T09:01:25.144883+00:00. Offline/no-simulation audit of terminal-profile consistency after the state-observable H10/H15 nested-CV result; validation64 and sealed test stayed closed. Robust target has value=True; mismatch bad=True. Decision: terminal-consistent labels retain some measured compute value, but cross-profile label transfer is unsafe; next intervention should calibrate terminal/objective targets before any learned selector rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_consistency_audit_v0_20260929T0915Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_consistency_audit_v0_20260929T0915Z/raw.json`.


## 2026-09-29T09:04:42.624373+00:00

<!-- vehicle-true-variable-H-robust-terminal-label-cv-v0-20260929T0925Z -->
## 2026-09-29 vehicle true-variable-H robust terminal-label CV v0

UTC: 2026-09-29T09:08:51.121000+00:00. Offline/no-simulation robust-terminal label diagnostic; validation64 and sealed test stayed closed. Pass summary: {'terminal_agreement_only': ['risk_anchor_source_all_profiles'], 'aggregate_terminal_robust': [], 'strict_per_profile_regret2': ['risk_anchor_source_all_profiles']}; strong pass summary: {'terminal_agreement_only': [], 'aggregate_terminal_robust': [], 'strict_per_profile_regret2': []}. Decision: robust terminal labels are learnable only on risk-anchor subset; next collect/freeze source-independent confirmation or value-calibration smoke, not broad validation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_robust_terminal_label_cv_v0_20260929T0925Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_robust_terminal_label_cv_v0_20260929T0925Z/raw.json`.


## 2026-09-29T09:12:28.857246+00:00



## 2026-09-29T09:27:20.078399+00:00

<!-- vehicle-true-variable-H-safety-gated-selector-cv-v0b-fast-20260929T0950Z -->
## 2026-09-29 vehicle true-variable-H safety-gated selector CV v0b fast

UTC: 2026-09-29T09:28:26.718027+00:00. Offline/no-simulation fast safety-gated robust-label selector diagnostic; validation64 and sealed test stayed closed. This repairs the v0 timeout by removing nested hyperparameter selection and evaluating five predeclared configs directly. Pass summary: {'terminal_agreement_only': ['matched_terminal_only'], 'aggregate_terminal_robust': ['matched_terminal_only'], 'strict_per_profile_regret2': ['matched_terminal_only']}; strong pass summary: {'terminal_agreement_only': [], 'aggregate_terminal_robust': [], 'strict_per_profile_regret2': []}. Decision: diagnostic same-config safety-gated CV passes beyond risk-only; next run an optimized nested/fresh-source confirmation before any selector rollout or validation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_safety_gated_selector_cv_v0b_fast_20260929T0950Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_safety_gated_selector_cv_v0b_fast_20260929T0950Z/raw.json`.


## 2026-09-29T09:32:23.364096+00:00

<!-- vehicle-true-variable-H-domain-profile-transfer-audit-v0-20260929T1000Z -->
## 2026-09-29 vehicle true-variable-H domain/profile transfer audit v0

UTC: 2026-09-29T09:33:02.747287+00:00. Development-only offline source/profile transfer diagnostic; validation64 and sealed test stayed closed. Pass summary: {'terminal_agreement_only': ['matched_terminal_only:train_risk_test_oracle', 'shared_h15_terminal_only:train_oracle_test_risk', 'shared_h15_terminal_only:train_risk_test_oracle', 'profile_transfer:train_matched_test_shared', 'profile_transfer:train_shared_test_matched'], 'aggregate_terminal_robust': ['matched_terminal_only:train_risk_test_oracle', 'shared_h15_terminal_only:train_oracle_test_risk', 'shared_h15_terminal_only:train_risk_test_oracle', 'profile_transfer:train_matched_test_shared', 'profile_transfer:train_shared_test_matched'], 'strict_per_profile_regret2': ['matched_terminal_only:train_risk_test_oracle', 'shared_h15_terminal_only:train_oracle_test_risk', 'shared_h15_terminal_only:train_risk_test_oracle', 'profile_transfer:train_matched_test_shared', 'profile_transfer:train_shared_test_matched']}; strong pass summary: {'terminal_agreement_only': ['shared_h15_terminal_only:train_oracle_test_risk', 'profile_transfer:train_matched_test_shared', 'profile_transfer:train_shared_test_matched'], 'aggregate_terminal_robust': ['shared_h15_terminal_only:train_oracle_test_risk', 'profile_transfer:train_matched_test_shared', 'profile_transfer:train_shared_test_matched'], 'strict_per_profile_regret2': ['shared_h15_terminal_only:train_oracle_test_risk', 'profile_transfer:train_matched_test_shared', 'profile_transfer:train_shared_test_matched']}; source-transfer passes: 9; profile-transfer passes: 6. Decision: At least one source-held-out transfer retained a safe >=5% measured decision saving; after backup, run a small fresh-source confirmation or optimized nested source-held-out selector before any validation rollout.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_domain_profile_transfer_audit_v0_20260929T1000Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_domain_profile_transfer_audit_v0_20260929T1000Z/raw.json`.


## 2026-09-29T09:36:43.745787+00:00

<!-- vehicle-true-variable-H-fresh-source-confirmation-freeze-v0-20260929T1015Z -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation freeze v0

UTC: 2026-09-29T09:41:53.773237+00:00. Metadata-only/no-simulation protocol freeze completed. Fresh source cases were selected from the stress-v1 candidate pool by metadata only, excluding all Stage1 H-outcome selected indices plus oracle/risk-anchor source_candidate_index values. Future runner budget is 136 episodes / 20400 control-step cap: 8 H15 trace scans followed by 128 blocked H10/H15 branch episodes under both terminal profiles. No validation64/test/training/refit. Further simulation/training/refit requires verified backup covering `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_FREEZE_V0_20260929T1015Z.json`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_20260929T1015Z/summary.md`, `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v0_frozen_20260929T1015Z.json`.


## 2026-09-29T09:42:40.268126+00:00



## 2026-09-29T09:47:37.998857+00:00



## 2026-09-29T09:51:45.667775+00:00



## 2026-09-29T09:56:04.150189+00:00



## 2026-09-29T09:58:34.906803+00:00
Cumulative elapsed service lifetime: ~71.1 h since 2026-09-26T10:55:29Z. Cumulative server API total_tokens: unavailable from exposed repository tools this cycle because `research.sqlite` was not listable/readable here; this excludes desktop conversation usage.

Concrete actions completed:

- Accepted the supervisor-provided verified backup gate as sufficient for simulation recovery:
  - commit `d93cd8a8f0b2d11c09cb8fa512422cb9cbe0b531`
  - package SHA256 `7cbbeb12c2536878035097f9d4b7dcdd744989b06b9573107b899d505c551bf7`
  - `remaining_changed_files: 0`
- Attempted to run the frozen fresh-source confirmation runner.
- The attempt failed immediately at CLI parsing, before any simulation:
  - script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v0_runner.py`
  - registry: `research_artifacts/aws_runs/20260929T095648_cf18ccf0/registry.json`
  - exit status: `2`
  - runtime: ~2.0 s
  - cause: omitted required `--backup-verified-commit`
- Verified from source that required arguments are:
  - `--run`
  - `--backup-verified-commit d93cd8a8f0b2d11c09cb8fa512422cb9cbe0b531`
  - `--i-accept-development-fresh-source-confirmation-v0`
- Checked for partial run output under the fixed run directory; none was present.
- Preserved handoff state in:
  - `research_artifacts/aws_state/continue_state_20260929T1000_after_fresh_source_runner_cli_arg_failure.md`
- Persisted durable state via `update_state`.

Safety/budget status:

- New simulations: `0`
- New control steps: `0`
- New training/refit/gradient steps: `0`
- validation64 opened: `false`
- sealed test accessed: `false`
- Research question unchanged.

Next concrete action for the next bounded iteration:

Run the same backed-up runner with corrected args:

```text
experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v0_runner.py
--run
--backup-verified-commit d93cd8a8f0b2d11c09cb8fa512422cb9cbe0b531
--i-accept-development-fresh-source-confirmation-v0
```

Declared budget remains: 136 development-only episodes, ≤20400 control steps, no validation64, no sealed test, no training/refit.

<!-- vehicle-true-variable-H-fresh-source-confirmation-v0-run-20260929T1025Z -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation v0 run

UTC: 2026-09-29T10:12:36.235821+00:00. Development-only fresh-source confirmation completed: 136 episodes (8 H15 traces + 128 blocked H10/H15 branches), 6660 control steps. No validation64, no sealed test, no training/refit. Primary shared-H15 source-trained guard gate: {'scope': 'shared_h15_terminal', 'groups': 16, 'h10_predictions': 5, 'confusion_vs_fresh_h10_beneficial_label': {'TP': 4, 'FP': 1, 'FN': 10, 'TN': 1}, 'false_positive_rows': [{'base_state_id': 'fresh_case03_slot1_mid_late', 'fresh_confirmation_group': 'fresh_high_heading_long_or_medium', 'h10_safe_all': True, 'h15_safe_all': True, 'physical_delta_h10_minus_h15': 6.663203456605871, 'decision_delta_h10_minus_h15': -0.7722706629283493}], 'catastrophic_false_positive_rows': [{'base_state_id': 'fresh_case03_slot1_mid_late', 'fresh_confirmation_group': 'fresh_high_heading_long_or_medium', 'h10_safe_all': True, 'h15_safe_all': True, 'physical_delta_h10_minus_h15': 6.663203456605871, 'decision_delta_h10_minus_h15': -0.7722706629283493}], 'physical_tolerance_vs_fixed_H15': 32.0, 'physical_gate': True, 'decision_saving_gate_5pct': True, 'decision_saving_gate_10pct_strong': False, 'primary_pass_5pct': False, 'primary_strong_10pct': False, 'solver_saving_reported_not_primary': 0.07401150723689377, 'train_or_refit_now': False}. Decision: fresh source H10 false positives/physical regression block selector rollout; prioritize terminal-value/objective/representation calibration before validation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v0_run_20260929T1025Z/completed.json`.


## 2026-09-29T10:22:30.899505+00:00

<!-- vehicle-true-variable-H-guard-veto-diagnostic-v0-20260929T1035Z -->
## 2026-09-29 vehicle true-variable-H guard/veto diagnostic v0

UTC: 2026-09-29T10:23:39.042896+00:00. No-simulation development diagnostic after fresh-source confirmation failure. Source labels: {'agreement_positive_h10': 13, 'agreement_non_h10': 1, 'terminal_disagreement_or_missing': 12}; total 26 examples. Passing weak fresh shared-H15 variants: ['allNonposVeto_s1_raw_abs_l2_m1.5_v1.0', 'disagreementVeto_s1_raw_abs_l2_m1.5_v1.0', 'recall_s1_raw_abs_l2_agreementNeg_m1.5']; passing strong variants: ['allNonposVeto_s1_raw_abs_l2_m1.5_v1.0', 'disagreementVeto_s1_raw_abs_l2_m1.5_v1.0', 'recall_s1_raw_abs_l2_agreementNeg_m1.5']. Decision: Exploratory source-label veto/representation variants can exceed the strong 10% fresh development gate without catastrophic H10 false positives. Because this is outcome-informed, freeze a new fresh-source confirmation block for the top simple variant before any validation64/test access.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_guard_veto_diagnostic_v0_20260929T1035Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_guard_veto_diagnostic_v0_20260929T1035Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_guard_veto_diagnostic_v0_20260929T1035Z/completed.json`.


## 2026-09-29T10:27:01.462398+00:00

<!-- vehicle-true-variable-H-fresh-source-confirmation-freeze-v1-20260929T1055Z -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation freeze v1

UTC: 2026-09-29T10:31:07.207110+00:00. Metadata-only/no-simulation protocol freeze completed for `disagreementVeto_s1_raw_abs_l2_m1.5_v1.0`. It excludes previous Stage1/oracle/risk sources and all v0 fresh-source candidate IDs [13, 37, 45, 155, 186, 205, 227, 245]; selected independent candidate IDs [161, 165, 27, 157, 171, 76, 111, 238]. Future runner budget is 136 episodes / 20400 control-step cap. No validation64/test/training/refit. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_20260929T1055Z/summary.md`, `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v1_frozen_20260929T1055Z.json`.


## 2026-09-29T10:33:29.560533+00:00

<!-- vehicle-true-variable-H-fresh-source-confirmation-v1-run-20260929T1110Z -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation v1 run

UTC: 2026-09-29T10:47:06.765333+00:00. Development-only independent fresh-source confirmation completed for `disagreementVeto_s1_raw_abs_l2_m1.5_v1.0`: 136 episodes (8 H15 traces + 128 blocked H10/H15 branches), 6654 control steps. No validation64, no sealed test, no training/refit. Primary gate: {'scope': 'shared_h15_terminal', 'groups': 16, 'h10_predictions': 5, 'h15_predictions': 11, 'nonconstant_horizons': True, 'confusion_vs_fresh_h10_beneficial_label': {'TP': 5, 'FP': 0, 'FN': 9, 'TN': 2}, 'false_positive_rows': [], 'catastrophic_false_positive_rows': [], 'unsafe_rows': [], 'physical_tolerance_sum_vs_fixed_H15': 32.0, 'physical_gate': True, 'fixed_H15_physical_sum': 87.81786992090244, 'comparison_policy_vs_fixed_H15': {'physical_delta_vs_fixed_H15': 0.22815922495708207, 'decision_relative_saving_vs_fixed_H15': 0.08207176592861759, 'solver_relative_saving_vs_fixed_H15': 0.09159857845021688}, 'comparison_oracle_vs_fixed_H15': {'physical_delta_vs_fixed_H15': 1.0362879287184228, 'decision_relative_saving_vs_fixed_H15': 0.24461430437627588, 'solver_relative_saving_vs_fixed_H15': 0.269078665753328}, 'oracle_chosen_counts': {'10': 14, '15': 2}, 'policy_chosen_counts': {'15': 11, '10': 5}, 'decision_saving_gate_5pct': True, 'decision_saving_gate_10pct_strong': False, 'primary_pass_5pct': True, 'primary_strong_10pct': False, 'solver_saving_reported_not_primary': 0.09159857845021688}. Decision: v1 independent fresh-source confirmation gives only weak measured-decision saving; inspect timing uncertainty and consider value/objective/representation calibration before validation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v1_run_20260929T1110Z/completed.json`.


## 2026-09-29T10:52:39.343618+00:00



## 2026-09-29T10:57:33.439513+00:00



## 2026-09-29T11:04:02.176390+00:00

<!-- vehicle-true-variable-H-selector-refit-v2c-relaxed-schema-20260929T1145Z -->
## 2026-09-29 vehicle true-variable-H selector-refit v2b schema-repair diagnostic

UTC: 2026-09-29T11:07:41.854535+00:00. Metadata/refit-only diagnostic completed with 0 simulations/control steps, no validation64, no sealed test, no gradient training. It cross-trained simple selector variants on fresh_v0/fresh_v1 development banks and evaluated on the opposite bank. Decision: A simple refit reaches cross-bank strong shared-H15 performance but has matched-terminal robustness concerns; either constrain claims/profile to shared-H15 terminal or prioritize terminal-value calibration before validation.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema_20260929T1145Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema_20260929T1145Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema_20260929T1145Z/completed.json`, candidate protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_selector_refit_v2c_relaxed_schema_20260929T1145Z_candidate_protocol.json`.


## 2026-09-29T11:12:37.166060+00:00

<!-- vehicle-true-variable-H-fresh-source-confirmation-freeze-v2-20260929T1120Z -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation freeze v2

UTC: 2026-09-29T11:13:49.017027+00:00. Metadata-only/no-simulation freeze for v2c top candidate `source_only::v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only` under shared-H15 terminal primary scope. Selected unused fresh-source candidate IDs [44, 26, 239, 159, 146, 203, 251, 12]; total future budget 136 episodes / cap 20400 control steps; no validation64/sealed test/training/refit. Matched-terminal remains secondary because v2c showed robustness failures. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_20260929T1120Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_20260929T1120Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_20260929T1120Z/completed.json`, protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json`.


## 2026-09-29T11:17:41.127534+00:00
Elapsed service lifetime: ~72.33 h since 2026-09-26T10:55:29.419331Z.  
Cumulative server API total_tokens: not available from repository-visible files in this API turn; must be filled from supervisor SQLite/JSONL, excluding this desktop/API conversation usage.

Concrete actions completed:

1. Inspected the current evidence state:
   - v1 fresh-source confirmation was safe/nonconstant but weak: 8.207% measured decision saving, below the 10% strong gate.
   - v2c selector-refit diagnostic found a development-promising source-only shared-H15-terminal selector with cross-bank minimum decision saving 10.466% and zero primary catastrophic false positives, but matched-terminal remained unsafe/fragile with 5 bad rows.
   - Therefore the most informative next step is unused fresh-source confirmation, not another label-density sweep and not validation/test.

2. Executed one bounded metadata-only experiment:
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2.py`
   - Registry: `research_artifacts/aws_runs/20260929T111348_2d45546c/registry.json`
   - Exit: success
   - Budget actual: 0 simulations, 0 control steps, 0 training/refit/gradient steps, 0 validation64, 0 sealed test.

3. Frozen v2 protocol:
   - Protocol: `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json`
   - Summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_20260929T1120Z/summary.md`
   - Completed: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_20260929T1120Z/completed.json`
   - Candidate: `source_only::v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only`
   - Primary scope: `shared_h15_terminal`; matched-terminal remains secondary diagnostic only.
   - Selected unused source IDs: `[44, 26, 239, 159, 146, 203, 251, 12]`
   - Future run budget: 136 development episodes, cap 20,400 control steps, no training/refit/validation/test.

4. Wrote v2 runner source, not executed yet:
   - `experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v2_runner.py`
   - sha256: `dabb3c582c1600674b47ddeb78214c02616150a85f6e3564b4d8d32a7e9cf866`

5. Preserved state and backup request:
   - State: `research_artifacts/aws_state/continue_state_20260929T1115_after_freeze_v2_and_runner_source.md`
   - Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_FRESH_SOURCE_CONFIRMATION_V2_FREEZE_AND_RUNNER_20260929T1115Z.json`

Safety/access status:
- Sealed test: not accessed.
- validation64 for revised selector: not opened.
- No mobile-robot MPPI work resumed.
- Current branch remains development-only IMPROVED evidence, not ORIGINAL SAC reproduction.

Next required action:
1. Verify external backup covering the v2 freeze outputs and v2 runner source.
2. Then run:
   `experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v2_runner.py --run --backup-verified-commit <verified_commit> --i-accept-development-fresh-source-confirmation-v2`
3. If v2 strong-passes, freeze validation64 with fair fixed true-H baselines and actual timing. If weak/fails, move to targeted terminal-value/objective/representation or bounded training/refit ablation rather than another unchanged selector sweep.

<!-- vehicle-true-variable-H-fresh-source-confirmation-v2-run-20260929T1135Z -->
## 2026-09-29 vehicle true-variable-H fresh-source confirmation v2 run

UTC: 2026-09-29T11:30:44.849870+00:00. Development-only unused fresh-source confirmation completed for `source_only::v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only`: 136 episodes (8 H15 traces + 128 blocked H10/H15 branches), 6355 control steps. No validation64, no sealed test, no training/refit. Primary gate: {'scope': 'shared_h15_terminal', 'groups': 16, 'h10_predictions': 7, 'h15_predictions': 9, 'nonconstant_horizons': True, 'confusion_vs_fresh_h10_beneficial_label': {'TP': 5, 'FP': 2, 'FN': 9, 'TN': 0}, 'false_positive_rows': [{'base_state_id': 'fresh_case04_slot0_early_mid', 'fresh_confirmation_group': 'fresh_high_heading_short', 'h10_safe_all': True, 'h15_safe_all': True, 'physical_delta_h10_minus_h15': 136.9152309816722, 'decision_delta_h10_minus_h15': -0.8018156842736062, 'row_tolerance': 2.0, 'diagnostics': {'nearest_pos': 0.3183019756765563, 'nearest_neg': 0.6693482442856566, 'nearest_veto': 0.5105084217465663, 'support': 4, 'radius': 0.6647337186896707, 'support_ok': True, 'neg_ok': True, 'veto_ok': True, 'reason': 'h10'}, 'catastrophic_threshold': 3.8139257703814136}, {'base_state_id': 'fresh_case05_slot0_early_mid', 'fresh_confirmation_group': 'fresh_low_heading_low_clearance_control', 'h10_safe_all': True, 'h15_safe_all': True, 'physical_delta_h10_minus_h15': 132.65816547206765, 'decision_delta_h10_minus_h15': -0.9841039822786115, 'row_tolerance': 2.0, 'diagnostics': {'nearest_pos': 0.3599241243371876, 'nearest_neg': 0.7252531108911741, 'nearest_veto': 0.3686667415340344, 'support': 4, 'radius': 0.6647337186896707, 'support_ok': True, 'neg_ok': True, 'veto_ok': True, 'reason': 'h10'}, 'catastrophic_threshold': 4.184115892836806}], 'catastrophic_false_positive_rows': [{'base_state_id': 'fresh_case04_slot0_early_mid', 'fresh_confirmation_group': 'fresh_high_heading_short', 'h10_safe_all': True, 'h15_safe_all': True, 'physical_delta_h10_minus_h15': 136.9152309816722, 'decision_delta_h10_minus_h15': -0.8018156842736062, 'row_tolerance': 2.0, 'diagnostics': {'nearest_pos': 0.3183019756765563, 'nearest_neg': 0.6693482442856566, 'nearest_veto': 0.5105084217465663, 'support': 4, 'radius': 0.6647337186896707, 'support_ok': True, 'neg_ok': True, 'veto_ok': True, 'reason': 'h10'}, 'catastrophic_threshold': 3.8139257703814136}, {'base_state_id': 'fresh_case05_slot0_early_mid', 'fresh_confirmation_group': 'fresh_low_heading_low_clearance_control', 'h10_safe_all': True, 'h15_safe_all': True, 'physical_delta_h10_minus_h15': 132.65816547206765, 'decision_delta_h10_minus_h15': -0.9841039822786115, 'row_tolerance': 2.0, 'diagnostics': {'nearest_pos': 0.3599241243371876, 'nearest_neg': 0.7252531108911741, 'nearest_veto': 0.3686667415340344, 'support': 4, 'radius': 0.6647337186896707, 'support_ok': True, 'neg_ok': True, 'veto_ok': True, 'reason': 'h10'}, 'catastrophic_threshold': 4.184115892836806}], 'unsafe_rows': [], 'physical_tolerance_sum_vs_fixed_H15': 32.0, 'physical_gate': False, 'fixed_H15_physical_sum': 114.41260760169516, 'comparison_policy_vs_fixed_H15': {'physical_delta_vs_fixed_H15': 269.6448236443954, 'decision_relative_saving_vs_fixed_H15': 0.11059728363795457, 'solver_relative_saving_vs_fixed_H15': 0.11713906602512027}, 'comparison_oracle_vs_fixed_H15': {'physical_delta_vs_fixed_H15': -1.6660502167614197, 'decision_relative_saving_vs_fixed_H15': 0.24975081536769275, 'solver_relative_saving_vs_fixed_H15': 0.2688011200020139}, 'oracle_chosen_counts': {'10': 14, '15': 2}, 'policy_chosen_counts': {'15': 9, '10': 7}, 'decision_saving_gate_5pct': True, 'decision_saving_gate_10pct_strong': True, 'primary_pass_5pct': False, 'primary_strong_10pct': False, 'solver_saving_reported_not_primary': 0.11713906602512027}. Decision: v2 independent fresh-source confirmation failed safety/physical gates; block selector rollout and prioritize value/objective/representation refit or training ablation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/completed.json`.


## 2026-09-29T11:32:40.356032+00:00
Elapsed service lifetime: ~72.7 h since 2026-09-26T10:55:29.419331Z.  
Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; not estimated and excludes this desktop/API conversation.

Concrete actions completed:

1. **Verified/inspected frozen v2 protocol and runner**
   - Protocol: `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fresh_source_confirmation_freeze_v2_frozen_20260929T1120Z.json`
   - Runner: `experiments/bohn2021_aws/vehicle_true_variable_horizon_fresh_source_confirmation_v2_runner.py`
   - Used supervisor-provided verified backup commit before running: `e31dfe9ae30a8e5d0656a2e545124dc490860807`.

2. **Executed the frozen development-only v2 fresh-source confirmation**
   - Registry: `research_artifacts/aws_runs/20260929T111808_0f8e0897/registry.json`
   - Run dir: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fresh_source_confirmation_v2_run_20260929T1135Z/`
   - Exit: success
   - Budget actual:
     - 136 development episodes
     - 6,355 control steps
     - 0 training episodes
     - 0 gradient steps
     - 0 selector refit steps
     - 0 validation64 episodes
     - 0 sealed-test episodes
   - Access status: validation64 unopened; sealed final test unopened.

3. **Scientific result: v2 selector failed physical/safety-performance gate**
   - Candidate: `source_only::v2b_raw_abs_l2_prs1.25_nm1.5_vm0.75_agreement_only_disagreement_only`
   - Primary profile: shared-H15 terminal.
   - The selector was nonconstant and achieved measured compute savings, but produced catastrophic H10 false positives:
     - H10 predictions: 7 / 16 states
     - H15 abstentions: 9 / 16 states
     - Confusion vs fresh H10-beneficial labels: TP=5, FP=2, FN=9, TN=0
     - Policy vs fixed H15:
       - Physical delta: **+269.645**, tolerance sum 32.0 → **physical gate failed**
       - Whole-decision saving: **11.06%**
       - Solver saving: **11.71%**
     - Strong/weak pass: both **false** because physical gate failed.

4. **Failure anatomy preserved**
   - Catastrophic false positives:
     1. `fresh_case04_slot0_early_mid`
        - H10 physical ≈ 152.171 vs H15 ≈ 15.256
        - Delta ≈ +136.915
        - Decision saving ≈ 0.802 s
        - Both safe/successful; solver failures 0.
     2. `fresh_case05_slot0_early_mid`
        - H10 physical ≈ 149.395 vs H15 ≈ 16.737
        - Delta ≈ +132.658
        - Decision saving ≈ 0.984 s
        - Both safe/successful; solver failures 0.

   These are real control/compute tradeoff failures, not solver failure, unsafe termination, or early-termination artifacts.

5. **Important positive diagnostic evidence**
   - Same fresh block still has strong adaptive opportunity under an oracle:
     - Oracle H10 choices: 14 / 16
     - Oracle physical delta vs fixed H15: **−1.666**
     - Oracle whole-decision saving: **24.98%**
     - Oracle solver saving: **26.88%**
   - Interpretation: the scenario/branch states do contain useful state-dependent H10/H15 opportunities. The current bottleneck is the selector/objective/representation, not absence of adaptive-horizon opportunity.

6. **State preserved**
   - New state file:
     - `research_artifacts/aws_state/continue_state_20260929T1135_after_fresh_source_confirmation_v2_run_inspected.md`
   - Durable supervisor state updated via `update_state`.
   - Backup request written by runner:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_VEHICLE_TRUE_VARIABLE_HORIZON_FRESH_SOURCE_CONFIRMATION_V2_RUN_20260929T113044.849870+0000.json`

Decision:

- **Do not advance this v2c selector to validation64.**
- Treat v2 as development-only negative evidence.
- Next informative action after verified backup is not another unchanged label-density sweep. It should be a no-simulation opened-bank failure-aware analysis/refit design over v0/v1/v2, specifically testing whether richer pre-decision features/objectives can eliminate catastrophic H10 false positives while retaining measured compute savings.
- If opened-bank leave-one-bank-out analysis cannot find a robust zero-catastrophic-FP rule with adequate timing savings, move to bounded value/objective/representation retraining or refit ablation before any validation rollout.

Current blocker for further simulation/training:

- Need externally verified backup covering the new v2 run outputs and state. No user input is needed unless backup infrastructure fails repeatedly.

## vehicle-true-variable-H-failure-aware-selector-refit-v3b-compact-lobo-20260929T113906Z

Development-only failure-aware selector refit/LOBO diagnostic over opened fresh_v0/v1/v2 banks. No simulations, no validation64, no sealed test, no gradient training.

Headline: best deployable strong pass = `False`, strong deployable candidates = `0`, best variant = `v3_deploy_obs_state_step_risk_std_primary_only_negall_nonpositive_vetocatastrophic_only_s2_q0.75_rs1_nm1_vm0.75`, decision = No deployable-feature selector class passed opened-bank LOBO strong gates; prioritize bounded value/objective/representation retraining or terminal-value refit rather than another nearest-neighbour threshold sweep..

Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_failure_aware_selector_refit_v3b_compact_lobo_20260929T113906Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_failure_aware_selector_refit_v3b_compact_lobo_20260929T113906Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_failure_aware_selector_refit_v3b_compact_lobo_20260929T113906Z/completed.json`.


## 2026-09-29T11:59:35.225998+00:00



## 2026-09-29T12:04:05.486819+00:00



## 2026-09-29T12:09:52.932665+00:00


- 2026-09-29T12:11:15.604110+00:00 `vehicle_true_variable_horizon_delta_value_model_v4_lobo`: closed-form delta/risk value LOBO over opened fresh_v0/v1/v2; strong=0, weak=0, best_min_save=0.03178, bad=0; no validation64/test/simulation; artifacts `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_delta_value_model_v4_lobo_20260929T121112Z/summary.md`.


## 2026-09-29T12:14:49.677322+00:00
- 2026-09-29T12:15:33.708169+00:00 `vehicle_true_variable_horizon_terminal_causality_fresh_v5`: fresh-bank terminal-causality diagnostic; shared_cat=6, resolved_by_matched=1, persistent=5, flips=13; no simulation/validation/test; artifacts `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_terminal_causality_fresh_v5_20260929T121533Z/summary.md`.


## 2026-09-29T12:19:07.562580+00:00
- 2026-09-29T12:22:23.348640+00:00 `vehicle_true_variable_horizon_risk_tree_representation_v6`: nonlinear risk-tree representation diagnostic; deploy_strong=0, deploy_weak=0, best_deploy_min_save=0.1439, best_deploy_bad=1; no simulation/validation/test; artifacts `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_tree_representation_v6_20260929T122023Z/summary.md`.


## 2026-09-29T12:27:30.456846+00:00

<!-- vehicle-true-variable-H-history-representation-v7-20260929T122843Z -->
## 2026-09-29 true-variable-H history/static representation v7

UTC: 2026-09-29T12:49:03.130935+00:00. Development-only opened-fresh-bank diagnostic, no simulations/control steps, no validation64, no sealed test, no gradient training. Headline: `{"best_online_history_bad": 1, "best_online_history_min_save": 0.1439090695782106, "best_online_history_variant": "v7_online_obs_pose_baseline_recomputed_d1_ml2_mp1_mc0_mg0", "best_same_step_bad": 3, "best_same_step_min_save": 0.12353000768917985, "best_same_step_variant": "v7_diagnostic_same_step_h15_solve_history_static_d1_ml2_mp1_mc0_mg0", "best_variant": "v7_online_obs_pose_baseline_recomputed_d1_ml2_mp1_mc0_mg0", "families": 5, "online_history_strong_candidates": 0, "online_history_weak_candidates": 0, "rows": 48, "same_step_strong_candidates": 0, "tree_fits": 3240, "tree_variants": 1080}`. Decision: History/static deployable features still fail opened-bank weak/strong gates. Prioritize targeted source-independent risk acquisition or value-function training/refit with richer risk/clearance representation rather than another selector-only sweep.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_representation_v7_lobo_20260929T122843Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_representation_v7_lobo_20260929T122843Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_representation_v7_lobo_20260929T122843Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_HISTORY_REPRESENTATION_V7_20260929T122843Z.json`.


## 2026-09-29T12:51:34.623707+00:00



## 2026-09-29T12:56:59.705244+00:00



## 2026-09-29T13:00:26.125847+00:00



## 2026-09-29T13:03:00.292196+00:00
Elapsed service lifetime: >73.9 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; not estimated; excludes desktop/API conversation.

Concrete action completed this iteration:

- Verified the latest supervisor-provided backup was externally recoverable before attempting new work:
  - commit `9045afb78cda6f6406859c23c4c9649ac77e0d79`
  - package sha256 `831ae6d59b7831577a7bdd7c6574d4c72d6d3d36dcc993a35b96e13803b5e04e`
  - `remaining_changed_files=0`
- Attempted the prepared v8 diagnostic:
  - `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8.py`
- The run failed immediately before any simulation because the script requires `--backup-verified-commit`.
  - Failed registry: `research_artifacts/aws_runs/20260929T130126_7323a6e1/registry.json`
  - Exit status: `2`
  - Runtime: ~2.0 s
- Prepared the corrected invocation, but the one-experiment-per-iteration guard correctly blocked a second run in the same bounded cycle.

Safety/access accounting:

- New development simulation episodes: `0`
- New control steps: `0`
- New training episodes / gradient steps / refit steps: `0`
- Validation64 opened: `false`
- Sealed test opened: `false`
- Mobile-robot MPPI resumed: `false`
- No scientific result changed; this was an invocation/protocol-gate failure only.

State preserved:

- Wrote handoff:
  - `research_artifacts/aws_state/continue_state_20260929T1302_after_v8_arg_failure.md`
  - sha256 `af751f1b71a860c0642e639ed1dc0c3e6afd58817a03db281c24db869f47c198`
- Wrote backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V8_ARG_FAILURE_20260929T1302Z.json`
  - sha256 `ba7075650e953b9ad19bb940b0ded93ebfbcddee3fdafd97229610c4d333f538`
- Persisted next action via `update_state`.

Next concrete action:

1. Verify/request external backup covering the failed v8 registry and new handoff files.
2. Then run corrected v8 exactly once:

```text
experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8.py
--run
--backup-verified-commit <latest_verified_backup_commit>
--i-accept-development-risk-probe-v8
```

Purpose remains unchanged: bounded development-only H10/H15 risk-probe acquisition to decide between terminal-value/objective refit, online risk/representation/training smoke, or a scenario-opportunity/design conclusion. No validation64 or sealed final-test access is authorized for this step.


## 2026-09-29T13:07:08.556417+00:00



## 2026-09-29T13:11:57.851419+00:00



## 2026-09-29T13:15:29.292363+00:00
Elapsed service lifetime: >74.3 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; not estimated and excludes this desktop/API conversation.

Concrete progress this iteration:

- Tried the required external backup before running the prepared v8b risk-probe diagnostic.
  - `scripts/research_service/backup.py`
  - Registry: `research_artifacts/aws_runs/20260929T131215_23fa83ef/registry.json`
  - Failed with exit status 1 after ~36 s; stdout/stderr empty.
  - No simulations, no control steps, no training/refit, no validation64 access, no sealed-test access.

- Inspected the backup registry and backup script behavior.
  - `backup.py` records the detailed sanitized exception into supervisor state outside the repository, so the repo-visible registry alone cannot explain the failure.
  - This is an infrastructure/backup blocker, not a scientific result.

- Wrote a zero-budget metadata-capture script for the next iteration:
  - `experiments/bohn2021_aws/backup_failure_status_capture_v8b_v0.py`
  - sha256: `3ae47ca4aa134b46a894c0e76f0f1b5e2388da6e2f37baad4e8a0f770f6f4ad3`
  - Purpose: safely copy sanitized backup status from supervisor state into repository evidence, without simulations/training/refit/validation/test or secrets.

- Attempted to run that metadata-capture script, but the tool layer enforced “one experiment per iteration,” so it was not run. I persisted the next step instead.

- Persisted handoff:
  - `research_artifacts/aws_state/continue_state_20260929T1316_after_backup_failure_before_v8b_capture.md`

Current blocker:

- Latest verified external backup is still:
  - Time: `2026-09-29T13:07:51.958731+00:00`
  - Commit: `3f47aff3df355ffecba9eb17d705ffdcae6fa1e1`
  - Package: `20260929T130748_fe2bead3.tar.gz`
  - SHA256: `470c391fb3944bbc5678673b3b291b48ade0c5fee495d39a6cf7c634bdf1f54c`
- New v8 failure evidence, v8b source, the failed backup registry, the new backup-capture source, and this state are not yet externally backed up.
- Therefore v8b simulations remain blocked until backup is verified or the backup failure is diagnosed/resolved.

Next concrete action:

1. Run the metadata-only capture:
   - `experiments/bohn2021_aws/backup_failure_status_capture_v8b_v0.py`
   - modern interpreter
   - args: `--i-accept-no-simulation-backup-status-capture`
   - all budgets zero.

2. If it confirms backup is still inadequate, diagnose/repair or retry `scripts/research_service/backup.py` before any further scientific simulations.

3. Once backup is verified, run:
   - `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry.py`
   - legacy interpreter
   - with `--run --backup-verified-commit <verified_commit> --i-accept-development-risk-probe-v8`.

No validation64 or sealed test was accessed. Research is not complete.


## 2026-09-29T13:21:04.633423+00:00
Elapsed service lifetime: >74.4 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; do not estimate; excludes desktop/API conversation.

Concrete actions completed this iteration:

- Ran the prepared v8b true-variable-H risk-probe diagnostic under the required legacy Python/TF1 runtime after verified backup commit `6226c4882a07f5d39c66f3ffd7a7bb36436a74fe`.
  - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry.py`
  - Registry: `research_artifacts/aws_runs/20260929T131628_ec8e3d21/registry.json`
  - Result: failed after ~68 s with:
    - `ContractError('expected 16 selected branch states, got 8')`
  - Failure artifact:
    - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8b_legacy_retry_20260929T1312Z/failed.json`

- Inspected the failure evidence.
  - `progress.json` shows v8b completed Stage-A only:
    - 4 development H15 trace episodes
    - 300 control steps
    - no Stage-B H10/H15 paired continuation comparisons
    - no training/refit
    - no validation64 access
    - no sealed-test access
  - The issue is an execution-contract bug, not a scientific negative result.
  - Diagnosis: v8 was intentionally designed for 4 fresh cases × 2 branch states/case = 8 branch states, but it reused a helper from an earlier 8-case protocol that hard-coded an expected total of 16 selected states.

- Prepared a narrow versioned repair:
  - `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count.py`
  - sha256: `c9c4ca8923afdf8ac41e5d32287fba8d3269ec019e79991a108a1b1846b89296`
  - Repair scope:
    - preserves v8/v8b evidence;
    - uses a fresh output namespace;
    - keeps case selection, risk scoring, selection windows, branch template, and scientific protocol unchanged;
    - only generalizes the stale selected-state count assertion from `16` to `len(stage_a_episodes) * branch_states_per_case`.

- Persisted state and backup request:
  - State:
    - `research_artifacts/aws_state/continue_state_20260929T1320_after_v8b_failure_v8c_prebackup.md`
  - Backup request:
    - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V8B_PARTIAL_FAILURE_BEFORE_V8C_20260929T1320Z.json`
  - Also updated durable supervisor state via `update_state`.

Budget consumed this iteration:

- Development simulation episodes: 4  
- Development control steps: 300  
- Training episodes / gradient steps / selector refit: 0  
- validation64 episodes: 0  
- sealed-test episodes: 0  

Current status:

- Latest verified backup before v8b was commit `6226c4882a07f5d39c66f3ffd7a7bb36436a74fe`.
- New v8b partial evidence, v8c repair source, and the latest state/backup-request files are not yet externally backed up.
- Therefore the next scientific simulation is blocked until backup is verified.

Next concrete action:

1. Run/verify external backup covering the new v8b/v8c artifacts.
2. Then run v8c under the legacy interpreter:

```bash
python experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count.py \
  --run \
  --backup-verified-commit <new_verified_commit> \
  --i-accept-development-risk-probe-v8
```

3. If v8c completes, use its frozen decision rule to choose the next bounded intervention: risk/value representation training or value-refit if fresh H10 catastrophic modes reproduce; otherwise reassess scenario/control-compute opportunity.

No validation64 or sealed final test was accessed. Research is not complete.


## 2026-09-29T13:24:20.620733+00:00
Elapsed service lifetime: >74.5 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; do not estimate; excludes desktop/API conversation.

Concrete actions completed this iteration:

- Attempted the required external backup before running the repaired v8c risk-probe simulation.
  - Script: `scripts/research_service/backup.py`
  - Registry: `research_artifacts/aws_runs/20260929T132120_54f69c3b/registry.json`
  - Result: failed after ~36.0 s, exit status `1`, with empty stdout/stderr.
  - Scientific effect: none — no simulation, no training/refit, no validation64 access, no sealed-test access.

- Inspected the failed backup evidence and backup script.
  - Confirmed this is an infrastructure/backup-gate issue, not a scientific result.
  - Did not read or log credentials/secrets.
  - Latest verified backup remains the earlier commit `6226c4882a07f5d39c66f3ffd7a7bb36436a74fe`; it does not cover the newest failed-backup record and newly written capture script.

- Prepared a metadata-only backup-status capture for the next bounded cycle:
  - `experiments/bohn2021_aws/backup_failure_status_capture_after_v8c_prebackup_v0.py`
  - sha256: `e767aa1ba9968db937ef12298b16ae4e1990c418eea637e123d57757c1f1ba8c`
  - Purpose: sanitize and record `/data/openai-agent/state/backup_status.json` and decide whether backup is adequate despite the silent backup failure.
  - It is explicitly no-simulation/no-training/no-validation/no-test.
  - I attempted to run it, but the tool layer enforced the one-experiment-per-iteration limit after the backup attempt, so it is queued for the next cycle.

- Persisted handoff state and backup retry request:
  - `research_artifacts/aws_state/continue_state_20260929T1326_after_backup_failure_before_v8c.md`
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_RETRY_AFTER_V8C_PREBACKUP_FAILURE_20260929T1326Z.json`
  - Updated durable supervisor state via `update_state`.

Budget consumed this iteration:

- Development simulation episodes: 0
- Development control steps: 0
- Training episodes / gradient steps / selector refit: 0
- validation64 episodes: 0
- sealed-test episodes: 0
- Backup attempts: 1
- New source files written: 1 metadata-only infrastructure script

Current blocker:

- Further scientific simulation/training/refit is blocked until external backup coverage is verified for:
  - v8b partial evidence,
  - v8c repair source,
  - the failed backup attempt record,
  - the new backup-status capture script and handoff files.

Next concrete action:

1. Run the metadata-only capture script:
   ```bash
   python experiments/bohn2021_aws/backup_failure_status_capture_after_v8c_prebackup_v0.py \
     --i-accept-no-simulation-backup-status-capture
   ```

2. If it shows backup is inadequate, retry/fix `scripts/research_service/backup.py` without changing IAM/resources or exposing credentials.

3. Once verified backup is available, run v8c under the legacy interpreter:
   ```bash
   python experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count.py \
     --run \
     --backup-verified-commit <new_verified_commit> \
     --i-accept-development-risk-probe-v8
   ```

No validation64 or sealed final test was accessed. Research is not complete.

<!-- vehicle-true-variable-H-risk-probe-acquisition-v8c-flexible-state-count-20260929T1320Z -->
## 2026-09-29 vehicle true-variable-H targeted risk-probe acquisition v8

UTC: 2026-09-29T13:28:53.781213+00:00. Development-only fresh-source risk probe completed: 36 episodes, 1596 control steps, validation64 closed, sealed test closed, no training/refit. Catastrophic H10 rows: 2; early catastrophics: 1; beneficial H10 rows: 6. Decision: adaptive opportunity exists but requires risk-aware value/representation learning before selector rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/completed.json`.


## 2026-09-29T13:30:37.065233+00:00
Elapsed service lifetime: >74.6 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; do not estimate; excludes desktop/API conversation.

Concrete actions completed this iteration:

- Ran the queued v8c true-variable-H risk-probe diagnostic after verified backup coverage.
  - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count.py`
  - Interpreter: legacy Python/TF1 runtime
  - Registry: `research_artifacts/aws_runs/20260929T132516_01401074/registry.json`
  - Exit status: `0`
  - Runtime: ~220.1 s
  - Development budget: 36 episodes / 1596 control steps
  - Training/refit: 0
  - validation64 access: false
  - sealed-test access: false

Key evidence inspected from new v8c outputs:

- Output summary:
  - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/summary.md`
- Completion/hash record:
  - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_probe_acquisition_v8c_flexible_state_count_20260929T1320Z/completed.json`

v8c result:

- 8 fresh branch states probed.
- H10 beneficial on 6 / 8 states.
- H10 catastrophic on 2 / 8 states.
- Fixed H15 physical cost sum: `58.98`
- Fixed H10 physical cost sum: `562.66`
- Fixed H10 measured decision time: `15.46 s`
- Fixed H15 measured decision time: `19.00 s`
- Fixed H10 measured decision-time saving vs H15: `18.6%`, but with unacceptable catastrophic physical failures.
- Oracle over these branch states:
  - Physical cost sum: `59.49`
  - Decision time: `15.85 s`
  - Measured decision-time saving vs H15: `16.6%`

Scientific interpretation:

- This is not merely a label-density issue.
- v8c shows a real control-vs-measured-compute opportunity from identical branch states: many states can safely use H10 and save actual decision time.
- But fixed H10 is not acceptable because low-clearance/risk states caused catastrophic physical degradation.
- The next informative direction is therefore a risk-aware value/representation selector/refit, not another unchanged threshold/nearest-neighbor/label-density sweep.

Persisted state:

- Wrote handoff:
  - `research_artifacts/aws_state/continue_state_20260929T1332_after_v8c_risk_probe.md`
- Updated durable supervisor state via `update_state`.

Current backup state:

- Latest verified backup before v8c:
  - Commit: `7e530cb17ad32a4d0c8d2cc881fd623c5a6438a0`
  - Package: `20260929T132500_c9d50dc3.tar.gz`
  - SHA256: `5a19fe190b52278d37e5d2db97a89261a43f8159aae19f7aff29397b0645e333`
- New v8c outputs and this handoff are not yet covered by a verified backup.
- Further scientific simulation/training/refit should wait until backup verification covers these new artifacts.

Next concrete action:

1. Verify external backup for the v8c outputs and new state.
2. Then implement/freeze/run a bounded IMPROVED v9 development-only risk-aware selector/refit smoke.
3. v9 hypothesis: a conservative risk-aware selector can choose H10 on safe branch states and H15 on catastrophic low-clearance states, preserving near-H15 physical performance while reducing measured decision time.
4. No validation64 or sealed final test access until an independently justified gate is met.

Research is not complete.


## 2026-09-29T13:41:05.554996+00:00
- 2026-09-29T13:48:19.461516+00:00 `vehicle_true_variable_horizon_risk_refit_v9_opened_banks`: opened-bank risk-aware finite selector refit including v8c; pure_deploy_pass10=0, pure_deploy_pass5=0, best_deploy=v9_deploy_obs_pose_step_no_risk_d1_ml2_mp1_mc0_mg0 min_save=0.0594 bad=3 v8c_bad=2; no simulation/validation/test; artifacts `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_refit_v9_opened_banks_20260929T1345Z/summary.md`.


## 2026-09-29T13:50:27.198415+00:00



## 2026-09-29T14:01:53.332829+00:00



## 2026-09-29T14:07:09.388193+00:00



## 2026-09-29T18:14:52.589056+00:00

<!-- vehicle-true-variable-H-risk-value-fast-refit-v10b-20260929T1815Z -->
## 2026-09-29 vehicle true-variable-H v10b fast risk/value refit

UTC: 2026-09-29T18:20:01.772872+00:00. Development-only opened-bank supervised/nonparametric refit after v10 timeout; no MPC simulation, no validation64, no sealed test. configs=1152; global_pass5=0; nested_save=0.0843; nested_bad=2; decision: Deployable risk/value refit still permits catastrophic H10 false positives; next freeze targeted risk-data acquisition around ambiguous catastrophic/near-safe states or richer terminal/risk value training, not validation rollout.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/completed.json`.


## 2026-09-29T18:20:49.873884+00:00
Elapsed service lifetime: >79.4 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; do not estimate; excludes desktop/API conversation.

Concrete work completed this iteration:

- Checked for Astra reviewer handoff: `docs/bohn2021_takeover/astra_reviews/LATEST.md` is not present, so no external-review recommendations were available to verify/action.
- Used the verified backup from supervisor context before running science:
  - commit `28446b7bb8e64ff83bfb07170ad3ad4a36f83ee7`
  - backup time `2026-09-29T18:15:45.561429+00:00`
  - `remaining_changed_files: 0`
- Ran the bounded development-only refit:
  - script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_value_fast_refit_v10b.py`
  - registry: `research_artifacts/aws_runs/20260929T181625_f8de167a/registry.json`
  - exit status: `0`
  - runtime: `218.1 s`
  - no MPC simulation, no training, no validation64, no sealed-test access.

Scientific result from v10b:

- Opened development rows: `56`
- Banks: `fresh_v0`, `fresh_v1`, `fresh_v2`, `fresh_v8c`
- Positive rows: `48`
- Catastrophic H10 rows: `8`
- Configs evaluated: `1152`
- Selector/refit evaluations: `18436`
- Global LOBO:
  - pass10 configs: `0`
  - pass5 configs: `0`
  - best global config had zero catastrophic false positives but only `1.20%` minimum decision saving, below the weak 5% gate.
- Nested opened-bank selection:
  - decision saving vs fixed H15: `8.43%`
  - H10 chosen rows: `20`
  - catastrophic H10 false positives: `2`
  - physical gate: `False`
  - pass5: `False`
  - pass10: `False`

Interpretation:

- v10b provides further negative evidence against simple deployable tree/kNN/poly risk-value selectors on the current 56 opened rows.
- There is still branch-level compute opportunity, but current deployable representations cannot safely separate beneficial H10 from catastrophic H10 under nested bank holdout.
- This supports the current hypothesis that the blocker is risk/terminal-value representation and/or missing targeted boundary examples, not simply sparse labels or absence of a control-compute tradeoff.
- This does **not** justify validation64 rollout or sealed-test access.

Artifacts produced:

- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_20260929T1815Z/completed.json`
- `research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_value_fast_refit_v10b_preoutcome_frozen_20260929T1815Z.json`
- `research_artifacts/aws_state/continue_state_20260929T1815_after_risk_value_fast_refit_v10b.md`
- `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_RISK_VALUE_FAST_REFIT_V10B_20260929T1815Z.json`

Budget/accounting this iteration:

- development MPC simulation episodes: `0`
- development control steps: `0`
- training episodes: `0`
- gradient steps: `0`
- selector/refit evaluations: `18436`
- validation64 episodes: `0`
- sealed-test episodes: `0`

Current blocker / next safe action:

- New v10b artifacts and doc/registry updates are not yet externally verified.
- Per project policy, next iteration must first run/verify backup coverage for v10b before any further scientific simulation/training/refit.
- After backup, the next scientific action should be a frozen v11 targeted risk-data acquisition or richer terminal/risk-value refit focused on ambiguous catastrophic/near-safe H10 boundary states, using development-only sources and no validation64/sealed-test access.


## 2026-09-29T18:26:56.080559+00:00

<!-- vehicle-true-variable-H-risk-boundary-acquisition-v11-20260929T1835Z -->
## 2026-09-29 vehicle true-variable-H targeted risk-boundary acquisition v11

UTC: 2026-09-29T18:34:06.725804+00:00. Development-only source-independent risk-boundary acquisition completed: 54 episodes, 2929 control steps, validation64 closed, sealed test closed, no training/refit. Catastrophic H10 rows: 2; early catastrophics: 1; beneficial H10 rows: 8. Decision: adaptive opportunity exists but requires risk-aware value/representation learning before selector rollout. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/completed.json`.


## 2026-09-29T18:35:48.219567+00:00
Elapsed service lifetime: >79.6 h since 2026-09-26T10:55:29.419331Z.  
Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; not estimated; excludes desktop/API conversation.

Concrete actions completed this iteration:

1. **Astra review check**
   - Checked `docs/bohn2021_takeover/astra_reviews/LATEST.md`.
   - It does not exist, so there were no Astra recommendations to verify or log.

2. **Inspected v11 source**
   - Read `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_boundary_acquisition_v11.py`.
   - Confirmed it is a development-only IMPROVED targeted risk-boundary acquisition wrapper around the audited v8 true-H10/H15 branch runner.
   - Confirmed intended split restrictions: no validation64 and no sealed test access.

3. **Ran v11 targeted risk-boundary acquisition**
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_risk_boundary_acquisition_v11.py`
   - Interpreter: legacy
   - Registry: `research_artifacts/aws_runs/20260929T182803_0d6ab828/registry.json`
   - Exit status: 0
   - Runtime: ~368 s
   - Budget used:
     - 54 development-only MPC simulation episodes
     - 2929 control steps
     - 0 training episodes
     - 0 gradient steps
     - 0 selector refit evaluations
     - 0 validation64 episodes
     - 0 sealed-test episodes
   - Access flags:
     - `validation64_bank_opened=false`
     - `sealed_test_accessed=false`

4. **v11 scientific result**
   - Summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/summary.md`
   - Completed marker: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_20260929T1835Z/completed.json`
   - Protocol: `research_artifacts/aws_protocols/vehicle_true_variable_horizon_risk_boundary_acquisition_v11_preoutcome_frozen_20260929T1835Z.json`
   - Headline:
     - 12 branch states from 6 fresh source-independent cases.
     - H10-beneficial rows: 8/12.
     - Catastrophic H10 rows: 2/12.
     - Early catastrophic rows: 1/12.
     - Risk-role catastrophics: 0.
     - Oracle decision saving vs fixed H15: 12.51%.
     - Fixed H10 decision saving vs H15: 55.23%, but unsafe at the per-state level because of catastrophic H10 rows.
   - Interpretation:
     - v11 strengthens evidence that true variable-horizon branch-level compute opportunity exists.
     - The remaining blocker is safe deployable selection/risk-value representation, not absence of adaptive opportunity.
     - Catastrophic H10 can occur outside the originally targeted high-risk roles, so the next method must be conservative and explicitly learn/penalize catastrophic false positives.

5. **Backup status**
   - Latest verified backup before v11:
     - Commit: `43c983f1e0d0d43bc0166f1b5153390b71e7218a`
     - Package: `20260929T182736_556bd907.tar.gz`
     - SHA256: `2c3b32d3cf642a8ae4eb005d96ad8d77df2185a8b456b04b17327594fc780cce`
   - v11 artifacts were created after that backup and are not yet externally backed up.
   - I attempted to run `scripts/research_service/backup.py`, but the one-experiment-per-iteration guard blocked it:
     - `ValueError: One experiment per iteration: persist next action for the next bounded cycle`

6. **State preservation**
   - Wrote:
     - `research_artifacts/aws_state/continue_state_20260929T1836_after_v11_unbacked.md`
   - Updated durable supervisor state with:
     - v11 result summary
     - current backup blocker
     - next queue
     - explicit prohibition on further science until backup verifies.

Current blocker: **external backup is not yet verified after v11 artifacts and the new continue-state file.**  
Next action must be to run `scripts/research_service/backup.py` before any further scientific simulation, training, selector refit, or validation.

Next scientific action after backup: freeze/implement a bounded IMPROVED v12 risk/value representation refit using opened development banks including v11, with nested/LOBO gates and no validation64/sealed-test access.

<!-- vehicle-true-variable-H-outcome-aligned-refit-v12-20260929T1855Z -->
## 2026-09-29 vehicle true-variable-H v12 outcome-aligned refit

UTC: 2026-09-29T20:41:46.782259+00:00. Development-only IMPROVED relabel/refit over opened banks including v11; no MPC simulation, no validation64, no sealed test, no gradient training. label_flips=2; aligned_oracle_save=0.3080; nested_deploy_save=0.0674, nested_deploy_bad=5, nested_deploy_pass5=False; nested_all_save=0.1300, nested_all_bad=1. Decision: Objective/label mismatch is verified but deployable selector still fails nested gate; next intervention should train/refit richer deployable risk/value representation using the aligned objective, not validation rollout.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_outcome_aligned_refit_v12_20260929T1855Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_outcome_aligned_refit_v12_20260929T1855Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_outcome_aligned_refit_v12_20260929T1855Z/completed.json`.


## 2026-09-29T20:42:19.530496+00:00



## 2026-09-29T20:48:54.312054+00:00



## 2026-09-29T21:55:03.866674+00:00

<!-- vehicle-true-variable-H-history-feature-refit-v13b-20260929T2150Z -->
## 2026-09-29 vehicle true-variable-H v13b compact/cached history-feature refit

UTC: 2026-09-29T21:56:27.648118+00:00. Development-only IMPROVED compact/cached H15-prefix history-feature conservative support refit; no MPC simulation, no validation64, no sealed test, no gradient training. rows=68; traces=68/68; global_pass5=0; nested_save=0.0150; nested_bad=1; nested_pass5=False; decision: H15-prefix history/static engineered features still do not yield a safe deployable selector; next prioritize explicit risk/terminal-value learning/refit with uncertainty or targeted boundary acquisition, not another static-feature sweep or validation rollout.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/completed.json`.


## 2026-09-29T21:57:25.372566+00:00
Elapsed service lifetime: >83.0 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; do not estimate; excludes desktop/API conversation.

Concrete actions completed this iteration:

1. **Astra review check**
   - Checked `docs/bohn2021_takeover/astra_reviews/LATEST.md`.
   - No file exists, so no Astra recommendations required disposition this cycle.

2. **Backup gate before science**
   - Confirmed latest verified backup from supervisor context before the run:
     - commit `bbc343c32a9f29ca049ac952bfcae221b9c09450`
     - package SHA256 `c9690f98d8edf16497a1cb6970286593cfcc67115e35b87960f20462e4d3423e`
     - remaining changed files `0`
   - This cleared the prior backup blocker for running v13b.

3. **Ran v13b compact/cached history-feature refit**
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_history_feature_refit_v13b_fast.py`
   - Registry: `research_artifacts/aws_runs/20260929T215618_84fa94cb/registry.json`
   - Runtime: ~10.0 s
   - Classification: development-only IMPROVED refit; no MPC simulation, no gradient training, no validation64, no sealed test.
   - Actual budget:
     - development MPC episodes: `0`
     - control steps: `0`
     - training episodes: `0`
     - gradient steps: `0`
     - selector refit evaluations: `24,305`
     - validation64 episodes: `0`
     - sealed test episodes: `0`

4. **v13b result**
   - Rows: `68`
   - Banks: `fresh_v0`, `fresh_v1`, `fresh_v11`, `fresh_v2`, `fresh_v8c`
   - Positive H10-beneficial rows: `58`
   - Catastrophic H10 rows: `10`
   - H15-prefix traces loaded: `68/68`
   - Feature counts:
     - base_no_history: `48`
     - history_no_risk: `136`
     - history_with_risk: `137`
   - Compact configs: `972`
   - Global pass10 count: `0`
   - Global pass5 count: `0`
   - History-family global pass5 count: `0`
   - Best global config:
     - `v13b_base_no_history_k1_cg1.25_q0.5_m0.75_g0.0_p2.0`
     - bad/catastrophic false positives: `0`
     - H10 choices: `10`
     - min decision saving vs fixed H15: `0.00%`
     - average decision saving vs fixed H15: `4.53%`
   - Nested aggregate:
     - decision saving vs fixed H15: `1.50%`
     - catastrophic H10 false positives: `1`
     - physical gate: `True`
     - H10 choices: `7`
     - pass5: `False`
     - pass10: `False`

5. **Scientific conclusion**
   - v13b directly tested the H15-prefix history-feature hypothesis after v13 timed out.
   - Result: **failed weak and strong opened-development gates**.
   - Interpretation: engineered deployable static/history support features still do not yield a safe useful selector. The bottleneck is more likely calibrated catastrophic-risk / continuation-value estimation under uncertainty, and/or insufficient boundary labels—not simply sparse positive labels or lack of adaptive opportunity.
   - No validation rollout is justified.

6. **Artifacts produced**
   - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/summary.md`
   - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/raw.json`
   - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_20260929T2150Z/completed.json`
   - `research_artifacts/aws_protocols/vehicle_true_variable_horizon_history_feature_refit_v13b_fast_preoutcome_frozen_20260929T2150Z.json`
   - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_HISTORY_FEATURE_REFIT_V13B_20260929T2150Z.json`
   - `research_artifacts/aws_state/continue_state_20260929T2150_after_history_feature_refit_v13b.md`
   - Docs/registry were appended by the script.

7. **State preserved**
   - Durable supervisor state updated with:
     - v13b result
     - budget accounting
     - new backup blocker
     - next hypothesis and queue
     - failure record for v13 timeout and v13b gate failure

Current blocker:

- v13b artifacts are **not yet covered by a verified external backup** in repository-visible state.
- Backup request exists: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_HISTORY_FEATURE_REFIT_V13B_20260929T2150Z.json`
- Per contract, no further scientific simulation/refit should run until this backup is verified.

Next action after backup verification:

1. Inspect v13b raw only as needed to identify the nested false positive and ambiguous neighborhoods.
2. Freeze v14 as an explicit calibrated risk/terminal-value intervention, not another static feature sweep.
3. Preferred v14 direction:
   - uncertainty-aware catastrophic-H10 risk model and continuation-cost/value predictor from deployable observation/history features;
   - conservative abstention to H15;
   - nested opened-bank LOBO selection;
   - no validation64 or sealed test.
4. If data remain too sparse or ambiguous, run a small targeted development-only boundary acquisition protocol with declared MPC episode/control-step budget before fitting.


## 2026-09-29T22:01:48.308750+00:00

<!-- vehicle-true-variable-H-calibrated-risk-value-refit-v14-20260929T2215Z -->
## 2026-09-29 vehicle true-variable-H v14 calibrated risk/value refit

Elapsed service lifetime at write: >83.2 h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED calibrated risk/value refit over opened banks; no MPC simulation, validation64 or sealed test. configs=1296; selector_refit_evaluations=32405; global_pass5=0; nested_save=0.0292; nested_bad=2; nested_pass5=False; decision: Calibrated uncertainty over the current deployable feature/data representation still cannot safely extract >=5% opened-bank savings; next run targeted boundary acquisition or a richer terminal/risk-value learning refit rather than another static/history feature sweep or validation rollout.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_20260929T2215Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_20260929T2215Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_20260929T2215Z/completed.json`.


## 2026-09-29T22:09:40.645890+00:00
Elapsed service lifetime: >83.2 h since 2026-09-26T10:55:29.419331Z. Cumulative server API total_tokens: unavailable from repository-visible state in this API turn; not estimated; excludes desktop/API conversation.

Concrete actions completed this iteration:

1. **Astra review check**
   - Checked `docs/bohn2021_takeover/astra_reviews/LATEST.md`.
   - Result: absent. No Astra recommendations to disposition; continued without waiting.

2. **Backup precondition verified before new science**
   - Latest verified external backup from supervisor context:
     - Commit: `69aa148a3fa11d0304f7fb7fae415a9992690061`
     - Time: `2026-09-29T22:02:33Z`
     - Package SHA256: `32b80ff5c10c5c4cbf05b643a11077dea39508e0cc5be830caee544f8b204b3f`
   - This covered the v13b/audit state before starting v14.

3. **Implemented and ran v14 calibrated risk/value refit**
   - New source:
     - `experiments/bohn2021_aws/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14.py`
     - SHA256: `718f21e564541b4c5ef0ec3e05946782ffc00d92dcc5dace3328bf659e3780f2`
   - Frozen protocol:
     - `research_artifacts/aws_protocols/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_preoutcome_frozen_20260929T2215Z.json`
   - Registry:
     - `research_artifacts/aws_runs/20260929T220747_9784c9c3/registry.json`
   - Result artifacts:
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_calibrated_risk_value_refit_v14_20260929T2215Z/summary.md`
     - `.../raw.json`
     - `.../completed.json`

4. **v14 result**
   - Classification: development-only IMPROVED calibrated risk/value selector refit.
   - No MPC simulation, no gradient/RL training, no validation64 access, no sealed-test access.
   - Rows: `68`
   - Banks: `fresh_v0`, `fresh_v1`, `fresh_v11`, `fresh_v2`, `fresh_v8c`
   - Positives: `58`
   - Catastrophic H10 rows: `10`
   - H15-prefix traces loaded: `68/68`
   - Configs: `1296`
   - Selector-refit evaluations: `32,405`

   Key outcomes:
   - Global best config had **0 catastrophic false positives** but only **2.08% decision-time saving**, below the 5% gate.
   - Global pass5 count: `0`; pass10 count: `0`.
   - Nested deployment:
     - Decision saving vs fixed H15: `2.92%`
     - Solver saving vs fixed H15: `3.25%`
     - Physical gate: `True`
     - H10 choices: `9`
     - Catastrophic H10 false positives: `2`
     - pass5: `False`
     - pass10: `False`

   Nested false-positive rows:
   - `fresh_v8c/fresh_case02_slot1_mid_late_control`
     - group: `risk_low_heading_low_clearance_prior_catastrophic_mode`
     - phys delta: about `247.867`
     - decision gain: about `0.122 s`
   - `fresh_v11/fresh_case05_slot1_mid_late_control`
     - group: `lower_stress_specificity_control`
     - phys delta: about `40.0565`
     - decision gain: about `0.361 s`

5. **Scientific decision**
   - v14 failed the predeclared development gate.
   - Do **not** open validation64 or sealed test.
   - New evidence strengthens the diagnosis: real H10/H15 opportunity exists from v11/v12 oracle evidence, but current deployable instantaneous/H15-prefix feature representations plus calibrated uncertainty still cannot safely extract it.
   - Next best intervention is **targeted boundary acquisition** around the catastrophic/safe lookalike modes, or a richer terminal/risk-value learning/refit if boundary acquisition remains insufficient.

6. **State preserved**
   - Durable state updated via `update_state`.
   - Current phase: `after_v14_calibrated_risk_value_refit_failed_opened_development_gate_backup_required`.
   - Backup request created:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_CALIBRATED_RISK_VALUE_REFIT_V14_20260929T2215Z.json`

Current blocker for next iteration:
- v14 source/protocol/results/docs are **not yet externally backed up** after the prior verified commit. Satisfy the new backup request before further scientific simulation/refit or new protocol/source generation if possible.

Next queued action after backup:
1. Inspect v14 raw prediction scores for the two nested false positives and nearest safe/catastrophic neighbors.
2. Freeze a bounded v15 targeted boundary-acquisition protocol around those ambiguous mid-late/control states and nearest safe lookalikes.
3. Run only development MPC continuations with explicit budget caps; still no validation64 or sealed test.


## 2026-09-29T22:14:09.238218+00:00

<!-- vehicle-true-variable-H-v14-fp-neighbor-audit-v0-20260929T2225Z -->
## 2026-09-29 vehicle true-variable-H v14 false-positive neighbor audit

Elapsed service lifetime at write: >83.3 h since 2026-09-26T10:55:29.419331Z. Development-only metadata diagnostic; no MPC simulation, validation64 or sealed test. Audited v14 nested false positives=2 with reconstructed selected configs and nearest safe/catastrophic training lookalikes. Candidate v15 boundary plan contains 24 branch states (96 H10/H15 repeat episodes if later frozen). Decision: Freeze and run a small v15 development-only boundary acquisition using the candidate saved H15-prefix centers/offsets, after this audit is externally backed up; do not open validation64/sealed test.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/audit.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v14_false_positive_neighbor_audit_v0_20260929T2225Z/completed.json`.


## 2026-09-29T22:18:33.015824+00:00



## 2026-09-29T22:22:44.465710+00:00

<!-- vehicle-true-variable-H-v15-candidate-trace-preflight-v0b-20260929T2235Z -->
## 2026-09-29 vehicle true-variable-H v15 candidate trace preflight v0b

Elapsed service lifetime at write: >83.5 h since 2026-09-26T10:55:29.419331Z. Development-only structural preflight; no MPC simulation, no selector refit/search, no gradient training, no validation64 or sealed test. Candidate branch states=24 across 8 traces; existing trace paths=8/8; parsed step coverage=8/8; blocked candidates=0. Decision: Inputs are structurally sufficient for a very small v15 smoke: implement/freeze paired H10/H15 continuation on the two false-positive center states first, then expand only if replay fidelity checks pass.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/preflight.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_candidate_trace_preflight_v0b_20260929T2235Z/completed.json`.


## 2026-09-29T22:33:56.446718+00:00

<!-- vehicle-true-variable-H-v15-boundary-smoke-v0-20260929T2245Z -->
## 2026-09-29 vehicle true-variable-H v15 boundary-centre smoke v0

Elapsed service lifetime at write: >83.7 h since 2026-09-26T10:55:29.419331Z. Development-only four-episode smoke on two v14 nested false-positive centres; no validation64/sealed-test access, no selector refit/search, no gradient training. Budget actual: 4 episodes, 150 control steps. Aggregate: {'pair_count': 2, 'all_pairs_present': True, 'all_h15_safe': True, 'catastrophic_h10_rows': 2, 'beneficial_h10_rows': 0, 'fixed_H15_physical_sum': 24.84919767019776, 'fixed_H10_physical_sum': 312.7722167189375, 'fixed_H15_decision_sum_s': 4.555505791970063, 'fixed_H10_decision_sum_s': 4.120034996827599, 'fixed_H10_decision_relative_saving_vs_H15': 0.09559219437500521}. Decision: v15 centre smoke reproduced both false-positive centre H10 catastrophes with valid H15 pairs; next freeze the full boundary acquisition around centres/lookalikes, then refit only if added labels improve zero-catastrophe selection.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_smoke_v0_20260929T2245Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_smoke_v0_20260929T2245Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_smoke_v0_20260929T2245Z/completed.json`.


## 2026-09-29T22:37:57.527041+00:00



## 2026-09-29T22:43:45.167023+00:00

<!-- vehicle-true-variable-H-v15-case-snapshot-recovery-diagnostic-v0-20260929T2305Z -->
## 2026-09-29 v15 case-snapshot recovery diagnostic v0

Elapsed service lifetime at write: >83.8 h since 2026-09-26T10:55:29.419331Z. Metadata-only diagnostic after the v15 full-acquisition preparation failure; no MPC simulation, no selector refit/search, no training, no validation64/sealed-test access. Required candidates=24; direct/raw case recovery full=False; recoverable candidates=9/24; missing candidates=15. Decision: Not all v15 candidates have recoverable case/TVP snapshots via raw/manifest scan; do not rerun v0. Next inspect source runner case-bank construction and either repair deterministic source-case reconstruction or freeze a reduced recoverable-candidate acquisition.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_case_snapshot_recovery_diagnostic_v0_20260929T2305Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_case_snapshot_recovery_diagnostic_v0_20260929T2305Z/completed.json`.


## 2026-09-29T22:48:03.624529+00:00

<!-- vehicle-true-variable-H-v15-protocol-case-snapshot-recovery-v0b-20260929T2315Z -->
## 2026-09-29 v15 protocol-aware case-snapshot recovery v0b

Elapsed service lifetime at write: >83.9 h since 2026-09-26T10:55:29.419331Z. Metadata-only recovery diagnostic after the v15 acquisition v0 pre-simulation failure and the raw-only recovery miss; no MPC simulation, no selector refit/search, no training, no validation64/sealed-test access. Candidate recovery: 24/24 recoverable, missing=0; routes={'raw_direct_case_snapshot': 9, 'freeze_protocol_selected_fresh_case': 15}. Decision: All 24 predeclared v15 candidates are recoverable without case reselection; write a v0b acquisition runner that falls back to frozen fresh-source protocols for fresh_v0/v1/v2 case snapshots, then run the 96-episode acquisition only after these recovery artifacts and source repair are externally backed up.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b_20260929T2315Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b_20260929T2315Z/completed.json`.


## 2026-09-29T22:51:55.409160+00:00
Elapsed service lifetime: ~84.6 h since 2026-09-26T10:55:29.419331Z. Server API `total_tokens` from `research.sqlite` excluding this desktop conversation: unavailable in the provided supervisor/tool context, so not restated numerically.

Concrete actions completed:

1. Checked Astra reviewer handoff path: `docs/bohn2021_takeover/astra_reviews/LATEST.md` is absent; no reviewer recommendations pending.

2. Ran metadata-only v15 protocol-aware case snapshot recovery diagnostic:
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b.py`
   - Registry: `research_artifacts/aws_runs/20260929T224854_63ad7a96/registry.json`
   - Summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b_20260929T2315Z/summary.md`
   - Completed: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_protocol_case_snapshot_recovery_v0b_20260929T2315Z/completed.json`
   - Result: all 24 predeclared v15 candidates are recoverable without case reselection.
   - Recovery routes: 9 via raw direct case snapshots, 15 via frozen fresh-source protocols.
   - No MPC simulation, no selector refit, no training, no validation64, no sealed test.

3. Wrote repaired v15 boundary acquisition source:
   - Path: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b.py`
   - SHA256: `eb66383ffb0326da6fb793a3939c4f5e6fa8eef7a93f63f539e87fcc8c28e91e`
   - Change: minimal v0 repair only; raw case lookup first, then frozen-protocol fallback for `fresh_v0/fresh_v1/fresh_v2`.
   - Preserves: 24 candidates, H10/H15 x2 schedule, 96-episode / 14400-step cap, shared H15 terminal, and v0 decision rules.
   - Not yet executed because the one-experiment-per-iteration guard was already used.

4. Preserved next-iteration state:
   - State file: `research_artifacts/aws_state/continue_state_20260929T2330_after_v15_recovery_run_and_v0b_source_written.md`
   - Durable supervisor state updated via `update_state`.

Budget this cycle:
- Development MPC episodes: 0
- Development control steps: 0
- Training episodes / gradient steps: 0 / 0
- Selector refit evaluations: 0
- Validation64 / sealed-test episodes: 0 / 0
- Metadata experiments completed: 1
- New source files written: 1

Next concrete action:
1. Verify external backup for the new recovery outputs/docs/registry and `vehicle_true_variable_horizon_v15_boundary_acquisition_v0b.py`.
2. Then run, with legacy interpreter, exactly one bounded development experiment:

`experiments/bohn2021_aws/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b.py --run --backup-verified-commit <verified_commit> --i-accept-development-v15-boundary-acquisition`

Declared budget: 96 development MPC episodes, ≤14400 control steps, no training/refit, no validation64, no sealed test.

<!-- vehicle-true-variable-H-v15-boundary-acquisition-v0b-20260929T2325Z -->
## 2026-09-29 vehicle true-variable-H v15 local boundary acquisition v0b

Elapsed service lifetime at write: >84.1 h since 2026-09-26T10:55:29.419331Z. Development-only local boundary acquisition on 24 saved H15-prefix candidates x H10/H15 x2 repeats; no validation64/sealed-test access, no selector refit/search, no gradient training. v0b repairs only case-snapshot lookup using frozen fresh-source protocols where raw outputs omit selected-source snapshots. Budget actual: 96 episodes, 4486 control steps. Aggregate: pairs=48/48, all_h15_safe=True, catastrophic_h10=24, beneficial_h10=24, decision_saving_H10_vs_H15=0.24805757791328023, label_data_sufficient_for_refit=True. Decision: v15 boundary acquisition produced interpretable local labels with both catastrophic and safe-beneficial H10 outcomes; next freeze a boundary-augmented conservative refit, still development-only and requiring zero catastrophic H10 plus >=5% opened-development saving before any unused-source confirmation.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v15_boundary_acquisition_v0b_20260929T2325Z/completed.json`.


## 2026-09-29T23:07:00.464747+00:00



## 2026-09-29T23:22:51.484004+00:00

<!-- vehicle-true-variable-H-boundary-augmented-refit-v16b-fast-20260929T2320Z -->
## 2026-09-29 vehicle true-variable-H v16b-fast boundary-augmented refit

Elapsed service lifetime at write: >84.5 h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED runtime repair after v16 timeout; no MPC simulation, no validation64/sealed-test access, no gradient training. rows=92; configs=1296; equivalent_selector_refit_evaluations=32406; cached_score_fits=1356; strict_nested_save=0.015308; strict_nested_bad=4; strict_nested_pass5=False; global_pass5=0; decision: v16b boundary labels repair local/in-sample separability but not held-out generalization; boundary-risk information is relevant but static/refit features remain insufficient. Pivot to richer learned risk/terminal-value representation rather than another static-feature sweep.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast_20260929T2320Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast_20260929T2320Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_boundary_augmented_refit_v16b_fast_20260929T2320Z/completed.json`.


## 2026-09-29T23:30:12.348462+00:00



## 2026-09-29T23:35:27.854883+00:00

<!-- vehicle-true-variable-H-bank-consensus-v17-20260929T2345Z -->
## 2026-09-29 vehicle true-variable-H v17 bank-consensus uncertainty veto

Elapsed service lifetime at write: >84.7 h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline/refit diagnostic; no MPC simulation, no validation64/sealed-test access, no gradient training. candidate_configs=21; strict_nested_save=0.000000; strict_nested_solver_save=0.000000; strict_nested_bad=0; strict_nested_h10=0; strict_nested_pass5=False; cache_fits=1705. Decision: v17 bank-consensus veto eliminates catastrophic H10 false positives but is too conservative for the >=5% gate; current deployable representation/data appear insufficient for useful safe compute savings. Pivot to terminal/risk-value learning or new source-independent state coverage, not another static selector sweep.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_bank_consensus_v17_20260929T2345Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_bank_consensus_v17_20260929T2345Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_bank_consensus_v17_20260929T2345Z/completed.json`. Astra response log updated at `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md` if latest review existed.


## 2026-09-29T23:41:00.065265+00:00



## 2026-09-29T23:46:58.541340+00:00

<!-- vehicle-true-variable-H-probe-telemetry-v18d-20260930T0035Z -->
## 2026-09-30 vehicle true-variable-H v18d H10 probe-telemetry diagnostic

Elapsed service lifetime at write: >85.1 h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline diagnostic over existing v15 true-H branch traces; no MPC simulation, no validation64/sealed-test access, no gradient training. v18d executes the v18c repaired telemetry diagnostic and fixes the predeclared offline-evaluation cap. rows=24; configs=729; offline_model_evaluations=61249; leave_bank_save=0.004876; leave_bank_bad=2; leave_bank_h10=3; leave_source_save=0.016919; leave_source_bad=2; leave_source_h10=4; best_global=v18_dual_probe_objective_k3_r0.4_g0.0_p20.0_sq1.0 save=0.032141 bad=0 H10=4. Decision: Existing H10 probe telemetry is insufficient for a safe useful selector under strict bank/source splits; prioritize richer terminal-value/risk instrumentation or scenario/source-independent data, not another static threshold sweep.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_probe_telemetry_v18d_20260930T0035Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_probe_telemetry_v18d_20260930T0035Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_probe_telemetry_v18d_20260930T0035Z/completed.json`. Canonical experiment registry row is written by run_experiment, not by this script.


## 2026-09-30T00:00:13.436785+00:00

<!-- vehicle-true-variable-H-intermediate-h12-boundary-v19-20260930T0015Z -->
## 2026-09-30 vehicle true-variable-H v19 intermediate H12 boundary acquisition

Elapsed service lifetime at write: >85.3 h since 2026-09-26T10:55:29.419331Z. Development-only paired true-H12/H15 run on the 24 v15 boundary candidates; no validation64/sealed-test access, no selector refit/search, no gradient training. Budget actual: 96 episodes, 4434 control steps. Fixed H12 vs H15: decision saving=0.1432193357727001, solver saving=0.15865718235683493, physical delta=566.3996689048843 (tolerance 96.0), catastrophic_H12=6, beneficial_H12=42, fixed_pass5=False. Oracle H12/H15 decision saving=0.15723702644285859, physical delta=-1.2650899319502855, value5=True. Decision: True-H12 still has localized catastrophic/high-cost rows, but an oracle H12/H15 switch has >=5% measured decision saving with physical gate; next freeze an H12/H15 selector/refit using strict bank/source splits rather than more H10 sweeps.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_intermediate_h12_boundary_v19_20260930T0015Z/completed.json`.


## 2026-09-30T00:15:12.216941+00:00



## 2026-09-30T00:34:01.261120+00:00

<!-- vehicle-true-variable-H-h12-h15-selector-refit-v20b-fast-20260930T0100Z -->
## 2026-09-30 vehicle true-variable-H v20 H12/H15 selector/refit

Elapsed service lifetime at write: >85.7 h since 2026-09-26T10:55:29.419331Z. Development-only IMPROVED offline H12/H15 selector/refit over v19 opened boundary rows; no MPC simulation, validation64, sealed test, or gradient training. rows=24; positives=21; catastrophics=3; configs=1728; selector_refit_evaluations=153805; strict_bank_save=0.052056, bad=0, H12=7, pass5=True; strict_source_save=0.072057, bad=0, H12=9, pass5=True; decision: v20b H12/H15 conservative selector passes both strict opened bank and source nested >=5% zero-catastrophe gates, but the single held-out negative source is protected by a no-cat-training uncertainty fallback. Next do not validate yet: freeze source-independent H12-negative acquisition and an online overhead smoke to test whether this safety rule generalizes beyond one negative cluster.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_SELECTOR_REFIT_V20B_FAST_20260930T0100Z.json`.


## 2026-09-30T00:37:45.463791+00:00
累计服务寿命：约 85.7 h（自 2026-09-26T10:55:29.419331Z 起）；累计服务器 API total_tokens（research.sqlite，不含本桌面对话）：当前 API 工具上下文不可读，未伪造，下一轮若有 supervisor/sqlite 上下文需补读后以 M/B tokens 报告。单次实验运行时间：80.06 s。

本轮已完成一个具体实验并持久化状态：

## 已执行

运行：

`experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast.py`

- run registry: `research_artifacts/aws_runs/20260930T003516_9eb2ad62/registry.json`
- exit status: 0
- runtime: 80.055 s
- 类型：development-only IMPROVED H12/H15 offline selector/refit runtime repair
- 无 MPC 仿真、无 validation64、无 sealed test、无梯度训练
- 使用已验证备份 commit：`a8713ed1eda0cc6388926564fd026a454ba7729a`

## 关键结果

v20b 修复了 v20 超时问题，仅改变缓存/计算方式，不改变科学假设、输入、严格 split 或门槛。

在 v19 opened-development H12/H15 boundary rows 上：

- rows: 24
- H12 beneficial rows: 21
- H12 catastrophic/high-cost rows: 3
- fixed H12 vs H15:
  - decision saving: 14.32%
  - solver saving: 15.87%
  - bad rows: 3

严格 nested CV：

- leave-bank nested:
  - decision saving: 5.21%
  - solver saving: 5.70%
  - catastrophic false positives: 0
  - H12 choices: 7
  - physical gate: true
  - pass5: true

- leave-source nested:
  - decision saving: 7.21%
  - solver saving: 7.91%
  - catastrophic false positives: 0
  - H12 choices: 9
  - physical gate: true
  - pass5: true

解释：这是第一个在当前 opened boundary 机制数据上同时通过 strict leave-bank/source ≥5% 且零灾难 H12 的 H12/H15 selector 结果，但它仍然不是 validation-ready，因为唯一 H12-catastrophic source 是 `fresh_v11/fresh_case05_slot1_mid_late_control`，其 held-out 安全主要依赖 “outer training fold 无 catastrophic examples => uncertainty fallback to H15”。这不能证明 source-independent 风险泛化。

## 已检查/保存的证据

主要产物：

- summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/summary.md`
- raw: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/raw.json`
- completed: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_20260930T0100Z/completed.json`
- protocol: `research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_selector_refit_v20b_fast_preoutcome_frozen_20260930T0100Z.json`
- continue state: `research_artifacts/aws_state/continue_state_20260930T0100_after_h12_h15_selector_refit_v20b_fast.md`
- backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_SELECTOR_REFIT_V20B_FAST_20260930T0100Z.json`

脚本已更新：

- `STATUS.md`
- `RESEARCH_LOG.md`
- `DECISIONS.md`
- `RESULTS_AUDIT.md`
- `REPRODUCTION_PROTOCOL.md`
- `EXPERIMENT_REGISTRY.csv`

并已通过 `update_state` 持久化下一步状态。

## 当前门控/阻塞

本轮后检查：

`research_artifacts/aws_backup_proofs/backup_proof_20260930*`

结果为空。  
因此 v20b 新产物尚无 post-v20b verified external backup proof。

结论：在备份验证前，不应继续运行新的唯一科学实验或新仿真。sealed test 仍关闭且未授权；validation64 未被本轮打开。

## 下一轮优先动作

1. 先检查/等待 supervisor 产生覆盖 v20b artifacts 的 verified external backup proof。
2. 备份通过后，不直接进 validation。应冻结并运行 v21 development-only source-independent H12 negative/support acquisition：
   - 预注册新的 fresh sources/cases；
   - 主要比较 H12/H15；
   - 目标是检验 v20b 安全性是否能跨 source 泛化，而不是只靠单一 negative-source uncertainty fallback。
3. 若 acquisition 受限，则次优是做在线 overhead smoke，测量 deployable selector feature+decision overhead；但当前信息价值更高的是 source-independent H12 negative/support 数据获取。
4. 继续保持 ORIGINAL/IMPROVED 区分：v20b 是 IMPROVED，不是 ORIGINAL SAC；无 final-test 或正式复现成功声明。


## 2026-09-30T00:42:30.730008+00:00



## 2026-09-30T00:47:56.835545+00:00



## 2026-09-30T00:53:20.006065+00:00



## 2026-09-30T01:00:18.087677+00:00

<!-- vehicle-true-variable-H-h12-h15-source-independent-acquisition-v21-20260930T0130Z -->
## 2026-09-30 vehicle true-variable-H H12/H15 source-independent acquisition v21

UTC: 2026-09-30T01:06:32.171731+00:00. Development-only source-independent H12/H15 acquisition completed: 40 episodes, 1962 control steps, validation64 closed, sealed test closed, no training/refit. States=16, H12-beneficial=15, H12-catastrophic/high-cost=0, fixed-H12 decision saving vs H15=0.15049729897916866, oracle decision saving=0.16474542892945737. Decision: fresh source-independent fixed H12 itself passes the H15-referenced 5% safety/physical gate on this development batch; next measure online selector overhead and treat fixed H12 as a strong baseline. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_source_independent_acquisition_v21_20260930T0130Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V21_SOURCE_INDEPENDENT_H12_H15_ACQUISITION_20260930T010632.171731+0000.json`.


## 2026-09-30T01:09:44.735225+00:00



## 2026-09-30T01:17:10.641390+00:00

<!-- vehicle-h12-h15-online-overhead-v22-20260930T0125Z -->
## 2026-09-30 v22 H12/H15 online-overhead diagnostic

UTC: 2026-09-30T01:19:07.174133+00:00. Development-only IMPROVED analysis; no new MPC simulation, no validation64 access, no sealed-test access, no training/refit. Inputs were v20b and v21 opened-development artifacts backed up by supervisor commit `7677039b1e7c718e07b5ea1c011b09cb4b794c6c`.

Measured Python in-memory history feature + selector overhead over 20000 repetitions: mean 0.001956693s, median 0.001802578s, p95 0.002653475s. v21 fixed H12 remains a strong baseline: 15.05% decision saving vs H15, zero bad, physical gate=True. The v20b history selector trained on v19 chose {'12': 9, '15': 7} on v21 and achieved 5.41% overhead-adjusted branch saving, zero bad. The combined development proxy (v19 strict-source CV choices + v21 v19-trained selector) achieved 6.50% saving, zero bad=True, pass5=True; combined fixed H12 still failed safety/physical due to v19 negatives (bad=3, physical gate=False).

Decision: Selector overhead is negligible at branch scale and the combined development proxy passes vs H15 while fixed H12 remains unsafe on v19 negatives; however v21-only fixed H12 outperforms/equals the selector as a simple baseline, so the next experiment should broaden fresh confirmation with fixed H12/H15/selector all predeclared before any validation.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/completed.json`. Backup requested: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_ONLINE_OVERHEAD_V22_20260930T0125Z.json`.


## 2026-09-30T01:20:25.722125+00:00
Elapsed service lifetime: **311,018 s / 86.39 h** since `2026-09-26T10:55:29.419331Z` as of the v22 artifact timestamp. Cumulative server API `total_tokens` from supervisor `research.sqlite`: **unavailable from repository tools** (`research.sqlite` not found in `.`, `research_artifacts/`, or `state/`; v22 raw recorded `null`), so I am reporting it as **unknown, not zero**.

Concrete actions completed this iteration:

1. **Inspected coordination/reviewer state**
   - Read `docs/bohn2021_takeover/astra_reviews/COORDINATION.md`.
   - Confirmed `LATEST.md` still points to `20260929T153837Z`.
   - Verified `RESPONSE_LOG.md` already had v21 follow-up and was updated by the v22 script with stable Astra IDs.

2. **Ran v22 overhead/baseline diagnostic**
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_online_overhead_v22.py`
   - Run registry: `research_artifacts/aws_runs/20260930T011826_0aa46fe4/registry.json`
   - Classification: development-only IMPROVED diagnostic.
   - No MPC simulation, no validation64, no sealed test, no training/refit.
   - Budget: `20,000` selector microbenchmark repetitions.

3. **Key v22 evidence**
   - Python in-memory history feature + selector overhead:
     - mean: `0.001956693 s`
     - median: `0.001802578 s`
     - p95: `0.002653475 s`
   - v21-only fixed H12:
     - decision saving vs H15: `15.05%`
     - bad/catastrophic rows: `0`
     - physical gate: `True`
   - v21-only v20b/v22 history selector:
     - H counts: `{'12': 9, '15': 7}`
     - overhead-adjusted decision saving vs H15: `5.41%`
     - bad rows: `0`
   - Combined v19 strict-source proxy + v21 selector:
     - H counts: `{'12': 18, '15': 22}`
     - overhead-adjusted decision saving vs H15: `6.50%`
     - bad rows: `0`
     - pass5: `True`
   - Combined fixed H12:
     - decision saving vs H15: `14.61%`
     - bad rows: `3`
     - physical gate: `False`
     - bad source remains `fresh_v11/fresh_case05_slot1_mid_late_control`.

4. **Scientific decision**
   - Selector micro-overhead does **not** erase the branch-level development gain.
   - However, fixed true H12 is now a **strong baseline** on v21-like fresh rows and must be treated as primary comparator.
   - Current adaptive value is safety against known v19 H12-negative states while retaining net saving, not superiority over fixed H12 on every fresh subset.
   - No validation/final-test claim is justified; all v19/v21/v22 evidence remains opened development evidence.

5. **Artifacts updated/preserved**
   - Summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/summary.md`
   - Raw: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/raw.json`
   - Completed: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_20260930T0125Z/completed.json`
   - Protocol: `research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_online_overhead_v22_preanalysis_frozen_20260930T0125Z.json`
   - Continue state: `research_artifacts/aws_state/continue_state_20260930T0125_after_h12_h15_online_overhead_v22.md`
   - Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_ONLINE_OVERHEAD_V22_20260930T0125Z.json`
   - Updated docs/registry include `STATUS.md`, `RESEARCH_LOG.md`, `DECISIONS.md`, `RESULTS_AUDIT.md`, `REPRODUCTION_PROTOCOL.md`, `EXPERIMENT_REGISTRY.csv`, and Astra `RESPONSE_LOG.md`.

6. **Current blocker**
   - Backup gate is active before more unique science.
   - I checked `research_artifacts/aws_backup_proofs/backup_proof_20260930*`; no post-v22 proof was present.
   - Need verified external backup covering v22 source/results/docs/state/registry/response log and the backup request before running v23.

7. **Next preserved action**
   - After verified v22 backup: freeze a broader source-independent H12/H15/selector confirmation (`v23`) with fixed H12, fixed H15, and the v20b/v22 history selector all predeclared.
   - No validation64 or sealed test.
   - If new H12 negatives appear: pivot to richer terminal-risk/value refit or bounded training rather than another static sweep.
   - If fixed H12 remains broadly safe and dominates: treat fixed H12 as the main baseline/result and reassess adaptivity/scenario opportunity rather than forcing switching.

<!-- vehicle-true-variable-H-h12-h15-broader-confirmation-v23-20260930T0145Z -->
## 2026-09-30 vehicle true-variable-H H12/H15 broader confirmation v23

UTC: 2026-09-30T01:32:17.706415+00:00. Development-only broader source-independent H12/H15/selector confirmation completed after verified post-v22 backup. No validation64, no sealed test, no training/refit. Budget 60 episodes / 2832 control steps plus 24 preoutcome selector decisions. States=24, H12-beneficial=23, H12-catastrophic/high-cost=0. Fixed H12 save=0.17080323994754293, bad=0, pass5=True; selector save=0.08455613439452371, H counts={'12': 11, '15': 13}, bad=0, pass5=True. Decision: v23 fixed H12 is safe and faster than the selector on this broader fresh batch; fixed H12 must be the primary simple baseline and adaptivity is not yet justified for this stress distribution.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_broader_confirmation_v23_20260930T0145Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_H15_BROADER_CONFIRMATION_V23_20260930T0145Z.json`.


## 2026-09-30T01:32:57.912915+00:00



## 2026-09-30T01:39:54.297951+00:00

<!-- vehicle-true-variable-H-h12-adaptivity-opportunity-audit-v24-20260930T0205Z -->
## 2026-09-30 vehicle H12/H15 adaptivity-opportunity audit v24

UTC: 2026-09-30T01:41:17.634738+00:00. Offline/no-simulation diagnostic over already-opened v19/v21/v23 H12/H15 development rows; no validation64, no sealed test, no training/refit. Fresh source-independent v21+v23 rows: fixed H12 bad=0, pass5=True, decision saving=0.1622243713202727; v19 opened-boundary rows: fixed H12 bad=3, pass5=False, oracle pass5=True; all opened rows: fixed H12 bad=3, oracle pass5=True. Decision: Fixed H12 dominates the fresh source-independent stress-pool rows, while adaptivity is only required by the opened v19 lower-stress mid-late negative cluster. The next experiment should not be another broad fresh support sweep; it should specifically test whether v19-like H12-risk states reproduce in source-independent neighbors selected pre-outcome from H15 traces. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_ADAPTIVITY_OPPORTUNITY_AUDIT_V24_20260930T0205Z.json`.


## 2026-09-30T01:42:50.189616+00:00
Elapsed service lifetime since first supervisor event: ~86h45m48s (to v24 completion at 2026-09-30T01:41:17Z). Cumulative server API `total_tokens` from `research.sqlite`: unavailable in repository-accessible tools here; I did not infer or treat it as zero.

Concrete actions completed:

1. Read coordination/reviewer handoff requirements:
   - `docs/bohn2021_takeover/astra_reviews/COORDINATION.md`
   - `docs/bohn2021_takeover/astra_reviews/LATEST.md`
   - Confirmed latest Astra report remains `20260929T153837Z`.

2. Executed the previously written v24 offline audit:
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24.py`
   - Run registry: `research_artifacts/aws_runs/20260930T014117_86f32862/registry.json`
   - Exit status: `0`
   - Runtime: ~2.0s
   - New simulations/control steps/training/refit: `0 / 0 / 0 / 0`
   - Validation64 opened: `false`
   - Sealed final test accessed: `false`

3. v24 key result:
   - Fresh source-independent rows v21+v23:
     - Rows: `40`
     - H12 beneficial: `38`
     - H12 catastrophic: `0`
     - Fixed H12 decision saving vs H15: `16.22%`
     - Fixed H12 pass5: `True`
   - Opened v19 boundary rows:
     - Repeat-averaged rows: `24`
     - H12 catastrophic: `3` averaged rows, corresponding to v19’s repeat-level `6`
     - All catastrophic rows remain concentrated in `fresh_v11/fresh_case05_slot1_mid_late_control`
     - Fixed H12 pass5: `False`
     - Oracle H12/H15 pass5: `True`
   - Combined opened H12/H15 rows:
     - Fixed H12 bad: `3`, pass5 `False`
     - Oracle H12/H15 pass5 `True`, decision saving `16.80%`

4. Scientific decision:
   - Current evidence is mainly a **fixed-H12 compute-saving result** on the fresh stress-pool rows.
   - Adaptivity is not yet justified broadly; it is only motivated by the localized v19/v11 lower-stress mid-late H12-risk cluster.
   - Therefore the next useful experiment is not another broad fresh support sweep and not H10 label-density work.
   - Next discriminating experiment should specifically test whether v19-like H12-risk states reproduce in fresh source-independent source72-neighborhood / lower-stress mid-late states selected pre-outcome from H15 traces.

5. Documents/state updated by v24:
   - `STATUS.md`
   - `RESEARCH_LOG.md`
   - `DECISIONS.md`
   - `RESULTS_AUDIT.md`
   - `REPRODUCTION_PROTOCOL.md`
   - `EXPERIMENT_REGISTRY.csv`
   - `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`
   - Stable Astra IDs preserved through v24 for A4/A6/A7/A8/A11/A12.

6. Main new artifacts:
   - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/summary.md`
   - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/raw.json`
   - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_adaptivity_opportunity_audit_v24_20260930T0205Z/completed.json`
   - `research_artifacts/aws_state/continue_state_20260930T0205_after_h12_adaptivity_opportunity_audit_v24.md`
   - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_ADAPTIVITY_OPPORTUNITY_AUDIT_V24_20260930T0205Z.json`

7. State preserved for next iteration via `update_state`.

Current blocker / next step:

- Further unique science is backup-gated.
- Need verified external backup covering v24 source/results/docs/registry/response log/run registry before v25 simulation or refit work.
- Once backup is verified, next queued experiment is v25 H12-risk-family acquisition:
  - metadata/H15-trace-only pre-outcome selection,
  - source72-neighborhood / lower-stress mid-late morphology,
  - exclude all prior H-outcome sources,
  - compare fixed H12, fixed H15 and v20b/v22 selector with measured decision/solver timing,
  - no validation64 or sealed test.


## 2026-09-30T01:50:36.668091+00:00

<!-- vehicle-h12-h15-risk-family-acquisition-v25-dryrun-20260930T0210Z -->
## 2026-09-30 vehicle H12/H15 risk-family acquisition v25 dry-run/protocol freeze

UTC: 2026-09-30T01:57:29.103230+00:00. Dry-run only; no simulations, no validation64, no sealed test, no training/refit. Froze source72-neighborhood metadata-only case selection for v25 after verified post-v24 backup. Selected source_candidate_index values: [234, 17, 222, 162, 11, 135, 177, 167]. Planned budget: 40 episodes, max 6000 control steps, 16 selector choices. Protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_preoutcome_frozen_20260930T0210Z.json` sha256 `abadaede67336961e4a85cf148c70807320d0d7d3d413248668d244b66f05a90`; dry-run summary `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/dry_run_summary.md`. Backup required before actual Stage-A/B simulation: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_DRYRUN_20260930T0210Z.json`.


## 2026-09-30T01:58:29.334458+00:00

<!-- vehicle-h12-h15-risk-family-acquisition-v25-run-20260930T0210Z -->
## 2026-09-30 vehicle H12/H15 risk-family acquisition v25 run

UTC: 2026-09-30T02:04:13.379250+00:00. Development-only source72-neighborhood H12-risk-family acquisition completed; validation64 closed, sealed test closed, no training/refit. Budget 40 episodes / 1642 control steps plus 16 preoutcome selector decisions. States=16, H12-beneficial=16, H12-catastrophic/high-cost=0. Fixed H12 save=0.19159715606423805, bad=0, pass5=True; selector save=0.12110167355602902, H counts={'12': 10, '15': 6}, bad=0, pass5=True. Decision: v25 source72-neighborhood risk-family rows still favor safe fixed H12; adaptivity is not justified for this stress-pool family, so prioritize scenario/comparison design or fixed-H12-primary confirmation.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_RUN_20260930T0210Z.json`.


## 2026-09-30T02:06:21.789293+00:00
Elapsed service lifetime since first supervisor event: ~87h10m. Server API total_tokens from research.sqlite: unavailable via repository tools this cycle (unknown, not zero; desktop conversation excluded).

Concrete actions completed:

1. Verified reviewer coordination context
   - Re-read `docs/bohn2021_takeover/astra_reviews/COORDINATION.md`.
   - Re-read `LATEST.md`; latest Astra report remains `20260929T153837Z`.
   - Preserved stable Astra recommendation IDs in `RESPONSE_LOG.md`.

2. Materialized backup proof before new v25 simulation
   - Wrote local proof from supervisor/user context:
     - `research_artifacts/aws_backup_proofs/backup_proof_20260930T015912_from_supervisor_context_after_v25_dryrun.json`
   - This covered the v25 dry-run/protocol freeze before running v25 simulations.

3. Ran v25 H12/H15 source72-risk-family acquisition
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25.py`
   - Run registry: `research_artifacts/aws_runs/20260930T020004_53a68db1/registry.json`
   - Runtime: ~252 s, exit status 0.
   - Legacy interpreter used.
   - No validation64 access.
   - No sealed final test access.
   - No mobile-robot MPPI.
   - No new training, gradients, or selector grid refit.
   - Budget used: 40 development episodes, 1642 control steps, 16 preoutcome selector decisions.

4. Main v25 result
   - Artifacts:
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_h12_h15_risk_family_acquisition_v25_20260930T0210Z/summary.md`
     - `.../completed.json`
     - `.../raw.json`
     - `.../preoutcome_selector_choices.json`
   - 16 source-independent source72-neighborhood branch states.
   - H12-beneficial rows: 16/16.
   - H12 catastrophic/high-cost rows: 0/16.
   - H15 unsafe rows: 0/16.
   - Fixed H12 vs H15:
     - decision saving: 19.16%
     - solver saving: 21.76%
     - bad rows: 0
     - physical gate: true
     - pass5: true
   - v20b/v22 preoutcome selector:
     - selected H12 on 10/16 states, H15 on 6/16
     - overhead-adjusted decision saving: 12.11%
     - solver saving: 13.76%
     - bad rows: 0
     - pass5: true
     - measured feature+selector overhead mean: 0.002596 s, p95: 0.002809 s.

Scientific interpretation:

- v25 did not reproduce the opened v19/source72 H12-negative cluster in unused source72-neighborhood stress-bank cases.
- Together, v21+v23+v25 now provide fresh source-independent development evidence where fixed H12 is safe and faster on 56 branch states.
- The current selector is safe but worse than fixed H12 on these fresh rows because it conservatively chooses H15 in states where H12 was actually beneficial.
- Therefore, adaptivity is not currently justified for this stress-pool family; the evidence favors fixed-H12-primary comparison/scenario-design diagnosis over immediate terminal-risk/value retraining.
- This is still development/stress-pool evidence only, not population validation, not sealed-test evidence, and not ORIGINAL SAC reproduction.

State and docs preserved:

- Updated by script and inspection:
  - `STATUS.md`
  - `RESEARCH_LOG.md`
  - `DECISIONS.md`
  - `RESULTS_AUDIT.md`
  - `REPRODUCTION_PROTOCOL.md`
  - `EXPERIMENT_REGISTRY.csv`
  - `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`
  - `research_artifacts/aws_state/continue_state_20260930T0210_after_h12_risk_family_acquisition_v25.md`
  - `research_artifacts/aws_state/continue_state_20260930T0205_after_v25_run_inspection.md`

Backup gate:

- Wrote post-v25 backup requests:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_H12_RISK_FAMILY_ACQUISITION_V25_RUN_20260930T0210Z.json`
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V25_RUN_INSPECTION_FINAL_ADDENDUM_20260930T0205Z.json`
- Further unique science is blocked until a verified external backup covers v25 run artifacts, docs, registry, response log, and the new backup-proof materialization.

Next planned action after verified backup:

- Freeze/run a v26 fixed-H12-primary source-independent comparison/confirmation protocol, excluding v19/v21/v23/v25 sources.
- Primary baseline should be fixed true H12, with H15 and adaptive/oracle H12/H15 as secondary comparisons.
- Measure physical/complete cost, safety, success, solver failures, whole-decision timing, solver timing, and p50/p95 timing.
- If fixed H12 remains safe and faster, pivot toward scenario redesign or a credible negative adaptive-opportunity conclusion for this distribution.
- If fixed H12 failures recur, inspect telemetry and then run terminal-risk/value refit or bounded training rather than another static sweep.


## 2026-09-30T02:13:30.298126+00:00

<!-- vehicle-fixed-h12-primary-diagnostic-v26-20260930T0225Z -->
## 2026-09-30 vehicle fixed-H12-primary diagnostic v26

UTC: 2026-09-30T02:15:08.178820+00:00. Offline audit and protocol freeze only; no simulation, no validation64, no sealed test, no training/refit. Fresh v21+v23+v25 H12/H15 rows: n=40, H12-bad=0, fixed-H12 decision saving=0.17839771731752374, pass5=True. v19 remains opened counterevidence with H12-bad=0. Froze v26 fixed-H12-primary source-independent confirmation protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26_preoutcome_fixed_H12_primary_confirmation_20260930T0225Z.json` selecting source_candidate_index values [107, 53, 217, 190, 134, 30, 94, 108, 148, 242, 151, 195] for a future 60-episode development run after backup. Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_FIXED_H12_PRIMARY_DIAGNOSTIC_V26_20260930T0225Z.json`.


## 2026-09-30T02:19:43.934052+00:00

<!-- vehicle-fixed-h12-primary-diagnostic-v26b-repair-20260930T0235Z -->
## 2026-09-30 vehicle v26b fixed-H12 audit repair

UTC: 2026-09-30T02:22:19.269520+00:00. Offline correction only; no simulation, no validation64, no sealed test. v26's parser omitted v19/v21 schemas, so v26's before-evidence table/risk bound was incomplete. Corrected fresh v21+v23+v25: n=56, H12_bad=0, fixed_H12 decision saving=0.1695502100120484, pass5=True. Corrected all opened v19+v21+v23+v25: n=80, H12_bad=3, fixed_H12 pass5=False; v19 counterevidence remains. Repaired protocol amendment `research_artifacts/aws_protocols/vehicle_true_variable_horizon_fixed_h12_primary_diagnostic_v26b_repair_corrected_fixed_H12_primary_confirmation_amendment_20260930T0235Z.json` preserves v26 selected source indices [107, 53, 217, 190, 134, 30, 94, 108, 148, 242, 151, 195]. Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V26B_REPAIR_20260930T0235Z.json`.


## 2026-09-30T02:25:42.056436+00:00

<!-- vehicle-fixed-h12-primary-v27-preflight-20260930T0240Z -->
## 2026-09-30 v27 fixed-H12-primary preflight

UTC: 2026-09-30T02:32:42.975997+00:00. Metadata-only; no simulation/training/validation/test. v26b protocol integrity checks passed=True. Corrected fresh evidence remains n=56, H12_bad=0, fixed_H12_save=16.96%, pass5=True; all-opened evidence remains n=80, H12_bad=3, pass5=False. Adequate post-v26b backup present=True; input sufficient for unique simulation now=True. Next action: run repaired fixed-H12-primary 60-episode development confirmation. Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_PREFLIGHT_20260930T0240Z.json`.


## 2026-09-30T02:35:07.933642+00:00

<!-- vehicle-fixed-h12-primary-confirmation-v27-20260930T0250Z -->
## 2026-09-30 v27 fixed-H12-primary development confirmation

UTC: 2026-09-30T02:43:28.789811+00:00. Development-only v26b-repaired fixed-H12-primary confirmation completed; validation64 closed, sealed test closed, no training/refit/grid search. Budget 60 episodes / 3502 control steps plus 24 preoutcome selector choices. States=24, H12-beneficial=22, H12-catastrophic/high-cost=0. Fixed H12 save=0.19619167689373251, bad=0, pass5=True; selector save=0.03717074660898639, H counts={'12': 6, '15': 18}, bad=0, pass5=False. Decision: v27 fixed-H12-primary confirmation passes: fixed true H12 remains the stronger simple baseline on this fresh stress-bank batch; do not validate adaptive H12/H15 on this distribution without new scenario-opportunity evidence. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_confirmation_v27_20260930T0250Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_FIXED_H12_CONFIRMATION_20260930T0250Z.json`.


## 2026-09-30T02:47:02.985398+00:00

<!-- vehicle-fixed-h12-primary-v27-postdiagnostic-20260930T0300Z -->
## 2026-09-30 v27 fixed-H12-primary postdiagnostic

UTC: 2026-09-30T02:48:21.906171+00:00. Analysis-only audit of v27 completed; no simulations/control steps/training, no validation64-bank access, no sealed-test access. It preserves v27's relative result (fixed H12 decision saving 19.62%, relative bad=0, H15-referenced pass5=True) but clarifies failure accounting: absolute unsafe rows for fixed H12=2, fixed H15=2, both-fail rows=2 from source_candidate_index [242]; success-sensitive fixed-H12 pass5=False. Combined fresh v21+v23+v25+v27 relative rows=80 with H12_bad=0 and decision saving=17.90%, but this remains stress-pool development evidence. Decision: v27 strengthens the comparison-design conclusion: on fresh stress-pool rows, true fixed H12 is the dominant H12/H15 relative-compute baseline and the current adaptive selector is too conservative. However two source242 rows are absolute failures for both H12 and H15, so the v27 pass is H15-referenced rather than a deployable success/safety pass. After external backup, do not validate this adaptive H12/H15 selector; freeze a success-aware scenario/comparison audit/protocol that separates feasible states, both-fail hard cases, and the opened v19 H12-negative cluster before deciding on scenario redesign or terminal-risk/value training. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/completed.json`. Backup required: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_POSTDIAGNOSTIC_20260930T0300Z.json`.


## 2026-09-30T02:49:26.157483+00:00
Elapsed service lifetime since first supervisor event: ~87.88 h (2026-09-26T10:55:29Z → 2026-09-30T02:48:21Z). Cumulative server API total_tokens from `research.sqlite`: unavailable via current repository tools (`research.sqlite` not visible); not estimated, desktop conversation usage excluded.

Concrete action completed: ran the previously blocked v27 postdiagnostic as the sole experiment this iteration.

## Experiment completed

- Experiment ID: `20260930T024821_8b3c3b30`
- Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic.py`
- Type: analysis-only postdiagnostic
- Runtime: ~2.01 s
- New simulations: 0
- New control steps: 0
- New training / gradient / refit steps: 0
- Validation64 opened: no
- Sealed test accessed: no
- Mobile robot MPPI resumed: no

## Verified result

The audit clarified that v27’s fixed-H12 result is only an H15-referenced relative tradeoff success, not an absolute deployability/safety success.

Key numbers:

- v27 fixed H12 vs H15:
  - decision saving: 19.62%
  - solver saving: 21.17%
  - relative catastrophic bad rows: 0
  - H15-referenced pass5: true
- Success-sensitive accounting:
  - fixed H12 absolute unsafe rows: 2
  - fixed H15 absolute unsafe rows: 2
  - both-fail rows: 2
  - both-fail source candidate index: `[242]`
  - success-sensitive fixed-H12 pass5: false
- Current adaptive selector:
  - H counts: `{'12': 6, '15': 18}`
  - decision saving: 3.72%
  - absolute unsafe rows: 2
  - success-sensitive pass5: false

Combined development context now recorded:

- Fresh v21+v23+v25+v27: 80 rows, relative H12-bad = 0, H12-beneficial = 76, decision saving = 17.90%.
- All opened v19+v21+v23+v25+v27: 104 rows, relative H12-bad = 3, decision saving = 17.06%, physical delta still unfavorable due v19 counterevidence.

## Scientific decision

Do not validate the current adaptive H12/H15 selector on this distribution.

The evidence now points to a comparison/scenario/failure-accounting issue before more selector retraining: fresh stress-pool rows mostly favor fixed true H12, but hard both-fail cases can make an H15-referenced gate look successful while absolute safety fails. The next useful step is not another label-density sweep and not validation; it is a success-aware scenario/comparison audit/protocol that separates:

1. feasible rows where both H12 and H15 succeed,
2. both-fail hard rows such as v27 source242,
3. H12-only failure / adaptive-opportunity rows such as opened v19,
4. H15-only failures if present.

Terminal-risk/value refit or bounded retraining remains deferred only until that audit shows whether adaptive opportunity persists beyond fixed H12 and whether failures are separable by state/scenario features.

## Artifacts updated

Main outputs:

- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_fixed_h12_primary_v27_postdiagnostic_20260930T0300Z/completed.json`
- `research_artifacts/aws_runs/20260930T024821_8b3c3b30/registry.json`
- `research_artifacts/aws_state/continue_state_20260930T0300_after_v27_postdiagnostic.md`

Docs/state updated by the script:

- `STATUS.md`
- `RESEARCH_LOG.md`
- `DECISIONS.md`
- `RESULTS_AUDIT.md`
- `REPRODUCTION_PROTOCOL.md`
- `EXPERIMENT_REGISTRY.csv`
- `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`

Astra response log now includes new stable recommendation:

- `A13_both_fail_rows_must_not_count_as_successful_fixed_H12_pass`

## Backup / next-iteration gate

Latest verified backup before this work covered commit `78fa84b66a43fc263ba3aa4d9c2077da1acb4e31`.

New artifacts now require external backup before more unique science:

- Backup request written: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V27_POSTDIAGNOSTIC_20260930T0300Z.json`

Next after verified backup: freeze a v28 success-aware scenario/comparison audit/protocol; no validation64 or sealed test access.


## 2026-09-30T02:55:39.835099+00:00



## 2026-09-30T03:02:08.464223+00:00

<!-- vehicle-success-aware-scenario-comparison-audit-v28-20260930T0315Z -->
## 2026-09-30 v28 success-aware scenario/comparison audit

UTC: 2026-09-30T03:04:16.008112+00:00. Analysis-only development audit completed with no simulations/control steps/training/refit, no validation64-bank access and no sealed-test access. It partitions the opened H12/H15 evidence by success semantics: fresh v21+v23+v25+v27 has 80 collapsed rows with H12-only relative failures=0 and both-fail rows=2 (v27 source242), while opened v19 has 24 collapsed rows with H12-only failures=3 (6 repeat rows) and H15 unsafe=0. All opened collapsed evidence therefore contains H12-only failures=3 and both-fail rows=2; relative timing gates and absolute deployability are now explicitly separated. Decision: do not validate the current adaptive H12/H15 selector. Freeze success-aware longer-H feasibility protocol `research_artifacts/aws_protocols/vehicle_true_variable_horizon_success_aware_scenario_comparison_audit_v28_frozen_success_aware_followup_20260930T0315Z.json` for a bounded H12/H15/H25/H35 identical-state probe on source242 both-fail rows, v19 H12-only failures and safe matched controls after verified backup. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V28_SUCCESS_AWARE_SCENARIO_COMPARISON_AUDIT_20260930T0315Z.json`.


## 2026-09-30T03:06:22.950728+00:00



## 2026-09-30T03:13:22.088319+00:00

<!-- vehicle-success-aware-longer-H-feasibility-v29-20260930T0340Z -->
## 2026-09-30 v29 success-aware longer-H feasibility probe

UTC: 2026-09-30T03:22:13.345024+00:00. Development-only identical-state H12/H15/H25/H35 probe completed; validation64 closed, sealed test closed, no training/refit. Budget 44 episodes / 2557 control steps. Source242 H12/H15 both-fail rows rescued by H25/H35=2; v19 H12-only failures reproduced=3; no-safe states=0; safe controls fastest H12=6. Decision: Longer fixed-H rescues at least one source242 H12/H15 both-fail row; broaden fixed-H/scenario comparison before any adaptive selector validation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_success_aware_longer_H_feasibility_probe_v29_20260930T0340Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V29_SUCCESS_AWARE_LONGER_H_PROBE_20260930T0340Z.json`.


## 2026-09-30T03:25:50.122213+00:00



## 2026-09-30T03:42:40.605472+00:00

<!-- vehicle-three-way-selector-feature-audit-v30b-fast-20260930T0410Z -->
## 2026-09-30 v30b three-way selector feature/separability audit

UTC: 2026-09-30T03:46:34.079983+00:00. Analysis-only over v29; validation64 closed, sealed test closed, simulations=0, control_steps=0, training/refit=0. This supersedes the failed v30 brute-force analysis attempt, which timed out with exit_status=-15 and no scientific outputs. On the 11 opened v29 states, success-sensitive bad rows: fixed H12=5, fixed H15=2, fixed H25=4, fixed H35=4, oracle fastest-safe H12/H15/H35=0. Two-threshold feature rule: in-sample bad=0, LOO bad=1, LOO saving vs fixed H35=48.30% if safety holds. Decision: Oracle triage opportunity is real on v29, but the simple deployable feature rule did not pass leave-one-out. After backup, acquire more pre-outcome source-independent labels or run a bounded terminal-risk/value refit before validating any selector. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_three_way_selector_feature_audit_v30b_fast_20260930T0410Z/completed.json`. Backup required before further unique science: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_THREE_WAY_SELECTOR_FEATURE_AUDIT_20260930T0410Z.json`.


## 2026-09-30T03:49:58.631315+00:00

<!-- vehicle-v30b-feature-stability-postdiagnostic-20260930T0355Z -->
## 2026-09-30 v30b feature-stability postdiagnostic

UTC: 2026-09-30T03:51:36.708427+00:00. Analysis-only over opened v29/v30b; simulations=0, control_steps=0, training/refit=0, validation64 closed, sealed test closed. Diagnostic result: opened features are separable in-sample with a low-complexity margin rule, but leave-one-out model selection is unstable on the tiny risk cluster; this is not yet deployable evidence. Margin-rule same-opened-row metric bad=0, H counts={'12': 6, '15': 3, '35': 2}, decision saving vs fixed H35=50.57%; v30b LOO remained bad=1. H35 margin on `abs_obs_00` is 0.027297 from only 2 rows; H15 margin on `abs_obs_07` after H35 guard is 0.167916 from 3 rows. This is not independent confirmation; it is a handoff diagnostic for Astra. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_FEATURE_STABILITY_POSTDIAGNOSTIC_20260930T0355Z.json`.


## 2026-09-30T03:52:30.851161+00:00
Elapsed since first supervisor event: **3d 16h 56m 7.289s** at `2026-09-30T03:51:36.708427Z` (`320167.289096 s`; desktop conversation excluded). Cumulative server API `total_tokens`: **unavailable via repository tools in this cycle**; treated as unknown, not zero.

Concrete actions completed:

1. **Verified coordination state**
   - Re-read `COORDINATION.md`, `LATEST.md`, `NEXT_REVIEW_REQUEST.json`, and recent `RESPONSE_LOG.md`.
   - Checked for `docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json`; none was present.

2. **Materialized latest supervisor-reported backup proof**
   - Wrote:
     - `research_artifacts/aws_backup_proofs/backup_proof_20260930T035044_from_supervisor_context_after_v30b_and_feature_stability_source.json`
   - SHA256:
     - `bd6ffe2f133b1b2f31967c595f270c5840db87b0bef4f5ce7bca331f72c198e1`
   - This records the verified backup reported in supervisor context at `2026-09-30T03:50:44Z`, commit `7845b847e31745d589d769a85d0211cd2eb4bbc1`.

3. **Executed bounded analysis-only postdiagnostic**
   - Experiment:
     - `20260930T035136_ee411d20`
   - Script:
     - `experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic.py`
   - Runtime:
     - `2.006 s`
   - Classification:
     - Development analysis only over already-opened v29/v30b artifacts.
     - No simulations.
     - No control steps.
     - No training/refit.
     - No validation64 access.
     - No sealed-test access.

4. **Postdiagnostic result**
   - Artifacts:
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/summary.md`
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/raw.json`
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_feature_stability_postdiagnostic_20260930T0355Z/completed.json`
   - Headline:
     - v30b oracle H12/H15/H35 triage: `bad=0`, saving vs fixed H35 `50.57%` on 11 opened states.
     - v30b LOO feature rule: `bad=1`, nominal saving vs fixed H35 `48.30%`; still not deployable/safe evidence.
     - Same-opened-row margin rule: `bad=0`, H counts `{'12': 6, '15': 3, '35': 2}`, saving vs fixed H35 `50.57%`.
     - H35 separability margin on `abs_obs_00`: `0.027297`, based on only 2 H35 rows.
     - H15 separability margin on `abs_obs_07`: `0.167916`, based on 3 H15 rows.
   - Execution-level interpretation:
     - Opened features are separable in-sample with a low-complexity margin rule, but leave-one-out model selection is unstable on the tiny risk cluster.
     - This is **not independent confirmation** and does **not** authorize validation64 or sealed-test use.

5. **Astra handoff updated**
   - Updated:
     - `docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json`
     - New request ID: `v30b-feature-stability-postdiagnostic-20260930T0355Z`
   - Updated:
     - `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`
   - The request asks Astra to choose among:
     - fresh source-independent triage label acquisition,
     - bounded terminal-risk/value refit/training,
     - scenario/comparison redesign or stratification.

6. **State preserved**
   - Persisted state via `update_state`.
   - Key blocker:
     - A new external backup is required before any unique simulations, refits/training, validation64 access, or final-test access.
   - Backup request written:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V30B_FEATURE_STABILITY_POSTDIAGNOSTIC_20260930T0355Z.json`

Next safe iteration should begin by verifying/obtaining backup coverage for the postdiagnostic outputs and updated Astra handoff, then checking for `ANALYSIS_READY.json` and implementing Astra’s selected direction.


## 2026-09-30T03:57:35.748299+00:00

<!-- vehicle_true_variable_horizon_v30b_postdiagnostic_gate_status_v0-20260930T0400Z -->
## 2026-09-30 v30b postdiagnostic backup/Astra gate status

UTC: 2026-09-30T03:59:35.067037+00:00. Metadata-only gate status completed; no simulation/control/training/refit/validation64/sealed-test access. Adequate backup=`False` with reasons=`['backup_status_not_verified_or_missing', 'remaining_changed_files_not_zero_or_missing', 'missing_verified_package_metadata', 'backup_time_predates_postdiagnostic_or_unparseable']`. Astra ready for current request `v30b-feature-stability-postdiagnostic-20260930T0355Z`=`False`. Next action: Wait for/obtain verified backup covering this gate status and v30b postdiagnostic, then re-check ANALYSIS_READY.json.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_postdiagnostic_gate_status_v0_20260930T0400Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_postdiagnostic_gate_status_v0_20260930T0400Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_postdiagnostic_gate_status_v0_20260930T0400Z/completed.json`.


## 2026-09-30T04:04:17.901544+00:00

<!-- vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31-20260930T0405Z -->
## 2026-09-30 v31 cluster-stability diagnostic

UTC: 2026-09-30T04:06:16.160475+00:00. Analysis-only over already-opened development artifacts; no simulations/control/training/refit/validation64/sealed-test access. Row-level two-feature LOO bad=0, but leave-one-source-family-out bad=5 with non-default oracle H15/H35 represented by one source-family each. This is not deployable validation; it is evidence for Astra's pending direction choice. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_CLUSTER_STABILITY_DIAGNOSTIC_20260930T0405Z.json`.


## 2026-09-30T04:07:31.557516+00:00
Elapsed service lifetime since first supervisor event: **3d 17h 10m 46.741s** at v31 completion (`2026-09-30T04:06:16.160475Z`). Cumulative server API `total_tokens`: **unavailable via repository tools this cycle; not treated as zero**. Desktop conversation tokens excluded.

Concrete actions completed this iteration:

1. **Verified coordination/evidence context**
   - Read `COORDINATION.md`, current `NEXT_REVIEW_REQUEST.json`, `LATEST.md`, and prior postdiagnostic state.
   - Confirmed **no `ANALYSIS_READY.json`** was present before/after the action, so no fresh Astra report was available for v29/v30b/v31.

2. **Materialized pre-run backup proof from supervisor context**
   - Wrote:
     - `research_artifacts/aws_backup_proofs/backup_proof_20260930T040506_from_user_context_after_v31_source_before_v31_run.json`
   - This proof covers repository state through the v31 source before execution, **not v31 outputs**.

3. **Ran v31 cluster-stability diagnostic**
   - Script:
     - `experiments/bohn2021_aws/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31.py`
   - Run registry:
     - `research_artifacts/aws_runs/20260930T040615_4f9729e6/registry.json`
   - Runtime: **6.01 s**
   - Classification: **development analysis only**
   - New simulations/control steps/training/refits/validation64/sealed-test access: **0 / 0 / 0 / 0 / false / false**

4. **v31 verified result**
   - Outputs:
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/summary.md`
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/raw.json`
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v30b_cluster_stability_diagnostic_v31_20260930T0405Z/completed.json`
   - Headline:
     - 11 opened rows collapse to **5 source-family groups**.
     - Oracle-label family counts: **H12=3**, **H15=1**, **H35=1**.
     - Two-feature in-sample rule: **0 bad rows**, **50.57%** decision saving vs fixed H35 on opened rows.
     - Two-feature row-level LOO: **0 bad rows**, **50.57%** saving vs fixed H35.
     - Leave-one-source-family-out: **5 bad rows**, nominal **19.75%** saving, not a valid speed claim.
     - Failing grouped folds:
       - `v19_case05_H15_risk_family`: 3 bad rows, held-out class H15 missing from training.
       - `v27_case09_H35_rescue_family`: 2 bad rows, held-out class H35 missing from training.

5. **Scientific status preserved**
   - Execution-level conclusion only: the opened rows contain feature signal, but H15/H35 regimes are represented by too few independent source families for deployable/model-selection claims.
   - No validation64 or sealed final test was opened.
   - No reproduction or speed claim is supported.

6. **Astra handoff updated**
   - Updated:
     - `docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json`
   - New request id:
     - `v31-cluster-stability-diagnostic-20260930T0405Z`
   - Updated:
     - `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`
   - Request asks Astra to choose the next scientific direction among fresh source-independent triage labels, terminal-risk/value refit/training, or scenario/comparison redesign/stratification.

7. **State preserved**
   - Wrote:
     - `research_artifacts/aws_state/continue_state_20260930T0407_after_v31_executor_state.md`
   - Persisted structured state with `update_state`.

Current blocker:

- **Post-v31 external backup is not yet verified.**
- v31 wrote backup request:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_CLUSTER_STABILITY_DIAGNOSTIC_20260930T0405Z.json`
- Until a verified backup covers v31 outputs/docs/state/registry/response log, I should not run new simulations, refits, training, validation64, or sealed-test work.

Next iteration queue:

1. Verify/obtain post-v31 external backup.
2. Check `docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json`.
3. If Astra has returned a matching/superseding report, read it and execute the selected plan.
4. If Astra is still pending, continue only reversible integrity/preparation work within the already approved scope.

<!-- vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0-20260930T0415Z -->
## 2026-09-30 v31 backup/Astra gate status

UTC: 2026-09-30T04:10:28.271978+00:00. Metadata-only gate status completed; no simulation/control/training/refit/validation64/sealed-test access. Adequate v31 backup=`False` with reasons=`['backup_status_not_verified_or_missing', 'remaining_changed_files_not_zero_or_missing', 'missing_backup_commit', 'missing_verified_package_metadata', 'backup_time_predates_v31_artifacts_or_unparseable']`. Astra ready for current request `v31-cluster-stability-diagnostic-20260930T0405Z`=`False`. Next action: Obtain verified external backup covering v31 outputs/handoff/state/registry before any new simulation/refit/validation. Continue only reversible integrity checks if needed.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/completed.json`.


## 2026-09-30T04:11:30.203922+00:00
Elapsed service lifetime since first supervisor event: **3d 17h 14m 58.853s** at `2026-09-30T04:10:28.271978Z`. Cumulative server API `total_tokens`: **unavailable via repository tools this cycle; not treated as zero**. Desktop conversation tokens excluded.

Concrete actions completed:

1. **Checked current evidence/coordination state**
   - Confirmed `docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json` is still absent.
   - Current `NEXT_REVIEW_REQUEST.json` remains:
     - `v31-cluster-stability-diagnostic-20260930T0405Z`
   - Latest old Astra report still predates v29/v30b/v31 and was not treated as analysis of new results.

2. **Created and ran a metadata-only v31 backup/Astra gate status script**
   - Script:
     - `experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0.py`
   - Run registry:
     - `research_artifacts/aws_runs/20260930T041028_4dccdd03/registry.json`
   - Runtime: **2.01 s**
   - Classification: **metadata-only backup/Astra gate; no simulations, no control steps, no training, no selector refit, no validation64, no sealed test**

3. **Verified gate result**
   - Outputs:
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/summary.md`
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/raw.json`
     - `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_backup_astra_gate_status_v0_20260930T0415Z/completed.json`
     - `research_artifacts/aws_state/continue_state_20260930T0415Z_after_v31_backup_astra_gate_status.md`
   - Backup gate outcome:
     - `adequate_v31_backup = false`
     - Reasons:
       - `backup_status_not_verified_or_missing`
       - `remaining_changed_files_not_zero_or_missing`
       - `missing_backup_commit`
       - `missing_verified_package_metadata`
       - `backup_time_predates_v31_artifacts_or_unparseable`
   - New backup request written:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_BACKUP_ASTRA_GATE_STATUS_20260930T0415Z.json`

4. **Astra gate outcome**
   - `ANALYSIS_READY.json`: **not present**
   - No matching/superseding Astra report for:
     - `v31-cluster-stability-diagnostic-20260930T0405Z`
   - Therefore I did **not** select a new scientific branch among fresh source labels, selector/value refit, or scenario redesign.

5. **Scientific evidence preserved**
   - v31 remains development-only evidence:
     - 11 opened rows collapse to 5 source-family groups.
     - H12 has 3 source-family groups; H15 and H35 each have only 1.
     - Row-level LOO can separate opened rows, but leave-one-source-family-out fails with 5 bad rows.
   - No deployable selector, validation, reproduction, or speed claim is supported.

6. **Budgets/access this iteration**
   - New development simulation episodes: **0**
   - New control steps: **0**
   - New training/gradient steps: **0**
   - Selector refits: **0**
   - Validation64 opened: **false**
   - Sealed test accessed: **false**

7. **State persisted**
   - Structured state updated via `update_state`.
   - Current blockers and next queue preserved.

Current blocker:

- A verified external backup after v31/gate-status outputs is still required before any new simulation, selector refit, training, validation64 access, or final-test work.

Next safe queue:

1. Verify/obtain external backup covering v31 outputs, gate-status outputs, updated docs/state/registry, and backup requests.
2. Re-check `docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json`.
3. If Astra returns a matching/superseding report, read it and implement its selected plan.
4. If Astra remains pending, continue only reversible integrity/preparation work; do not independently choose a new scientific direction.


## 2026-09-30T04:16:49.261030+00:00

<!-- vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0-20260930T042029Z -->
## 2026-09-30 v31 post-backup/Astra gate recheck

UTC: 2026-09-30T04:20:29.247346+00:00. Metadata-only/reversible audit completed; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 17h 24m 59.828s`; server API total_tokens `unknown (research.sqlite not found)`. Pre-existing v31 outputs backed up=`False` (proof `None`), but this recheck's source/output/proof require follow-up backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_POSTBACKUP_ASTRA_GATE_RECHECK_20260930T042029Z.json`. Astra ANALYSIS_READY for `v31-cluster-stability-diagnostic-20260930T0405Z` present/matching=`False`. Next action: Backup remains inadequate for pre-existing v31 outputs; do not run new simulation/refit/validation. Diagnose backup only.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_postbackup_astra_gate_recheck_v0_20260930T042029Z/completed.json`.


## 2026-09-30T04:22:51.350079+00:00

<!-- vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0-20260930T042732Z -->
## 2026-09-30 v31 04:23 backup/Astra materialization

UTC: 2026-09-30T04:27:32.786046+00:00. Metadata-only/reversible step; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 17h 32m 3.367s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. The explicit user-context backup `2026-09-30T04:23:35.649634+00:00` / commit `122927b8b338861842766471537414af8020c480` / package SHA256 `2c6b3cf0693e34d8229f38b9c801b7fcc279d4918af45876599e353306de9835` is materialized at `research_artifacts/aws_backup_proofs/backup_proof_20260930T042335_from_user_context_after_v31_postbackup_recheck.json`; known pre-04:23 v31/postbackup-recheck artifacts covered=`True`. This run's source/output/proof/doc edits require `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0423_BACKUP_ASTRA_MATERIALIZE_20260930T042732Z.json` before unique simulation/refit/validation. Astra ANALYSIS_READY for `v31-cluster-stability-diagnostic-20260930T0405Z` matching=`False`. Next action: Astra analysis for v31 is still pending. Continue only reversible preparation/integrity work and do not choose a fresh scientific branch until a matching/superseding report is read.. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0_20260930T042732Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0_20260930T042732Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0423_backup_astra_materialize_v0_20260930T042732Z/completed.json`.


## 2026-09-30T04:28:30.904894+00:00



## 2026-09-30T04:32:44.839446+00:00
Elapsed service lifetime (latest verified supervisor backup): **3d 17h 33m 44.665s** since 2026-09-26T10:55:29.419331Z.  
Cumulative server API total_tokens: **unknown** via repository tools (`research.sqlite` unavailable); desktop conversation tokens excluded.

Concrete actions completed this iteration:

1. **Rechecked Astra coordination state**
   - Read `COORDINATION.md` and current `NEXT_REVIEW_REQUEST.json`.
   - Confirmed current request remains:
     - `v31-cluster-stability-diagnostic-20260930T0405Z`
   - Checked `docs/bohn2021_takeover/astra_reviews/`: **no `ANALYSIS_READY.json` present**.
   - `LATEST.md` still points to old `20260929T153837Z.md`, which predates v29/v30b/v31 and is **not** valid analysis of the new results.

2. **Re-inspected key evidence**
   - v29: 11 opened states; H35 rescues both H12/H15 both-fail rows; H12/H15/H35 tradeoff is development-only.
   - v30b: oracle H12/H15/H35 triage bad=0 with 50.5688% decision-time saving vs fixed H35 on opened rows, but not deployable.
   - v31: 11 rows collapse to 5 source-family groups; oracle family counts H12=3, H15=1, H35=1; row-level LOO bad=0 but leave-one-source-family-out bad=5. This remains evidence of inadequate source-independent non-default label coverage, not validation success.

3. **Materialized the latest user-context backup proof**
   - Wrote:
     - `research_artifacts/aws_backup_proofs/backup_proof_20260930T042914_from_user_context_after_v31_0427_materialization.json`
   - It records the verified supervisor backup:
     - time: `2026-09-30T04:29:14.084561+00:00`
     - commit: `15a24f749073d360ceb96d1f563b6a909c71fc84`
     - package SHA256: `2dd145451f04afc4b2f36176bc4542bbecfa0515393874b7ff60ba8a3c910d1f`
   - Interpreted as covering known pre-04:29 v29/v30b/v31 and 04:27 materialization outputs per explicit supervisor context.

4. **Preserved next-iteration state**
   - Wrote:
     - `research_artifacts/aws_state/continue_state_20260930_after_user_context_0429_backup_and_astra_pending.md`
   - Wrote backup request for files created after the 04:29 backup:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_USER_CONTEXT_0429_PROOF_AND_STATE_20260930.json`
   - Updated durable supervisor state via `update_state`.

Budgets/access this iteration:
- New development simulations: **0**
- New control steps: **0**
- Training/gradient steps: **0**
- Selector refits: **0**
- Validation64 opened: **false**
- Sealed test accessed: **false**
- `run_experiment` calls: **0**

Current blockers / next safe action:
- **Astra analysis is still pending** for `v31-cluster-stability-diagnostic-20260930T0405Z`; I did not select a fresh scientific branch.
- Newly written post-04:29 proof/state/request files need another external backup before unique simulation/refit/training/validation/final-test work.
- Next iteration should first check for a matching/superseding `ANALYSIS_READY.json`; if present, verify cited evidence and execute Astra’s plan. If still absent, continue only reversible integrity/preparation work.


## 2026-09-30T04:35:33.145173+00:00
Elapsed service lifetime: **latest verified available = 3d 17h 33m 44.665s** since `2026-09-26T10:55:29.419331Z` at the 2026-09-30 04:29 backup. Current exact wall time is unavailable via repository tools.  
Cumulative server API total_tokens: **unknown**, because `research.sqlite` was not found via repository tools; desktop conversation tokens excluded.

Concrete actions completed this iteration:

1. **Rechecked Astra coordination**
   - Read/rechecked:
     - `docs/bohn2021_takeover/astra_reviews/COORDINATION.md`
     - `docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json`
     - `docs/bohn2021_takeover/astra_reviews/LATEST.md`
     - `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md` tail
   - Confirmed current request remains:
     - `v31-cluster-stability-diagnostic-20260930T0405Z`
   - Repository listing still shows **no `ANALYSIS_READY.json`**.
   - `LATEST.md` still points to old `20260929T153837Z.md`, which predates v29/v30b/v31 and must not be treated as analysis of the new evidence.

2. **Rechecked backup state**
   - Latest verified backup remains the 04:29 supervisor backup:
     - time: `2026-09-30T04:29:14.084561+00:00`
     - commit: `15a24f749073d360ceb96d1f563b6a909c71fc84`
     - package SHA256: `2dd145451f04afc4b2f36176bc4542bbecfa0515393874b7ff60ba8a3c910d1f`
   - No later `backup_proof_20260930T043*` proof was present.
   - This backup covers known pre-04:29 v29/v30b/v31 and 04:27 materialization artifacts per supervisor context.
   - Files written after that backup now need follow-up external backup before unique simulation/refit/training/validation/final-test work.

3. **Re-inspected key evidence**
   - v29: development-only H12/H15/H25/H35 identical-state probe, 11 opened states, 44 episodes, 2557 control steps; H35 rescues both source242 H12/H15 both-fail rows; H12 fastest safe on 6 safe-control rows; H15 fastest on 3 v19 rows.
   - v30b: analysis-only; oracle H12/H15/H35 triage bad=0 with 50.5688% decision-time saving vs fixed H35 on opened rows, but simple feature rule is not deployable due LOO failure.
   - v31: analysis-only; 11 rows collapse to 5 source-family groups; oracle family counts H12=3, H15=1, H35=1; row-level LOO bad=0 but leave-one-source-family-out bad=5. This remains evidence of insufficient source-independent non-default label coverage, not validation success.

4. **Preserved state for next iteration**
   - Wrote:
     - `research_artifacts/aws_state/continue_state_20260930_after_0429_backup_astra_absent_recheck_v2.md`
   - Wrote backup request:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_0429_BACKUP_ASTRA_ABSENT_RECHECK_V2_20260930.json`
   - Updated durable supervisor state with phase, evidence, blockers and next action.

Budgets/access this iteration:
- New development simulations: **0**
- New control steps: **0**
- Training/gradient steps: **0**
- Selector refits: **0**
- Validation64 opened: **false**
- Sealed test accessed: **false**
- `run_experiment` calls: **0**

Current blockers / next safe action:
- **Astra analysis is still pending** for `v31-cluster-stability-diagnostic-20260930T0405Z`; I did not choose a fresh scientific branch.
- **Backup needed** for post-04:29 proof/state/request files before any unique simulation, refit, training, validation64 access, or final-test work.
- Next iteration should first check for a matching/superseding `ANALYSIS_READY.json`; if present, verify the linked Astra report against raw evidence and execute its plan after backup. If still absent, continue only reversible integrity/preparation work.

<!-- vehicle_true_variable_horizon_v31_0436_backup_astra_recheck_v0-20260930T043931Z -->
## 2026-09-30 v31 04:36 backup/Astra gate recheck

UTC: 2026-09-30T04:39:31.989213+00:00. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 17h 44m 2.570s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. User-context backup `2026-09-30T04:36:20.681764+00:00` / commit `a035b38e4a01581045c745ee3956f4c8c4e2414f` / package SHA256 `59ebb7023f5ce9a1b4a3bcef37ba47af5c21e22aa1055af83226635fef8477b5` is materialized at `research_artifacts/aws_backup_proofs/backup_proof_20260930T043620_from_user_context_after_0429_state.json` and covers known pre-04:36 artifacts=`True`. Current Astra request `v31-cluster-stability-diagnostic-20260930T0405Z`; ANALYSIS_READY present/matching=`False`. New outputs from this audit require backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0436_BACKUP_ASTRA_RECHECK_20260930T043931Z.json` before unique science. Evidence status remains: v29/v30b/v31 are development/opened-row diagnostics only; v31 shows source-family coverage insufficiency, not a deployable selector or reproduction claim. Next action: Astra analysis remains pending for v31. Do not select a fresh scientific branch. Continue only reversible integrity/preparation work until a matching/superseding ANALYSIS_READY.json arrives and post-04:36 metadata outputs are backed up.


## 2026-09-30T04:41:23.402313+00:00

<!-- vehicle_true_variable_horizon_v31_label_coverage_lower_bound_v0-20260930T044251Z -->
## 2026-09-30 v31 label-coverage lower-bound diagnostic

UTC: 2026-09-30T04:42:51.420306+00:00. Development-analysis only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 17h 47m 22.001s`. Current source-family counts by oracle label `{'12': 3, '15': 1, '35': 1}` imply lower-bound basic grouped-CV class-presence deficits `{'12': 0, '15': 1, '35': 1}`; non-default total deficit `2`. This is not a selector validation or a new branch decision; carry to Astra request `v31-cluster-stability-diagnostic-20260930T0405Z`. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_label_coverage_lower_bound_v0_20260930T044251Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_label_coverage_lower_bound_v0_20260930T044251Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_label_coverage_lower_bound_v0_20260930T044251Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_LABEL_COVERAGE_LOWER_BOUND_20260930T044251Z.json`.


## 2026-09-30T04:47:07.871793+00:00

<!-- vehicle_true_variable_horizon_v31_source_budget_bounds_v0-20260930T044819Z -->
## 2026-09-30 v31 source-coverage budget/identifiability bounds

UTC: 2026-09-30T04:48:19.103132+00:00. Analysis-only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 17h 52m 49.684s`. Current source-family counts `{'12': 3, '15': 1, '35': 1}`. Same-family densification cannot fix LOGO missing-class coverage. If Astra chooses fresh source-independent labels, basic grouped-CV lower bound is `{'12': 0, '15': 1, '35': 1}` with `6` H12/H15/H35 rollout episodes (`8` including H25); three-source target lower bound is `{'12': 0, '15': 2, '35': 2}` with `12` or `16` episodes. Not a branch decision or validation. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_SOURCE_BUDGET_BOUNDS_20260930T044819Z.json`. Astra request `v31-source-coverage-budget-bounds-20260930T044819Z`.


## 2026-09-30T04:49:32.258779+00:00
Elapsed service lifetime (latest measured): 3d 17h 52m 49.684s since 2026-09-26T10:55:29.419331Z.  
Cumulative server API total_tokens: unknown via repository tools; desktop conversation tokens excluded.

Concrete actions completed:

1. Ran bounded analysis-only diagnostic:
   - Experiment ID: `20260930T044818_81f75082`
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_source_budget_bounds_v0.py`
   - Runtime: 2.01s
   - No simulations, no control steps, no selector refits, no training/gradient steps.
   - No validation64 bank access.
   - No sealed-test access.

2. Verified/quantified v31 source-family identifiability limits:
   - Current independent source-family counts by oracle label:
     - H12: 3
     - H15: 1
     - H35: 1
   - Same-family extra rows cannot fix leave-one-source-family-out missing-class structure for H15/H35.
   - If Astra selects fresh source-independent triage-label acquisition:
     - Basic LOGO class-presence lower bound: +1 H15-like and +1 H35-like independent family.
       - H12/H15/H35 only: minimum 6 rollout episodes, estimated 349 control steps.
       - Including H25 reference: minimum 8 rollout episodes, estimated 465 control steps.
     - Three-source stability target: +2 H15-like and +2 H35-like independent families.
       - H12/H15/H35 only: minimum 12 rollout episodes, estimated 698 control steps.
       - Including H25 reference: minimum 16 rollout episodes, estimated 930 control steps.

3. Refreshed Astra handoff:
   - `docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json`
   - New request ID: `v31-source-coverage-budget-bounds-20260930T044819Z`
   - Supersedes: `v31-label-coverage-lower-bound-20260930T044251Z`
   - Asked Astra to choose between:
     - fresh source-independent triage-label acquisition,
     - terminal-risk/value refit/training,
     - scenario/comparison redesign.

4. Updated/preserved state and response logs:
   - `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`
   - `research_artifacts/aws_state/continue_state_20260930T044819Z_after_v31_source_budget_bounds.md`
   - Supervisor state via `update_state`.

Key artifacts:
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_source_budget_bounds_v0_20260930T044819Z/completed.json`
- `research_artifacts/aws_runs/20260930T044818_81f75082/registry.json`
- `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_SOURCE_BUDGET_BOUNDS_20260930T044819Z.json`

Current blockers:
1. Backup gate: post-04:48 artifacts require verified external backup before any unique simulation/refit/training/validation/final-test work.
2. Astra gate: `ANALYSIS_READY.json` was absent after the run; need matching/superseding Astra analysis for request `v31-source-coverage-budget-bounds-20260930T044819Z` before selecting the next scientific branch.
3. Sealed test remains closed and unauthorized.

Next safe action: once backup and Astra analysis are available, verify the report’s cited evidence, log dispositions in `RESPONSE_LOG.md`, then implement Astra’s selected plan.

<!-- vehicle_true_variable_horizon_v31_0447_backup_astra_gate_recheck_v0-20260930T045206Z -->
## 2026-09-30 v31 04:47 backup/Astra gate recheck

UTC: 2026-09-30T04:52:06.544655+00:00. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 17h 56m 37.125s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. Materialized supervisor backup claim `2026-09-30T04:47:52.479502+00:00` at `research_artifacts/aws_backup_proofs/backup_proof_20260930T044752_from_user_context_after_v31_source_budget_bounds.json` for request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_SOURCE_BUDGET_BOUNDS_20260930T044819Z.json`; requested files present=`True`, temporal consistency with local mtimes=`False`, prior gate cleared by local check=`False`. Current Astra request `v31-source-coverage-budget-bounds-20260930T044819Z`; ANALYSIS_READY matching=`False`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0447_BACKUP_ASTRA_GATE_RECHECK_20260930T045206Z.json`. Evidence status unchanged: v31 source-budget result is analysis-only and awaits Astra branch selection.


## 2026-09-30T04:52:59.709010+00:00
Elapsed service lifetime (latest measured): **3d 17h 56m 37.125s** since `2026-09-26T10:55:29.419331Z`.  
Cumulative server API total_tokens: **unknown** (`research.sqlite` not found by repository tools); desktop conversation tokens excluded.

Concrete actions completed:

1. Ran one bounded metadata-only integrity/gate recheck:
   - Experiment ID: `20260930T045206_4372119b`
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v31_0447_backup_astra_gate_recheck_v0.py`
   - Runtime: 2.01s
   - New simulations/control/training/refits: **0**
   - validation64 bank opened: **false**
   - sealed test accessed: **false**

2. Rechecked current v31 source-budget evidence:
   - Independent source-family counts remain:
     - H12: 3
     - H15: 1
     - H35: 1
   - Basic grouped-CV class-presence lower bound remains:
     - +1 independent H15-like family
     - +1 independent H35-like family
   - Three-source stability lower bound remains:
     - +2 independent H15-like families
     - +2 independent H35-like families
   - This is still development/opened-row analysis only, not validation/test evidence and not a branch decision.

3. Checked Astra handoff:
   - Current request ID: `v31-source-coverage-budget-bounds-20260930T044819Z`
   - `ANALYSIS_READY.json`: not present
   - Latest old review still predates v29/v30b/v31, so it is not usable as analysis of current results.
   - I did **not** choose between source-label acquisition, refit/training, or scenario/comparison redesign.

4. Materialized the user/supervisor 04:47 backup claim:
   - Claimed backup time: `2026-09-30T04:47:52.479502+00:00`
   - Commit: `aa4fba19cba24fa6447df8dc43d2b16bfde4e2bd`
   - Package SHA256: `5a200a111a3cf587527f2ca4116331591ffc9218c581ce928add7140e19fdbc8`
   - All files requested by `REQUEST_BACKUP_AFTER_V31_SOURCE_BUDGET_BOUNDS_20260930T044819Z.json` were locally present.
   - However, local mtime audit found 14 requested files modified after the claimed backup time, so I conservatively **did not treat this as independently clearing the unique-science backup gate**.

Key artifacts:
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0447_backup_astra_gate_recheck_v0_20260930T045206Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0447_backup_astra_gate_recheck_v0_20260930T045206Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v31_0447_backup_astra_gate_recheck_v0_20260930T045206Z/completed.json`
- `research_artifacts/aws_backup_proofs/backup_proof_20260930T044752_from_user_context_after_v31_source_budget_bounds.json`
- `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0447_BACKUP_ASTRA_GATE_RECHECK_20260930T045206Z.json`
- `research_artifacts/aws_state/continue_state_20260930T045206Z_after_v31_0447_backup_astra_gate_recheck.md`
- Registry: `research_artifacts/aws_runs/20260930T045206_4372119b/registry.json`

Current blockers:
1. **Astra gate:** no matching/superseding Astra analysis yet for `v31-source-coverage-budget-bounds-20260930T044819Z`.
2. **Backup gate:** post-04:52 metadata outputs require external backup; local audit did not independently clear the 04:47 backup for unique science.
3. **Sealed test remains closed/unauthorized.**

State was persisted with `update_state`. Next safe action is to wait/check for Astra analysis and verified backup, then verify Astra’s report against raw evidence and implement its selected plan.

<!-- vehicle_true_variable_horizon_v31_0453_backup_astra_gate_recheck_v0-20260930T045656Z -->
## 2026-09-30 v31 04:53 backup/Astra gate recheck

UTC: 2026-09-30T04:56:56.855113+00:00. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 18h 1m 27.436s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. Materialized supervisor backup claim `2026-09-30T04:53:42.986183+00:00` at `research_artifacts/aws_backup_proofs/backup_proof_20260930T045342_from_user_context_after_v31_0447_gate_recheck.json` for request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0447_BACKUP_ASTRA_GATE_RECHECK_20260930T045206Z.json`; requested files present=`True`, temporal consistency with local mtimes=`True`, prior 04:52 metadata backup gate cleared by local check=`True`. Current Astra request `v31-source-coverage-budget-bounds-20260930T044819Z`; ANALYSIS_READY matching=`False`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0453_BACKUP_ASTRA_GATE_RECHECK_20260930T045656Z.json`. Evidence status unchanged: v31 source-budget result is analysis-only and awaits Astra branch selection.


## 2026-09-30T04:58:41.222880+00:00

<!-- vehicle_true_variable_horizon_v31_0459_backup_astra_gate_recheck_v0-20260930T050247Z -->
## 2026-09-30 v31 04:59 backup/Astra gate recheck

UTC: 2026-09-30T05:02:47.169450+00:00. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 18h 7m 17.750s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. Materialized supervisor backup claim `2026-09-30T04:59:24.496081+00:00` at `research_artifacts/aws_backup_proofs/backup_proof_20260930T045924_from_user_context_after_v31_0453_gate_recheck.json` for request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0453_BACKUP_ASTRA_GATE_RECHECK_20260930T045656Z.json` plus branch-neutral note check; requested/known files present=`True`, temporal consistency with local mtimes=`True`, prior 04:56 metadata backup gate cleared by local check=`True`. Current Astra request `v31-source-coverage-budget-bounds-20260930T044819Z`; ANALYSIS_READY matching=`False`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0459_BACKUP_ASTRA_GATE_RECHECK_20260930T050247Z.json`. Evidence status unchanged: v31 source-budget result is analysis-only and awaits Astra branch selection.


## 2026-09-30T05:04:20.690358+00:00

<!-- vehicle_true_variable_horizon_v31_0505_backup_astra_gate_recheck_v0-20260930T050748Z -->
## 2026-09-30 v31 05:05 backup/Astra gate recheck

UTC: 2026-09-30T05:07:48.230056+00:00. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 18h 12m 18.811s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. Materialized supervisor backup claim `2026-09-30T05:05:06.407045+00:00` at `research_artifacts/aws_backup_proofs/backup_proof_20260930T050506_from_user_context_after_v31_0459_gate_recheck.json` for request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0459_BACKUP_ASTRA_GATE_RECHECK_20260930T050247Z.json`; requested files present=`True`, temporal consistency with local mtimes=`True`, prior 05:02 metadata backup gate cleared by local check=`True`. Current Astra request `v31-source-coverage-budget-bounds-20260930T044819Z`; ANALYSIS_READY matching=`False`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_0505_BACKUP_ASTRA_GATE_RECHECK_20260930T050748Z.json`. Evidence status unchanged: v31 source-budget result is analysis-only and awaits Astra branch selection.


## 2026-09-30T05:09:07.144113+00:00

<!-- vehicle_true_variable_horizon_v31_astra_pending_evidence_packet_v0-20260930T051219Z -->
## 2026-09-30 v31 pending-Astra evidence packet

UTC: 2026-09-30T05:12:19.416907+00:00. Branch-neutral metadata analysis only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 18h 16m 49.998s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. Rechecked Astra gate for `v31-source-coverage-budget-bounds-20260930T044819Z`: ANALYSIS_READY matching=`False`. Inspected v29/v30b/v31/source-budget raw files; consistency checks `{'v31_cluster_and_source_counts_match': True, 'expected_current_counts_H12_3_H15_1_H35_1': True, 'v30b_oracle_h_counts_match_expected_rows': True, 'all_required_input_files_exist': True}`. Wrote packet `docs/bohn2021_takeover/astra_reviews/v31_pending_evidence_packet_20260930T051219Z.md` and backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V31_PENDING_EVIDENCE_PACKET_20260930T051219Z.json`. Latest pending metadata files are not fully externally backed up=`True`. Scientific state unchanged: development-only adaptive triage opportunity exists, but non-default H15/H35 source-family coverage is insufficient; await Astra before branch selection.


## 2026-09-30T05:14:51.976435+00:00

<!-- vehicle_true_variable_horizon_v32_h12_supported_default_h35_diagnostic_v0-20260930T051611Z -->
## 2026-09-30 v32 H12-supported/H35-default diagnostic

UTC: 2026-09-30T05:16:11.288834+00:00. Development analysis only; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 18h 20m 41.870s`; server API total_tokens `unknown (research.sqlite not found in checked locations)`. On the same opened v29/v30b/v31 rows, a diagnostic rule H12 if `abs_obs_07 <= 0.776202689723` else H35 gave LOGO bad_count `1`, h_counts `{'12': 6, '15': 0, '35': 5}`, decision_sum_s `34.244503`, saving vs fixed H35 `0.399251`. Residual bad rows `[('v19_c13', 'v19_case05_H15_risk_family', 15, 35, 'large_physical_excess_vs_best_safe')]`. Interpretation: opened data contain source-family-supported H12-shortening signal, but H15 remains unsupported; not validation/final evidence and not a branch decision. Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_H12_SUPPORTED_DEFAULT_H35_DIAGNOSTIC_20260930T051611Z.json`. Astra gate matching=`False` for `v31-source-coverage-budget-bounds-20260930T044819Z`.


## 2026-09-30T05:18:12.073763+00:00
3d 18h 20m 41.870s elapsed since first supervisor event (2026-09-26T10:55:29.419331Z). Server API total_tokens remain unknown because `research.sqlite` was not found; desktop conversation tokens excluded.

Concrete actions completed:

1. Checked Astra coordination state:
   - `ANALYSIS_READY.json` is still absent.
   - `LATEST.md` still points to old report `20260929T153837Z.md`, which predates v29/v30b/v31/v32 and is not current analysis for these findings.

2. Ran the prepared branch-neutral v32 diagnostic:
   - Experiment: `20260930T051611_9f01149d`
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v32_h12_supported_default_h35_diagnostic_v0.py`
   - Classification: development analysis only.
   - No simulations, no control steps, no selector refits, no training, no validation64 access, no sealed-test access.

3. v32 numerical finding:
   - Diagnostic rule: `H12 if abs_obs_07 <= 0.776202689723 else H35`
   - It deliberately never predicts H15, so it is not deployable as a three-way selector.
   - On opened v29/v30b/v31 rows:
     - LOGO bad_count: `1`
     - horizon counts: `H12=6, H15=0, H35=5`
     - decision_sum_s: `34.244503`
     - decision saving vs fixed H35: `39.9251%`
   - Residual bad row:
     - `v19_c13`, family `v19_case05_H15_risk_family`, oracle H15, predicted H35, reason `large_physical_excess_vs_best_safe`.
   - Same opened-row comparators:
     - Fixed H35: bad_count `4`, decision_sum_s `57.002972`
     - Oracle H12/H15/H35: bad_count `0`, decision_sum_s `28.177248`

4. Interpretation limits preserved:
   - v32 suggests there is a source-family-supported H12-vs-non-H12 shortening signal, so v31’s grouped-CV failure was not total feature absence.
   - H15 remains under-supported and unresolved.
   - Current independent source-family counts remain `H12=3, H15=1, H35=1`.
   - This is not validation evidence, not final-test evidence, not a reproduction/speed claim, and not a branch decision.

5. Refreshed Astra handoff:
   - Wrote new `docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json`
   - New request id: `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`
   - It supersedes v31 request `v31-source-coverage-budget-bounds-20260930T044819Z`.
   - It asks Astra to choose the next scientific action: fresh source-independent labels vs terminal-risk/value refit/training vs scenario/comparison redesign vs another specified bounded action.

6. Updated/preserved state:
   - v32 outcomes were logged into `RESPONSE_LOG.md`, `STATUS.md`, `RESEARCH_LOG.md`, `DECISIONS.md`, `RESULTS_AUDIT.md`, `REPRODUCTION_PROTOCOL.md`, and `EXPERIMENT_REGISTRY.csv`.
   - Wrote backup request:
     - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_NEXT_REVIEW_REFRESH_20260930T051840Z.json`

Current blockers before unique science:
- Need verified external backup covering v32 outputs and refreshed Astra request.
- Need matching/superseding Astra analysis for request `v32-h12-supported-default-h35-diagnostic-20260930T051611Z` before selecting a substantive new scientific branch.

Next safe action:
- At next iteration, check for backup proof covering `REQUEST_BACKUP_AFTER_V32_NEXT_REVIEW_REFRESH_20260930T051840Z.json`, then check `ANALYSIS_READY.json`. If Astra report is available, verify cited evidence, update `RESPONSE_LOG.md`, and implement Astra’s selected plan.

<!-- vehicle_true_variable_horizon_v32_gate_preflight_state_v0-20260930T052039Z -->
## 2026-09-30 v32 gate preflight/state preservation

UTC: 2026-09-30T05:20:39.377935+00:00. Metadata-only/reversible; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 18h 25m 9.959s`; server API total_tokens `unknown (sqlite present but no total_tokens-like column found)`. Verified current Astra request `v32-h12-supported-default-h35-diagnostic-20260930T051611Z` has `12` evidence paths with missing=`[]`. ANALYSIS_READY matching current=`False`. Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_NEXT_REVIEW_REFRESH_20260930T051840Z.json` present=`True` and files present=`True`, but latest verified backup candidate `{'path': 'current_supervisor_context', 'time': '2026-09-30T05:15:38.012838+00:00', 'status': 'verified', 'remaining_changed_files': 0, 'commit': 'f2227d24fada4c7e7ba5fb3e7cbe060b47457cc1', 'package_sha256': '081ce0831d187e40aa419262e36ff7a7fcd2a08208a2fc45e7b1196b36f9e876', 'package_ok': True}` is after request time=`False` and files-after-backup count=`16`, so v32 backup gate clear=`False`. V32 digest remains opened-development only: counts `{'12': 3, '15': 1, '35': 1}`, H12/H35 LOGO bad `1`, saving vs fixed H35 `0.3992505755588043`. Next: await backup plus matching Astra report before unique science; if Astra appears, read/verify/report dispositions.


## 2026-09-30T05:23:11.390851+00:00

<!-- api_token_usage_audit_v0-20260930T052434Z -->
## 2026-09-30 API token usage audit

UTC: 2026-09-30T05:24:34.851424+00:00. Metadata-only operational audit; no simulation/control/training/refit/validation64/sealed-test access. Service elapsed `3d 18h 29m 5.432s`. Server API total_tokens status: `277,410,897 (277.411M) from calls.usage.total_tokens`. Details: `research_artifacts/aws_diagnostics/api_token_usage_audit_v0_20260930T052434Z/raw.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_20260930T052434Z.json`.


## 2026-09-30T05:25:26.287344+00:00
3d 18h 29m 5.432s since first supervisor event; server API total_tokens are now audited at 277.411M from `/data/openai-agent/state/research.sqlite` (`calls.usage.total_tokens`), excluding this desktop conversation.

Concrete actions completed:
- Inspected current Astra handoff state:
  - `NEXT_REVIEW_REQUEST.json` is still `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`.
  - `ANALYSIS_READY.json` is absent.
  - `LATEST.md` still points to the older 20260929 report, which predates v29–v32 and is not current analysis.
- Verified the latest v32 preflight evidence and carried forward its constraints:
  - v32 remains opened-development analysis only.
  - No validation/test/reproduction/deployable-selector claim is allowed.
  - Astra remains scientific lead for selecting the next substantive branch.
- Ran one metadata-only operational experiment:
  - Experiment: `20260930T052434_8369e758`
  - Script: `experiments/bohn2021_aws/api_token_usage_audit_v0.py`
  - Output summary: `research_artifacts/aws_diagnostics/api_token_usage_audit_v0_20260930T052434Z/summary.md`
  - Raw: `research_artifacts/aws_diagnostics/api_token_usage_audit_v0_20260930T052434Z/raw.json`
  - Completed: `research_artifacts/aws_diagnostics/api_token_usage_audit_v0_20260930T052434Z/completed.json`
  - Result: SQLite present with 2 tables; selected cumulative token source is `calls.usage.total_tokens = 277,410,897`.
- Wrote/preserved next-state via `update_state`.

Access/budget this iteration:
- New simulation episodes: 0
- New control steps: 0
- Training/gradient steps: 0
- Selector refits: 0
- Validation64 bank opened: false
- Sealed test accessed: false
- Run experiments: 1 successful, 0 failed

Current blockers / gates:
- Astra v32 analysis is still pending (`ANALYSIS_READY.json` absent).
- New token-audit outputs require external backup before unique scientific simulation/refit/training/validation/test work.
  - Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_20260930T052434Z.json`

Next safe action:
1. Check for verified backup covering the token-audit request above.
2. Check for matching/superseding Astra `ANALYSIS_READY.json` for `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`.
3. If available, read and verify Astra’s report, update `RESPONSE_LOG.md`, and implement its selected plan.
4. If still absent, continue only reversible integrity/preparation; do not start fresh labels, refit/training, scenario redesign, validation, or final test.

# v32/token backup + Astra gate recheck

UTC: `2026-09-30T05:29:01.564790+00:00`. Metadata-only operational check; no simulations, no control steps, no selector refit, no training, no validation64 bank access and no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `3d 18h 33m 32.145s`.
- Cumulative server API total_tokens from research.sqlite: `277,902,212 (277.902M) from calls.usage.total_tokens`; desktop conversation tokens excluded.

## Gate findings
- Materialized prompt-supplied verified backup proof: `research_artifacts/aws_backup_proofs/backup_proof_20260930T052356_from_user_context_after_v32_pre_token_audit.json`.
- Current Astra request: `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`; ANALYSIS_READY exists=`False`, matches current=`False`.
- Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_H12_SUPPORTED_DEFAULT_H35_DIAGNOSTIC_20260930T051611Z.json`: clear=`False`; best_candidate_time=`2026-09-30T05:23:56.296257+00:00`; missing=0; files_after_backup=3.
- Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_NEXT_REVIEW_REFRESH_20260930T051840Z.json`: clear=`False`; best_candidate_time=`2026-09-30T05:23:56.296257+00:00`; missing=0; files_after_backup=3.
- Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_GATE_PREFLIGHT_STATE_20260930T052039Z.json`: clear=`False`; best_candidate_time=`2026-09-30T05:23:56.296257+00:00`; missing=0; files_after_backup=3.
- Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_SOURCE_20260930T0522Z.json`: clear=`False`; best_candidate_time=`2026-09-30T05:23:56.296257+00:00`; missing=0; files_after_backup=3.
- Backup request `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_20260930T052434Z.json`: clear=`False`; best_candidate_time=`2026-09-30T05:23:56.296257+00:00`; missing=0; files_after_backup=9.

## Current decision
- Unique scientific simulation/refit/training/validation/final-test work remains blocked: token-audit backup clear=`False`, Astra current report ready=`False`.
- Continue only reversible integrity/preparation until a later verified backup covers this recheck/token-audit output and a matching/superseding Astra report is available.

## Preserved scientific constraints
- v29/v30b/v31/v32 remain opened-development diagnostics only, not validation/test/reproduction/deployed-speed evidence.
- v32 suggests an H12-vs-non-H12 signal but leaves the single H15-family residual unresolved; Astra remains scientific lead for branch selection.
- Sealed final test remains unopened and unauthorized.



## 2026-09-30T05:29:54.201276+00:00
3d 18h 33m 32.145s since first supervisor event; server API total_tokens are now audited at 277.902M from `/data/openai-agent/state/research.sqlite` (`calls.usage.total_tokens`), excluding this desktop conversation.

Concrete actions completed:
- Checked current Astra handoff:
  - Current request remains `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`.
  - `docs/bohn2021_takeover/astra_reviews/ANALYSIS_READY.json` is still absent.
  - `LATEST.md` still points to the older 20260929 report, which predates v29–v32 and is not current analysis.
- Wrote and ran a metadata-only gate recheck:
  - Experiment: `20260930T052901_1043ae98`
  - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v32_token_backup_astra_gate_recheck_v0.py`
  - Summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_token_backup_astra_gate_recheck_v0_20260930T052901Z/summary.md`
  - Raw: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_token_backup_astra_gate_recheck_v0_20260930T052901Z/raw.json`
  - Completed: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_token_backup_astra_gate_recheck_v0_20260930T052901Z/completed.json`
- Materialized the prompt-supplied verified backup claim:
  - `research_artifacts/aws_backup_proofs/backup_proof_20260930T052356_from_user_context_after_v32_pre_token_audit.json`
  - Backup time: `2026-09-30T05:23:56.296257+00:00`
  - Commit: `76e9cc71bca2ce0b4dee1c7d07a2610829884c3d`
  - Package SHA256: `2db0fd58b9068be7cf5d7711d05151d50c3b2444d1cb26f43b506fb2b591ed59`
- Verified that this backup is not sufficient for unique new science because it predates the 05:24 token-audit outputs and the 05:29 gate-recheck outputs.
- Updated/preserved state via `update_state`, `STATUS.md`, `RESEARCH_LOG.md`, and `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`.

Access/budget this iteration:
- New simulation episodes: 0
- New control steps: 0
- Training/gradient steps: 0
- Selector refits: 0
- Validation64 bank opened: false
- Sealed test accessed: false
- Run experiments: 1 successful, 0 failed

Current gates:
- Astra v32 analysis is still pending.
- Unique scientific simulation/refit/training/validation/final-test work remains blocked until a later verified external backup covers:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_20260930T052434Z.json`
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_TOKEN_BACKUP_ASTRA_GATE_RECHECK_20260930T052901Z.json`

Scientific constraint carried forward:
- v29/v30b/v31/v32 remain opened-development diagnostics only.
- v32 suggests an H12-vs-non-H12 signal, but H15/H35 source-family coverage remains insufficient for deployable three-way selector evidence.
- No validation/test/reproduction/deployed-speed claim is justified.

Next safe action:
1. Check for verified backup after `REQUEST_BACKUP_AFTER_V32_TOKEN_BACKUP_ASTRA_GATE_RECHECK_20260930T052901Z.json`.
2. Check for matching/superseding Astra `ANALYSIS_READY.json` for `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`.
3. If both gates clear, read and verify Astra’s report, update `RESPONSE_LOG.md`, then implement Astra’s selected plan.
4. If either gate remains absent, continue only reversible integrity/preparation; do not start fresh labels, refit/training, scenario redesign, validation, or final test.

<!-- vehicle_true_variable_horizon_v32_postbackup_astra_state_recheck_v1-20260930T053333Z -->
# v32 post-backup / Astra state recheck

UTC: `2026-09-30T05:33:33.166784+00:00`. Metadata-only integrity/handoff check; no simulations, no control steps, no selector refits, no training, no validation64 access and no sealed-test access.

## Required status-line values
- Service lifetime elapsed since `2026-09-26T10:55:29.419331Z`: `3d 18h 38m 3.747s`.
- Cumulative server API total_tokens from research.sqlite: `278,598,709 (278.599M) from calls.usage.total_tokens`; desktop conversation tokens excluded.

## Backup gate
- Materialized prompt-supplied verified backup proof: `research_artifacts/aws_backup_proofs/backup_proof_20260930T053041_from_user_context_after_v32_token_gate_recheck.json`.
- Latest backup candidate: `{'path': 'research_artifacts/aws_backup_proofs/backup_proof_20260930T053041_from_user_context_after_v32_token_gate_recheck.json', 'time': '2026-09-30T05:30:41.005783+00:00', 'status': 'verified', 'remaining_changed_files': 0, 'commit': '2ff567eeaced865b54be46711d4eb9853a832156', 'package_sha256': '1ef95bdbb2e9d77eed32706a471f06983fd4eda26f60305898b82d14709501af', 'package_verification': 'github_server_sha256', 'package_ok': True}`.
- Prior v32/API-token backup requests checked: `6`; all clear=`True`.
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_20260930T052434Z.json` clear=`True`; best_time=`2026-09-30T05:30:41.005783+00:00`; missing=0; files_after_backup=0
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_API_TOKEN_USAGE_AUDIT_SOURCE_20260930T0522Z.json` clear=`True`; best_time=`2026-09-30T05:30:41.005783+00:00`; missing=0; files_after_backup=0
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_GATE_PREFLIGHT_STATE_20260930T052039Z.json` clear=`True`; best_time=`2026-09-30T05:30:41.005783+00:00`; missing=0; files_after_backup=0
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_H12_SUPPORTED_DEFAULT_H35_DIAGNOSTIC_20260930T051611Z.json` clear=`True`; best_time=`2026-09-30T05:30:41.005783+00:00`; missing=0; files_after_backup=0
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_NEXT_REVIEW_REFRESH_20260930T051840Z.json` clear=`True`; best_time=`2026-09-30T05:30:41.005783+00:00`; missing=0; files_after_backup=0
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_TOKEN_BACKUP_ASTRA_GATE_RECHECK_20260930T052901Z.json` clear=`True`; best_time=`2026-09-30T05:30:41.005783+00:00`; missing=0; files_after_backup=0

## Astra gate
- Current request: `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`; matches expected=`True`.
- ANALYSIS_READY exists=`False`; matches current=`False`.
- LATEST.md still old 20260929 report marker=`True`.

## Evidence digest rechecked
- v31/v32 family counts: v31 `{'12': 3, '15': 1, '35': 1}`, v32 `{'12': 3, '15': 1, '35': 1}`.
- v32 H12-if-low-abs_obs_07 else H35 LOGO bad_count=`1`, h_counts=`{'12': 6, '15': 0, '35': 5}`, decision saving vs fixed H35=`0.3992505755588043`.
- v32 residual bad rows: `[{'base_state_id': 'v15c13_false_positive_center_fresh_v11_fresh_case05_slot1_mid_late_control_step045_off_0', 'best_safe_physical': 11.368420858363953, 'category': 'v19_h12_only_failure_h15_safe', 'family_group': 'v19_case05_H15_risk_family', 'oracle_h': 15, 'physical_constraint_cost': 64.58471587439642, 'predicted_horizon': 35, 'reason': 'large_physical_excess_vs_best_safe', 'state_label': 'v19_c13'}]`.

## Decision
- Prior backup gate appears clear from the prompt-supplied 05:30:41 verified backup, but current unique scientific work remains blocked because Astra analysis for v32 is not ready.
- v29/v30b/v31/v32 remain opened-development diagnostics only; do not claim validation/test/reproduction/deployed-speed/deployable-selector success.
- This metadata recheck output itself requires a later verified external backup before unique scientific work.
- Sealed final test remains unopened and unauthorized.


## 2026-09-30T05:34:13.318407+00:00
3d 18h 38m 3.747s since first supervisor event; cumulative server API total_tokens are 278.599M from `/data/openai-agent/state/research.sqlite` (`calls.usage.total_tokens`), excluding this desktop conversation.

Concrete actions completed:
- Wrote and ran a metadata-only post-backup/Astra gate recheck:
  - Experiment: `20260930T053332_07ab2871`
  - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v32_postbackup_astra_state_recheck_v1.py`
  - Summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_postbackup_astra_state_recheck_v1_20260930T053333Z/summary.md`
  - Raw: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_postbackup_astra_state_recheck_v1_20260930T053333Z/raw.json`
  - Completed: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v32_postbackup_astra_state_recheck_v1_20260930T053333Z/completed.json`
- Materialized the prompt-supplied verified backup claim:
  - `research_artifacts/aws_backup_proofs/backup_proof_20260930T053041_from_user_context_after_v32_token_gate_recheck.json`
  - Backup time: `2026-09-30T05:30:41.005783+00:00`
  - Commit: `2ff567eeaced865b54be46711d4eb9853a832156`
  - Package SHA256: `1ef95bdbb2e9d77eed32706a471f06983fd4eda26f60305898b82d14709501af`
- Rechecked outstanding v32/API-token backup requests:
  - 6 prior requests checked.
  - All 6 are clear against the 05:30:41 verified backup.
- Rechecked Astra handoff:
  - Current request remains `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`.
  - `ANALYSIS_READY.json` is still absent.
  - `LATEST.md` still points to the old 20260929 report, which predates v29–v32 and is not current analysis.
- Updated durable state via `update_state`, `STATUS.md`, `RESEARCH_LOG.md`, and `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`.

Access/budget this iteration:
- New simulation episodes: 0
- New control steps: 0
- Training/gradient steps: 0
- Selector refits: 0
- Validation64 bank opened: false
- Sealed test accessed: false
- Run experiments: 1 successful, 0 failed

Scientific evidence preserved:
- v31/v32 family counts remain `H12=3, H15=1, H35=1`.
- v32 opened-development diagnostic remains:
  - Rule: H12 if low `abs_obs_07`, else H35.
  - LOGO bad_count: 1
  - Horizon counts: `H12=6, H15=0, H35=5`
  - Decision saving vs fixed H35 on opened rows: 39.925%
  - Residual bad row: `v19_c13`, oracle H15 but predicted H35, with large physical excess.
- This remains development-only evidence, not validation/test/reproduction/deployed-speed/deployable-selector evidence.

Current gates:
- Prior v32/API-token backup gate is now clear.
- The new 05:33 metadata recheck outputs themselves require a later verified external backup:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V32_POSTBACKUP_ASTRA_STATE_RECHECK_20260930T053333Z.json`
- Astra v32 analysis is still pending.
- Therefore unique scientific simulation/refit/training/validation/final-test work remains blocked for now.

Next safe action:
1. Check for a verified backup covering the 05:33 recheck outputs and log updates.
2. Check for matching/superseding Astra `ANALYSIS_READY.json` for `v32-h12-supported-default-h35-diagnostic-20260930T051611Z`.
3. If both gates clear, read and verify Astra’s report, update `RESPONSE_LOG.md`, then implement Astra’s selected plan.
4. If either gate remains absent, continue only reversible integrity/preparation; do not choose a new scientific branch independently.


## 2026-09-30T05:47:07.870074+00:00

<!-- vehicle-true-variable-horizon-v33-terminal-h-cross-causal-probe-v0-20260930T054827Z -->
## 2026-09-30 v33 terminal × H causal cross-probe

UTC: 2026-09-30T05:57:42.452614+00:00. Executed Astra-selected Task 1 as development-only IMPROVED evidence: terminal contracts zero/V15_shared/V35_shared × H12/H15/H25/H35 on six already-opened branch states. Budget 72 episodes and 4013 control steps; validation64 closed, sealed test closed, no training/refit. Executor numeric readout: {'source242_read': 'at least one short-H source242 rescue appears after terminal swap; terminal contract can change rescue and should be repaired before selector work.', 'c13_read': 'v19_c13 H35 cost anomaly is terminal-sensitive: V35 is materially worse than V15/zero.', 'alternate_initialization_solver_calls_performed': 0, 'alternate_initialization_deferred_reason': 'requires a separate low-level warm-start manipulation path; not needed for the first H x terminal continuation matrix and not executed in v33'}. Source242 summary: {'zero': {'states': 2, 'H35_safe_count': 2, 'H12_safe_count': 0, 'H15_safe_count': 0, 'best_physical_horizons': Counter({'35': 2}), 'fastest_near_best_horizons': Counter({'35': 2})}, 'V15_shared': {'states': 2, 'H35_safe_count': 2, 'H12_safe_count': 0, 'H15_safe_count': 0, 'best_physical_horizons': Counter({'35': 2}), 'fastest_near_best_horizons': Counter({'35': 2})}, 'V35_shared': {'states': 2, 'H35_safe_count': 2, 'H12_safe_count': 1, 'H15_safe_count': 2, 'best_physical_horizons': Counter({'35': 2}), 'fastest_near_best_horizons': Counter({'15': 1, '12': 1})}}. v19_c13 H35 terminal effect: {'V35_physical': 64.58471587439642, 'V15_physical': 11.169277744886342, 'zero_physical': 11.168923987264899, 'V35_minus_best_other': 53.415791887131526, 'read': 'v19_c13 H35 cost anomaly is terminal-sensitive: V35 is materially worse than V15/zero.'}. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z/completed.json`. New backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V33_TERMINAL_H_CROSS_CAUSAL_PROBE_20260930T054827Z.json`. New Astra request: `v33-terminal-h-cross-causal-probe-20260930T054827Z`.


## 2026-09-30T05:59:43.630651+00:00
Elapsed service lifetime: **3d 19h 02m 13.033s** since `2026-09-26T10:55:29.419331Z` (from v33 completion timestamp).  
Cumulative server API total_tokens: **latest audited 278.599M** from `/data/openai-agent/state/research.sqlite`; not re-audited after v33 because this iteration used the single allowed `run_experiment` slot. Desktop conversation tokens excluded.

Concrete actions completed:

1. **Verified coordination state**
   - Read `ANALYSIS_READY.json`, `LATEST.md`, Astra reports `20260930T034449Z.md` and `20260930T054126Z.md`, `COORDINATION.md`, and `RESPONSE_LOG.md`.
   - Confirmed Astra’s v32 direction: execute terminal × H causal cross-probe before new selector training, threshold sweeps, fresh-label acquisition, validation, or test.
   - Verified pre-run backup context from supervisor: backup at `2026-09-30T05:47:51.591996+00:00`, commit `e8312dac70d4f94a1a3bf6ceec70dfb6b8749613`, package SHA256 `bbcc24abcb7275c5acbf4a10184075a3c440e20142c03b070640705bb55fbcf7`.

2. **Executed Astra-selected v33 terminal × horizon causal probe**
   - Experiment ID: `20260930T054827_b3e91ea1`
   - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0.py`
   - Classification: **development IMPROVED diagnostic only**, not ORIGINAL SAC, not validation/test.
   - Budget used: **72 development episodes**, **4013 control steps**.
   - No validation64 access, no sealed test access, no gradient training, no selector refit.

3. **Key numerical findings from v33**
   - **source242 states**
     - With `zero` terminal: H35 safe on 2/2 states; H12/H15 safe on 0/2.
     - With `V15_shared`: H35 safe on 2/2; H12/H15 safe on 0/2.
     - With `V35_shared`: H35 safe on 2/2, H15 safe on 2/2, H12 safe on 1/2.
   - Executor numeric readout: source242 rescue is not purely “H35 only”; terminal contract can change short-H rescue behavior, so terminal/value-contract issues remain causal candidates.
   - **v19_c13 H35 anomaly**
     - H35 + V35 physical cost: `64.58471587439642`
     - H35 + V15 physical cost: `11.169277744886342`
     - H35 + zero physical cost: `11.168923987264899`
     - V35 is materially worse than V15/zero for this row, supporting a terminal-sensitive cost anomaly.
   - Aggregate safe counts:
     - By H: H12 `11`, H15 `14`, H25 `14`, H35 `18`
     - By terminal: zero `18`, V15_shared `18`, V35_shared `21`

4. **Evidence and handoff artifacts**
   - Summary: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z/summary.md`
   - Raw: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z/raw.json`
   - Completed: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_terminal_h_cross_causal_probe_v0_20260930T054827Z/completed.json`
   - Registry: `research_artifacts/aws_runs/20260930T054827_b3e91ea1/registry.json`
   - Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V33_TERMINAL_H_CROSS_CAUSAL_PROBE_20260930T054827Z.json`

5. **Astra handoff preserved**
   - Refreshed `docs/bohn2021_takeover/astra_reviews/NEXT_REVIEW_REQUEST.json` with request:
     - `v33-terminal-h-cross-causal-probe-20260930T054827Z`
   - Updated `RESPONSE_LOG.md`, `STATUS.md`, `RESEARCH_LOG.md`, `DECISIONS.md`, `RESULTS_AUDIT.md`, `REPRODUCTION_PROTOCOL.md`, and `EXPERIMENT_REGISTRY.csv`.
   - Persisted state via `update_state`.

Current gates / next iteration:

- **External backup is required** before further unique scientific simulation/refit/training/validation:
  - `REQUEST_BACKUP_AFTER_V33_TERMINAL_H_CROSS_CAUSAL_PROBE_20260930T054827Z.json`
- **Astra analysis for v33 is pending.**
  - `ANALYSIS_READY.json` still points to the v32 report, not the new v33 request.
- Until both gates clear, the next safe work is reversible integrity/preparation only, especially the receipt final-hash/provenance issue Astra identified for v29/v32/v33.
- Sealed final test remains unopened and unauthorized.


## 2026-09-30T06:15:42.482146+00:00



## 2026-09-30T06:18:45.747149+00:00



## 2026-09-30T06:21:54.823071+00:00



## 2026-09-30T06:24:44.275230+00:00



## 2026-09-30T06:36:05.784389+00:00



## 2026-09-30T06:53:27.693885+00:00



## 2026-09-30T07:13:55.031933+00:00



## 2026-09-30T07:19:38.231091+00:00



## 2026-09-30T07:22:42.828081+00:00



## 2026-09-30T07:26:17.897150+00:00



## 2026-09-30T07:44:19.996750+00:00

<!-- vehicle-true-variable-horizon-v33-bookkeeping-preflight-v0-20260930T074544Z -->
## v33 bookkeeping preflight

UTC `2026-09-30T07:45:44.815804+00:00`: gate passed `True`; schedule/raw/trace counts `72`/`72`/`72`, trace lines `4013`; zero rollout/solver/training, validation64 and sealed test closed. Artifacts `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_bookkeeping_preflight_v0_20260930T074544Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v33_bookkeeping_preflight_v0_20260930T074544Z/raw.json`. Backup required: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V33_BOOKKEEPING_PREFLIGHT_20260930T074544Z.json`.


## 2026-09-30T07:48:12.741030+00:00



## 2026-09-30T07:52:14.471918+00:00



## 2026-09-30T07:55:15.539510+00:00



## 2026-09-30T08:03:58.923135+00:00



## 2026-09-30T08:43:01.902005+00:00



## 2026-09-30T08:49:02.417752+00:00



## 2026-09-30T09:00:03.769688+00:00



## 2026-09-30T09:03:00.701733+00:00



## 2026-09-30T09:07:44.917391+00:00



## 2026-09-30T09:10:53.087702+00:00



## 2026-09-30T09:16:15.337429+00:00



## 2026-09-30T09:19:21.031652+00:00



## 2026-09-30T09:24:40.488816+00:00



## 2026-09-30T09:30:47.726907+00:00



## 2026-09-30T09:34:37.676203+00:00



## 2026-09-30T09:36:43.184189+00:00

<!-- vehicle-v34c-contract-preflight-v0e-20260930T093751Z -->
## 2026-09-30 v34c zero-solve objective-basin contract preflight

UTC: 2026-09-30T09:39:32.043254+00:00. Executed Astra Task1 repair/preflight with 24 scheduled cells and 24 controlled intercepts before `mpc.solve`; lower-level solver calls=0, plant steps=0, training/refit=0, validation64=false, sealed test=false. hard_pass=True; vf4_ok=24/24; lterm_ok=24/24; group_failures=0. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V34C_CONTRACT_PREFLIGHT_V0E_20260930T093751Z.json`.


## 2026-09-30T09:40:43.070119+00:00
3d 22h 44m since first supervisor event; latest audited server API total_tokens remain ~278.599M from `/data/openai-agent/state/research.sqlite` as of the prior 2026-09-30 05:33/05:34 audit, excluding this desktop conversation.

Concrete action completed:

- Executed Astra Task1 v34c/v0e zero-solve contract preflight.
  - Experiment: `20260930T093751_899d9b5c`
  - Script: `experiments/bohn2021_aws/vehicle_true_variable_horizon_v34c_contract_preflight_v0e.py`
  - Exit status: `0`
  - Runtime: `102.07s`
  - Classification: development-only IMPROVED diagnostic, not validation/test.

Verified result:

- `hard_pass: true`
- Scheduled/intercepted rows: `24/24`
- Lower-level solver calls: `0`
- Plant steps: `0`
- Training/refit: `0`
- validation64 opened: `false`
- sealed/final test accessed: `false`
- env resets after construction: `0`
- `vf_fun` four-argument rows OK: `24/24`
- `lterm` rows OK: `24/24`
- Group failures: `0`

Key contract checks passed for all context/horizon groups:

- Initial primal identical across terminal modes within each initialization.
- Nonterminal `opt_p` identical across terminal modes and across initializations where required.
- Canonical and goal-facing initial primal hashes are distinct.
- Bounds have real names.
- Numeric shifted TVP conversion/readback metadata recorded.
- Direct no-reset context reconstruction recorded.
- Strict zero-solve intercepts confirmed before `mpc.solve`.

Evidence paths:

- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/completed.json`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/summary.md`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/raw.json`
- `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34c_contract_preflight_v0e_20260930T093751Z/arrays/`
- `research_artifacts/aws_runs/20260930T093751_899d9b5c/registry.json`

Backup/state:

- Backup request written:
  - `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V34C_CONTRACT_PREFLIGHT_V0E_20260930T093751Z.json`
- Durable state updated via `update_state`.
- Existing logs already contain v0e entries in:
  - `docs/bohn2021_takeover/astra_reviews/RESPONSE_LOG.md`
  - `STATUS.md`
  - `RESEARCH_LOG.md`
  - `RESULTS_AUDIT.md`
  - `EXPERIMENT_REGISTRY.csv`

Current gate:

- Do **not** start the v34 fixed 24-call objective-vs-basin solver probe until an external backup verifies coverage of the v0e source/output arrays/raw/summary/completed/docs/registry/backup request.
- After backup verification, proceed with Astra Task2: original fixed 24-call objective-vs-basin solver probe, still with:
  - 0 plant steps
  - max 24 lower-level solver attempts
  - no automatic retry/extra guesses
  - no validation64 or sealed/final test
  - no selector search/refit/training
  - no v33 rerun.


## 2026-09-30T09:46:51.646044+00:00



## 2026-09-30T09:54:04.298905+00:00

### vehicle-v34f-backup-gate-preflight-20260930T100335Z

UTC: 2026-09-30T10:03:35.369934+00:00

Opus task A backup-gate preflight passed with zero solver/plant/training/validation/test budget. Matched verified backup proof `research_artifacts/aws_backup_proofs/backup_proof_20260930T095445_from_user_context_after_v34e_source_and_docs.json` for commit `c376745cacb765419e0d31007a1d8e89b711dbb6` and package `8003dfb75ab3ec38b084c453faa58ac15956de55f5ee581c6706716ff9836cfd`. This only verifies the repaired gate; it is not objective-vs-basin science. External backup covering v34f source/preflight outputs is required before the one-cell objective-reconstruction smoke.

Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34f_backup_gate_preflight_v0_20260930T100335Z/completed.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34f_backup_gate_preflight_v0_20260930T100335Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34f_backup_gate_preflight_v0_20260930T100335Z/summary.md`.


## 2026-09-30T10:04:10.250376+00:00

<!-- vehicle-v34g-objective-reconstruction-smoke-20260930T100824Z -->
## v34g one-cell objective-reconstruction smoke

UTC: 2026-09-30T10:08:59.178066+00:00. Opus Task B executed one fixed-context solver call for `source242_slot0_branch_start|H15|V15_shared|canonical`. hard_pass=False; J_solver=1430.3365593819267; reconstructed=1429.8850674111343; relative_error=0.0003156543596896532; residual={'constraint_residual': None, 'bound_residual': None}; solver_status={'return_status': 'Solve_Succeeded', 'success': True, 'iterations': 29, 'solver_exception': None}. Budgets: solver_calls=1, plant_steps=0, env_reset_calls=0, training/refit=0, validation64=false, sealed_test=false. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34g_objective_reconstruction_smoke_v0_20260930T100824Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V34G_OBJECTIVE_RECONSTRUCTION_SMOKE_20260930T100824Z.json`.


## 2026-09-30T10:09:50.604670+00:00

<!-- vehicle-v34h-objective-localization-20260930T101852Z -->
## v34h objective/residual localization

UTC: 2026-09-30T10:19:19.491925+00:00. Executed Opus Task E/F as offline development-only localization from the already-opened v34g one-cell arrays. Budgets: solver_calls=0, plant_steps=0, env_reset_calls_after_construction=0, training/refit=0, validation64=false, sealed_test=false. G-E objective reconstruction pass=False with best candidate `None`, total=None, solver=None, rel_error=None. G-F residual alias offline pass=True; repaired residuals={'bound_residual_using_lb_opt_x_names': 0.0, 'constraint_residual_using_cons_lb_names': 8.612633157188794e-09}; legacy residuals={'bound_residual_using_opt_x_lb_names': None, 'constraint_residual_using_opt_g_lb_names': None}. Artifacts: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34h_objective_localization_v0_20260930T101852Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34h_objective_localization_v0_20260930T101852Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34h_objective_localization_v0_20260930T101852Z/candidate_residuals.csv`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34h_objective_localization_v0_20260930T101852Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V34H_OBJECTIVE_LOCALIZATION_20260930T101852Z.json`.


## 2026-09-30T10:20:00.885949+00:00



## 2026-09-30T10:31:32.277888+00:00

<!-- vehicle-v34i-objective-contract-localization-20260930T103304Z -->
## v34i objective contract localization

UTC: 2026-09-30T10:33:30.317279+00:00. Executed active Opus E′/F′ zero-solve diagnostic after verified post-v34h backup. Budgets: solver_calls=0, plant_steps=0, env_reset/env_step after construction=0, training/refit=0, validation64=false, sealed_test=false. Assembly/indexability pass=False; G-E pass=False; G-F pass=True; both_pass=False. Best candidate `None` rel_error=None total=None solver=None; components stage=None, eps=None, rterm=None, terminal=None. Evidence: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34i_objective_contract_localization_v0_20260930T103304Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34i_objective_contract_localization_v0_20260930T103304Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34i_objective_contract_localization_v0_20260930T103304Z/candidate_residuals.csv`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34i_objective_contract_localization_v0_20260930T103304Z/assembly_bool_table.csv`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V34I_OBJECTIVE_CONTRACT_LOCALIZATION_20260930T103304Z.json`.


## 2026-09-30T10:38:08.703692+00:00

<!-- vehicle-v34k-label-accessor-objective-localization-20260930T104148Z -->
## v34k label-accessor objective localization

UTC: 2026-09-30T10:42:15.100805+00:00. Ran a zero-solve/zero-plant operational diagnostic after v34i showed hash-correct saved arrays but failed partial do-mpc struct indexing. Budgets: solver_calls=0, plant_steps=0, env_reset/env_step after construction=0, training/refit=0, validation64=false, sealed_test=false. Label-accessor pass=True; G-E=True; G-F=True; both_pass=True. Best candidate `stage=x0_then_last_node|term=author_last_node|eps=False|r=False|discount=n_horizon_parameter_or_H|z=same_k_last` rel_error=2.556146357191739e-07 total=1430.3369249968853 solver=1430.3365593819267; components stage=1417.891241005867, eps=0.0, rterm=0.0, terminal=12.445683991018303. Evidence: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/candidate_residuals.csv`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34k_label_accessor_objective_localization_v0_20260930T104148Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V34K_LABEL_ACCESSOR_OBJECTIVE_LOCALIZATION_20260930T104148Z.json`.


## 2026-09-30T10:56:06.186988+00:00

<!-- vehicle-v34l-residual-eps-label-dissection-20260930T105720Z -->
## v34l/A13b residual, eps and label dissection

UTC: 2026-09-30T10:57:46.062710+00:00. Ran Opus A13b as a zero-solve/zero-plant diagnostic over already-opened v34g/v34k evidence. Budgets: solver_calls=0, plant_steps=0, env_reset/env_step after construction=0/0, training/refit=0, validation64=0, sealed_test=0. Formula `stage=x0_then_last_node|term=author_last_node|eps=False|r=False|discount=n_horizon_parameter_or_H|z=same_k_last` produced solver=1430.3365593819267, reconstructed=1430.3369249968853, abs_residual=0.0003656149585822277, rel_residual=2.556146357191739e-07; alternative summation deltas={'builtin_order_total': 1430.336924996885, 'builtin_order_minus_fsum': -2.2737367544323206e-13, 'reversed_order_total': 1430.3369249968855, 'reversed_order_minus_fsum': 2.2737367544323206e-13, 'numpy_float64_total': 1430.3369249968853, 'numpy_float64_minus_fsum': 0.0, 'numpy_float32_total': 1430.3369140625, 'numpy_float32_minus_fsum': -1.0934385272776126e-05}. eps classification=eps_labels_exact_and_objective_branch_active_but_current_solution_eps_zero with path_counts={'exact': 45, 'fallback': 0, 'missing_or_ambiguous': 0, 'zero_values': 45, 'nonzero_values': 0}; rterm probe={'has_rterm_factor': True, 'factor_values': [0.0, 0.0], 'factor_abs_sum': 0.0, 'factor_nonzero_count': 0, 'current_k0_unweighted': 0.0, 'unit_delta_unweighted': 0.0}. raw_live_label_match semantics=label_set_same_but_order_or_string_format_differs; max_colloc classification=v34k_raw_display_truncated_max_colloc_first_to_first_8_entries; full label table contains all k values. Mechanical flags={'a13b_completed': True, 'eps_accessor_or_zero_fill_blocker': False, 'raw_live_label_normalized_mismatch_blocker': False, 'collocation_missing_k_blocker': False, 'missing_formula_blocker_detected_by_a13b': False, 'double_precision_summation_explains_residual': False, 'max_float64_order_delta_vs_abs_residual': 2.2737367544323206e-13, 'a13c_allowed_by_mechanical_checks': True, 'v34j_23_call_probe_still_blocked_until_a13c_passes_under_latest_opus_plan': True}. Evidence: `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34l_residual_eps_label_dissection_v0_20260930T105720Z/summary.md`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34l_residual_eps_label_dissection_v0_20260930T105720Z/raw.json`, `research_artifacts/aws_diagnostics/vehicle_true_variable_horizon_v34l_residual_eps_label_dissection_v0_20260930T105720Z/completed.json`. Backup request: `research_artifacts/aws_backup_proofs/REQUEST_BACKUP_AFTER_V34L_RESIDUAL_EPS_LABEL_DISSECTION_20260930T105720Z.json`.


## 2026-09-30T11:01:04.348499+00:00

