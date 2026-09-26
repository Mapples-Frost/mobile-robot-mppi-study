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

