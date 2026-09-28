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
