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
