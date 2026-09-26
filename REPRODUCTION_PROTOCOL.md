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
