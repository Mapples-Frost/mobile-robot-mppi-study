# L249 Paper Optimizer 候选边界约束执行协议

日期：2026-07-21  
状态：预注册 development execution；尚未读取任何 L249 outcome

## 1. 背景

L248 的完整性来源核验通过，但 treatment fidelity 失败：候选过滤只进入标准 MPPI，未进入
Full Proposed 使用的 Paper RL-Driven elite optimizer；新增诊断也未被 episode metrics 持久化。
L248 已冻结为 engineering-invalid 记录，不覆盖、不删除、不作为方法效果证据。

## 2. 唯一修复

L249 不改变 L248 的研究 treatment，只完成实现传播：

1. 对 Paper optimizer 每轮固定的 50 个 guided/Gaussian 候选使用与标准 MPPI 相同的
   footprint corridor margin；
2. 保留一个确定性 braking candidate，K 仍为 50；
3. 不可行候选不得进入 elite set；
4. 加权均值 rollout 不可行时，回退到末轮最低成本可行候选；
5. 记录 candidate feasible fraction、no-feasible fraction、weighted-update feasibility、
   fallback 和 final predicted margin；
6. 约束对 ICODE-MPPI 和 Full Proposed 公平共享。

## 3. 全部冻结条件

继承 L248：同一 Actor SHA、ICODE/value checkpoints、三张 L239 W=4.0D Tracking 地图、
nominal_seen、development seed `923301001`、100 rollouts/decision、50×2 vs 100×1、
2210/1405/2030 episode budgets、所有 MPPI cost 权重、LaserScan、scan_guard、安全仲裁和
boundary termination。禁止使用 sealed seeds。

## 4. 执行完整性 Gate

读取效果前必须确认：

- 6 个唯一 MuJoCo 回合，Git/config/checkpoint/provenance 完整；
- 所有 resolved config 启用 `path_boundary_candidate_filter_enabled=true`；
- Full 的 metrics 中 filter enabled fraction 为 1；
- candidate feasible、no-feasible、fallback、final margin 字段均存在且有限；
- 无 Traceback、Exception、NaN、Inf；
- L248 目录只读保留。

## 5. 性能 Gate

沿用 L248 预注册阈值，不因已看到 L248 执行失败而改变：

1. 两 arms 均零碰撞；
2. Full 三场景均零 boundary violation，最小实际 footprint margin 不小于零；
3. Full 相对 L247 平均 completion 回退不超过 0.01；
4. Full 任一场景 completion 回退不超过 0.02；
5. Full 保持非平凡 RL proposal authority；
6. 完整报告约束激活率、可行率、fallback、planner time 与安全负担。

本轮仍是单 development seed。通过只允许进入多-seed qualification，不得写成论文确认性结论。

## 6. 禁止事项

不得训练/更换模型、修改核心耦合、地图、cost、预算或安全链；不得筛 seed、删除失败、
重跑挑结果、使用 sealed seeds 或伪造缺失诊断。

