# L249 执行与诊断链审计

日期：2026-07-21  
结论：`engineering-invalid`；保留全部结果，不进入性能 Gate

## 已完成内容

- 三张 Tracking 场景、两种方法共完成 6/6 个 MuJoCo development 回合；
- 回合键唯一，日志中未发现 Traceback、Exception、NaN 或 Inf；
- Git SHA、Actor checkpoint、MuJoCo 版本、随机种子、回合预算和 resolved config 均与
  L249 冻结协议一致；
- resolved config 中 `path_boundary_candidate_filter_enabled=true`。

## 阻断原因

L249 在读取性能结果前执行 treatment-fidelity 审计时发现，六个 `metrics.json` 的
`path_boundary_candidate_filter_enabled_fraction` 均为 0。进一步代码追踪确认：

1. 标准 MPPI 与 Paper RL-Driven optimizer 均生成了候选边界过滤诊断；
2. `EpisodeMetrics.summary()` 已尝试汇总这些诊断；
3. 但 `EpisodeMetrics.update()` 未把诊断从 planner diagnostics 写入逐步 records；
4. 因而汇总阶段只能读取默认值，导致运行时生效证据链缺失。

这属于科研记录链缺陷。虽然控制器代码可能实际执行了过滤，但现有工件无法证明每个决策
时刻的约束状态、候选可行率、无可行候选次数、fallback 和预测边界余量，因此禁止读取或
宣称 L249 的方法效果。

## 处置

- L249 原始目录与失败诊断永久保留，不覆盖、不删除；
- 补齐 `EpisodeMetrics.update()` 的全部候选边界诊断字段；
- 增加 planner diagnostics → step records → episode summary 的端到端单元测试；
- 建立 L250，仅修复科研仪器记录链，继承 L249 的全部方法、地图、预算、模型、seed、cost
  和安全条件；
- L250 使用新 Git SHA 和新输出目录重新执行 6 个回合，避免与 L249 混淆。

## 科研解释边界

L249 只能证明“六个回合完成且配置请求开启过滤”，不能证明 treatment 在每一规划周期按
协议工作，也不能作为正向或负向算法结论。L250 完成完整性 Gate 后，才允许读取性能指标。

