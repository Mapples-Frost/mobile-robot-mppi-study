# Full Proposed 路径跟踪迁移确认结果

日期：2026-07-19  
预注册：`docs/rl/183_full_proposed_path_tracking_prereg_2026-07-19.md`  
冻结实现：`917b867f8bb6c4818f3b676a8ab8dc4b6b42e60d`

## 1. 实验范围

本实验在不重新训练 Actor、critic 或 ICODE 的条件下，评估冻结方法从
point-goal 向无障碍 polyline path tracking 的迁移。实验包含：

- 确认种子：403--407；
- 路径：gentle S、double turn、slalom；
- 物理域：nominal seen、long-delay seen、combined unseen；
- 四个 2×2 factorial 实验臂；
- 每臂 45 个 episode，共 180 个 episode；
- 每拍严格 100 个 model rollouts、2 次 MPPI refinement；
- seed-cluster bootstrap 10,000 次。

原始与分析产物位于：

```text
results/research_platform/rl/full_proposed_path_transfer_confirm_l183/
results/research_platform/rl/full_proposed_path_transfer_confirm_l183_analysis/
```

## 2. 完整性审计

- 180/180 episode 完成；
- 45/45 区组包含全部四臂；
- 所有分片 provenance 均指向冻结 SHA `917b867`；
- 四臂 rollout budget 均为 100；
- 无 NaN/Inf；
- 0 次碰撞；
- tracking 阶段 completion floor 触发率为 0；
- terminal 阶段 completion floor 有实际激活；
- adaptive HSS 的 raw guided fraction 实际到达过 0。

## 3. 四臂结果

| 方法 | Cross-track RMSE (m) | Completion | Success | Collision | Jerk | Planner ms |
|---|---:|---:|---:|---:|---:|---:|
| Ordinary fixed | 1.8969 | 0.9516 | 0/45 | 0/45 | 0.0890 | 386.9 |
| Value fixed | 1.9223 | 0.9711 | 0/45 | 0/45 | 0.0899 | 381.2 |
| Ordinary adaptive | 1.7266 | 0.9870 | 0/45 | 0/45 | 0.0862 | 322.2 |
| Full Proposed | 1.7419 | 0.9673 | 0/45 | 0/45 | 0.0868 | 321.9 |

## 4. 主要确认比较

`Full Proposed - ordinary fixed`：

| 指标 | 有利效应 | 95% seed-cluster CI | 相对变化 |
|---|---:|---:|---:|
| Cross-track RMSE | +0.1550 m | [0.1201, 0.1899] | 8.17% 降低 |
| Cross-track maximum | +0.1135 m | [0.0950, 0.1332] | 3.90% 降低 |
| Completion ratio | +0.0157 | [-0.0331, 0.0645] | 1.65% 提高 |
| Control jerk | +0.00226 | [0.00074, 0.00347] | 2.54% 降低 |
| Planner time | +65.0 ms | [54.7, 75.3] | 16.8% 降低 |
| Tangent-heading RMSE | -0.0272 rad | [-0.0755, 0.0285] | 未确认 |
| Success / collision | 0 / 0 | [0, 0] / [0, 0] | 无差异 |

Cross-track 改善在所有路径和物理域聚合中方向一致：

| 分层 | Ordinary fixed | Full Proposed | 改善 (m) |
|---|---:|---:|---:|
| Gentle S | 1.7862 | 1.6343 | 0.1519 |
| Double turn | 1.8097 | 1.7366 | 0.0731 |
| Slalom | 2.0949 | 1.8549 | 0.2401 |
| Nominal seen | 1.8756 | 1.7461 | 0.1296 |
| Long-delay seen | 1.8898 | 1.7425 | 0.1473 |
| Combined unseen | 1.9253 | 1.7371 | 0.1882 |

## 5. 因子解释

Cross-track RMSE 的 factorial 结果为：

- adaptive HSS 主效应：改善 0.1754 m，
  95% CI [0.1239, 0.2297]；
- value-aligned ICODE 主效应：恶化 0.0204 m，
  95% CI [0.0010, 0.0434]；
- ICODE × HSS interaction：估计有利 0.0100 m，
  95% CI [-0.0256, 0.0467]，不能确认。

因此，Full Proposed 相对 ordinary fixed 的确认优势主要由 adaptive HSS
驱动。本实验不支持 value-aligned ICODE 在当前路径任务上的独立改善，也不支持
超加性 ICODE × RL synergy。

## 6. Gate 与结论边界

预注册的统计迁移 Gate 通过：

- cross-track favorable CI 下界大于 0；
- collision 未增加；
- completion 的点估计未退化；
- success 无净损失；
- jerk 未恶化，反而改善。

但是应用层 Gate 未通过：所有方法均未在给定时限内到达最终路径终点，且绝对
cross-track RMSE 较大。`completion ratio` 是轨迹对 polyline 的单调投影进度，
不能替代成功到达；在车辆偏离路径后，它仍可能给出较高投影进度。

本轮结果只能支持：

> 在测试的无障碍 differential-drive polyline 任务和三类物理域中，
> 冻结 Full Proposed 相对 ordinary fixed 显著降低了 cross-track 误差、
> jerk 和计算耗时；该收益主要来自可靠性自适应 Actor 采样。

本轮结果不能支持：

- 可靠路径到达或闭环任务成功；
- value-aligned ICODE 的路径跟踪收益；
- ICODE × RL 协同；
- 动态障碍导航；
- 论文 bicycle 模型复现；
- 实车泛化；
- 稳定性或收敛性保证。

## 7. 后续决策

不得使用确认种子 403--407 调参。后续路径研究若继续，应建立新的开发种子和新
的封存确认集，优先解决“沿路径前进但不能收敛到终点”的 reference/policy
适配问题。该问题与已经确认的 point-goal Full Proposed 结果分开报告。
