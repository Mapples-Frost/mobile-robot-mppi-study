# L218 六类扩展场景 MuJoCo 开发验证报告

日期：2026-07-20
性质：**开发集资格验证（qualification），不是封存测试集上的正式论文结论**

## 1. 结论先行

本轮在 MuJoCo 3.2.3 差速轮物理后端中完成了六类 6.5 m × 6.5 m 复杂场景、三组开发随机种子、两种耦合方法的 36 个闭环回合。所有回合都使用 headless MuJoCo；headless 只是不显示窗口，不代表绕过物理仿真。

本轮得到一个清晰但边界明确的正向结果：

- 未经可靠性保护的 `Simple combination` 在六类大地图上 0/18 到达；
- `Full proposed` 在完全相同的场景—种子配对下 9/18 到达，成功率提高 0.50；
- 两种方法均为 0/18 碰撞；
- `Full proposed` 的平均路径完成度由 0.056 提高到 0.691；
- `Full proposed` 的平均横向误差由 0.439 m 降至 0.166 m；
- 三个开发种子中均重复出现 `Full proposed` 3/6、`Simple combination` 0/6 的结果。

机制诊断同时表明：残差条件 Actor 在这些相对训练分布很远的新场景上支持度为零；role-aware HSS 因此把 Actor proposal authority 降为 0，并 100% 回退到可信的 ICODE-MPPI proposal。这证明的是：

> 可靠性耦合能够阻止分布外 RL prior 破坏 ICODE-MPPI，并实现安全退化。

它**尚不能证明**“RL prior 在六类新地图上优于 ICODE-MPPI”，因为本轮 Full Proposed 中的 RL proposal 被可靠性机制正确拒绝。后续必须在代码冻结后，使用未见随机种子和完整方法矩阵确认这一结论，并另外用 Actor 支持域场景验证 RL 的正向引导价值。

## 2. 真实仿真链路

本轮执行链为：

```text
MuJoCo 3.2.3 differential-drive plant
    -> synthetic LaserScan generated from the MuJoCo scene
    -> scan_guard
    -> local_obstacle_layer
    -> MPPI candidate rollout (ICODE prediction model)
    -> control arbitration
    -> wheel actuation in MuJoCo
    -> next physical observation
```

需要区分两种模型：

1. **环境执行模型**是 MuJoCo 差速轮物理模型，负责轮胎、惯量、执行器动态和碰撞接触；
2. **规划器预测模型**是 nominal/ICODE dynamics，负责在 MPPI 内预测候选控制序列。

规划器没有直接读取障碍物真值作为局部障碍输入。障碍物仍通过 Synthetic LaserScan 进入 `scan_guard` 和 `local_obstacle_layer`。Polyline 只充当全局参考路线，用于把本研究聚焦于局部控制与动力学耦合，而不是把“缺失全局规划器”混成干扰变量。

## 3. 六类场景

| 场景 | 主要结构 | 障碍物 | 参考路线长度 | 设计目的 |
|---|---|---:|---:|---|
| Serpentine | 四层交错长板 | 4 | 25.69 m | 长时域、多次换向、累计误差 |
| Giant U | 大型 U 形板 | 3 | 9.19 m | 长绕行与 U 形局部极小值 |
| Opposed U | 两个相向 U 形板 | 6 | 9.33 m | 窄口选择和连续转向 |
| Nested U | 大 U 套小 U | 6 | 8.05 m | 多层凹障碍与进出策略 |
| Cylinder forest | 规则圆柱阵列 | 16 | 7.91 m | 连续避障和通道跟踪 |
| Cylinder rings | 多圈圆柱 | 21 | 6.06 m | 高密度、弯曲通道和局部拥挤 |

离线几何 Gate 检查了起点、终点、膨胀后的连通性、参考线连续净空。六张地图均通过；这只证明存在几何可行通道，不保证给定局部控制器一定到达。

## 4. 开发协议

- 开发随机种子：91001、91002、91003；
- 场景：六类全部纳入，每个种子完整重复；
- 物理域：`nominal_seen`；
- 方法：`Simple combination`、`Full proposed`；
- 每次决策总 rollout 预算：30；
- MPPI horizon：36；
- 最大闭环步数：900；
- 状态跟踪指标由保存的逐步 `trajectory.csv` 重新计算；
- 统计独立单位为 seed，六个场景是同一 seed 内的重复分层；
- 使用 10,000 次 seed-cluster bootstrap，禁止把 timestep 或 scene 当作独立样本扩充样本量；
- 所有失败均保留，未筛 seed、未删除失败回合。

这套预算是开发筛查预算，不是后续封存试验的最终高预算配置。

## 5. 总体结果

| 指标 | Simple combination | Full proposed | Full 相对效果 |
|---|---:|---:|---:|
| 成功 | 0/18 (0.0%) | 9/18 (50.0%) | +50.0 percentage points |
| 碰撞 | 0/18 | 0/18 | 无恶化 |
| 路径完成度 | 0.0555 | 0.6910 | +0.6355 |
| 横向 RMSE | 0.4386 m | 0.1658 m | -0.2728 m |
| 最终目标距离 | 4.5948 m | 1.4124 m | -3.1823 m |
| 闭环步数 | 900.0 | 729.6 | -170.4 |
| 平均规划耗时 | 455.9 ms | 353.1 ms | -102.8 ms |
| 控制 jerk | 0.2160 | 0.2188 | +0.0028（轻微变差） |
| 最小净空 | 0.5448 m | 0.2203 m | -0.3245 m（变差） |
| 安全干预次数 | 229.0 | 453.9 | +224.9（增多） |

Full Proposed 的较小净空和较多 `front_obstacle_slow` 干预不能被隐藏。Simple combination 多数时间停滞在起点附近，因此“净空更大、干预更少”不等于控制更好；Full Proposed 实际穿越障碍通道，暴露时间和接近障碍的机会更多。尽管如此，后续仍需报告按路径长度/时间归一化的干预率，并继续优化安全负担，不能仅用零碰撞替代分析。

## 6. 分场景结果

| 场景 | Simple 成功 | Full 成功 | Full 平均路径完成度 | 判读 |
|---|---:|---:|---:|---|
| Cylinder forest | 0/3 | 3/3 | 0.965 | 稳定到达 |
| Giant U | 0/3 | 3/3 | 0.969 | 稳定到达 |
| Opposed U | 0/3 | 3/3 | 0.968 | 稳定到达 |
| Cylinder rings | 0/3 | 0/3 | 0.612 | 有明显进展但未到达 |
| Nested U | 0/3 | 0/3 | 0.378 | 被局部结构限制 |
| Serpentine | 0/3 | 0/3 | 0.254 | 长时域局部规划能力不足 |

后三类失败是当前方法和短视局部规划器的真实边界。它们不否定 ICODE 或可靠性耦合，但说明“后续 RL/全局引导应解决复杂拓扑和长时域探索”仍未完成实验闭环。

## 7. 配对统计

下列效果定义为正值代表 Full Proposed 更有利。95% CI 来自 seed-cluster bootstrap；只有三个开发 seed，区间和标准化效应不可当作最终推断。

| 配对效果 | 均值 | 95% cluster-bootstrap CI |
|---|---:|---:|
| 成功率 | +0.5000 | [0.5000, 0.5000] |
| 路径完成度 | +0.6355 | [0.6286, 0.6445] |
| 横向 RMSE 改善 | +0.2728 m | [0.2310, 0.3180] |
| 最终目标距离改善 | +3.1823 m | [3.1021, 3.3122] |
| 步数减少 | +170.39 | [163.67, 175.33] |
| 平均计算时间减少 | +102.77 ms | [101.74, 103.56] |
| 控制 jerk 改善 | -0.0028 | [-0.0049, -0.0014] |
| 最小净空改善 | -0.3245 m | [-0.3361, -0.3029] |
| 安全干预改善 | -224.94 | [-239.17, -217.00] |

三个 seed 呈现完全相同的成功/失败模式，导致成功率区间退化为一点。这不是“证据无限强”，而是开发样本量较小且任务具有确定性结构的表现。

## 8. 发现并修复的核心实现问题

原实现中，HSS 的低可靠性只会降低 Actor-guided sample fraction，却没有降低 Actor 对 Gaussian proposal center/covariance 的控制权。因此即使诊断显示 authority=0，采样分布仍然围绕分布外 Actor 展开，造成所谓“关闭 RL 后仍被 RL 影响”的语义漏洞。

修复后定义：

```text
proposal_authority = completion_handover_authority * reliability_authority
```

Actor proposal 的均值和方差按 `proposal_authority` 与 GoalWarmStart/previous-sequence baseline 混合。可靠性为零时，Actor 不再暗中控制采样中心。新增逐步与回合级诊断：

- `reliability_proposal_authority`；
- `reliability_proposal_fallback_fraction`。

开发矩阵中：

- Simple combination：proposal authority = 1，fallback = 0；
- Full proposed：proposal authority = 0，fallback = 1；
- Full proposed 的 HSS 和 residual context 均在全部回合启用。

因此 0/18 到 9/18 的改善对应一个可检查的耦合机制，而不是更换方法或挑选随机种子。

## 9. 对论文主线的支持程度

### 已支持

1. 大型复杂地图确实会把离线 Actor 推入 OOD 区域；
2. 直接把 RL prior 与 ICODE-MPPI 串联会造成灾难性停滞；
3. 可靠性加权 proposal authority 能检测并切断不可信 RL 引导；
4. Full Proposed 在 OOD 时可以退化为安全、有效的 ICODE-MPPI，而非继续相信错误 Actor；
5. 这给“ICODE reliability -> RL guidance authority”这一跨层耦合提供了直接开发证据。

### 尚未支持

1. RL prior 在这些六张新地图上提供正向性能增益；
2. value-consistent ICODE 在这六张地图上的独立因果收益；
3. combined-unseen 物理失配域中的稳定优势；
4. 七方法完整消融和 ICODE × RL interaction effect；
5. 未见 seed 上的正式统计外推；
6. 实车性能或实时性结论。

## 10. 可复现性与数据边界

每个回合均保存：

- resolved YAML 配置；
- 逐时刻 `trajectory.csv`；
- `metrics.json`；
- `provenance.json`；
- 方法级 episode CSV；
- 随机化 schedule；
- checkpoint、配置和分析工件 SHA-256。

本轮运行时工作区包含尚未提交的机制修复。分析审计因此记录了 tracked patch SHA-256：

```text
168e74e7d43dcd0cad054e1b35169a1413cbc809909c5530442b5838d20ea64b
```

这正是本轮不得升级为正式论文结论的原因。正确流程是：先测试、提交并推送实现，再预注册从未使用的新 seed，然后只运行一次封存 benchmark。

## 11. 下一 Gate

1. 固化并提交当前实现与完整测试；
2. 预注册全新的 sealed seeds、七方法、六场景与两物理域矩阵；
3. 运行一次未见 seed 的 MuJoCo benchmark，不再根据封存结果调参；
4. seed 作为 cluster 做配对统计，完整保留负向结果；
5. 将 L217 的支持域结果和 L218 的极端 OOD 结果分层报告：前者评价 RL 引导收益，后者评价可靠性退化能力。

只有完成上述 Gate，才能决定这一模块能否进入论文主结果，而不是只进入工程消融或附录。
