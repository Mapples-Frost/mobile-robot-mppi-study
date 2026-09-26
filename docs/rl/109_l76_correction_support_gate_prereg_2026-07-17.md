# L76：冻结 BC 周围的 SAC 修正支持度门控预注册

日期：2026-07-17  
性质：development experiment；本文件在读取 L76 结果前冻结

状态说明：本文件的 `3.0/4.5` 初始尺度在正式 L76 数据生成前被短程 instrumentation
smoke 判定为逐 step 尺度不匹配。唯一阈值修订及其时间顺序记录在
`docs/rl/110_l76_prereg_amendment_and_l76a_calibration_2026-07-17.md`；正式判定以
修订后的 `3.0/7.0` 为准，其余 Gate 未改变。

## 1. 研究问题

L75 已否定“进度不足时才打开整个 learned prior”。L76 检验更窄、结构上可解释的
问题：

> 当观测偏离 SAC 训练支持域时，只连续衰减 SAC 相对冻结 BC 的增量修正，是否能
> 保留窄通道中的有效轨迹引导，同时减少单障碍和 U-trap 中的成功损失？

RL 仍只产生 MPPI sampling prior。MPPI、ICODE rollout、LaserScan、
`local_obstacle_layer`、`scan_guard` 和最终安全仲裁均不改变。

## 2. 冻结机制

训练后 correction policy 的确定性动作写成：

\[
a_{\mathrm{SAC}}(o)
=
a_{\mathrm{BC}}(o)
+
\Delta a_{\mathrm{SAC}}(o).
\]

target twin critic 的工程 LCB 先给出二值修正选择：

\[
g_{\mathrm{LCB}}\in\{0,1\}.
\]

训练 observation normalizer 给出已有的最大标准化偏离分数
\(s_{\mathrm{OOD}}(o)\)。L75 诊断集中，成功 episode 的均值主要位于约
`2.4–3.3`，失败 episode 约位于 `3.7–4.0`。该诊断只用于预先固定本轮的两个
开发阈值：

\[
c_{\mathrm{support}}(s)=
\begin{cases}
1, & s\le 3.0,\\
\dfrac{4.5-s}{4.5-3.0}, & 3.0<s<4.5,\\
0, & s\ge 4.5.
\end{cases}
\]

最终送入 prior decoder 的 latent action 为：

\[
a_{\mathrm{deploy}}
=
a_{\mathrm{BC}}
+
c_{\mathrm{support}}
g_{\mathrm{LCB}}
\Delta a_{\mathrm{SAC}}.
\]

外层 LaserScan complexity 权重和 near-goal fallback 仍按 L74 冻结配置作用于
decoded learned prior。与 L75 的关键区别是：支持度降低时仍保留冻结 BC，而不是
回退到无关的 `GoalWarmStartPrior`。

`s_OOD` 是工程支持度指标，不是已校准概率、置信区间或理论 OOD 保证。本轮不调整
`3.0/4.5`，也不扫描 beta、checkpoint 或 MPPI 参数。

## 3. 条件与受控变量

每个 `model block × scene × episode seed` 运行三个配对条件：

1. `complexity_bc_icode`：step-0 冻结 BC，correction 精确为零；
2. `gated_lcb_icode`：30k SAC correction，target LCB，支持度门关闭；
3. `support_gated_lcb_icode`：完全相同的 30k checkpoint 和 target LCB，只增加
   本轮连续支持度衰减。

三条件共享：

- 同一 ICODE checkpoint；
- 同一 MuJoCo plant 与传感配置；
- 同一 MPPI `K`、horizon、cost、noise；
- 同一 LaserScan complexity gate；
- 同一 near-goal fallback；
- memory 关闭；
- 同一安全链。

## 4. 实验单位、区组与随机化

- 训练随机性的独立重复：3 个 RL/ICODE model block；
- episode seed 是模型块内的重复测量；
- 控制 step 不是独立样本；
- 场景：single obstacle、narrow corridor、U-trap；
- 新开发 seed：`22200901–22200905`；
- 封存 seed：`22200911–22200915`；
- 总数：`3 × 3 × 5 × 3 = 135 episodes`；
- 每个模型块内的条件顺序由固定 schedule seed 随机打乱；
- L74、L75 和更早保护 seed 由 runner 拒绝复用。

## 5. 主要对比

主要比较是 `support_gated_lcb_icode − complexity_bc_icode`。同时报告：

- `gated_lcb_icode − complexity_bc_icode`；
- support gate 相对 always-LCB 保留了多少 gain、减少了多少 loss；
- episode 级离线 best-of-three oracle，仅作为可选择性上界，不是可部署方法。

## 6. 预注册开发 Gate

只有以下条件全部满足，L76 才允许进入封存确认：

1. 135 个 episode 完整、主键唯一、数值有限且未使用受保护 seed；
2. 至少 2/3 模型块的净成功变化非负；
3. 相对冻结 BC：success gain 至少 2，success loss 不超过 2，净 gain 至少 1；
4. U-trap success loss 为 0；
5. narrow corridor 净 success gain 至少 2；
6. 新增碰撞为 0；
7. 平均最终目标距离改善不小于 0；
8. success loss 不得多于 always-LCB；
9. success gain 最多允许比 always-LCB 少 1；
10. 有效 correction authority 相对 always-LCB 至少降低 15%；
11. mean support confidence 位于 `[0.15, 0.90]`，排除恒开或恒关；
12. 冻结 BC 条件的 correction gate alpha 必须精确为 0。

这些是开发资格门，不是显著性检验。任一项失败都保持封存种子关闭。

## 7. 可证伪条件

以下任一现象会否定当前实现：

- 三个模型块大多为负；
- 支持度衰减丢失窄通道中的 SAC gain；
- U-trap 或单障碍新增成功损失；
- correction authority 降低但性能没有改善；
- 支持度分数实际恒开或恒关；
- 出现碰撞回归。

若失败，不能继续事后扫描当前 L76 seed。应保留结果并重新定义下一轮机制。
