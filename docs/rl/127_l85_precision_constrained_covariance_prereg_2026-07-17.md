# L85：精度约束的协方差策略预注册

日期：2026-07-17  
状态：在查看任何 L85 训练或确认性结果之前冻结  
前序证据：L84 三个独立训练块与 240 回合闭环评估

## 1. 为什么进行 L85

L84 已证明 RL 可以学习 MPPI 采样协方差，并在全部四个场景中相对默认协方差减少到达时间；但相对最强固定设置 `speed=(1.75, 0.75)`，学习策略的平均到达时间减少 1.752 s，而 cross-track RMSE 增加 0.00581 m。因此当前证据支持“上下文自适应产生速度—精度权衡”，尚不支持“学习策略全面优于最强固定设置”。

L85 只解决这一处缺口，不增加新的网络、感知模块、RL prior 类型或理论声明。

## 2. 研究问题与可证伪假设

研究问题：在 ICODE 残差预测模型固定的条件下，SAC 是否可以通过在线调整 MPPI 的二维采样协方差，在满足路径跟踪精度约束的同时，保留相对强固定协方差的到达时间优势？

约束形式为：

$$
\min_{\pi}\; \mathbb{E}_{\pi}[T]
\quad\text{s.t.}\quad
\sqrt{\mathbb{E}_{\pi}[e_{\mathrm{ct}}^2]} \leq \epsilon,
$$

其中 $T$ 为到达时间，$e_{\mathrm{ct}}$ 为机器人到参考折线的横向距离，预注册阈值为

$$
\epsilon = 0.040\ \mathrm{m}.
$$

训练使用拉格朗日乘子：

$$
\mathcal{L}(\pi,\lambda)
= -J_{\mathrm{task}}(\pi)
+ \lambda\left(\mathbb{E}_{\pi}[e_{\mathrm{ct}}^2]-\epsilon^2\right),
\qquad \lambda\ge 0.
$$

可证伪假设 H1：相对 `speed=(1.75, 0.75)`，约束策略在确认集上的 cross-track RMSE 非劣，且到达时间显著更短。

若精度非劣性或时间优势任一失败，则 H1 不成立；不得用综合加权分数掩盖失败。

## 3. 干预、对照与保持不变的部分

唯一训练干预为：

- replay 记录每个 transition 的 $e_{\mathrm{ct}}^2$；
- critic 使用当前乘子形成的约束奖励 $r-\lambda e_{\mathrm{ct}}^2$；
- 每个完整 episode 结束后，根据 episode 均方横向误差更新 $\lambda$；
- checkpoint 选择首先要求精度可行，再按成功、安全和任务回报排序。

保持不变：

- ICODE checkpoint、MPPI horizon、采样数、基础噪声与成本；
- SAC 网络结构、观测、动作边界和四个训练/验证场景；
- memory 关闭；
- scan/safety 仲裁链保持启用；
- RL 只输出两个正的 covariance scale，不直接输出底盘控制。

对照：

1. 固定 `baseline=(1.0, 1.0)`；
2. 固定 `speed=(1.75, 0.75)`，主要强对照；
3. L84 无约束学习策略，用于解释约束的增量作用；
4. L85 精度约束学习策略。

## 4. 实验单位、阻断与随机化

独立实验单位是一个完整训练块：独立 SAC seed，并从相同冻结的 ICODE checkpoint 开始。物理 episode seed 是块内重复测量，不得被当作独立训练重复。

- 开发阶段：1 个新训练 seed，只用于调试和检查可学习性；
- 确认阶段：至少 3 个预先固定的独立训练 seed；
- 每个块在 4 个场景、每场景 5 个全新物理 seed 上与所有对照配对；
- 方法执行顺序由确定性 seed 打乱，避免顺序与方法完全混杂；
- L84 正式评估 seed 不得作为 L85 确认 seed。

统计汇总以训练块为最高层级；episode 只提供块内配对差值。置信区间使用分层 bootstrap：先重采样训练块，再在块内重采样场景和配对 seed。

## 5. 预注册训练规则

- `target_cross_track_rmse_m = 0.040`；
- `initial_multiplier = 0.0`；
- 乘子按 episode 更新并限制在 `[0, maximum_multiplier]`；
- replay 保存原始 task reward 和约束 cost，旧 checkpoint 缺失 cost 时按零恢复，保证兼容；
- 默认 `precision_constraint.enabled=false` 时数值路径保持原样；
- L85 环境中固定 cross-track reward 权重设为 0，避免同一误差被静态权重与对偶变量重复惩罚；
- 乘子、约束 violation、原始/约束 reward 分别记录；
- 训练中断后的恢复必须包含 replay cost、当前乘子和更新状态。

开发阶段允许调整一次 `dual_lr` 或 `maximum_multiplier`，但每次调整都必须形成新版本配置和结果目录；确认阶段开始后不得再调参。

## 6. Checkpoint 选择

验证 checkpoint 的选择顺序为：

1. 成功率最高；
2. 碰撞率最低；
3. `mean_cross_track_rmse <= 0.040 m` 的 checkpoint 才是精度可行解；
4. 可行解中优先较高 mean task return；
5. 如果训练期间从未出现可行 checkpoint，则保存最小 cross-track RMSE 的 checkpoint，并明确标记 `constraint_infeasible`，不得称为最佳可行策略。

初始随机策略不作为论文方法，但保留 step-zero 评估用于诊断学习是否发生。

## 7. 主要与次要指标

主要指标：

- cross-track RMSE；
- 到达时间；
- success rate；
- collision rate。

次要指标：trajectory length、control jerk、planner mean/max compute time、协方差两个维度的均值/方差、乘子轨迹和约束违反量。

## 8. 确认门槛

相对固定 `speed` 对照，L85 同时满足以下条件才算支持 H1：

1. success rate 不下降，collision rate 不上升；
2. cross-track RMSE 配对差的 95% 分层 bootstrap 上界不超过 `+0.002 m`；
3. 到达时间配对差的 95% 分层 bootstrap 上界小于 0；
4. 至少 3/3 独立训练块的平均到达时间差为负；
5. 结果不由单一场景反转或单一 seed 驱动。

若只满足精度条件而没有时间优势，则说明约束有效但 RL 没有带来控制效率收益；若只满足时间条件而精度失败，则复现 L84 的 Pareto 权衡，不能支持 H1。

## 9. 声明边界

L85 验证的是“ICODE 预测模型上的 RL 自适应 MPPI 采样协方差”，不是通用安全 RL，也不提供稳定性、收敛性或约束必然满足的数学保证。训练时的拉格朗日更新是工程型约束优化机制；所有安全结论仍以独立碰撞指标和现有 safety arbitration 为准。
