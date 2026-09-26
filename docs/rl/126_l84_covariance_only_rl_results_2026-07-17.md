# L84：ICODE 条件下 covariance-only RL 正式结果

日期：2026-07-17  
性质：已完成的训练与闭环确认；不是 ICRA 最终实验结论

## 1. 研究问题

在 ICODE 残差预测模型、MPPI 代价函数和安全仲裁均保持不变时，RL 是否可以只调节 MPPI 采样协方差，而不直接修改控制均值，并在高动态路径跟踪中产生稳定收益？

策略输出只有两个受限尺度：

\[
a_t^{\mathrm{RL}}=(s_{v,t},s_{\omega,t}),\qquad
s_{v,t},s_{\omega,t}\in[0.5,2.0].
\]

MPPI 的均值序列仍来自传统 goal warm start。策略仅修改：

\[
\Sigma_t=
\operatorname{diag}\!\left[
(s_{v,t}\sigma_v)^2,
(s_{\omega,t}\sigma_\omega)^2
\right].
\]

因此本实验验证的是“RL 引导 ICODE-MPPI 如何搜索”，而不是 RL 绕过 MPPI 直接控制小车。

## 2. 前置诊断与方向选择

L82 发现：ICODE 对完整开环控制序列真实代价的排序不优于参数量匹配 MLP，因此淘汰了直接使用 ICODE elite sequence 作为 proposal teacher 的路线。该结果不否定 ICODE 的闭环价值，只否定“完整开环排序 teacher”这一实现假设。

L83 对五组固定协方差进行精确 MuJoCo counterfactual rollout。相对默认协方差，最佳固定配置改善真实加权控制序列代价约 26.46%；逐状态 oracle 相对最佳固定配置仍有约 4.14% 的额外空间，且 3/3 模型块均为正。因此 L84 升级为真正的 covariance-only SAC 训练。

## 3. 固定实验条件

- prediction dynamics：三个独立 L57 ICODE checkpoint；
- RL training seeds：`20268801`、`20268802`、`20268803`；
- MPPI：`K=100`、`H=36`；
- memory：关闭；
- control mean：传统 prior，RL 不修改；
- safety：原有 scan guard 与 arbitration 保持启用；
- 训练场景：accel-straight、accel-turn、chicane、sweep；
- 验证场景：accel-turn、chicane、unseen hairpin、unseen reverse-S；
- 每个训练块：20,000 environment steps；
- 最佳 checkpoint：按安全不退化条件下的验证回报选择；
- 正式确认：3 blocks × 4 scenes × 5 plant seeds × 4 conditions，共 240 个闭环回合。

四个确认条件为：

1. learned covariance；
2. fixed baseline `(1.0, 1.0)`；
3. fixed speed-explore `(1.75, 0.75)`；
4. fixed broad `(1.75, 1.75)`。

## 4. 三个训练块的可训练性结果

三个训练块的最佳 checkpoint 均优于各自未训练初始策略，并且成功率、碰撞率没有退化。

- 平均横向 RMSE 相对下降：19.95%；
- 最弱训练块相对下降：14.82%；
- 正向训练块：3/3；
- 所有验证回合：成功率 100%，碰撞率 0。

训练过程中三个种子都出现先退化、后恢复的共同形态：约 5,000--7,500 步时策略过度探索；10,000--12,500 步后开始稳定改善。这说明当前实现可训练，但样本效率仍可优化，不能把任意最后一步模型当作最佳模型。

## 5. 240 回合正式闭环结果

以下为每个条件 60 个配对回合的总体均值。

| 条件 | 成功率 | 碰撞率 | 横向 RMSE (m) | 到达时间 (s) | 轨迹长度 (m) | control jerk | planner mean (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| learned covariance | 1.00 | 0.00 | 0.044895 | 28.610 | 6.550 | 0.1066 | 31.730 |
| fixed baseline | 1.00 | 0.00 | 0.055861 | 29.413 | 6.604 | 0.1176 | 32.034 |
| fixed broad | 1.00 | 0.00 | 0.069448 | 30.047 | 6.690 | 0.1384 | 31.904 |
| fixed speed-explore | 1.00 | 0.00 | 0.039087 | 30.362 | 6.542 | 0.0995 | 31.870 |

### 5.1 相对预注册主对照 fixed broad

learned covariance 的配对结果：

- 横向 RMSE 差：`-0.024553 m`；
- 95% bootstrap CI：`[-0.026685, -0.022420] m`；
- 到达时间差：`-1.4367 s`；
- 95% bootstrap CI：`[-1.6600, -1.2067] s`；
- 3/3 模型块横向 RMSE 改善；
- 成功率差：0；
- 碰撞率差：0。

预注册主门槛通过。

### 5.2 相对默认 fixed baseline

learned covariance 同时改善：

- 横向 RMSE：`-0.010966 m`，CI `[-0.013054, -0.009022]`；
- 到达时间：`-0.8033 s`，CI `[-1.0050, -0.6017]`；
- 轨迹长度：`-0.0540 m`，CI `[-0.0636, -0.0445]`；
- control jerk：`-0.01105`，CI `[-0.01316, -0.00890]`。

这说明收益不是仅由选择一个更宽的固定噪声产生。

### 5.3 相对 fixed speed-explore

该比较形成稳定的 Pareto 权衡：

- learned 到达时间更短：`-1.7517 s`，CI `[-1.9817, -1.5233]`；
- learned 横向 RMSE 更高：`+0.005808 m`，CI `[0.004054, 0.007656]`；
- learned control jerk 更高：`+0.00702`，CI `[0.00483, 0.00912]`；
- 轨迹长度差 `+0.00832 m`，CI 跨 0；
- planner compute time 差异 CI 跨 0。

四个场景中，learned 都更快，speed-explore 都有更低横向 RMSE。因此不能声称 learned 在所有指标上优于最强固定配置。

## 6. 能支持的结论

当前证据支持以下有限结论：

1. 在固定 ICODE-MPPI 和安全链条件下，RL 可以只通过状态相关协方差调节产生可复现的闭环收益；
2. 该收益跨三个独立 ICODE/RL 模型块、seen/unseen 路径和五个 plant seeds 保持；
3. learned covariance 全面优于默认和 fixed broad；
4. 相对 fixed speed-explore，learned 获得显著时间优势，但存在显著精度和平滑性代价；
5. RL 输出在回合内具有明显变化，不是退化为一个常数协方差。

## 7. 不能声称的结论

当前结果不支持：

- “RL 在所有指标上优于全部固定 MPPI 参数”；
- “已经完成 uncertainty gate”；
- “已经证明 OOD 安全保证”；
- “已经得到 ICRA 最终实验结论”；
- “当前单一 ICODE 网络提供理论置信区间”；
- “当前 unicycle/differential-drive 实验完整复现论文五状态 vehicle model”。

## 8. 下一阶段决策

下一阶段不回到 direct action correction，也不增加更多互相纠缠的模块。核心问题收敛为：

> 能否在保持 learned covariance 时间优势的同时，使横向误差不劣于 fixed speed-explore？

优先探索两个实现，按简单到复杂排序：

1. **精度约束 covariance RL**：保留二维 covariance-only action，使用显式横向误差约束或拉格朗日代价，而不是继续事后调一个标量 reward；
2. **preference-conditioned covariance policy**：给策略一个速度--精度偏好变量，学习一条可审计 Pareto 前沿，并允许简单场景退化到固定传统配置。

只有在精度约束版本仍无法跨种子稳定收回误差时，才启用更强的离散/一维受限协方差流形。该顺序保持论文 story 为三层：ICODE 预测、RL 调节搜索分布、MPPI 与安全链输出控制。

## 9. 证据位置

- 训练汇总：`results/research_platform/rl/l84_covariance_only_training_3blocks.json`；
- 正式矩阵：`results/research_platform/rl/l84_covariance_policy_evaluation_20260717_v1/`；
- 正式汇总：上述目录的 `summary.json`；
- 每回合指标：上述目录的 `episodes.csv`；
- 条件/场景汇总：上述目录的 `condition_scene_summary.csv`；
- 三个 best checkpoints：各 `l84_covariance_only_seed*/checkpoints/best.pt`；
- 运行配置：`configs/rl/covariance_policy_evaluation_l84.yaml`。
