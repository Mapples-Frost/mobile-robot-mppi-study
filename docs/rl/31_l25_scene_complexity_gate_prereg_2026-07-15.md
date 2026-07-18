# L25：可解释 LaserScan 场景复杂度门控预注册

日期：2026-07-15  
性质：development experiment；不打开 sealed test  
前置结果：L24 否定了“用当前一次候选轨迹预测未来十轮闭环效用”的假设。

## 1. 研究问题

本轮不训练新的效用网络，而检验一个更窄、可审计的问题：

> 仅使用机器人当时真实可获得的 LaserScan，能否在简单局部区域退化为传统 MPPI，并在局部几何复杂区域有选择地开放冻结的 RL sampling prior？

RL 仍然只修改 MPPI 的采样均值；MPPI 仍执行轨迹优化；`scan_guard`、`local_obstacle_layer` 和最终安全仲裁不改变。ICODE、Memory、动态障碍和动力学 OOD 本轮关闭，避免跨层混杂。

## 2. 复杂度定义

输入只来自 `LaserScan.ranges` 或同一传感链生成的 `obstacle_ranges`，不读取 MuJoCo 障碍物真值。定义三个归一化分量：

\[
s_{\mathrm{front}}=\operatorname{clip}\!\left(
\frac{d_{\mathrm{far}}-d_{\mathrm{front}}}
{d_{\mathrm{far}}-d_{\mathrm{near}}},0,1\right),
\]

\[
s_{\mathrm{constriction}}=
\min(s_{\mathrm{left}},s_{\mathrm{right}}),
\qquad
s_{\mathrm{density}}=
\operatorname{clip}\!\left(
\frac{\rho(d\le d_{\mathrm{density}})}{\rho_{\mathrm{full}}},0,1\right).
\]

最终分数为：

\[
s_{\mathrm{complexity}}=
\max\left(
s_{\mathrm{front}},
s_{\mathrm{constriction}},
s_{\mathrm{density}}
\right).
\]

RL 外层权重采用分段线性激活：

\[
\alpha_{\mathrm{RL}}=
\begin{cases}
0, & s\le 0.35,\\
\dfrac{s-0.35}{0.70-0.35}, & 0.35<s<0.70,\\
1, & s\ge 0.70.
\end{cases}
\]

距离阈值依据现有物理安全尺度预先固定：`near=0.45 m` 与 planner obstacle influence/soft block 一致，`far=1.30 m` 表示局部规划仍有明显绕行余量。缺失或无效 scan 必须令复杂度为零，使 RL fail closed，而不是猜测环境复杂。

该分数是可解释的工程几何指标，不是严格的概率、统计置信度，也不等同于“策略训练分布之外”的完整 OOD 定义。

## 3. 固定控制条件

每个 `training checkpoint × scene × episode seed` 区组运行四个配对条件：

1. `traditional_mppi`：`GoalWarmStartPrior`，完全不加载 RL；
2. `frozen_bc_prior`：冻结的 step-0 BC prior；
3. `lcb_always`：L17 step-20k correction，加 L20 已固定的 target twin-critic LCB `beta=2`，外层始终开放；
4. `complexity_lcb`：与 3 使用完全相同 checkpoint 和 beta，仅增加本轮 LaserScan complexity outer gate。

三个 independently trained correction checkpoints 为 `20260721`、`20260722`、`20260723`。不重新选择 actor、不扫描 beta、不依据本轮结果调整复杂度阈值。

## 4. 场景区组

简单场景：

- `clean_dynamics`：无障碍；
- `clean_single_obstacle`：单障碍绕行。

复杂场景：

- `narrow_corridor`；
- `u_trap_long_board`。

U-trap 是 actor 训练几何；其他几何未进入 L17 correction 训练。因此本轮同时能暴露“局部复杂度高”与“策略几何泛化不足”，但不把两者错误宣称为已经正交分离。动态障碍将在静态门控通过后作为独立因素加入。

## 5. 实验单位、随机化与封存

- 配对单位：`training checkpoint × scene × episode seed`；
- checkpoint 是训练随机性的独立重复；episode seed 是每个 checkpoint 内的环境重复；
- 四条件在每个区组中使用相同 seed，运行顺序由脚本按固定 seed 打乱；
- development seeds：`20281701--20281710`；
- sealed test seeds：`20281711--20281730`，Development Gate 失败时不得运行；
- 场景与 episode rows 不被当成额外独立训练重复，汇报时保留 checkpoint 分层。

## 6. 开发样本量依据

L24 的 108 个 checkpoint-episode group means 给出约 `2.499 cm` 的效应标准差。若最小有意义距离改善取 `1 cm`，则配对标准化效应约为：

\[
d_z=\frac{1.0}{2.499}\approx0.40.
\]

双侧 `alpha=0.05`、power `0.80` 的配对均值设计约需 52 个独立 pairs。L25 development 每个场景只有 `3×10=30` 个 checkpoint-episode blocks，因此明确定位为方法筛选而非论文确认性检验；不报告事后功效或把 360 个 step 当成 360 个独立样本。若 development gate 通过，预留的 20 个 test seeds 可形成每场景 60 个配对 blocks，同时正式稿仍应增加到至少五个独立训练 seeds，以检验训练过程而非单一 actor 的偶然性。

## 7. 冻结 Development Gate

只有以下条件全部满足才允许打开 test：

1. 每个 checkpoint、scene、seed 和四个条件完整，数值有限，无碰撞账目错误；
2. `complexity_lcb` 在两个简单场景合计的 mean alpha 不超过 `0.15`，active fraction 不超过 `0.25`；
3. 在两个复杂场景合计的 mean alpha 至少 `0.10`，active fraction 位于 `[0.15,0.85]`，避免永远关闭或永远开启；
4. complexity gate 相比 `lcb_always` 至少减少 50% 的平均 RL 权重；
5. 简单场景相对 `traditional_mppi` 不损失任何 paired success；
6. 所有场景相对 traditional 不产生新 collision；
7. 复杂场景相对 traditional 的 success gains 至少比 losses 多 1，且平均最终距离改善至少 `0.01 m`。

这些是开发资格门，不是显著性检验。任何一项失败都保留传统 MPPI fallback，并保持 test seeds 封存。

## 8. 主要输出

- episode-level paired CSV；
- step-level gate/complexity CSV；
- 按 checkpoint 和 scene 分层的 success、collision、final distance、clearance、jerk、compute time；
- complexity 分量、gate alpha、active fraction；
- frozen config、checkpoint hashes、git SHA 和 Development Gate decision。

不得把 development 结果称作正式论文结论，也不得因为结果接近阈值而事后修改门槛。

