# RL + ICODE 实现方向组合与决策树

日期：2026-07-17  
性质：方法设计记录；不包含新的实验结论

## 1. 不变的研究主线

本文档不改变导师确定的研究大方向：

> ICODE 学习真实小车相对名义模型的残差动力学，RL 在复杂或 OOD 情形下改善 MPPI 的搜索，MPPI 与安全仲裁保留最终控制权。

当前已有封存证据支持把 ICODE 作为固定底座：

- L59：ICODE 相对 nominal 的高动态路径跟踪误差改善跨模型块、跨场景成立；
- L62：在参数量匹配的比较中，ICODE 虽然 offline H36 RMSE 不优于 MLP，但闭环跟踪优于 MLP；
- L65：上述 ICODE 优势跨质量、摩擦、执行器、时延和组合 plant shift 保持；
- L68：上述优势在 clean、观测时延和噪声加时延条件下保持。

因此，后续不再反复把“ICODE 是否有效”作为主要问题，而研究：

> **RL 应该在 MPPI 的哪一层介入，才能稳定利用 ICODE 的预测能力？**

L79--L81 已否定一个更窄的实现假设：当前 always-on、共享参数的 SAC residual-correction prior 尚不能跨独立初始化稳定复现。该失败不否定 RL + ICODE，只说明需要改变 RL 的学习对象和介入位置。

## 2. 设计原则

1. ICODE 始终负责预测动力学，不让 RL 重复学习完整动力学。
2. RL 不直接绕过 MPPI 输出 `/cmd_vel`；MPPI 仍评价候选轨迹并给出提议控制。
3. `scan_guard` 和 safety arbitration 始终位于最高优先级。
4. 每次只改变一个主要机制，避免同时混合 residual、memory、RL prior 和多种 gate。
5. 简单场景允许 RL 退化为零作用；论文价值来自复杂/OOD 条件下的增益，而不是要求 RL 在所有场景都胜过 MPPI。
6. 先验证中间机制，再投入大规模 RL 训练；不能只看最终成功率猜测原因。

## 3. 候选实现池

评分范围为 1--5，5 表示更有利。分数用于安排实验顺序，不是论文结论。

| 编号 | RL 学习对象 | ICODE 的作用 | 预期优点 | 主要风险 | 论文契合 | 工程可行 | 首轮优先级 |
|---|---|---|---|---|---:|---:|---:|
| A | MPPI 控制序列的均值与结构化协方差 | 评价候选并产生 elite target | 学习信号密集，直接降低无效采样 | teacher 排序若偏差会被蒸馏 | 5 | 5 | 1 |
| B | 按状态分配 `K`、horizon、noise covariance | 提供 rollout 和预测不确定性特征 | 不直接改动作，安全且容易退化 | 可能只带来效率而非控制质量 | 4 | 5 | 2 |
| C | traditional prior / RL specialist 的混合权重 | 给 gate 提供模型可信度与 rollout cost | 符合“简单场景传统、复杂场景 RL” | gate 标签与校准可能复杂 | 5 | 4 | 3 |
| D | 候选轨迹 ranker / terminal value | 产生短期物理可行轨迹特征 | 可补 MPPI 有限 horizon 的远期判断 | 容易与手工 cost 重复或错配 | 4 | 4 | 4 |
| E | 主动探索策略，选择 residual 数据采集动作 | ICODE 是被改进的模型 | 直接体现 RL 探索功能 | 离在线控制收益链更长 | 5 | 3 | 5 |
| F | waypoint / subgoal policy | ICODE + MPPI 完成局部可行控制 | 能解决 U-trap 等长时域问题 | story 易变成全局规划论文 | 4 | 4 | 6 |
| G | residual authority / dynamics expert selector | 在多残差专家间切换 | 对多地面、多载荷实车有价值 | 当前单一 ICODE 已较强，需新域 | 3 | 3 | 7 |
| H | 在线 cost-weight adaptation | ICODE 负责 rollout | 接口简单、可解释 | 可能破坏 cost 语义和安全裕量 | 3 | 4 | 8 |

## 4. 首选方法：ICODE-guided proposal RL

首选方向不是让 RL 直接输出最终控制，而是让其输出 MPPI 的提案分布：

\[
q_\phi(U\mid o_t,g_t)
=
\mathcal N\!\left(\mu_\phi(o_t,g_t),\Sigma_\phi(o_t,g_t)\right),
\]

其中 (U=[u_t,\ldots,u_{t+H-1}])。在线仍由 ICODE rollout 评价从该分布采样的控制序列，并由 MPPI 重新加权：

\[
w_k\propto
\exp\left(-\frac{S_{\mathrm{ICODE}}(U_k)-\rho}{\lambda}\right).
\]

### 4.1 训练分两段

第一段是 planner-guided initialization：使用 ICODE-MPPI 的 elite 控制序列或加权均值提供密集初始化目标。它是初始化，不等于最终方法是纯监督学习。

第二段是 RL fine-tuning：策略根据真实 MuJoCo 闭环回报更新，允许超越 teacher；同时使用 KL 或 trust-region 约束，防止策略一次更新离开已经验证的可行区域。

### 4.2 这一设计为何不同于失败的 shared correction

- 旧方案学习“在 BC prior 上加多少动作修正”，回报链长且每个动作会改变后续状态分布；
- 新方案学习“哪些完整候选序列值得优先采样”，监督信号来自每次 MPPI 已经计算的整批候选；
- MPPI 仍能拒绝 RL 的坏提案，而不是把 RL correction 直接当作均值中心；
- ICODE 的价值从“隐含在 RL 环境里”变成“显式决定 elite 和排序”。

相关方法背景包括 path-integral policy improvement、学习型 MPC prior 以及 MPPI 协方差设计，但本项目的研究问题是 ICODE 残差结构、RL proposal 与跨层 gate 在差速小车物理平台上的组合与验证：

- [Learning Policy Improvements with Path Integrals](https://proceedings.mlr.press/v9/theodorou10a.html)
- [CoVO-MPC: optimal covariance design for sampling-based MPC](https://proceedings.mlr.press/v242/yi24b.html)
- [Combining Model-Based and Model-Free Updates for Trajectory-Centric RL](https://proceedings.mlr.press/v70/chebotar17a.html)

## 5. 第二候选：RL adaptive sampling allocation

RL 不输出动作均值，而输出受限的离散/连续采样参数：

\[
a_t^{\mathrm{RL}} = (K_t,\;H_t,\;s_{v,t},\;s_{\omega,t}),
\]

其中 (K_t) 是样本数，(H_t) 是 horizon，(s_v,s_\omega) 缩放基础协方差。所有输出均被限制在预注册的安全范围内。

适合的目标是以控制质量为约束最小化计算量，或在固定计算预算下提高成功率。该方向的优势是不会直接覆盖传统 prior；简单区域可以使用小 (K)，复杂/OOD 区域增加探索或 horizon。

## 6. 第三候选：complexity/OOD-gated specialist mixture

定义传统 prior 与 RL specialist 的混合：

\[
q(U\mid o)=
(1-\alpha(o))q_{\mathrm{traditional}}(U\mid o)
+\alpha(o)q_{\mathrm{RL}}(U\mid o),
\qquad \alpha(o)\in[0,1].
\]

`alpha` 不直接解释为真实概率，也不声称提供安全保证。它可以由以下可审计输入决定：

- ICODE ensemble disagreement 或 rollout residual support；
- LaserScan 几何复杂度、通道宽度、目标遮挡；
- recent MPPI effective sample size、elite concentration、stuck history；
- RL prior 相对 traditional prior 的离线/在线 competence calibration。

默认 `alpha=0`，只有在 development calibration 中被证明有益的区域才开放 RL authority。

## 7. 路线选择所需的 L82 诊断

在训练新策略前，先回答：

> 对完全相同的候选控制序列，ICODE 是否比 nominal 和参数量匹配 MLP 更准确地排序真实 MuJoCo 闭环成本？

如果成立，A 路线具有可信 teacher，随后测试 A；如果只在部分状态成立，优先 A+C；如果不成立，暂停 ICODE teacher distillation，优先 B 或 E。

这一步同时解释此前“MLP offline RMSE 更低、ICODE 闭环却更好”的现象：控制器真正需要的未必是所有状态分量的平均预测误差最小，而是把低成本候选排在前面。

## 8. 快速淘汰和升级规则

### 8.1 A：proposal RL

先做 imitation-only diagnostic，再做 RL fine-tuning。若 proposal 在相同 (K) 下不能提高 elite recall 或真实闭环质量，停止训练；不能靠扩大网络或事后挑 seed 挽救。

### 8.2 B：adaptive budget

必须形成 Pareto 改善：同等质量下更快，或同等计算时间下更好。仅改变平均 `K` 而没有质量/耗时收益则淘汰。

### 8.3 C：gated mixture

必须满足简单场景 `alpha` 接近零、复杂/OOD 场景在独立 seed 上产生非零且有益的 authority；若 gate 只学会永远关闭或永远打开，则不构成方法贡献。

### 8.4 E：active residual collection

必须在相同真实交互预算下，提高 unseen disturbance 的 ICODE rollout 或闭环指标；只增加数据量而没有 sample-efficiency 改善则淘汰。

## 9. 当前推荐顺序

1. L82：候选序列排序保真度；
2. 若通过，建立 ICODE elite dataset 与 proposal policy 的离线初始化；
3. 比较 traditional、BC proposal、RL-finetuned proposal；
4. 只在上述 proposal 确有贡献后加入 complexity/OOD gate；
5. adaptive budget 作为独立效率贡献或备选主线；
6. active collection 只在排序诊断或 proposal 路线失败时升级为主线。

该顺序保留了多个创新空间，同时保证论文 story 始终只有三层：**ICODE 预测、RL 引导搜索、MPPI 安全优化**。
