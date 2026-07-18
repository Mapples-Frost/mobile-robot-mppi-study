# 冻结 ICODE-style 控制仿射残差模型下的 RL 引导候选轨迹采样 MPPI

> ICRA 2027 研究与投稿计划（根据导师意见修订的讨论稿）

**暂定英文题目：** *RL-Guided Candidate-Trajectory Sampling for MPPI with Frozen Residual Dynamics*  
**版本日期：** 2026 年 7 月 12 日  
**内部投稿节点：** 2026 年 9 月 14 日（北京时间）  
**官方截稿节点：** 2026 年 9 月 15 日 23:59（Pacific Time）

---

老师您好。感谢您对前一版计划提出的具体建议。根据您的意见，我拟将论文主线进一步收敛：ICODE-style residual model 在主实验开始前完成校验并冻结；RL 只用于提出 MPPI 的候选控制序列或采样先验；MPPI 仍负责在线轨迹评价、加权更新与最终 proposed control；现有安全仲裁继续拥有最终执行权。

本修订稿将“RL 探索”暂时明确为**MPPI 候选控制轨迹空间中的探索**，而不是让 RL 专门负责 residual 数据采集。RL 训练轨迹可以用于覆盖范围和分布审计，但默认不回流更新 ICODE。若我对“探索”的理解仍有偏差，恳请老师指正。

本稿仍是研究讨论稿。文中“拟议方法”“预期关系”和阶段阈值均需要通过相关工作核查、代码审计和正式实验验证，不作为已经获得的论文结论。

---

## 一、修订后的总体方案

### 1.1 相比上一版的主要调整

| 项目 | 上一版计划 | 本次修订 |
|---|---|---|
| RL 的主要职责 | 输出 MPPI prior，并与多层置信度联合门控 | 保留 sampling prior/trajectory proposer，重点研究候选轨迹探索 |
| ICODE 的部署方式 | residual gating，并考虑 on-policy 更新 | 主实验前选定 checkpoint 并冻结，不持续在线训练 |
| 调节机制 | residual trust 与 policy trust 的双层复杂门控 | 先研究一个候选样本分配系数；复杂置信度降为可选扩展 |
| OOD 范围 | scene × dynamics 双轴大规模组合 | 只保留少量、可解释的未见场景或动力学变化 |
| 障碍物 | 静态与动态均进入主矩阵 | 静态障碍为主，动态障碍为条件性扩展 |
| 实车 | 多阶段完整部署 | 以 offline、shadow 和有限低速验证为主 |
| 理论表达 | 较多不确定性与可靠度形式化描述 | 以技术方法、受控消融和闭环实证为主 |
| 工作量 | 多模块、700–1000 次控制实验 | 最小充分 2×2、RL 学习曲线和分层实验 |

### 1.2 拟聚焦的核心问题

本研究拟集中回答：

> 在冻结的 residual-enhanced rollout model 下，RL 生成的候选控制轨迹能否在有限 MPPI samples 和固定实时预算内，提高有效候选轨迹覆盖与困难静态场景的闭环性能；一个保留传统 MPPI 锚点的简化探索分配机制，能否在简单场景中避免无必要地依赖 RL，并在困难或未见条件中发挥 RL proposal 的增量价值？

该问题不预设 RL 一定优于传统 MPPI。传统 MPPI 是从第一次运行即可工作的强基线；RL 的收益必须在完整训练曲线、相同搜索预算和相同计算预算下得到验证。

### 1.3 模块职责

~~~text
Frozen ICODE-style residual
    用于：修正候选控制序列的运动预测，并检验是否提高精度

RL trajectory proposer
    负责：提出值得 MPPI 优先评价的控制轨迹或低维 control knots

Traditional MPPI prior
    负责：保留 previous-sequence / goal warm-start 锚点

MPPI
    负责：统一评价候选轨迹、计算权重并更新控制序列

Safety arbitration
    负责：在最终执行前检查并覆盖危险控制
~~~

### 1.4 本轮明确不作为主线的内容

- 不让 RL 作为 residual 数据采集器；
- 不在 RL 训练过程中更新 ICODE 参数；
- 不进行持续在线 residual training；
- 不同时研究多个 RL 算法；
- 不将 Memory cost、memory sampling bias 和 RL prior 混入核心消融；
- 不把动态障碍预测作为主要算法创新；
- 不提出稳定性、安全性、收敛性或任意 OOD 泛化保证；
- 不设置形式化定理或命题章节；
- 不把 ICODE、RL prior 或二者的直接组合本身表述为首次创新。

### 1.5 拟议贡献及其边界

以下内容只是待验证的贡献候选。

1. **锚定式双源候选轨迹构造。** 在固定总 samples 下，同时保留 traditional MPPI candidates 与 RL-guided candidates，避免 RL proposal 完全取代强传统基线。
2. **相对候选质量感知的探索分配。** 分别评价 traditional 与 RL 两个候选池在上一周期的可行性和 elite cost，用一个有界标量调节下一周期的样本比例；只有 RL candidates 显示相对优势时才增加其预算。
3. **冻结学习动力学下的职责分离与系统实证。** 通过 Nominal/Frozen-ICODE × Traditional/RL-prior 的完整 2×2 消融，区分预测模型改进与候选生成改进，并报告学习曲线、实时开销和安全指标。

如果第二项机制不能在消融中产生稳定增益，则不得将其作为主要贡献；如果完整方法不能优于最佳单模块，则不得声称 ICODE 与 RL 具有增量互补性。

目前真正可能形成算法方法差异的主要是第二项 source-relative allocation；第一项更接近架构设计，第三项属于系统实证。该 allocation 原理上也可配合其他 rollout model，并非 ICODE 专属机制。因而最终题目和贡献排序需要在 07-16 相关工作 Gate 后再次确认，也不宜因标题强调冻结 ICODE 而弱化 adaptive sampling 的直接竞争压力。

### 1.6 投稿时间边界

[IEEE Robotics and Automation Society 的 ICRA 2027 页面](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/)目前列出的 full paper submission deadline 为 2026 年 9 月 15 日 23:59 Pacific Time。项目仍按更保守的内部节点执行：

- 2026-08-08：目标方法冻结；
- 2026-08-10：方法硬冻结；若未完成则缩减实验范围，不继续移动冻结日期；
- 2026-08-24：主结果冻结；
- 2026-08-28：完整论文初稿；
- 2026-09-06：实验冻结；
- 2026-09-10：格式、匿名材料、视频和代码检查；
- 2026-09-12：形成内部预提交包；
- 2026-09-14：北京时间内部正式提交；
- 2026-09-15：只保留为紧急缓冲，不安排新实验或新方法。

---

## 二、项目基础与当前缺口

### 2.1 已具备的工程基础

现有仓库已经完成一次兼容式重构，并保留原有 MPPI、MuJoCo、LaserScan、Memory 和实车安全链。目前可直接复用的基础包括：

- 分层的 `mobile_robot_mppi` 研究框架与旧入口兼容层；
- 三状态与五状态动力学接口，以及 Euler/RK4 积分；
- nominal、oracle、MLP residual 与 ICODE-style residual 接口；
- residual dataset、normalization、multi-step rollout loss 和 checkpoint；
- 可配置的 MuJoCo 差速轮物理对象、执行器、接触与传感器模型；
- MuJoCo ray-cast LaserScan、scan_guard 和 local_obstacle_layer；
- MPPI proposed control 与 safety executed control 的显式区分；
- SamplingPrior、PreviousSequencePrior 和 GoalWarmStartPrior 接口；
- 多场景配置、headless benchmark、CSV/JSON 和 provenance 输出；
- ROS Kinetic/Python 2 bridge 与 Python 3/PyTorch 研究栈的隔离边界。

正式 planner obstacles 仍必须来自：

~~~text
LaserScan
    ↓
scan_guard
    ↓
local_obstacle_layer
    ↓
planner obstacles
~~~

不得为了提高实验表现而把 MuJoCo 全局障碍物真值直接提供给 planner。

### 2.2 当前 smoke 结果

现有端到端 smoke 已经打通：

~~~text
MuJoCo 数据采集
→ residual 标签生成
→ ICODE 训练
→ checkpoint 保存与加载
→ ICODE 进入 MPPI rollout
→ MuJoCo 闭环执行
~~~

当前 smoke dataset 只有 120 transitions、6 episodes，ICODE 只训练了 2 epochs。相同场景、相同 seed、64 samples 下的现有结果为：

| 指标 | Nominal | ICODE smoke |
|---|---:|---:|
| Success | 否 | 否 |
| Collision | 否 | 否 |
| Final goal distance | 2.840 | 3.042 |
| Control jerk | 0.106 | 0.053 |
| Minimum clearance | 0.054 | 0.073 |
| Mean planner time | 约 2.6 ms | 约 72.9 ms |

该结果只能证明 ICODE 已实际进入 rollout 并改变控制行为，不能证明其优于 nominal。当前 ICODE 推理开销和目标推进表现也说明，正式训练、批量推理优化和公平预算比较仍是必要前提。

### 2.3 投稿前必须解决的基础问题

1. 对照标准 MPPI 推导审计 control perturbation、proposal distribution 与 importance correction；
2. 修正 40 ms command delay 被 100 ms control step 量化的问题；
3. 数据中严格区分 commanded control 与 applied control；
4. 确认 MuJoCo 轮速、扭矩 PI、饱和、摩擦和传感器时延趋势合理；
5. 完成正式 residual dataset 和 checkpoint 选择，而不是使用 smoke checkpoint；
6. 将 ICODE inference 优化到闭环实时预算内；
7. 实现并训练一种 RL trajectory proposer；
8. 建立传统 MPPI、RL-guided MPPI 和组合方法的统一计时与统计协议；
9. 明确动态障碍实验是否只有反应式 LaserScan 更新，避免误称为预测式动态避障。

---

## 三、问题定义与研究边界

### 3.1 状态、控制与真实对象

第一阶段沿用现有五状态 dynamic unicycle 表示：

$$
\mathbf{x}
=
\begin{bmatrix}
p_x & p_y & \theta & v & \omega
\end{bmatrix}^{\mathsf T},
$$

控制命令为：

$$
\mathbf{u}
=
\begin{bmatrix}
v_{\mathrm{cmd}} & \omega_{\mathrm{cmd}}
\end{bmatrix}^{\mathsf T}.
$$

名义预测模型写为：

$$
\dot{\mathbf{x}}_{\mathrm{nom}}
=
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u}).
$$

MuJoCo 或真实机器人对应的 plant 写为：

$$
\dot{\mathbf{x}}_{\mathrm{true}}
=
f_{\mathrm{true}}
\left(
\mathbf{x},\mathbf{u},\boldsymbol{\xi}
\right),
$$

其中 $\boldsymbol{\xi}$ 表示摩擦、载荷、执行器状态、轮胎滑移和延迟等未完全建模因素。

### 3.2 两个受控研究瓶颈

#### 瓶颈 A：候选轨迹预测误差

MPPI 需要使用预测模型展开大量候选控制序列。名义模型与 MuJoCo/真实小车之间的误差会在 horizon 内累积，导致候选轨迹排序错误。

#### 瓶颈 B：有限预算下的候选生成效率

传统 MPPI 在 $K$ 个 samples 内围绕 previous sequence 或 goal warm-start 加入扰动。在狭窄通道、U-trap 或复杂布局中，有限 samples 可能无法覆盖有效绕行方向。

### 3.3 “RL 探索”的明确含义

本稿中的 exploration 指：

> 在固定 MPPI samples 和固定控制周期内，RL proposer 是否能够产生传统局部高斯采样较少覆盖、但具有较高可行性或较低任务代价的候选控制轨迹。

它不表示：

- 在实车上无约束试错；
- 为训练 residual 主动采集新数据；
- 绕过 MPPI 直接执行 RL 动作；
- 以碰撞或触发 scan_guard 作为探索方式。

### 3.4 OOD 评估范围

本轮不再采用大规模 scene × dynamics 全组合。只保留三类受控条件：

1. **IID：** 场景和动力学均处于训练/调参范围；
2. **Scene shift：** 未见静态布局，但障碍表示与传感器链保持一致；
3. **Dynamics shift：** 只选择 1–2 个可解释变化，例如摩擦或执行器增益。

控制延迟依赖控制历史。若状态中不包含 applied-control/history features，则 delay 不属于标准 Markov residual，必须单独报告，不与“精确 residual”混称。

### 3.5 论文主张边界

- 当前 residual 结构称为 **ICODE-style control-affine residual**；
- 不声称已经复现原始 ICODE 的全部数学约束；
- 不声称经验性回退具有稳定性或安全性保证；
- 不声称 RL 在简单场景或训练初期优于传统 MPPI；
- 不声称未使用运动预测器的动态障碍实验属于 predictive avoidance；
- 实车结果只支持本文实际测试的速度、地面、载荷和场景范围。

### 3.6 直接相关工作与新颖性 Gate

截至本稿日期，直接相关工作已经分别覆盖 residual-enhanced MPPI、RL-guided MPPI 和 adaptive sampling。因此，“把 ICODE、RL 和 MPPI 放在一起”不能单独构成充分创新。

| 直接相关工作 | 已公开覆盖的核心内容 | 对本计划的约束 |
|---|---|---|
| [ICODE-MPPI](https://arxiv.org/abs/2605.03260) | Control-affine residual learning 与 MPPI path tracking | ICODE residual 接入 MPPI 不能单独作为新贡献 |
| [RGB](https://arxiv.org/abs/2606.25123) | 预训练 RL policy 作为 MPPI sampling prior | RL policy 作为 prior 不能单独作为新贡献 |
| [HOLO-MPPI](https://arxiv.org/abs/2606.16480) | 高层 policy 在线参数化 MPPI sampling distribution | Learned proposal 的一般概念已有直接竞争 |
| [PO-MPC](https://arxiv.org/abs/2510.04280) | 从 proposal optimization 角度学习或适配 MPC sampling distribution | 必须说明双源分配与一般 learned proposal optimization 的差异 |
| [Adaptive Hierarchical RL-MPC](https://arxiv.org/abs/2512.17091) | RL action 引导 MPPI sampler，并自适应利用 MPPI exploration | “adaptive exploration”表述必须进一步说明技术差异 |

因此，2026-07-16 前需要完成一张逐项对照表，至少核查：

- RL 输出是直接 action、mean sequence、control knots，还是完整 distribution；
- traditional prior 是否作为独立 anchor 保留；
- samples allocation 是否固定或自适应；
- 自适应信号来自 policy uncertainty、model uncertainty 还是 planner diagnostics；
- 是否明确说明 frozen learned dynamics 与 proposal mechanism 是职责分离设计，以及二者如何进行条件性、因子化评价；
- 是否在相同 samples 与相同 wall-clock 下评价 candidate quality；
- 是否存在完整 2×2 因子消融和 RL training curve。

只有当该对照能够形成清晰差异、且早期实验支持这一差异时，才保留 adaptive allocation 为拟议贡献。否则应在方法冻结前进一步收敛，而不是依靠平台规模或模块数量包装新颖性。

---

## 四、拟研究方法

### 4.1 冻结的 ICODE-style residual model

残差估计器采用 control-affine 结构：

$$
\widehat f_{\mathrm{res},\boldsymbol{\theta}}
(\mathbf{x},\mathbf{u})
=
f_{\boldsymbol{\theta}}(\phi(\mathbf{x}))
+
G_{\boldsymbol{\theta}}(\phi(\mathbf{x}))\mathbf{u},
$$

其中 $\phi(\mathbf{x})$ 为状态编码，默认使用航向角的 $\sin/\cos$ 表示，并根据消融决定是否保留全局位置。

组合预测模型为：

$$
\dot{\mathbf{x}}_{\mathrm{pred}}
=
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u})
+
\widehat f_{\mathrm{res},\boldsymbol{\theta}^{*}}
(\mathbf{x},\mathbf{u}),
$$

其中 $\boldsymbol{\theta}^{*}$ 是根据 validation multi-step RMSE 选定的 checkpoint。主实验开始后：

- 模型参数冻结；
- normalizer 冻结；
- 网络结构冻结；
- residual dataset 版本冻结；
- 记录 checkpoint hash、config hash 和 Git SHA；
- RL experience 不用于更新 $\boldsymbol{\theta}^{*}$。

一次性少量 fine-tuning 只作为独立消融，使用 checkpoint 副本和预先规定的 adaptation set；它不进入默认 pipeline，时间不足时优先删除。

### 4.2 RL trajectory proposer

RL policy 不直接输出最终 `/cmd_vel`，而输出少量 control knots：

$$
\mathbf{c}_{\mathrm{RL}}
=
\pi_{\boldsymbol{\varphi}}(\mathbf{o}),
$$

再通过插值得到 horizon 为 $H$ 的 prior sequence：

$$
\overline{\mathbf U}_{\mathrm{RL}}
=
\operatorname{Interp}
\left(
\mathbf{c}_{\mathrm{RL}},H
\right).
$$

观测 $\mathbf{o}$ 拟包含：

- 当前 pose/twist；
- goal relative pose；
- compressed LaserScan 或 local obstacle features；
- previous control/previous sequence；
- 可选的 MPPI 上一周期诊断量。

正式输入不得包含 MuJoCo 全局障碍物真值。RL 算法只选择 PPO 或 SAC 中的一种，并在早期 smoke 后立即冻结，不做多算法竞争。

默认 policy 只输出一个低维 proposal center；单周期内的随机覆盖主要来自 MPPI perturbation 以及 traditional/RL 两个不同 anchor。因此本文采用“candidate-trajectory sampling/proposal allocation”表述，不把单一均值 policy 夸大为 multimodal exploration policy。只有确实实现并消融 stochastic/multi-proposal output 后，才使用更强的多模态探索表述。

### 4.3 传统 prior 与 RL prior 的锚定式候选集

传统 prior 记为：

$$
\overline{\mathbf U}_{\mathrm{trad}}
=
\operatorname{Prior}_{\mathrm{previous/goal}}
(\mathbf{o}).
$$

在固定总预算 $K$ 下，候选集分为两部分：

$$
K_{\mathrm{RL}}
=
\left\lfloor
\eta_t K
\right\rfloor,
\qquad
K_{\mathrm{trad}}
=
K-K_{\mathrm{RL}},
$$

由于整数取整，实际 mixture proportion 记录为：

$$
\overline\eta_t
=
\frac{K_{\mathrm{RL}}}{K}.
$$

其中 $\eta_t$ 表示 RL-guided candidates 占总预算的比例。对拟议的 anchored method，约束为：

$$
0
<
\eta_{\min}
\leq
\eta_0
\leq
\eta_{\max}
<
1,
\qquad
\eta_t
\in
\left[
\eta_{\min},
\eta_{\max}
\right],
$$

其中首周期使用固定 $\eta_0$。从而在线主方法始终保留至少一部分 traditional candidates。$\eta=0$ 和 $\eta=1$ 只作为独立的单源诊断消融，不进入包含 $\operatorname{logit}(\eta_t)$ 的 adaptive update。

两组候选分别为：

$$
\mathbf U_k^{\mathrm{trad}}
=
\overline{\mathbf U}_{\mathrm{trad}}
+
\boldsymbol\epsilon_k^{\mathrm{trad}},
$$

$$
\mathbf U_j^{\mathrm{RL}}
=
\overline{\mathbf U}_{\mathrm{RL}}
+
\boldsymbol\epsilon_j^{\mathrm{RL}}.
$$

默认先使用已知、固定的 Gaussian perturbation，并记录完整 proposal density：

$$
\boldsymbol\epsilon^{s}
\sim
\mathcal N
\left(
\mathbf 0,\boldsymbol\Sigma_s
\right),
\qquad
s\in\{\mathrm{trad},\mathrm{RL}\},
$$

$$
p_t(\mathbf U)
=
(1-\overline\eta_t)
\mathcal N
\left(
\mathbf U;
\overline{\mathbf U}_{\mathrm{trad}},
\boldsymbol\Sigma_{\mathrm{trad}}
\right)
+
\overline\eta_t
\mathcal N
\left(
\mathbf U;
\overline{\mathbf U}_{\mathrm{RL}},
\boldsymbol\Sigma_{\mathrm{RL}}
\right).
$$

核心 2×2 初期令 $\boldsymbol\Sigma_{\mathrm{trad}}=\boldsymbol\Sigma_{\mathrm{RL}}$，避免把 covariance 差异混入 RL prior 因子。若后续让 RL 输出 bounded scale，必须另列消融并重新审计 proposal correction。

最终候选集合为：

$$
\mathcal U_t
=
\mathcal U_t^{\mathrm{trad}}
\uplus
\mathcal U_t^{\mathrm{RL}}.
$$

这里 $\uplus$ 表示把两个带 source label 和 sample index 的 candidate batches 连接起来；即使两个数值序列恰好相同，也不会像普通数学集合那样被去重，总基数始终为 $K$。

这种构造保留 traditional MPPI anchor，因此能够直接比较：

- $\eta=0$：完全传统 MPPI；
- $\eta=1$：完全围绕 RL prior，仅作诊断消融；
- 固定 $\eta$：简单双源混合；
- 自适应 $\eta_t$：拟议探索分配机制。

### 4.4 简化的相对候选质量感知探索分配

如果只根据“整个混合候选集表现较差”就增加 RL 比例，可能形成错误正反馈：当退化恰好由 RL candidates 引起时，下一周期反而会分配更多 RL samples。因此，拟分别统计两个 proposal source 的在线、planner-visible 指标。

为避免“某一候选池因为已经分到更多 samples，所以更容易出现低代价样本，继而继续获得更多预算”的样本数反馈偏差，每个 source 固定保留一个等大的 probe subset：

$$
K_{\mathrm{probe}}
\leq
\min
\left(
K_{\mathrm{trad}},
K_{\mathrm{RL}}
\right).
$$

$\eta_{\min}$ 与 $\eta_{\max}$ 必须保证所有正式 $K$ 下均满足该约束。在线 source score 只由两个等样本数的 probe subsets 计算，剩余 candidates 仍全部参与 MPPI evaluation。

对 $s\in\{\mathrm{trad},\mathrm{RL}\}$，记录：

- probe subset 上的 predicted feasible ratio $q_t^{s}$；
- probe subset 中固定数量 $m_{\mathrm{elite}}$ 的最佳 candidates 集合 $\mathcal E_t^{s}$；
- 在同一 prediction model 和同一 cost function 下的 normalized elite cost $\widetilde J_t^{s}$。

其中：

$$
0
\leq
q_t^{s}
\leq
1,
\qquad
1
\leq
m_{\mathrm{elite}}
\leq
K_{\mathrm{probe}},
\qquad
\varepsilon>0.
$$

归一化 elite cost 可写为：

$$
\widetilde J_t^{s}
=
\frac{
\operatorname{mean}_{k\in\mathcal E_t^{s}}S_k
-S_t^{\min}
}{
S_t^{\max}-S_t^{\min}+\varepsilon
}.
$$

每个 source 的相对质量分数为：

$$
Q_t^{s}
=
q_t^{s}
-
\lambda_J\widetilde J_t^{s},
$$

并计算 RL 相对 traditional 的优势：

$$
\Delta_t
=
Q_t^{\mathrm{RL}}
-
Q_t^{\mathrm{trad}}.
$$

样本分配使用有界、平滑更新：

其中 $\sigma(z)=1/(1+\exp(-z))$ 为 logistic sigmoid。

$$
\widehat\eta_{t+1}
=
\operatorname{clip}
\left[
\sigma
\left(
\operatorname{logit}(\eta_t)
+
\gamma\Delta_t
\right),
\eta_{\min},
\eta_{\max}
\right],
$$

$$
\eta_{t+1}
=
(1-\kappa)\eta_t
+
\kappa\widehat\eta_{t+1}.
$$

预期但尚未验证的行为为：

~~~text
Traditional candidates 的相对质量更高
→ Delta < 0
→ 降低 RL candidates 比例
→ 简单场景主要保留传统 MPPI 优势

RL candidates 的相对质量更高
→ Delta > 0
→ 增加 RL candidates 比例
→ 在困难场景利用已学习的 trajectory proposal
~~~

在 OOD 条件下不预设 $\eta$ 必然上升：如果 RL pool 确实提供更有价值的候选，分配器才增加其比例；如果 RL 同样因分布变化而退化，则 traditional pool 应保留或获得更多预算。这使“强调 RL 探索”成为可检验的行为，而不是无条件信任 RL。

初始 $\eta_0$、$K_{\mathrm{probe}}$、$m_{\mathrm{elite}}$、$\lambda_J$、$\gamma$、$\kappa$ 和上下界只使用 development/validation conditions 冻结，并约束 $\lambda_J\geq0$、$\gamma>0$、$0<\kappa\leq1$。这里的 feasible 只来自 planner 当前能够访问的预测轨迹和约束；MuJoCo ground truth 不得反馈给在线分配器。

ESS 仍可作为 MPPI 数值诊断，但在双源 proposal 的 importance correction 完成前，不用于在线分配或跨方法 Gate。该机制也不是安全保证或形式化 OOD detector；feature-distance OOD score 只作为可删除的对照实验。

### 4.5 MPPI 统一评价与更新

所有传统候选和 RL 候选都通过同一预测模型、同一 perception input 和同一 cost function 进行评价。

第 $k$ 条候选轨迹代价记为 $S_k$，基础数值稳定权重写为：

$$
w_k
=
\frac{
\exp
\left[
-\frac{1}{\lambda}(S_k-\rho)
\right]
}{
\sum_j
\exp
\left[
-\frac{1}{\lambda}(S_j-\rho)
\right]
},
$$

其中：

$$
\rho=\min_k S_k.
$$

正式实现仍需完成 proposal distribution、control perturbation 和 importance-sampling correction 的公式—代码审计。如果双源 proposal 未能满足严格 MPPI 权重条件，论文将使用更谨慎的 guided sampling-based MPC 表述，不以名称掩盖数学差异。

### 4.6 安全与执行边界

最终链路保持：

~~~text
RL / traditional proposals
    ↓
MPPI candidate evaluation
    ↓
MPPI proposed control
    ↓
scan_guard + safety arbitration
    ↓
executed control
    ↓
MuJoCo / robot
~~~

必须保持：

- RL 不直接发布 `/cmd_vel`；
- scan_guard 始终为最高优先级；
- local_obstacle_layer 继续消费 LaserScan；
- planner 不读取全局障碍真值；
- Memory 在核心消融中关闭；
- viewer、日志和绘图时间不计入 planner time，但 residual/RL inference 必须计入；
- deadline miss 时执行预先定义的受限 fallback，并继续经过 safety arbitration，而不是等待学习模块完成。

---

## 五、训练、数据与冻结协议

### 5.1 Residual 数据来源

正式 residual dataset 独立于 RL policy experience，主要包括：

- 独立随机控制激励；
- structured excitation；
- task-specific traditional MPPI trajectories；
- 必要的 hard-case trajectories。

Structured excitation 包括：

- 线速度和角速度阶跃；
- sine/chirp control；
- 固定圆弧；
- 加速—匀速—减速；
- 原地旋转；
- 多初始速度。

数据按 episode、seed 和物理配置拆分为 Train、Validation、IID Test 和 OOD-Dynamics Test，禁止按单个 timestep 随机拆分。

### 5.2 Residual checkpoint 选择

候选模型包括：

- nominal；
- MLP residual；
- ICODE-style residual；
- low-order Markov mismatch 下的 Oracle residual，仅作诊断上界。

最佳 checkpoint 主要依据 validation multi-step rollout RMSE，而不是 training derivative loss。重点 horizon 为：

$$
H\in\{1,5,10,20\}.
$$

在进入 RL 正式训练前，选定一个 ICODE checkpoint 并冻结。MLP 与 Oracle 主要停留在 prediction benchmark，不进入所有控制场景和所有 sample budgets。

### 5.3 RL proposer 训练

RL 训练目标是提高候选轨迹质量，而不是替代 MPPI。Reward 初步包含：

- goal progress；
- terminal success；
- collision penalty；
- clearance penalty；
- stuck/spin penalty；
- control magnitude；
- control jerk；
- safety override penalty。

默认训练闭环拟定义为：

~~~text
observation
→ RL 输出 control knots / prior parameters
→ 构造 RL-guided candidates
→ MPPI 与 traditional candidates 一起评价并更新
→ 执行 MPPI proposed control（仍经过 safety arbitration）
→ MuJoCo 返回 next observation 与 reward
~~~

因此 RL 学到的是“如何为内层 MPPI 提供 proposal”，而不是一套在训练结束后才临时改作 prior 的直接控制策略。若该内层训练计算量不可接受，是否改用预训练 direct policy 作为 prior 必须作为一次明确的方法选择，并重新检查与 RGB 等工作的差异；两条路线不在正式实验中同时展开。

默认内层 MPPI 使用已经冻结的 $\boldsymbol\theta^{*}$；RL gradient 不反向传播进入 ICODE，optimizer 也不持有 ICODE 参数。为完成核心 2×2，M2 与 M3 复用同一个 RL proposer checkpoint、同一个 fixed $\eta$ 和同一 proposal hyperparameters，只在在线 evaluation 时切换 nominal/frozen-ICODE rollout model。该设计用于分离运行时 prediction 因子；论文会同时说明 policy 的训练上下文，避免把它表述成完全对称的双因素训练实验。

需要专门检查：

- 原地不动以避免惩罚；
- 高速冲向目标并依赖 scan_guard；
- 通过输出极端 control knots 扩大 MPPI 搜索；
- 只在少数训练布局记忆路径；
- reward 增长但实际 success 不增长。

RL training curve 的横轴统一使用 environment transitions，并保存至少：

- 初始/未训练 checkpoint；
- early checkpoint；
- middle checkpoint；
- validation 选定的 final checkpoint；
- 3 个独立 training seeds。

### 5.4 三类数据严格隔离

| 数据 | 用途 | 是否更新 ICODE | 是否用于最终调参 |
|---|---|---:|---:|
| Residual Train/Validation | 训练与选择 ICODE | 是 | 仅 Validation |
| RL training experience | 训练 trajectory proposer | 否 | 仅 RL Validation |
| IID/OOD final test | 最终评价 | 否 | 否 |

RL 轨迹可以记录并分析其状态—控制覆盖，但默认不进入 residual Train。若设置一次 fine-tuning 消融，adaptation split 必须通过独立的预设控制激励采集，不使用 RL training experience，并重新保留未触碰的 final test。若另行研究“使用 RL 日志微调”，只能作为非默认敏感性分析单独命名和报告。

### 5.5 复现与 provenance

每个 dataset、checkpoint 和 experiment run 至少记录：

- 配置快照；
- seed 类型与数值；
- Git SHA；
- dataset/checkpoint hash；
- state/control dimensions；
- normalization statistics；
- device、线程和精度设置；
- MuJoCo、PyTorch 和 Python 版本；
- samples、horizon、control period；
- scene 和 dynamics parameters；
- wall-clock timestamps。

### 5.6 训练规模与计算上限

为避免在有限时间内无上限地扩大训练，暂定以下硬预算，并在 07-15 根据实际 GPU/CPU 资源只允许向下调整：

| 项目 | 第一阶段 | 硬上限 | 停止规则 |
|---|---:|---:|---|
| Residual dataset | 2 万 transitions | 10 万 transitions | H=10/20 validation curve 连续两次无实质改善 |
| ICODE initializations | 3 | 3 | 不追加第四个 seed |
| RL smoke | 10 万 transitions | 10 万 transitions | 只检查接口、reward 与吞吐 |
| RL formal per seed | 50 万 transitions | 200 万 transitions 或 24 GPU-hours | 先达到者停止 |
| RL training seeds | 3 | 3 | 不增加算法或 seed 数 |
| RL evaluation cadence | 每 10 万 transitions | 固定 | final test 不参与 checkpoint 选择 |

这些数值是工程预算，不是预设最优训练量。若 3 seeds 在硬上限前均无改善趋势，应触发 G2，而不是继续扩大搜索。

---

## 六、实验问题与可判定条件

本节不把预期写成已经成立的结论，而是提前规定什么结果支持或否定相应主张。

### 6.1 E1：Residual 是否具有闭环价值？

先在可解析、Markov 型 mismatch 下使用 Oracle residual 检查接口。如果正确 residual 仍不能改善候选预测或任何闭环指标，则优先排查：

- residual 是否正确进入 rollout；
- cost 是否主导失败；
- 感知或 safety arbitration 是否是主要瓶颈；
- 当前任务是否根本不是 dynamics-error-limited。

Oracle 诊断失败时，不继续扩大 ICODE 训练规模。

### 6.2 E2：Frozen ICODE 是否改善多步预测？

重点比较：

$$
\operatorname{RMSE}_{H}
(\mathrm{ICODE\text{-}style})
\quad\text{与}\quad
\operatorname{RMSE}_{H}
(\mathrm{Nominal}),
$$

其中 $H=10,20$ 为主要 horizon。

如果 ICODE 只改善 one-step error，却不能改善 multi-step error，或者推理时间导致闭环 deadline miss，则不能声称它为 MPPI 带来有效预测改进。

内部 Gate 暂定为：主要 unseen-dynamics validation 条件下 $H=10/20$ 误差相对 nominal 至少降低 15%，同时核心闭环场景不出现明显安全指标退化。该阈值只用于工程决策，不替代正式统计报告。

### 6.3 E3：RL proposer 是否改善候选轨迹探索？

固定 prediction model、$K$、$H$ 和随机种子后，比较 traditional prior 与 RL prior：

- feasible candidate ratio；
- best sampled trajectory cost；
- elite trajectory diversity；
- MPPI 对 prior 的修正幅度；
- closed-loop success 与 time-to-goal；
- planner latency。

如果 RL 只提高训练 reward，却不能改善候选轨迹或闭环指标，则不能声称它提高了 MPPI exploration。

### 6.4 E4：RL 需要多少训练才有价值？

传统 MPPI 作为无需训练的水平基线，与多个 RL checkpoints 比较：

~~~text
Traditional MPPI performance
        ───────────────────────────

RL-guided performance
        early → middle → final
~~~

需要报告：

- RL 何时开始接近传统 MPPI；
- 是否存在稳定超过传统 MPPI 的阶段；
- 收益是否只出现在复杂场景或低 sample budget；
- 达到该收益需要多少 transitions 和训练时间；
- 3 个 training seeds 是否趋势一致。

若只有一个 seed 或一个最终 checkpoint 表现较好，不作为稳定结论。

### 6.5 E5：Residual 与 RL 是否具有增量互补性？

核心 2×2 为：

| Prediction model | Traditional prior | Fixed dual-source RL-guided prior |
|---|---:|---:|
| Nominal | A | C |
| Frozen ICODE | B | D |

判定逻辑为：

- $B>A$：residual 在 traditional prior 下有独立价值；
- $C>A$：RL 在 nominal prediction 下有独立价值；
- $D>B$：固定 residual 后，RL 仍有增量价值；
- $D>C$：固定 RL 后，residual 仍有增量价值；
- $D$ 未超过 $B$ 或 $C$：不得声称组合具有增量互补性。

若论文需要使用“协同”一词，还必须对预注册、方向统一的性能量定义 interaction contrast：

$$
\Delta_{\mathrm{interaction}}
=
(D-B)
-
(C-A).
$$

只有该 interaction 在 paired/clustered analysis 中方向稳定且置信区间支持时，才讨论 synergy；否则只使用“两个模块的增量贡献”或“factorized evaluation”表述。

### 6.6 E6：自适应探索分配是否优于固定混合？

在相同 $K$ 下比较：

- $\eta=0$；
- $\eta=1$；
- 固定 $\eta$；
- source-relative adaptive $\eta_t$。

需要验证：

1. 简单场景中不应明显弱于 traditional MPPI；
2. 困难静态场景中应优于固定或单源 proposal；
3. 增益不能仅来自更多网络推理时间；
4. $\eta_t$ 变化应与 $q_t^{s}$、$\widetilde J_t^{s}$ 和 $\Delta_t$ 的相对关系一致，而不是由某一 source 的样本数优势造成。

如果自适应版本只在个别 seed 有效，或者频繁振荡并增加 deadline miss，则删除该主张，保留固定 mixture 作为基线分析。

### 6.7 E7：动态障碍是否值得纳入？

动态障碍只在静态主实验和实时 Gate 均通过后执行。若系统没有显式障碍速度估计与运动预测，则该实验只评价基于逐周期 LaserScan 更新的反应式行为。

如果动态障碍结果受到感知、追踪或预测缺失主导，应将其作为局限性或未来工作，不强行用于证明 residual 或 RL proposal 的贡献。

---

## 七、正式实验设计

### 7.1 核心方法组

| 编号 | Prediction | Proposal | 用途 |
|---|---|---|---|
| M0 | Nominal | Traditional | 强基础 MPPI |
| M1 | Frozen ICODE | Traditional | Residual-only |
| M2 | Nominal | Fixed dual-source | 检验 RL proposal 在 nominal prediction 下的作用 |
| M3 | Frozen ICODE | Fixed dual-source | 简单组合 |
| M4 | Frozen ICODE | Adaptive dual-source | 拟议完整方法 |

M2 与 M3 使用完全相同的 fixed $\eta$、noise covariance、$K$、$H$ 和 proposal construction，二者只替换 prediction model，使其运行时差异主要对应 prediction model，并按第 5.3 节所述训练上下文限制解释。M4 不属于核心 2×2，而是在 M3 基础上检验 adaptive allocation。

附加诊断：

- Oracle residual：只用于低阶 Markov mismatch 上界；
- MLP residual：主要用于 prediction ablation，并在 2 个代表场景、$K=100$、5 seeds 下做一次有限闭环结构消融；
- Frozen ICODE + $\eta=1$：只用于 E6 的 RL-only proposal 诊断，避免用 nominal prediction 的 M2 代替该消融；
- fine-tuned ICODE：只用于一次可删除的更新消融；
- Memory-Augmented MPPI：不进入核心表。

### 7.2 场景分层

#### 层 S0：Clean dynamics benchmark

- 无障碍或极少障碍；
- 直线、圆弧、转向和目标跟踪；
- 用于隔离动力学预测、delay 和 actuator mismatch。

#### 层 S1：简单静态障碍

- `simple`；
- 稀疏障碍；
- 用于验证传统 MPPI 是否已经足够强，以及自适应机制是否避免不必要依赖 RL。

#### 层 S2：困难静态障碍

- `narrow_corridor`；
- `u_trap_long_board`；
- 资源允许时加入 `lab_complex`；
- 用于检验 RL candidate exploration 的主要价值。

#### 层 S3：动态障碍扩展

- 单一 crossing obstacle；
- 固定速度与起始时刻；
- 只比较 M0、M3、M4；
- 不扩展成多种动态行为的大矩阵。

#### Episode 终止与 timeout 协议

所有正式场景固定 $\mathrm{control\_dt}=0.10\,\mathrm{s}$，并在 final test 前冻结：

| 场景层 | Max control steps | Max simulated time |
|---|---:|---:|
| S0 Clean dynamics | 120 | 12 s |
| S1 Simple static | 180 | 18 s |
| S2 Difficult static | 240 | 24 s |
| S3 Dynamic extension | 200 | 20 s |

到目标位置误差不大于 0.20 m 记为 success；接触碰撞按配置立即终止并记为 failure；达到 max steps 仍未成功记为 timeout failure。Stuck/spin 作为过程指标记录，不通过不同方法专属的提前终止规则改变 episode length。

### 7.3 动力学条件

主实验只保留：

1. nominal/IID；
2. unseen friction；
3. unseen actuator gain 或 left/right asymmetry。

以下条件只单独诊断：

- command delay；
- 复合多扰动；
- 极端低摩擦；
- 传感器大规模 dropout。

### 7.4 Sample budgets

主点为：

$$
K\in\{50,100,400\}.
$$

$K=25$ 和 $K=200$ 只在主要趋势已经形成且算力允许时补充。论文不以“samples 更少”等同于“计算更快”，必须同时报告 wall-clock。

### 7.5 RL 学习曲线协议

每个 RL training seed 保存相同 transitions 位置的 checkpoints。评价时：

- 使用独立 validation scene seeds；
- 不根据 final test 选择 checkpoint；
- 固定 residual checkpoint；
- 固定 MPPI $K/H$；
- 同时画 return、success、collision、candidate quality 和 latency；
- 报告训练 wall-clock、硬件与 environment transitions。

### 7.6 候选轨迹质量指标

RL exploration 不能只通过最终 success 间接判断，至少报告：

- feasible candidate ratio；
- collision-free candidate ratio；
- best-of-$K$ cost；
- elite-set cost mean/std；
- elite trajectory diversity；
- effective sample size；
- MPPI update magnitude；
- RL proposal 被 MPPI 修正的幅度；
- 左/右绕行或其他可识别 trajectory mode 的覆盖率（仅在定义可靠时使用）。

指标定义提前冻结：

- **collision-free ratio**：预测轨迹未与当前 local obstacle representation 相交；
- **feasible ratio**：在 collision-free 基础上，同时满足 boundary、control limit、数值有效性及预定义动态约束；
- **elite set**：每个 source 内按预测代价排序的前 10%，且每组不少于预先规定的最小数量；
- **elite diversity**：对 control sequence 按 $v/\omega$ 训练范围归一化后，计算 elite pairs 的平均序列距离。

不同 prediction models 产生的 predicted cost 不能直接当作公平的 proposal quality 证据，因为乐观模型可能人为给出较低代价。正式实验将在一小部分冻结的**完整仿真快照**上，用同一个离线 MuJoCo/common evaluator 重放各方法的候选控制序列，统一计算 ground-truth feasibility 与 realized cost。

完整快照至少包含：

- MuJoCo `qpos`、`qvel`、`act`、`ctrl`、simulation time 与必要的 solver warm-start state；
- torque PI integral、上一时刻 torque/command 和 actuator internal state；
- command-delay queue、applied-control history 与 sensor-latency buffers；
- time-varying disturbance phase 和动态障碍状态；
- NumPy/Python/MuJoCo 相关 RNG state。

Common-evaluator snapshot library 在产生任何待比较方法的 candidates 之前预注册：

~~~text
2 个 validation 场景
× 5 个固定 snapshot seeds
× 每个 scene-seed 10 个完整快照
= 100 个 validation snapshots
~~~

每个 scene-seed 的 10 个快照由方法无关的固定 collector protocol 生成：5 个来自冻结的 traditional-MPPI reference rollout，5 个来自有界 structured excitation；分别从各自 pre-terminal trajectory 按等归一化时间位置选择。快照索引和 hash 在 G2/G3 前冻结，所有方法使用完全相同的 library、candidate budget 和 replay plant。另用独立 test seeds 生成同规模 final-test library，只用于最终报告，不参与 Gate 或调参。

只保存五维 $[p_x,p_y,\theta,v,\omega]$ 不足以重放包含 actuator、delay 和隐藏状态的 plant。该 privileged evaluator 只用于离线评价，绝不反馈给 planner、RL policy 或在线 $\eta_t$。

### 7.7 闭环控制指标

指标决策层级预先规定为：

- **主要 efficacy 指标：** success rate；
- **安全非劣指标：** collision/contact rate；
- **部署约束：** p95 planner latency 与 deadline miss rate；
- **次要效率指标：** time-to-goal、path length 与 control smoothness。

次要指标：

- final goal distance；
- trajectory length；
- minimum clearance；
- mean absolute $\omega$；
- control jerk；
- stuck/spin steps；
- scan hard-stop 次数；
- safety intervention rate；
- commanded/applied control difference。

Time-to-goal 同时报告两种口径：成功 episodes 的条件分布，以及将失败 episode 统一记为 episode horizon 的 penalized time；不得只删除失败样本后比较平均时间。

对于动态障碍扩展，额外报告：

- minimum time-to-collision（只作为 evaluator metric）；
- encounter success rate；
- minimum moving-obstacle clearance；
- stop/wait duration；
- replanning latency；
- scan hard-stop 与 safety override 次数。

动态障碍 ground truth 只能用于离线评价，不能作为正式 planner observation。

同一初始条件下还将保存 candidate trajectories、MPPI selected trajectory 与 executed trajectory 的配对可视化，用于解释不同 residual model 和 RL proposal 为什么产生不同结果；轨迹“好坏”仍由预注册指标判断，不以个别漂亮案例替代统计。

### 7.8 Prediction 与物理指标

Prediction：

- derivative RMSE；
- one-step state RMSE；
- $H=5,10,20$ rollout RMSE；
- position RMSE、wrapped heading RMSE、velocity RMSE 与 yaw-rate RMSE 分开报告；
- unseen-dynamics RMSE；
- residual inference mean/p95/max。

MuJoCo physical diagnostics：

- wheel tracking error；
- slip ratio；
- actuator effort；
- rise/settling time；
- commanded/applied delay；
- Odom–GroundTruth drift。

### 7.9 公平计算预算

正式结果同时采用两种协议。

#### 协议 A：等搜索预算

- 相同 $K$；
- 相同 $H$；
- 相同 control period；
- 相同 cost evaluation；
- 相同 state dimension、integrator、numerical precision；
- 记录 Euler/RK4 derivative-call 数；
- 记录 residual、RL 与 rollout 的分项耗时。

#### 协议 B：等 wall-clock

- 在独立 calibration seeds 上测量每种方法满足控制周期的最大 $K$；
- 冻结各方法的 $K_{\mathrm{method}}$；
- final test 不再调整；
- viewer、绘图和磁盘写入不计入 planner time；
- model inference 和 proposal generation 必须计入。

计时同时报告：

1. planner kernel latency：observation feature 已准备后，到 proposed control 输出；
2. end-to-end latency：LaserScan preprocessing/local obstacle update 开始，到 proposed control 输出。

每种方法使用 5 个专用 calibration seeds；每种配置先运行 100 个 warm-up cycles，再累计至少 1000 个 measured cycles。$K$ 搜索网格固定为：

$$
K_{\mathrm{calib}}
\in
\{25,50,75,100,150,200,300,400\}.
$$

在相同硬件、线程、device、precision、timer boundary 和 scan preprocessing 设置下，选择满足 G5 的最大 $K_{\mathrm{method}}$，并在 final test 前冻结。ESS 只有在 proposal correction 审计通过后才作为跨方法诊断，否则仅在各自方法内部展示。

### 7.10 Seeds 与统计

需要区分：

1. environment/scenario seed；
2. residual initialization seed；
3. RL training seed。

正式核心比较计划采用：

- 10 个 paired environment seeds；
- 3 个 RL training seeds；
- 至少 3 个 residual initializations 用于确认 checkpoint 稳定性，但宽场景控制实验可在说明条件后使用冻结的最终 checkpoint。

报告：

- mean、standard deviation 和 95% confidence interval；
- paired effect size；
- success/collision 的比例置信区间；
- 每个 RL seed 的结果，而不是只展示最好 seed；
- 失败 episode 与异常原因；
- 原始 trajectory 和自动生成图表的脚本。

对于包含 RL 的方法，environment episodes 先在各自 RL training seed 内汇总，再报告跨 training seeds 的变化；不得把同一 policy 下的大量 environment episodes 当作彼此独立的 RL 训练重复，从而人为缩小置信区间。

### 7.11 分层运行量

完整笛卡尔积不执行。建议规模为：

#### A. 核心主表

核心主表使用一个简单场景和一个困难静态场景。没有 RL 的方法不重复计算 RL training seeds；包含 RL 的方法对 3 个 training seeds 分别评价：

~~~text
M0/M1:
2 methods × 2 scenarios × 10 environment seeds
= 40 episodes

M2/M3/M4:
3 methods × 3 RL training seeds × 2 scenarios × 10 environment seeds
= 180 episodes

核心主表合计约 220 episodes
~~~

#### B. Sample-efficiency

为避免再次形成全因子组合，sample curve 只保留 M0 与 M4：

~~~text
M0:
3 sample budgets × 2 scenarios × 10 environment seeds
= 60 episodes

M4:
3 RL training seeds × 3 sample budgets × 2 scenarios × 5 paired environment seeds
= 90 episodes

合计约 150 episodes；与核心主表重合的 K=100 不重复运行
~~~

#### C. RL learning curve

学习曲线统一使用 M3 的 fixed $\eta$，避免 policy checkpoint 变化与 adaptive allocation 同时变化：

~~~text
4 checkpoints
× 2 scenarios
× 3 RL training seeds
× 5 evaluation seeds
= 120 episodes
~~~

#### D. Dynamics shift

只在一个代表性静态场景运行 M0、M1 与 M4：

~~~text
M0/M1:
2 methods × 2 shifts × 1 scenario × 10 environment seeds
= 40 episodes

M4:
3 RL training seeds × 2 shifts × 1 scenario × 5 paired environment seeds
= 30 episodes

合计约 70 episodes
~~~

A、B、C 均使用 IID dynamics；D 单独覆盖 unseen friction 与 unseen actuator gain/asymmetry，因此 IID 条件不在 A 中再次乘以 dynamics factor。

#### E. Equal-wall-clock confirmation

在 calibration seeds 上为各方法锁定满足实时约束的 $K_{\mathrm{method}}$ 后，只对强 baseline M0 与完整方法 M4 运行两个代表场景：

~~~text
M0:
2 scenarios × 10 environment seeds
= 20 episodes

M4:
3 RL training seeds × 2 scenarios × 5 paired environment seeds
= 30 episodes

合计约 50 episodes
~~~

只有 scene、environment seed、RL checkpoint、config hash、dynamics parameters 和所有 planner settings 完全一致的 $K=100$ run 才允许与前述矩阵去重。

#### F. 必要诊断组

~~~text
MLP residual limited closed-loop:
2 scenarios × 5 environment seeds
= 10 episodes

Frozen ICODE + eta=1:
3 RL training seeds × 2 scenarios × 5 environment seeds
= 30 episodes

必要诊断合计约 40 episodes
~~~

#### G. 可选扩展

- 第三个静态场景：只比较 M0/M4，M0 使用 10 environment seeds，M4 使用 3 RL seeds × 5 environment seeds，共约 25 episodes；
- 动态障碍：M0/M3/M4 × 10 environment seeds，使用预注册规则选定的单一 frozen RL checkpoint，共约 30 episodes；只作扩展/可行性证据，不声称跨 RL training seeds 的稳定性；
- 一次性 offline fine-tuning：M1/fine-tuned ICODE × clean/simple × 5 seeds，共约 20 episodes；
- 实车：有限重复，不承担大样本统计结论。

A–E 的 run manifest 明确复用两类完全相同的 runs：B 中的 $K=100$ 子集直接引用 A 的对应 run IDs；C 中 final RL checkpoint 的 fixed-$\eta$ 子集直接引用 A 中 M3 的对应 run IDs。E 的 equal-wall-clock confirmation 始终作为独立 runs 计算，即使某个 $K_{\mathrm{method}}$ 恰好等于 100 也不再扣除。损坏或无效的 run 使用相同 run ID 重跑并替换，不同时保留为新增样本。

按上述固定复用规则，A–E 为 530 个不重复 episodes；加上 F 的必要诊断后，正式 MuJoCo 计划固定为 570 episodes。G 的仿真扩展上限为 75 episodes，因此仿真总上限为 645 episodes。先使用 development seeds 淘汰明显无效实现，再运行冻结后的 final test seeds。

---

## 八、实施工作包与 Gate

### WP1：MPPI 与 proposal correctness

**任务：** 审计 legacy/strict MPPI、双源 proposal、importance correction、ESS 和 batch rollout。  
**验收：** 固定 seed 可复现；无 NaN/Inf；手工小例可复核；proposal 改变后的数学表述准确。  
**最迟日期：** 2026-07-19。

### WP2：MuJoCo 物理与观测可信度

**任务：** 修正 sub-step delay，校验 commanded/applied control、轮速、扭矩、摩擦、Odom 与 LaserScan。  
**验收：** 参数变化趋势可解释；20/40/65/100 ms delay 可区分；planner 不读取障碍真值。  
**最迟日期：** 2026-07-20。

### WP2B：强 Traditional MPPI baseline

**任务：** 只在 development seeds 上调节并冻结 $H$、temperature、noise covariance、control limits、cost weights 与 previous/goal warm-start。  
**验收：** simple/static validation 表现稳定；所有参数搜索范围和选择规则有记录；final test 不再调整。  
**最迟日期：** 2026-07-23。

### WP3：Frozen ICODE checkpoint

**任务：** G0 通过后采集正式 residual 数据，训练 nominal/MLP/ICODE，完成 H-step prediction benchmark 与 inference 优化。  
**验收：** Oracle 接口成立；ICODE 在主要 validation 条件改善 H=10/20；checkpoint、normalizer 和数据版本冻结。  
**目标日期：** 2026-07-24；**最迟日期：** 2026-07-27。

正式 RL 训练只在该 hash 实际冻结后的下一天启动。若 WP3 延迟到 07-27，后续通过删除可选场景/扩展吸收延迟，08-10 方法硬冻结日期不后移。

### WP4：RL trajectory proposer

**任务：** 冻结 observation/action/reward 和一种 RL 算法；ICODE hash 冻结前只做接口/provisional smoke，之后完成 3 seeds 正式训练与定期 checkpoint evaluation。  
**验收：** 至少 2/3 seeds 出现一致改善趋势；预设诊断未观察到明显依赖安全层的 reward hacking，并如实报告相关指标；完整训练曲线可复现。  
**目标日期：** 2026-08-02；**最迟日期：** 2026-08-05。

### WP5：双源候选集与自适应探索分配

**任务：** 实现 $\eta\in\{0,1\}$、fixed $\eta$ 与 adaptive $\eta_t$，完成核心 2×2 和候选质量指标。  
**验收：** 简单场景无明显退化；至少一个困难场景显示稳定增量；p95 latency 满足控制周期。  
**目标日期：** 2026-08-08；**方法硬冻结：** 2026-08-10。若未通过则删除可选机制或缩减矩阵，不使用 08-11 之后的 test runs 继续调参。

### WP6：正式 benchmark

**任务：** 运行静态主表、sample-efficiency、RL curve 和有限 dynamics shifts。  
**验收：** paired seeds 完整；原始轨迹、summary、配置和 hash 齐全；测试集未参与调参。  
**目标日期：** 2026-08-21；**主结果最迟：** 2026-08-24。

### WP7：实车证据与论文材料

**任务：** offline/shadow、条件性低速验证、论文、视频、匿名代码和复现说明。  
**验收：** 不破坏 ROS bridge；scan_guard 保持启用并通过回归与触发测试；论文图表由冻结结果自动生成。  
**实验冻结：** 2026-09-06；**预提交包：** 2026-09-12。

### Gate 与 Go/No-Go 原则

| Gate | 通过条件 | 未通过时的同年处理 |
|---|---|---|
| G0 Foundation（07-20） | WP1/WP2、固定 seed、计时边界和 Oracle interface 通过 | 暂停正式数据与训练，先修基础 |
| G1 Residual（07-27） | H=10/20 RMSE 至少改善 15%；dev collision 不增加超过 2 percentage points；可实时推理 | 不做联合主张，及时向老师汇报，并恳请指导是否收敛为 RL-guided nominal MPPI |
| G2 RL（08-05） | 至少 2/3 seeds 在困难 validation 场景使 common-evaluator feasible ratio 提高 10 percentage points 或 realized elite cost 降低 10%；success 非劣界 -5 points，collision 非劣界 +2 points，deadline 合格 | 停止扩大超参搜索，保留学习曲线并重新评估主线 |
| G3 Joint（08-08） | M3 相对 M1、M2 中的较优者，在至少 2/3 RL seeds 上达到 success +5 points 或 common-evaluator realized cost -10%，同时 collision/deadline 不退化 | 不声称 synergy，只报告成立的单模块关系 |
| G4 Adaptive allocation（08-10） | 相对 M3 改善困难场景，同时 simple success 非劣界为 -5 percentage points，且不增加 collision/deadline | 删除 adaptive 主张，fixed mixture 只作分析基线 |
| G5 Real-time（正式 benchmark 前） | p95 planner latency $\leq0.8\,dt$，deadline miss rate $<1\%$ | 降低 samples；仍不合格则不进入实车闭环 |

上述处理均以 2026-09-15 投稿节点不变为前提，不通过增加新模块弥补失败结果。

---

## 九、2026 年 9 月 15 日倒排时间表

### 9.1 三条并行工作流

为压缩时间，以下工作并行推进：

1. **方法与代码流：** MPPI/MuJoCo 审计、Frozen ICODE、RL proposer 和双源 proposal；
2. **数据与实验流：** 数据集、benchmark、统计、图表和实车准备；
3. **论文流：** related work、method、experiment setup、结果和视频同步写作。

### 9.2 每周计划

| 日期 | 方法与代码 | 数据与实验 | 论文与汇报 |
|---|---|---|---|
| 07-12 至 07-15 | 冻结修订主线；定义 RL exploration；确认核心接口 | 锁定场景、seeds、指标、预算上限和结果 schema | 完成根据老师意见修订的方案稿与论文一页 outline |
| 07-13 至 07-19 | MPPI/proposal 数学审计；修复关键 regression | 建立 legacy golden 与计时基线；只做数据管线 smoke | 完成 Problem Statement 与相关工作逐项对照 |
| 07-14 至 07-20 | 修正 MuJoCo delay/applied control；搭建 RL env 接口 | 物理阶跃、圆弧、摩擦和延迟校验；G0 前不生成正式数据 | 完成 Experiment Setup 框架 |
| 07-20 至 07-24 | G0 后训练/优化 ICODE；RL 只做 provisional smoke | 正式 residual dataset、prediction benchmark；dev-only MPPI tuning | 完成 Dynamics/Residual 方法初稿；并行采集早期实车 offline logs |
| 07-25 至 08-05 | ICODE hash 冻结后启动一种 RL 算法 3-seed 正式训练 | 固定间隔评价 RL learning curve；继续 offline/shadow 接口检查 | 完成 RL Prior、Reward 与训练协议初稿 |
| 07-26 至 08-06 | 实现双源候选集、fixed $\eta$ 和候选质量指标 | 核心 2×2 development seeds | 完成 Ablation/Metric 协议 |
| 08-05 至 08-08 | 仅在 G2/fixed mixture 通过后实现 adaptive $\eta_t$ | 中期消融与实时检查；否则直接删除 adaptive 分支 | 形成 Method v1 和主要示意图 |
| 08-09 至 08-10 | 只修阻塞性 bug并硬冻结方法 | 锁定 configs/checkpoints/test seeds；shadow smoke | Method 和 Setup 定稿 |
| 08-11 至 08-21 | 不新增功能 | 核心主表、sample curve、RL curve | 同步生成图表与结果段落 |
| 08-18 至 08-24 | 只处理实验缺陷 | dynamics shifts、缺失 paired runs | 主结果冻结，完成 Results 初稿 |
| 08-22 至 08-28 | 动态障碍/实车仅在核心完成时执行 | 可选扩展和失败案例复核 | 形成完整论文初稿 v1 |
| 08-29 至 09-03 | 只修影响结论的实现错误 | 导师建议的定向补充实验 | 完成 v2、摘要、引言和局限性 |
| 09-04 至 09-06 | 代码与配置 freeze | 实验 freeze、artifact 审计 | 图表、引用和结论冻结 |
| 09-07 至 09-10 | 匿名代码与运行脚本检查 | 视频素材和复现抽查 | 格式、页数、匿名和补充材料检查 |
| 09-11 至 09-12 | 不再改算法 | 只修数据/图表错误 | 形成内部预提交包 |
| 09-13 至 09-14 | 最终 checksum 与备份 | 最终一致性核对 | 北京时间内部正式提交 |
| 09-15 | 仅处理 portal 或文件损坏等紧急问题 | 不安排实验 | 官方截稿缓冲 |

### 9.3 写作交付节点

- 07-15：一页论文 outline；
- 07-20：Introduction/Related Work 结构稿；
- 07-27：Method 主体初稿；
- 08-05：Experiment Setup 初稿；
- 08-18：第一批正式图表；
- 08-24：主结果与分析冻结；
- 08-28：完整初稿 v1；
- 09-03：根据老师意见完成 v2；
- 09-06：正文、图表和实验冻结；
- 09-10：匿名、格式、视频和代码检查；
- 09-12：预提交版本；
- 09-14：内部提交。

---

## 十、实车验证计划

### 10.1 原则

实车验证使用现有小车平台，不更换平台、不改写安全链。Python 2 ROS bridge 不直接 import PyTorch；如需推理，采用独立 Python 3 node、进程间接口或经过验证的模型导出方式。

### 10.2 阶段 R1：Offline logs

低速、空场采集：

- 直线；
- 原地旋转；
- 固定圆弧；
- 不同 $v_{\mathrm{cmd}}$ 和 $\omega_{\mathrm{cmd}}$；
- 不同载荷或地面；
- commanded/applied control、odom、scan summary 和 safety state。

比较 nominal 与 frozen ICODE 的 one-step/H-step prediction，RL 不参与该阶段。

### 10.3 阶段 R2：Shadow mode

在线运行但不影响 `/cmd_vel`，记录：

- traditional prior；
- RL proposal；
- MPPI selected trajectory；
- candidate feasible ratio 与 ESS；
- residual/RL inference time；
- proposed/executed control difference；
- scan_guard 和 safety arbitration state。

### 10.4 阶段 R3：条件性 guarded rollout

只有 MuJoCo、offline 和 shadow 均通过时才执行：

- 固定低速上限；
- 空场后再进入简单静态障碍；
- 先 M0，再 M4；
- 每种方法有限重复；
- 人工急停始终可用；
- 发生接触、连续 deadline miss 或异常安全覆盖时立即停止。

实车样本量不足时，只作为 feasibility evidence，不做强统计结论。

---

## 十一、资源、风险与缩域顺序

### 11.1 需要确认的资源

| 资源 | 最迟确认 | 用途 |
|---|---|---|
| GPU 与可用时段 | 07-15 | RL 3 seeds、ICODE 训练 |
| 并行 CPU/仿真进程 | 07-15 | 多 seed MuJoCo benchmark |
| 存储空间 | 07-15 | trajectories、checkpoints、视频 |
| 实车与场地时段 | 07-20 | offline/shadow/低速验证 |
| 拟请老师审阅方法的时间点 | 08-10 前（以老师方便时间为准） | 方法冻结前获得指导 |
| 拟请老师审阅初稿的时间点 | 08-28 后（以老师方便时间为准） | 根据意见修改 v1/v2 |

### 11.2 主要风险

| 风险 | 早期信号 | 处理方式 |
|---|---|---|
| Traditional MPPI 已足够强 | RL 在所有简单场景均无增益 | 将主分析集中到困难场景和训练成本，不贬低 baseline |
| RL 不稳定 | 3 seeds 差异过大 | 只保留一种算法，减少 action 维度，尽早 Gate |
| RL candidate pool 缺少模式多样性 | 双源 candidates 仍集中在相近控制模式 | 调整 proposal representation 或 bounded entropy，而不是扩大模块数量 |
| ICODE prediction 无收益 | H-step RMSE 不降 | 回到数据、标签、delay 和模型假设审计 |
| ICODE 太慢 | p95 超过 control period | batch、缩小网络、等 wall-clock 比较 |
| 双源 proposal 数学不严格 | importance correction 无法闭合 | 修正推导或改用 guided sampling-based MPC 表述 |
| adaptive $\eta$ 过于启发式 | 只在个别 seed 有效 | 删除 adaptive 主张，保留固定混合分析 |
| 动态障碍成为新项目 | 需要 tracking/prediction 重构 | 降为反应式扩展或删除 |
| 实车时间不足 | 场地/硬件不可用 | 优先完成 offline/shadow，闭环不作刚性主结论 |
| 工作量再次膨胀 | 新增模型、场景、指标 | 按优先级立即缩域，不延后方法冻结 |

### 11.3 缩域优先级

若时间不足，按以下顺序删除：

1. 一次性 offline fine-tuned ICODE 消融；
2. $K=25$、$K=200$ 补充点；
3. `lab_complex` 等非核心场景；
4. 动态障碍扩展；
5. 实车闭环统计，保留 offline/shadow；
6. MLP/Oracle 的完整闭环，只保留 prediction；
7. 额外 uncertainty detector；
8. 非核心 seeds 的宽矩阵。

不得删除：

- 强 traditional MPPI baseline；
- Frozen ICODE × RL prior 的核心 2×2；
- RL 3-seed learning curve；
- 等 samples 与等 wall-clock 公平比较；
- 静态 MuJoCo 正式感知链；
- collision、safety intervention 和 deadline 指标；
- 配置、seed、checkpoint 与 Git SHA 记录。

---

## 十二、论文结构建议

1. **Introduction**：有限 MPPI 预算下的 prediction error 与 candidate generation inefficiency；
2. **Related Work**：residual dynamics MPPI、RL-guided MPPI、adaptive sampling；
3. **Problem Formulation**：frozen prediction model、trajectory proposer 和 safety boundary；
4. **Method**：ICODE rollout、dual-source candidates、adaptive exploration allocation；
5. **Experimental Setup**：MuJoCo、小车、LaserScan、场景、预算和统计；
6. **Results**：prediction、RL curve、2×2、allocation、real-time、OOD/extension；
7. **Real-Robot Evidence**：根据实际完成程度决定独立章节或实验小节；
8. **Limitations**：无理论安全保证、静态主场景、frozen residual、动态障碍边界；
9. **Conclusion**。

论文叙事顺序拟为：

~~~text
传统 MPPI 是强而无需训练的 baseline
    ↓
检验 Frozen residual 是否改善候选轨迹预测
    ↓
检验 RL proposer 能否提供有增量价值的候选轨迹
    ↓
若成立，再以 Traditional anchor 避免完全依赖 RL
    ↓
检验单一 eta 是否能合理调节候选预算
    ↓
MPPI 统一评价并修正 proposal
    ↓
通过 2×2、learning curve 和 allocation ablation 判断各项主张是否成立
~~~

---

## 十三、恳请老师指导的事项

1. 我将“RL 探索”理解为候选控制轨迹空间的探索，并继续保留 RL sampling prior；请问这一理解是否符合您的建议？
2. 将 ICODE 在主实验前冻结、一次微调仅作为可删除消融，是否足以回应计算效率和工作量问题？
3. 将静态障碍作为主要闭环实验、动态障碍作为条件性扩展，是否合适？
4. 以“traditional/RL 双源候选集 + 单一探索分配系数 $\eta$”替代原来的复杂双层门控，是否是更合适的方法复杂度？
5. 实车部分以 offline、shadow 和有限低速验证为主，是否符合本次投稿对实验完整性的预期？

我的初步倾向是先在 7 月中旬尽快冻结上述边界，避免再次增加模块；随后用 Gate 结果决定是否保留 adaptive allocation、动态障碍和 ICODE fine-tuning。恳请老师审阅并指导，我会根据您的意见继续修改。

---

## 附录 A：Residual 训练公式

Residual target 为：

$$
\mathbf r_t
=
\dot{\mathbf x}_{\mathrm{obs},t}
-
f_{\mathrm{nom}}
\left(
\mathbf x_t,
\mathbf u_{\mathrm{applied},t}
\right).
$$

Residual label 默认使用与该 transition 实际对应的 applied control；commanded control 作为独立字段保存。该标签只用于状态定义下可表示的 Markov mismatch。若控制延迟没有通过 applied-control/history features 进入扩展状态，则 delay 单独评价，不把历史依赖误写成瞬时 residual。

总损失为：

$$
\mathcal L
=
\lambda_{\mathrm{res}}\mathcal L_{\mathrm{res}}
+
\lambda_1\mathcal L_1
+
\lambda_H\mathcal L_H
+
\lambda_{\mathrm{reg}}\mathcal L_{\mathrm{reg}}.
$$

多步损失为：

$$
\mathcal L_H
=
\sum_{h=1}^{H}
w_h
\left\|
\widehat{\mathbf x}_{t+h}
-
\mathbf x_{t+h}
\right\|_{W}^{2}.
$$

航向误差统一使用：

$$
e_{\theta}
=
\operatorname{atan2}
\left(
\sin\Delta\theta,
\cos\Delta\theta
\right).
$$

训练与评估支持 Euler/RK4；正式比较必须报告积分器、$dt$ 和 derivative-call 数。

---

## 附录 B：实验产物清单

每次正式 experiment run 输出：

~~~text
config snapshot
seed manifest
Git SHA
dataset/checkpoint hash
summary.csv
trajectory.csv
candidate_metrics.csv
metrics.json
timing.json
figures/
logs/
~~~

每张图至少标明：

- 方法；
- 场景；
- seed/seeds；
- dynamics condition；
- samples 与 horizon；
- residual/RL checkpoint；
- equal-sample 或 equal-wall-clock protocol。

---

## 附录 C：提交前检查清单

- [ ] 标题、摘要和贡献与最终方法一致；
- [ ] 不把 smoke 结果写成正式结果；
- [ ] 不把 guided sampling 错称为严格 MPPI；
- [ ] 不把 ICODE-style 结构写成完整 ICODE 理论复现；
- [ ] 不把 RL prior 写成最终执行控制；
- [ ] 不把 adaptive allocation 写成安全保证；
- [ ] 不把反应式动态障碍实验写成预测式避障；
- [ ] 所有主表包含 strong traditional MPPI baseline；
- [ ] RL learning curve 包含 3 training seeds；
- [ ] equal-sample 与 equal-wall-clock 均已报告；
- [ ] test/OOD test 未参与调参；
- [ ] 所有结果均可由原始 CSV/JSON 自动重建；
- [ ] 实车 bridge、scan_guard 和 local obstacle chain 未被削弱；
- [ ] 匿名代码中没有作者、单位或机器路径泄漏；
- [ ] PDF 页数、字体、引用、视频与 portal 要求已复核；
- [ ] 9 月 14 日北京时间完成内部提交并保存校验副本。
