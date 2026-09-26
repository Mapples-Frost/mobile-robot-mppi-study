# 跨层不确定性门控的学习动力学—策略引导 MPPI：研究计划与 ICRA 2027 投稿安排

> 汇报对象：导师  
> 文档性质：研究方案讨论稿，恳请老师审阅并批评指正  
> 项目仓库：mobile-robot-mppi-study  
> 计划投稿：ICRA 2027 主会  
> 固定截稿日期：2026 年 9 月 15 日  
> 计划起始日期：2026 年 7 月 11 日  
> 可用周期：约 66 天

---

## 一、汇报说明

老师您好：

结合前期已经完成的 MPPI、Memory-Augmented MPPI、MuJoCo、LaserScan、安全仲裁、ROS bridge 以及 ICODE 残差学习框架，我重新梳理了一份面向 ICRA 2027 的研究与投稿计划。

本计划希望在保留已有成果和实车安全链的基础上，围绕“动力学模型误差”和“MPPI 采样效率”两个相对独立的问题，探索 ICODE 残差动力学学习与 RL sampling prior 的协同方法。

由于距离 2026 年 9 月 15 日截稿仅约 66 天，时间非常紧张。为尽可能提高完成质量，我计划遵循以下原则：

1. 研究、实验和论文写作并行推进；
2. 优先验证核心科学假设，不盲目扩展功能；
3. 每一项结论均由可追溯、可复现的实验支持；
4. 如果某个假设不成立，及时缩小论文主张，不人为挑选有利结果；
5. MuJoCo、LaserScan、scan_guard、ROS bridge 和实车安全链保持独立且完整；
6. 9 月 15 日投稿节点固定，所有开发与写作均按该日期倒排；
7. 能够提前完成的任务随即前移，为必要的补充实验、重复验证，以及根据老师意见修改预留时间。

以下内容仍属于我的初步构思，特别是创新点边界、RL 算法选择、实车实验范围以及论文主线，恳请老师进一步指导。

### 1.1 导师快速阅读摘要

| 项目 | 当前计划 |
|---|---|
| 暂定题目 | 跨层不确定性门控的学习动力学—策略引导 MPPI |
| 唯一主问题 | 在模型失配与有限采样预算下，统一 horizon-risk 是否能够同时调节 residual trust 与 learned proposal trust，并在 OOD 时连续回退 |
| 拟议贡献 1 | 校准不确定性门控的 ICODE-style control-affine residual |
| 拟议贡献 2 | 由 policy OOD 与 horizon model uncertainty 联合调节的 RL prior/fallback |
| 拟议贡献 3 | Scene × dynamics 双轴 OOD、低采样预算和实时约束下的系统验证 |
| 理论边界 | 不宣称 contraction、stability 或 safety guarantee；先完成 strict MPPI/proposal correction 审计 |
| 关键 Gate | 07-14 新颖性、07-20 Foundation、07-27 Residual、08-03 RL、08-10 Joint |
| 方法冻结 | 目标 08-10，最迟 08-17 |
| 主结果冻结 | 目标 08-21，最迟 08-24 |
| 完整初稿 | 目标 08-28，最迟 08-31 |
| 实验冻结 | 目标 09-06，最迟 09-08 |
| 内部提交 | 09-14（北京时间） |
| 正式截稿 | 2026-09-15，具体时区与 portal 政策待官网最终核验 |

目前最希望请老师判断的三点是：

1. 将 cross-layer reliability coupling 作为唯一核心方法是否妥当；
2. 我初步建议以 PPO 为默认方案；仅当环境、reward 和接口已经通过，而 48 小时优化仍不稳定时切换一次 SAC，恳请老师判断是否妥当；
3. 实车证据应作为主文核心，还是设置明确停止条件后作为加分项。

后续章节保留完整技术细节、实验矩阵与执行安排，便于进一步讨论。

---

## 二、项目现状与已有基础

### 2.1 已有基础

目前仓库已经具备以下模块：

- 统一的 mobile_robot_mppi 分层研究框架；
- 三状态和五状态动力学接口；
- Euler 与 RK4 数值积分；
- nominal、oracle、MLP residual、ICODE-style residual 模型接口；
- 可配置的 MuJoCo 差速轮物理对象；
- 轮半径、轮距、质量、惯量、摩擦、扭矩限制、PI 执行器等物理参数；
- MuJoCo ray-cast LaserScan；
- Odom、传感器噪声、dropout 与 latency 接口；
- LaserScan → scan_guard → local_obstacle_layer → MPPI 感知链；
- MPPI proposed control 与 safety executed control 的显式区分；
- residual dataset、normalization、multi-step rollout loss 与 checkpoint；
- Memory-Augmented MPPI 兼容层；
- RL sampling prior 接口和 Gymnasium 风格接口；
- 多场景配置、headless benchmark、CSV/JSON/provenance 输出；
- ROS Kinetic/Python 2 实车 bridge 兼容边界。

### 2.2 当前 smoke 结果

目前已经完成以下端到端 smoke 验证：

~~~text
MuJoCo 数据采集
→ residual 标签生成
→ ICODE 训练
→ checkpoint 保存与加载
→ ICODE 进入 MPPI rollout
→ MuJoCo 闭环执行
~~~

但当前结果仅能证明框架连通，尚不能作为正式论文结论。

现有 smoke dataset 规模如下：

| 项目 | 当前数量 |
|---|---:|
| 总 transitions | 120 |
| Train | 80 |
| Validation | 20 |
| Test | 20 |
| Unseen-dynamics test | 0 |
| Episodes | 6 |
| 每个 episode | 20 steps |
| ICODE 训练 | 2 epochs |

在相同场景、相同 seed、64 samples 条件下，当前 smoke 对比如下：

| 指标 | Nominal | ICODE smoke |
|---|---:|---:|
| Success | 否 | 否 |
| Collision | 否 | 否 |
| Final goal distance | 2.840 | 3.042 |
| Control jerk | 0.106 | 0.053 |
| Minimum clearance | 0.054 | 0.073 |
| Mean planner time | 约 2.6 ms | 约 72.9 ms |

上述结果说明：

- ICODE 已经实际影响 MPPI rollout 和控制行为；
- 当前 smoke ICODE 控制较平滑，但目标推进较差；
- 当前数据量、训练轮数和推理实现不足以证明 ICODE 优势；
- 正式研究需要重新采集数据、训练模型并完成公平消融。

### 2.3 正式实验前需要优先解决的问题

1. 当前通用 MPPI 需要进一步对照经典公式审计 importance-sampling 和 control-noise correction；
2. 当前 40 ms command delay 会被 100 ms 控制周期量化，需要改为基于时间戳或 MuJoCo 物理子步的延迟队列；
3. 数据标签需要严格区分 commanded control 与 applied control；
4. 当前 ICODE 输入包含全局位置，可能影响跨场景泛化，需要进行 translation-invariant encoding 消融；
5. 当前 ICODE 推理开销较大，需要优化 batch inference 和网络规模；
6. 当前尚无正式 RL policy、uncertainty estimation 与 gating；
7. 当前尚无论文级多 seed、OOD 和实车结果。

---

## 三、拟研究的核心问题

### 3.1 总体研究问题

本研究拟回答：

> 在移动机器人同时面临动力学模型失配、复杂障碍场景和有限 MPPI 采样预算时，可靠度门控的 residual dynamics 与 learned sampling prior 能否提高 OOD 闭环表现，并在学习模块失效时保持接近 nominal MPPI 的可控回退行为？

### 3.2 两个主要瓶颈

#### 瓶颈 A：Rollout model error

MPPI 使用低阶 nominal model 预测候选控制序列对应的未来轨迹，但真实机器人包含：

- 电机与执行器动态；
- 质量和转动惯量；
- 轮胎与地面摩擦；
- 轮胎滑移；
- 扭矩饱和；
- 控制延迟；
- 左右轮不对称；
- 未建模扰动。

因此，MPPI 的预测轨迹可能偏离真实执行轨迹。

#### 瓶颈 B：Sampling inefficiency

MPPI 在实时预算内只能尝试有限数量的控制序列。传统高斯采样可能产生大量明显无效的样本，在狭窄通道、U-trap 或复杂障碍环境中尤其明显。

### 3.3 拟采用的职责分工

~~~text
ICODE-style residual
负责回答：这个候选控制真正执行后，机器人会怎样运动？

RL sampling prior
负责回答：当前场景下，优先从哪些控制方向开始搜索？

MPPI
负责在线比较候选轨迹，并根据任务代价与障碍代价更新控制序列。

Safety arbitration
负责在最终执行前检查并覆盖危险控制。
~~~

---

## 四、研究对象与数学定义

### 4.1 状态与控制

第一阶段拟采用五状态 dynamic unicycle：

$$
\mathbf{x}
=
\begin{bmatrix}
p_x & p_y & \theta & v & \omega
\end{bmatrix}^{\mathsf T}.
$$

其中：

- $p_x,p_y$：机器人在世界坐标系中的位置；
- $\theta$：机器人航向角；
- $v$：当前线速度；
- $\omega$：当前角速度。

控制命令为：

$$
\mathbf{u}
=
\begin{bmatrix}
v_{\mathrm{cmd}} & \omega_{\mathrm{cmd}}
\end{bmatrix}^{\mathsf T}.
$$

### 4.2 名义动力学与真实对象

名义动力学写为：

$$
\dot{\mathbf{x}}_{\mathrm{nom}}
=
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u}).
$$

MuJoCo 或真实机器人对应的真实动力学写为：

$$
\dot{\mathbf{x}}_{\mathrm{true}}
=
f_{\mathrm{true}}(\mathbf{x},\mathbf{u},\boldsymbol{\xi}),
$$

其中 $\boldsymbol{\xi}$ 表示未完全建模的物理因素，例如摩擦、延迟、负载和执行器状态。

### 4.3 残差动力学

如果 $\boldsymbol{\xi}$ 可以被观测并作为上下文输入，则真实瞬时残差可以定义为：

$$
f_{\mathrm{res}}
\left(
\mathbf{x},\mathbf{u},\boldsymbol{\xi}
\right)
=
f_{\mathrm{true}}
\left(
\mathbf{x},\mathbf{u},\boldsymbol{\xi}
\right)
-
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u}).
$$

但在当前系统中，部分物理因素不可直接观测。因此，仅以 $(\mathbf{x},\mathbf{u})$ 为输入的模型实际学习的是条件平均意义下的残差：

$$
f_{\mathrm{res}}^{*}(\mathbf{x},\mathbf{u})
=
\mathbb{E}
\left[
f_{\mathrm{true}}
\left(
\mathbf{x},\mathbf{u},\boldsymbol{\xi}
\right)
-
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u})
\;\middle|\;
\mathbf{x},\mathbf{u}
\right].
$$

组合预测模型相应写为：

$$
\dot{\mathbf{x}}_{\mathrm{pred}}
=
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u})
+
f_{\mathrm{res}}^{*}(\mathbf{x},\mathbf{u}).
$$

控制延迟依赖控制历史，因此对未扩展状态的 $(\mathbf{x}_t,\mathbf{u}_t)$ 并非 Markov。正式实验中拟采用以下边界：

- 对瞬时、Markov 型物理失配评估 residual；
- 对 delay 单独报告；
- 如需让模型显式学习 delay，则将 applied-control/history features 加入扩展状态；
- 数据中同时保存 commanded control 与 applied control；
- 实车噪声较强时，同时报告离散转移预测，避免只依赖有限差分 derivative 标签。

### 4.4 ICODE control-affine residual

拟使用的 ICODE-style control-affine residual estimator 为：

$$
\widehat{f}_{\mathrm{res},\boldsymbol{\theta}}
\left(
\mathbf{x},\mathbf{u}
\right)
=
f_{\boldsymbol{\theta}}(\mathbf{x})
+
G_{\boldsymbol{\theta}}(\mathbf{x})\mathbf{u}.
$$

其中 $f_{\mathrm{res}}^{*}$ 表示希望逼近的条件平均 residual，$\widehat{f}_{\mathrm{res},\boldsymbol{\theta}}$ 表示由有限数据训练得到的估计器，二者在全文中保持区分。

其中：

$$
f_{\boldsymbol{\theta}}(\mathbf{x})
\in
\mathbb{R}^{n_x}
$$

表示与当前控制无关的 residual drift；

$$
G_{\boldsymbol{\theta}}(\mathbf{x})
\in
\mathbb{R}^{n_x \times n_u}
$$

表示状态相关的 residual control gain。

### 4.5 不确定性感知的 residual gating

为降低 OOD 条件下错误 residual 的风险，拟研究：

$$
\dot{\mathbf{x}}_{\mathrm{pred}}
=
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u})
+
\alpha(\mathbf{x},\mathbf{u})
\widehat{f}_{\mathrm{res},\boldsymbol{\theta}}
\left(
\mathbf{x},\mathbf{u}
\right),
$$

其中：

$$
0 \leq \alpha(\mathbf{x},\mathbf{u}) \leq 1.
$$

对于由 $M$ 个 residual estimators 组成的 ensemble，拟用预测分歧构造第 $t$ 步的 model-risk score：

$$
r_t
=
\operatorname{tr}
\left[
\operatorname{Cov}_{m=1,\ldots,M}
\left(
\widehat{f}_{\mathrm{res},\boldsymbol{\theta}_m}
\left(
\mathbf{x}_t,\mathbf{u}_t
\right)
\right)
\right].
$$

Residual trust 由同一个经过 Validation calibration 的单调递减映射给出：

$$
\alpha(\mathbf{x}_t,\mathbf{u}_t)
=
\alpha_t
=
g_{\mathrm{res}}(r_t),
\qquad
0 \leq g_{\mathrm{res}}(r_t) \leq 1.
$$

预期行为为：

~~~text
Residual 置信度高
→ alpha 接近 1
→ 充分使用 learned correction

Residual 置信度低
→ alpha 下降
→ 逐渐退化到 nominal model
~~~

需要特别说明的是，该 gating 只能作为经验性可靠度调节机制。在没有完成相应理论推导和验证前，本研究不会将其表述为稳定性或安全性保证。

### 4.6 RL sampling prior

RL policy 不直接输出最终执行控制，而是生成 MPPI proposal 的均值控制序列或低维控制 knots：

$$
\boldsymbol{\mu}_{\mathrm{RL}}
=
\pi_{\phi}(\mathbf{o}),
$$

其中 $\mathbf{o}$ 表示机器人观测、目标相对位置和 LaserScan/局部障碍特征。

为与当前实现能力以及 MPPI 权重审计保持一致，主线暂不直接使用两个概率密度的 mixture sampling，而优先采用均值序列的可信度混合：

$$
\overline{\mathbf{U}}_0
=
\beta(\mathbf{o})
\overline{\mathbf{U}}_{\mathrm{RL}}
+
\left[1-\beta(\mathbf{o})\right]
\overline{\mathbf{U}}_{\mathrm{fallback}}.
$$

随后在审计后的固定协方差下采样：

$$
\boldsymbol{\epsilon}_k
\sim
\mathcal{N}(\mathbf{0},\boldsymbol{\Sigma}),
$$

$$
\mathbf{U}_k
=
\overline{\mathbf{U}}_0
+
\boldsymbol{\epsilon}_k.
$$

其中 $0 \leq \beta(\mathbf{o}) \leq 1$。Fallback 拟采用 PreviousSequencePrior、GoalWarmStartPrior 或二者混合。

沿 RL proposal 预测得到的 horizon model risk 定义为：

$$
R_H
\left(
\overline{\mathbf{U}}_{\mathrm{RL}}
\right)
=
\operatorname{Agg}
\left(
r_0,r_1,\ldots,r_{H-1}
\right),
$$

其中 $\operatorname{Agg}$ 初步比较 discounted mean 与 conservative maximum，并在 Validation 上冻结。

Horizon model trust 定义为：

$$
T_{\mathrm{model}}
\left(
\overline{\mathbf{U}}_{\mathrm{RL}}
\right)
=
g_H
\left[
R_H
\left(
\overline{\mathbf{U}}_{\mathrm{RL}}
\right)
\right].
$$

为形成真正的 residual–policy coupling，proposal trust 进一步定义为：

$$
\beta(\mathbf{o})
=
T_{\mathrm{policy}}(\mathbf{o})
\,
T_{\mathrm{model}}
\left(
\overline{\mathbf{U}}_{\mathrm{RL}}
\right),
$$

并分别约束：

$$
0
\leq
T_{\mathrm{policy}}(\mathbf{o})
\leq
1,
\qquad
0
\leq
T_{\mathrm{model}}
\left(
\overline{\mathbf{U}}_{\mathrm{RL}}
\right)
\leq
1,
$$

从而保证 $0 \leq \beta \leq 1$。

其中：

- $T_{\mathrm{policy}}$ 表示 policy 对当前观测的可信度；
- $T_{\mathrm{model}}$ 表示 residual model 沿 RL 候选 rollout horizon 的可信度；
- 如果 RL proposal 会把系统带入模型不熟悉的区域，则降低 $\beta$，增加 fallback influence。

因此，同一 ensemble-derived step risk $r_t$ 一方面通过 $g_{\mathrm{res}}$ 逐步调节 residual correction，另一方面沿 RL proposal 聚合为 $R_H$，再通过 $g_H$ 调节 RL proposal trust。该 cross-layer coupling 是本研究拟重点验证的机制。

Residual trust $\alpha$ 与 proposal trust $\beta$ 拟由同一 Validation calibration protocol 标定，并以 feature-distance OOD score 作为低成本对照。置信度到 $[0,1]$ 的映射、校准误差和推理开销将在方法冻结前明确。

### 4.7 MPPI 轨迹权重

对第 $k$ 条候选轨迹，其代价记为 $S_k$。MPPI 权重形式为：

$$
w_k
=
\frac{
\exp\left[-\frac{1}{\lambda}(S_k-\rho)\right]
}{
\sum_j
\exp\left[-\frac{1}{\lambda}(S_j-\rho)\right]
},
$$

其中：

$$
\rho = \min_k S_k
$$

用于数值稳定，$\lambda$ 为温度参数。

严格 MPPI 中还需要对 control perturbation、proposal distribution 和 importance-sampling correction 进行一致推导。这部分计划在第一周完成公式—代码逐项审计后形成相对稳定版本。

---

## 五、相关工作定位与拟议创新点

### 5.1 直接相关工作的竞争压力

截至 2026 年 7 月，以下方向已经存在较接近的工作：

| 相关工作 | 已覆盖内容 | 本研究不能重复声称的内容 |
|---|---|---|
| ICODE-MPPI | Control-affine residual + MPPI | 仅将 ICODE-style residual 接入 MPPI |
| RGB | Pretrained RL policy 作为 MPPI sampling prior | 仅让 RL policy 引导 MPPI |
| HOLO-MPPI | Policy 在线参数化 MPPI sampling distribution | 仅学习 sampling distribution |
| Adaptive Hierarchical RL-MPC | RL action 引导 MPPI sampler，并进行 adaptive sampling | 仅以 uncertainty 调节探索 |
| PO-MPC | Learned proposal/adaptive prior 的统一优化视角 | 仅做 learned proposal |
| Uncertainty-Averse MPPI | GP/ensemble uncertainty 与 MPPI | 仅将模型 uncertainty 加入 MPPI |
| URPPI | Learned dynamics uncertainty 调节探索与利用 | 仅做 uncertainty-adaptive exploration |

计划持续核对的资料包括：

- ICODE-MPPI：<https://arxiv.org/abs/2605.03260>
- RGB：<https://arxiv.org/abs/2606.25123>
- HOLO-MPPI：<https://arxiv.org/abs/2606.16480>
- Adaptive Hierarchical RL-MPC：<https://arxiv.org/abs/2512.17091>
- PO-MPC：<https://arxiv.org/abs/2510.04280>
- Uncertainty-Averse MPPI：<https://arxiv.org/abs/1710.04005>
- URPPI：<https://arxiv.org/abs/2509.03839>

因此，本研究不拟将以下内容单独表述为创新：

- Residual 与 RL 的职责分解；
- RL policy 作为 sampling prior；
- ICODE-style residual 进入 MPPI；
- 少 samples 实验；
- 多场景测试；
- 实车迁移流程。

这些内容是必要的系统组成或验证维度，但不足以单独构成方法新颖性。

### 5.2 7 月 14 日 Related-Work Gate

计划在 7 月 14 日前完成一张方法对比表，至少回答：

1. 是否已有方法同时使用 learned residual、policy prior 和 uncertainty gating；
2. 是否已有方法沿 policy-proposed horizon 累积 model uncertainty；
3. 是否已有方法用同一 risk signal 同时调节 residual trust 与 proposal trust；
4. 所采用的 proposal update 是否与 MPPI importance correction 一致；
5. 本研究与 cost-based uncertainty、exploration-only adaptation 和 always-on residual 的差别是什么。

如果届时无法用一张对比表清楚说明差异，我计划在同一 9 月 15 日投稿时间线内立即收窄到证据最强的单一主线，而不继续堆叠模块。

### 5.3 拟议贡献

以下三项目前属于拟议贡献，而不是已经被结果证明的结论。是否具有足够的新颖性，还需要结合上述 Gate 和实验结果进一步收敛，恳请老师指导。

#### 创新点 1：校准不确定性门控的控制仿射残差动力学

拟采用受 ICODE 启发的 control-affine residual structure，并通过校准后的 horizon uncertainty 调节 residual trust：

$$
\dot{\mathbf{x}}_{\mathrm{pred}}
=
f_{\mathrm{nom}}(\mathbf{x},\mathbf{u})
+
\alpha(\mathbf{x},\mathbf{u})
\widehat{f}_{\mathrm{res},\boldsymbol{\theta}}
\left(
\mathbf{x},\mathbf{u}
\right).
$$

研究重点不是只增加一个 residual network，而是：

- 识别 residual 何时可信；
- 在高误差/OOD 区域降低 learned correction；
- 比较 always-on residual、固定权重 residual 与 calibrated gating；
- 报告 uncertainty calibration、risk–coverage 和推理开销。

#### 创新点 2：由 policy OOD 与 horizon model uncertainty 联合调节的 RL sampling prior

拟让 RL 提供 MPPI 的 prior mean，而不直接替代控制器。Proposal trust 同时依赖：

- policy 对当前 observation 的熟悉程度；
- residual model 沿 RL 候选轨迹的 horizon uncertainty。

当 RL proposal 会进入动力学模型不熟悉的区域时，系统降低对 RL prior 的信任并增加 fallback influence。该 coupling 是本研究区别于两个独立启发式开关的关键。

#### 创新点 3：有限采样预算下的 scene × dynamics 双轴 OOD 验证

拟在相同 wall-clock 和安全设置下，系统比较：

- nominal；
- residual-only；
- RL-prior-only；
- fixed-weight combination；
- dual-gated proposed method；
- Oracle（仅限可诚实构造的低阶模型）或 privileged diagnostic upper bound。

重点验证：

> 在 scene shift 与 dynamics shift 同时存在时，组合方法是否能够用更少 MPPI samples 获得更好的闭环表现，并在学习模块不可靠时回退到接近 nominal MPPI 的行为。

实车 offline、shadow 和低速闭环属于该方法的重要工程验证，但不单独列为算法创新，也不将经验性 gating 表述为安全、稳定或收敛保证。

---

## 六、研究假设与可证伪条件

### H1：Oracle/privileged diagnostic 应当证明动力学修正具有潜在闭环价值

精确 Oracle residual 先限定在可解析、瞬时且 Markov 的低阶 model mismatch。当前 MuJoCo plant 包含执行器、接触和隐藏状态，普通 MPPI 不具备访问完整真动力学的权限，因此不把 identified model 或事后有限差分标签称为 exact Oracle。

MuJoCo 中拟采用两类独立诊断：

- privileged MuJoCo rollout：只用于上界/接口诊断，并明确其真值权限与计算开销；
- identified-model baseline：只使用训练数据，不拥有 Oracle 权限。

控制延迟另设 augmented-state/单独实验，不把未建模历史错误地包装为“精确 Oracle residual”。

如果在 Markov mismatch 下，完全正确的 residual 都无法改善 MPPI，则可能说明：

- 当前任务不是动力学误差主导；
- residual 未正确进入 rollout；
- cost、感知或 safety 可能是主要瓶颈。

在该情况下，我计划暂停扩大神经网络训练，优先排查实验设计。

### H2：ICODE-style residual 应改善多步预测

计划验证：

$$
\operatorname{RMSE}_{H}
\left(
\mathrm{ICODE}
\right)
<
\operatorname{RMSE}_{H}
\left(
\mathrm{Nominal}
\right),
$$

重点关注 $H=10$ 和 $H=20$，而不是只比较 one-step error。

计划同时比较 nominal、普通 MLP residual、always-on ICODE-style residual 与 gated residual。内部判断以 paired bootstrap 的误差差值置信区间为主，并要求 OOD 条件下不存在明显灾难性退化，而不只观察单个平均数。

### H3：RL prior 应改善少样本 MPPI

在固定 prediction model 条件下，RL prior 应在 25、50 或 100 samples 时优于传统 prior。

比较时同时固定或报告 wall-clock budget、mean/p95 planner latency 和 deadline miss rate。如果 learned prior 的 50-sample 控制器比 400-sample nominal 更慢，则不能只依据 samples 数量声称 sample efficiency。

### H4：组合方法应优于两个单独模块

论文的核心协同结论计划同时与 residual-only、RL-prior-only、fixed-weight fusion 和 always-on joint model 比较，而不只与 nominal baseline 比较。最低关系为：

~~~text
ICODE + RL prior
优于
ICODE + traditional prior

并且

ICODE + RL prior
优于
Nominal + RL prior
~~~

### H5：Gating 应在 OOD 条件下降低失败

Always-on learned model 可能在 OOD 条件下产生错误修正。Gating 版本应尽量减少：

- collision；
- stuck；
- spin；
- safety override；
- prediction divergence；
- deadline overrun。

同时计划报告：

- uncertainty calibration error；
- OOD detection AUROC；
- error–uncertainty correlation；
- risk–coverage curve；
- fallback activation rate。

Nominal fallback 只是经验性的可控回退路径，本身不构成安全或稳定性保证。

---

## 七、总体技术路线

~~~text
MuJoCo / Real Robot
    ↓
Odom + LaserScan
    ↓
scan_guard + local_obstacle_layer
    ↓
RobotObservation
    ↓
RL sampling prior + confidence
    ↓
MPPI candidate sequences
    ↓
Nominal dynamics + gated ICODE-style residual
    ↓
Predicted trajectories
    ↓
Goal / obstacle / smoothness / control costs
    ↓
MPPI weighted update
    ↓
Proposed control
    ↓
Safety arbitration
    ↓
Executed control
    ↓
MuJoCo / Real Robot
~~~

以下为方法上的逻辑依赖关系；日历执行中，方法、数据/实验和论文写作三条工作流将并行推进：

1. 修正 MPPI 与 MuJoCo 基础问题；
2. 建立 clean dynamics benchmark；
3. 验证低阶 Oracle 与 MuJoCo privileged diagnostic；
4. 建立正式数据集；
5. 比较 nominal、MLP、ICODE；
6. 加入 residual uncertainty 与 gating；
7. 固定 residual 后训练 RL prior；
8. 验证 RL sample efficiency；
9. 联合 residual 与 RL；
10. 完成 dual gating；
11. 运行正式多 seed benchmark；
12. 在时间和安全条件允许时完成实车验证；
13. 在请老师审阅并根据意见充分修改后，按节点完成实验、论文、视频与匿名代码定稿。

---

## 八、工作包与阶段验收

### WP1：MPPI 数学与实现审计

#### 计划任务

- 对照经典 MPPI 推导检查当前实现；
- 明确 nominal sequence、control perturbation、proposal distribution 和 importance weights；
- 检查 RL prior 改变采样均值或协方差后，权重修正是否仍然严格；
- 实现 strict MPPI mode；
- 保留 legacy-compatible mode 用于回归；
- 增加 effective sample size；
- 增加权重退化与数值稳定诊断；
- 完成 batch rollout 性能分析。

#### 计划验收

- 小规模样例能够手工复核；
- 固定 seed 可复现；
- Euler/RK4 行为符合预期；
- 无 NaN/Inf；
- strict 与 legacy mode 的差异有清晰记录；
- 计算时间随 samples/horizon 的变化可解释；
- 如果 RL-guided proposal 无法满足严格 MPPI 推导，则谨慎使用 guided sampling-based MPC 表述，避免过度声称。

### WP2：MuJoCo 物理可信度

#### 计划任务

- 修正 sub-control-step command delay；
- 区分 commanded control 与 applied control；
- 校验左右轮目标速度转换；
- 校验 torque PI、饱和、slew rate 和 deadband；
- 校验质量、惯量、摩擦与接触；
- 校验 Odom、LaserScan 和 sensor latency；
- 增加 target、LaserScan、samples 和 predicted trajectory overlay。

#### 标准物理实验

| 实验 | 输入 | 主要指标 |
|---|---|---|
| 直线阶跃 | 固定 $v_{\mathrm{cmd}}$ | rise time、steady-state error |
| 原地旋转 | 固定 $\omega_{\mathrm{cmd}}$ | yaw response、overshoot |
| 固定圆弧 | 固定 $v/\omega$ | radius error、heading error |
| 刹车 | $v_{\mathrm{cmd}} \rightarrow 0$ | stopping time、distance |
| 低摩擦 | 多组 friction | slip、trajectory deviation |
| 扭矩限制 | 多组 torque limit | acceleration、tracking |
| 控制延迟 | 0–200 ms | lag、prediction error |
| 左右轮失配 | asymmetric response | drift、yaw bias |

#### 计划验收

- 20、40、65、100 ms delay 能够区分；
- 参数变化产生可解释趋势；
- planner 不读取全局障碍真值；
- sensor observation 与 ground truth 的边界清晰。

### WP3：正式 residual 数据集

#### 数据来源

- random exploration；
- structured excitation；
- task-specific MPPI；
- hard-case/on-policy data。

#### 计划规模

| 阶段 | Episodes | Transitions | 用途 |
|---|---:|---:|---|
| 快速基线 | 50–100 | 2 万–5 万 | 验证模型趋势 |
| 正式训练 | 150–300 | 8 万–20 万 | 最终模型与消融 |

数据量不作为唯一目标，是否继续扩充将依据 learning curve 决定。

#### Structured excitation

- 线速度阶跃；
- 角速度阶跃；
- sine/chirp 控制；
- 固定圆弧；
- 左右对称转向；
- 加速—匀速—减速；
- 原地旋转；
- 多初始速度。

#### 数据拆分

- Train：已见参数组合；
- Validation：独立 episodes；
- IID Test：训练范围内独立测试；
- OOD-Dynamics：未见物理参数；
- OOD-Combination：未见参数组合；
- On-policy Test：由 RL prior 产生的状态—控制分布。

#### 计划验收

- 按 episode/物理配置拆分；
- 无 timestep leakage；
- normalization 只使用 Train；
- OOD split 非空；
- 每条数据记录 config hash、model hash、seed 和 Git SHA；
- commanded/applied control 均被保存；
- 训练、验证和测试的状态—控制覆盖范围有可视化审计。

### WP4：Residual 模型与不确定性

#### 主要 baselines

- Nominal；
- Full MLP dynamics；
- MLP residual；
- ICODE-style control-affine residual；
- Gated ICODE-style residual；
- Oracle（低阶可解析模型）与 identified-model baseline（权限分开报告）。

这里使用 ICODE-style 表述，是因为当前实现复现的是公开的 control-affine residual structure；除非后续确实实现并验证相应数学约束，否则不宣称复现原始 ICODE 的全部 contraction、stability 或 convergence 保证。

#### State encoding 消融

- raw angle；
- sin/cos angle；
- 包含全局 $p_x,p_y$；
- translation-invariant encoding；
- 可选 actuator/history features。

#### Loss

总损失拟写为：

$$
\mathcal{L}
=
\lambda_{\mathrm{res}}\mathcal{L}_{\mathrm{res}}
+
\lambda_1\mathcal{L}_{1}
+
\lambda_H\mathcal{L}_{H}
+
\lambda_{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}.
$$

Residual derivative loss 为：

$$
\mathcal{L}_{\mathrm{res}}
=
\left\|
\hat{\mathbf{r}}_t-\mathbf{r}_t
\right\|_2^2.
$$

One-step state loss 为：

$$
\mathcal{L}_{1}
=
\left\|
\hat{\mathbf{x}}_{t+1}-\mathbf{x}_{t+1}
\right\|_2^2.
$$

Multi-step rollout loss 为：

$$
\mathcal{L}_{H}
=
\sum_{h=1}^{H}
w_h
\left\|
\hat{\mathbf{x}}_{t+h}-\mathbf{x}_{t+h}
\right\|_2^2.
$$

角度误差统一采用：

$$
e_{\theta}
=
\operatorname{atan2}
\left(
\sin\Delta\theta,
\cos\Delta\theta
\right).
$$

#### Uncertainty 方案

为控制时间与实现风险，主线计划只选择一种 uncertainty 方法：

1. deep ensemble disagreement；或
2. normalized feature-distance OOD score。

选择依据包括：

- calibration；
- error–uncertainty correlation；
- inference overhead；
- 实现稳定性；
- 能否在 MPPI rollout 中批量计算。

#### 计划验收

- Oracle/privileged diagnostic 的闭环收益成立；
- ICODE-style model 多步预测优于 nominal；
- ICODE-style 与 MLP 的差异在 OOD 上可解释；
- uncertainty 能够识别高误差区域；
- gated residual 在 OOD 下优于 always-on residual。

### WP5：RL sampling prior

#### 接口边界

RL 仅输出：

- mean control sequence 或低维 control knots；
- 可选 covariance；
- confidence/OOD score。

RL 不越过 MPPI 和 safety arbitration 直接发布最终控制。

#### Observation

候选输入包括：

- 当前 pose/twist；
- goal relative pose；
- compressed LaserScan 或局部障碍特征；
- previous action/sequence；
- residual uncertainty。

正式输入不使用 MuJoCo 障碍全局真值。

#### Action representation

考虑到完整 $24 \times 2$ 控制序列维度较高，优先采用：

- 少量 control knots；
- knot interpolation；
- 或低维 prior parameters。

#### RL 算法冻结

由于时间有限，计划只选择一个主算法：

- PPO：并行训练较直接，工程实现相对稳定；
- SAC：适合连续动作，sample efficiency 通常较好。

将在小规模 smoke 中依据稳定性、训练速度和接口复杂度尽快冻结，不进行多算法大规模扩展。

#### Reward

拟包含：

- goal progress；
- terminal success；
- collision penalty；
- clearance penalty；
- stuck/spin penalty；
- control magnitude；
- control jerk；
- safety override penalty。

将重点检查 reward hacking，例如原地不动、依赖 scan_guard 修正危险动作等。

#### 计划验收

- 25/50/100 samples 下优于传统 prior；
- OOD scene 不出现明显灾难性退化；
- RL prior 失败时 MPPI/fallback 能够纠正；
- safety override 不显著增加；
- 总计算预算保持公平。

### WP6：联合方法

#### 固定训练顺序

1. 训练并冻结 residual；
2. 在固定 residual 下训练 RL prior；
3. 收集 RL-on-policy transitions；
4. 如时间允许，仅更新一轮 residual；
5. 重新评估 RL distribution shift；
6. 冻结最终组合。

#### 核心 2×2

| Prediction | Traditional prior | RL prior |
|---|---:|---:|
| Nominal | A | B |
| ICODE-style residual | C | D |

核心判断为：

- $D>B$：residual 在 RL 下仍有独立价值；
- $D>C$：RL 在 residual 下仍有独立价值；
- $D>A$：组合优于基础 baseline。

#### Gating 扩展

| Residual gating | RL gating | 用途 |
|---|---|---|
| Off | Off | Simple combination |
| On | Off | Residual trust |
| Off | On | Policy trust |
| On | On | Proposed dual gating |

为避免 dual gating 只是两个独立启发式开关，计划进一步研究统一的 horizon uncertainty 是否能够同时调节 residual trust、proposal trust、sampling covariance 或温度参数。

---

## 九、正式实验设计

### 9.1 最小必要方法组

考虑到 66 天时间约束，主论文优先保证以下方法：

1. Nominal MPPI；
2. Residual-only MPPI；
3. RL-prior-only MPPI；
4. Fixed-weight residual + RL combination；
5. Dual-gated proposed method；
6. Oracle/privileged diagnostic MPPI。

MLP 与 ICODE-style 的比较主要作为 model-structure ablation，不对所有场景和所有 sample budgets 做不可承受的全因子组合。

### 9.2 模型预测实验

指标包括：

- residual derivative RMSE；
- one-step state RMSE；
- $H=5,10,20,40$ rollout RMSE；
- position RMSE；
- heading RMSE；
- velocity/yaw-rate RMSE；
- IID/OOD RMSE；
- inference mean/p95/max；
- parameter count；
- checkpoint size；
- uncertainty calibration；
- OOD detection AUROC 或等价指标。

### 9.3 控制场景

主场景拟包括：

- clean dynamics；
- simple；
- lab_complex；
- narrow_corridor；
- u_trap_long_board。

Ellipse、sine 和 figure-eight path tracking 仅在核心任务提前完成时加入。

### 9.4 动力学变化

- nominal friction；
- low/high friction；
- nominal/heavy chassis；
- actuator gain/torque limit；
- command delay；
- left/right asymmetry；
- unseen combined disturbance。

### 9.5 Sample budgets

$$
K
\in
\left\{
25,50,100,200,400
\right\}.
$$

### 9.6 Seeds

- 开发 smoke：3 seeds；
- 中期判断：5 seeds；
- 正式 paired comparison：10 seeds；
- 最终主方法如算力允许扩展至 20 seeds。

### 9.7 控制指标

- success rate；
- final goal distance；
- time to goal；
- trajectory length；
- collision rate；
- minimum clearance；
- control jerk；
- mean absolute omega；
- stuck steps；
- spin steps；
- safety intervention rate；
- fallback activation rate。

### 9.8 计算指标

- model inference time；
- rollout time；
- planner mean/p50/p95/p99/max；
- deadline miss rate；
- effective sample size；
- CPU/GPU 与线程设置；
- peak memory。

### 9.9 物理指标

- slip ratio；
- wheel tracking error；
- actuator effort；
- commanded/applied delay；
- Odom–GroundTruth drift。

### 9.10 统计规范

- 同一比较使用 paired seeds；
- 报告 mean、standard deviation、95% confidence interval 与 effect size；
- success/collision 报告比例置信区间；
- 预先指定 primary metrics；
- 不只展示最好 seed；
- 保留每次运行的原始 trajectory；
- 如实报告失败案例和异常；
- 图表由脚本从原始 CSV/JSON 自动生成。

### 9.11 分层实验矩阵与预计运行量

完整笛卡尔积会超过可用时间，因此正式实验拟采用分层设计，而不是让所有模型、场景、扰动、sample budgets 和 seeds 全组合。

#### 层 1：协同主表

~~~text
5 个关键方法
× 3 个代表性条件
× K=100
× 10 paired environment seeds
≈ 150 episodes
~~~

关键方法包括 nominal、residual-only、RL-prior-only、fixed fusion 和 proposed dual gating。

#### 层 2：Sample-efficiency 曲线

~~~text
3 个关键方法
× K={50,100,400}
× 2 个代表性条件
× 10 paired environment seeds
≈ 180 episodes
~~~

25 和 200 samples 作为算力允许时的补充点，不作为第一批刚性矩阵。

#### 层 3：OOD 与 gating

~~~text
4 个关键方法
× 4 类 dynamics shifts
× 2 个代表场景
× 10 paired environment seeds
≈ 320 episodes
~~~

#### 层 4：模型结构、Oracle 与实车

- MLP vs ICODE-style 主要在 prediction benchmark 中比较；
- Oracle 单独作为接口诊断和上界，不与 identified model 混称；
- 实车 offline/shadow/closed-loop 单独成表；
- 全部正式控制实验预计控制在约 700–1000 episodes；
- 先以 3–5 seeds 筛选实现与明显无效方法，再对冻结方法进行正式 paired evaluation。

### 9.12 Seed 层级

需要区分三类随机性：

1. environment/scenario seed；
2. residual initialization/ensemble seed；
3. RL training seed。

主结果计划至少使用：

- 10 个 paired environment seeds；
- 3 个 RL training seeds 用于核心条件的训练稳定性；
- 3-member residual ensemble 或等价的多初始化检查。

如果宽场景 sweep 只能使用一个固定 checkpoint，论文会明确将结论限定为“给定该 checkpoint 的条件性控制表现”，并另外报告训练稳定性，避免把一个偶然较好的 checkpoint 当作算法稳定性证据。

### 9.13 数据使用纪律

- 超参数只根据 Train/Validation 决定；
- IID Test 与 OOD Test 不用于回调阈值和模型结构；
- On-policy data 如用于更新 residual，另留最终未触碰的 on-policy holdout；
- Oracle 只能读取仿真真值并明确标注；
- identified-model baseline 使用可训练数据，不拥有 Oracle 权限；
- smoke 结果与正式结果目录完全隔离。

---

## 十、实车迁移计划

### 10.1 数据采集

在低速、安全、空场条件下采集：

- 直线；
- 原地旋转；
- 固定圆弧；
- 不同 $v_{\mathrm{cmd}}$；
- 不同 $\omega_{\mathrm{cmd}}$；
- 不同地面；
- 不同载重；
- 重复实验。

### 10.2 Offline prediction

使用同一实车日志比较：

- nominal；
- MLP residual；
- ICODE-style residual；
- gated residual。

### 10.3 Shadow mode

学习模块实时运行并记录：

- predicted next state；
- residual correction；
- uncertainty；
- RL prior；
- fallback；
- computation deadline。

Shadow mode 不影响 /cmd_vel。

### 10.4 Guarded rollout

只有在 offline 和 shadow mode 通过后，才计划进行低速闭环：

- 最大速度和角速度严格限制；
- scan_guard 保持最高优先级；
- 保留人工急停；
- 先空场，再简单障碍；
- 一次只启用一个学习模块；
- 每次运行前后完成安全检查。

### 10.5 实车实验的范围边界与停止条件

考虑到 9 月 15 日固定截稿，实车计划设置明确边界：

- 7 月 20 日前完成实车资源与场地确认；
- 8 月 15 日前无法获得稳定日志，则主论文仅保留 MuJoCo，实车内容不作为核心 claim；
- 8 月 24 日前 offline/shadow 未通过，则不进行 learned closed-loop；
- 低速闭环属于高价值加分项，但不会以削弱安全链或牺牲主仿真实验为代价。

---

## 十一、ICRA 2027 固定截稿倒排计划

### 11.1 总体原则

- 9 月 15 日投稿日期固定；
- 核心方法尽量在 8 月 17 日前冻结；
- 主要 MuJoCo 结果尽量在 8 月 24 日前冻结；
- 8 月 31 日前形成完整论文初稿；
- 9 月 8 日后原则上不新增非关键实验；
- 9 月 12 日冻结正文；
- 9 月 14 日北京时间完成内部正式提交并保存回执；
- 每项任务如能提前完成，随即进入下一阶段。

为体现“能压缩则压缩”，每个关键节点同时设置目标日期和绝对最迟日期：

| 事项 | 目标日期 | 绝对最迟 |
|---|---:|---:|
| Method Freeze | 08-10 | 08-17 |
| Main Result Freeze | 08-21 | 08-24 |
| Full Paper V1 | 08-28 | 08-31 |
| Experiment Freeze | 09-06 | 09-08 |
| Portal/format pre-check | 09-10 | 09-10 |
| Pre-submission package | 09-12 | 09-12 |
| 内部完成提交 | 09-14 北京时间 | 09-14 北京时间 |
| 官方截止 | 09-15 | 具体时区待官网核验 |

9 月 15 日只保留给不可预见的紧急修复，不计划将常规写作或实验留到截止当天。

### 11.2 三条并行工作流

#### A. 方法与代码

~~~text
MPPI审计
→ MuJoCo修正
→ residual/uncertainty
→ RL prior
→ dual gating
~~~

#### B. 数据与实验

~~~text
物理benchmark
→ 数据采集
→ prediction benchmark
→ control benchmark
→ OOD/multi-seed/real-time
~~~

#### C. 论文与材料

~~~text
论文骨架
→ Related Work
→ 方法图
→ 实验模板
→ 结果填充
→ 视频/匿名代码
~~~

### 11.3 详细日程

| 日期 | 阶段 | 主要任务 | 阶段交付物 |
|---|---|---|---|
| 07-11 至 07-13 | 研究冻结 | 问题、贡献、baseline、指标、论文目录 | 研究一页纸、论文 skeleton |
| 07-14 至 07-20 | 基础冻结 | MPPI 审计；delay/applied control；clean benchmark | Foundation tests、baseline report |
| 07-21 至 07-27 | 模型 Gate | Oracle；collector；IID/OOD split；第一批数据 | Dataset v1、Oracle/Residual report |
| 07-28 至 08-03 | RL Gate | RL prior；traditional prior 对照；少样本 smoke | RL sample-efficiency v1 |
| 08-04 至 08-10 | 联合 Gate | Residual/RL combination；第一版 gating；核心 2×2；从 08-04 起滚动运行已冻结 baselines | 联合方法趋势报告 |
| 08-11 至 08-17 | Method Freeze | 统一 uncertainty；dual gating；公平性审计；目标 08-10、最迟 08-17 冻结 | 方法定义冻结、系统视频 v1 |
| 08-18 至 08-24 | Main Result Freeze | 5 场景；dynamics shifts；paired seeds；sample budgets；目标 08-21、最迟 08-24 | 主图、主表 v1 |
| 08-25 至 08-31 | Paper V1 | 完整论文；实车 offline/shadow；失败案例；目标 08-28、最迟 08-31 | 完整可读稿，提交老师进行第一轮审阅；第一次 clean-room 复现 |
| 09-01 至 09-07 | Evidence Freeze | 关键消融；扩展 seeds；可行时实车低速闭环；目标 09-06 冻结实验 | 最终证据、论文 v2；最迟 09-05 完成第二次 clean-room 检查 |
| 09-08 至 09-10 | Experiment/Portal Freeze | 最迟 09-08 固化 configs、CSV、JSON、figures、tables；09-10 完成 portal、作者、格式预检 | 结果归档与 hash；提交元数据草稿 |
| 09-11 至 09-12 | Manuscript/Pre-submission Freeze | 提交老师审阅，并根据意见调整 claim、篇幅与图表；完成预提交包 | Final draft 与完整提交包 |
| 09-13 | 最终复现复核 | 抽查主要结果、匿名性、引用与视频链接 | Final reproducibility checklist |
| 09-14 | 内部正式提交 | 北京时间完成实际提交并下载回执 | Submission receipt |
| 09-15 | 紧急缓冲 | 仅处理不可预见的 portal/文件问题并复核最终状态 | ICRA 2027 submission confirmed |

### 11.4 Gate 与同年分支动作

所有 Gate 只决定论文范围，不改变 9 月 15 日投稿节点。

| 日期 | Gate | 预设判断依据 | 未通过时的同年分支 |
|---|---|---|---|
| 07-14 | Novelty | 一张对比表能清楚区分 cross-layer coupling 与已有工作 | 立即收窄为证据最强的单一方法 |
| 07-20 | Foundation | strict MPPI 手算/回归通过；delay 与 applied-control tests 通过 | 暂停 learned modules，先修基础正确性 |
| 07-27 | Residual | Markov mismatch 下 Oracle 有闭环收益；H20 OOD 误差达到预先设定的 practical improvement；p95 推理可控 | 主线转为 RL-guided MPPI，residual 作为负面/辅助消融 |
| 08-03 | RL Prior | 50/100 samples 下相对 traditional prior 有一致趋势；wall-clock 仍有收益；safety override 不升 | 主线转为 gated residual MPPI，RL 作为负面消融 |
| 08-10 | Joint | Proposed 相对 residual-only、RL-only 和 fixed fusion 均有增量 | 选择更强单线，组合保留为诚实消融 |
| 08-17 | Method Freeze | 公式、代码、baselines、claim 和公平性检查完成 | 不再增加方法，只缩小 claim 并完成实验 |

H20 OOD improvement 的具体 practical threshold 将在不查看最终 Test/OOD Test 的前提下，根据 Validation 和计算预算预先固定。初步希望达到至少约 10% 的相对误差改善，并以 paired confidence interval 与 effect size 共同判断，而不是只看单一百分比。

### 11.5 每周目标

#### 第 1 周：基础可信度

- 完成 MPPI 公式—代码审计；
- 修正 command delay；
- 增加 commanded/applied control tests；
- 完成物理阶跃测试；
- 输出 clean dynamics baseline；
- 开始 Introduction 和 Related Work。

#### 第 2 周：Oracle 与 residual model Gate

- 构造权限清晰的低阶 Oracle、MuJoCo privileged diagnostic 与 identified baseline；
- 固定 dataset schema；
- 实现 structured excitation；
- 采集第一批数据；
- 建立 OOD split；
- 训练第一轮 MLP/ICODE-style model；
- 完成 residual Go/No-Go。

#### 第 3 周：RL prior Gate

- 冻结 RL observation/action/reward；
- 选择 PPO 或 SAC 中的一个；
- 训练 prior；
- 与 Goal/Previous prior 比较；
- 完成 25/50/100 samples smoke；
- 检查 reward hacking。

#### 第 4 周：联合方法 Gate

- 固定 residual 后联合 RL；
- 完成核心 2×2；
- 验证是否存在独立协同；
- 完成第一版 residual/RL confidence；
- 根据结果决定 dual-gating 的最终形式。

#### 第 5 周：Method Freeze

- 完成 residual gating；
- 完成 RL prior gating；
- 完成统一 uncertainty 或耦合机制；
- 冻结论文方法；
- 完成公平性检查；
- 录制第一版视频。

#### 第 6 周：正式仿真

- 完成五场景；
- 完成主要 dynamics shifts；
- 完成 sample budgets；
- 完成 paired 5–10 seeds；
- 输出主图与主表；
- 完成实验部分 v1。

#### 第 7 周：论文与实车证据

- 完整论文 v1；
- 实车 offline prediction；
- shadow mode；
- 整理失败案例；
- 根据老师意见补充关键消融；
- 开始匿名代码整理。

#### 第 8 周：最终证据

- 主要方法扩展 seeds；
- 完成统计；
- 安全和时间允许时进行低速 guarded rollout；
- 固化主图和主表；
- 完成论文 v2 与视频。

#### 最后 8 天

- 停止非关键开发；
- 仅修复阻断性 bug；
- 完成结果冻结；
- 提交老师审阅，并根据意见完成修改；
- 完成干净环境复现；
- 完成匿名性、格式和引用检查；
- 提前完成 portal 与格式预检，并于 9 月 14 日北京时间完成内部正式提交；
- 9 月 15 日仅用于紧急修复和最终状态确认。

---

## 十二、时间压缩与范围控制

### 12.1 并行推进

- 数据采集与论文写作并行；
- RL 环境准备与 residual 正式训练并行；
- 多 seed benchmark 与实车 offline 分析并行；
- 视频和匿名代码从 8 月中旬开始，不等待最后一周。

### 12.2 自动化

- 多进程 MuJoCo 数据采集；
- 自动生成 config snapshot；
- 自动记录 seed、Git SHA、model hash；
- 自动运行 multi-seed sweep；
- 自动汇总 CSV/JSON；
- 自动生成 figures/tables；
- 自动检查缺失运行和异常 seed。

### 12.3 提前冻结

- 7 月 27 日后原则上不增加 residual model 类型；
- 8 月 3 日后不更换 RL 主算法；
- 8 月 17 日后不改变核心方法定义；
- 8 月 24 日后不增加主场景；
- 9 月 8 日后不新增非关键实验。

### 12.4 主动裁剪

如时间不足，优先从主论文移除：

- Memory 与 RL/ICODE 的联合实验；
- 多种 RL 算法比较；
- 非核心 path-tracking 场景；
- 多轮在线持续学习；
- 非核心 UI 美化；
- 未实现的稳定性/收敛性理论声明。

以下内容优先保留：

- MPPI 正确性；
- MuJoCo/data label 正确性；
- Nominal、residual、RL-only、combined 与 privileged diagnostic；
- 多 seed 与计算时间；
- OOD/fallback；
- safety chain；
- 至少完整 MuJoCo 证据。

### 12.5 资源假设与资源不足时的裁剪顺序

以下资源尚需在第一周尽快确认，否则 700–1000 个正式 episodes 和多个 RL training seeds 的时间可行性无法可靠判断。

| 资源 | 初步假设 | 确认期限 |
|---|---|---|
| GPU | 至少 1 张可用于 PyTorch 训练的 CUDA GPU，建议 12 GB 以上显存 | 07-14 前确认型号、显存与可独占时段 |
| CPU | 至少 8–16 个可用逻辑核心 | 07-14 前确认 MuJoCo 并行进程上限 |
| 存储 | 至少预留 100 GB | 07-14 前确认数据、checkpoints、trajectories 与视频位置 |
| 并行环境 | 目标 4–8 个 MuJoCo workers | 07-14 前确认稳定性与实际吞吐 |
| 实车 | 8 月内至少 2–3 个可预约时段 | 07-20 前确认场地、安全人员、急停和日志接口 |
| 论文工具 | LaTeX、BibTeX、匿名仓库与视频存储 | 07-14 前确认模板与权限 |

WP1 完成后，将根据实测的单 episode mean/p95 时间重新计算总机时，并为数据采集、RL training、正式 benchmark 和失败重跑分别预留预算。

如果资源不足，计划依次裁剪：

1. 将最终 20 environment seeds 缩减为 10 paired seeds；
2. 删除 25 和 200 samples 两个非核心采样点，保留 50/100/400；
3. 删除非核心 path-tracking 和额外随机场景；
4. 将实车低速闭环缩减为 offline/shadow；
5. 减少非核心模型结构消融。

不因资源不足裁剪：

- MPPI 与数据标签正确性；
- residual-only、RL-only、fixed fusion 与 proposed method 的核心消融；
- 计算时间与 deadline 报告；
- OOD/fallback 验证；
- 结果可复现和科研诚信要求。

---

## 十三、投稿前内部验收

以下不是 ICRA 官方条件，而是为避免结论不足而设置的内部检查项。

### 13.1 方法正确性

- MPPI 数学与代码一致；
- command delay 和 applied control 正确；
- residual label 可追溯；
- 无 truth leakage；
- 所有模型使用公平数据预算。

### 13.2 科学证据

- Oracle/privileged diagnostic 有明确收益；
- ICODE-style model 多步预测优于 nominal；
- RL prior 在少 samples 下有明确收益；
- Combined 优于两个单独模块；
- Gating 在 OOD 下有明确收益；
- 结果在 paired multi-seed 下稳定。

### 13.3 工程证据

- p95/p99 计算时间被报告；
- deadline miss rate 被报告；
- safety intervention 被报告；
- nominal fallback 可运行；
- 实车结果如纳入主文，至少通过 offline/shadow。

### 13.4 复现证据

- 所有实验保存 config、seed、Git SHA；
- 图表由脚本自动重建；
- 正式结果与 smoke 结果分离；
- 匿名代码可从零运行主要流程；
- 失败案例不被删除。

---

## 十四、风险与预案

| 风险 | 可能影响 | 计划应对 |
|---|---|---|
| Oracle 无闭环收益 | Residual 主线不成立 | 回到 clean benchmark、cost 和 rollout 接口 |
| ICODE-style 不优于 MLP | 结构贡献不足 | 检查数据覆盖、encoding 和 control-affine 假设 |
| RL prior 无少样本收益 | RL 主线不足 | 简化 action representation、BC warm start、检查 reward |
| RL 与 residual 相互抵消 | 协同结论失败 | 固定训练顺序、on-policy data、gating |
| ICODE 推理过慢 | 无法实时 | 小网络、batch、TorchScript/ONNX、减少 samples |
| OOD 时 residual 错误 | 泛化风险 | uncertainty、alpha gating、nominal fallback |
| RL reward hacking | 结果不可信 | 行为诊断、失败视频、safety override 指标 |
| 感知成为主要瓶颈 | 难以观察动力学收益 | clean benchmark 与复杂场景分开 |
| 实车排期不足 | 缺少实车证据 | 第一周预约；优先 offline/shadow；按范围边界执行 |
| 实验数量过大 | 无法按期完成 | 先 3/5 seeds 筛选，再扩展最终方法 |
| 同期工作影响新颖性 | 贡献边界变弱 | 持续文献跟踪，及时调整 gating/shift 主线 |
| 论文篇幅不足 | 无法完整表达 | 从第一周按六页结构组织主结果 |

---

## 十五、固定截稿下的论文主张分级

投稿日期固定为 2026 年 9 月 15 日，但论文最终主张应与证据强度一致。

### Level A：完整目标

- Dual gating 成立；
- Combined 显著优于两个单独模块；
- Sample efficiency 成立；
- 双重 OOD 泛化成立；
- 实车 evidence 完整。

### Level B：核心协同成立

- Residual 与 RL 协同成立；
- 仅保留证据充分的 gating；
- 实车以 offline/shadow 为主或不作为核心 claim；
- 适当缩小理论和泛化主张。

### Level C：诚实缩域

- 如果 RL 协同不成立，则聚焦 uncertainty-gated residual MPPI；
- 如果 residual 不成立，则聚焦 RL-guided sample-efficient MPPI；
- 不将失败的组合包装为协同；
- 标题、摘要与贡献保持与最终证据一致。

我希望尽力完成 Level A，但会以实验事实为准，避免为了赶投稿而扩大不被数据支持的结论。

---

## 十六、预期论文结构与材料

### 16.1 正文

1. Introduction；
2. Related Work；
3. Problem Formulation；
4. Residual and Policy-Prior MPPI；
5. Uncertainty Gating；
6. Experimental Setup；
7. Results；
8. Limitations and Conclusion。

### 16.2 建议主图

- Figure 1：完整系统架构；
- Figure 2：Residual/RL dual gating；
- Figure 3：Prediction RMSE vs rollout horizon；
- Figure 4：Success vs sample budget；
- Figure 5：Scene shift × dynamics shift；
- Figure 6：MuJoCo 与可用的实车 evidence。

### 16.3 建议主表

- Table I：Prediction accuracy 与 inference time；
- Table II：2×2 独立协同消融；
- Table III：Gating 与 OOD；
- Table IV：实时性与可用的实车结果。

### 16.4 视频

- Nominal prediction failure；
- Oracle/privileged diagnostic；
- ICODE-style correction；
- Traditional vs RL prior；
- OOD gating/fallback；
- 多场景 MuJoCo；
- 可用的实车 offline/shadow/低速闭环；
- 失败案例。

---

## 十七、每日与每周管理方式

### 每日记录

- 完成内容；
- 当前阻塞问题；
- 新增/失败实验；
- 训练与 benchmark 队列；
- 是否影响关键节点；
- 次日最重要任务。

### 每三天更新

- 最好与最差结果；
- 主表进度；
- 关键图进度；
- 计算预算；
- 风险变化。

### 每周向老师汇报

1. 本周研究问题；
2. 本周 Gate；
3. 公平实验设置；
4. 正面结果；
5. 失败结果；
6. 当前结论是否仍然成立；
7. 下周计划；
8. 希望老师指导的事项。

---

## 十八、近期 72 小时行动计划

### 第一天

- P0：冻结研究问题、primary metrics 与 Related-Work Gate；
- P0：完成 MPPI 公式—代码审计清单；
- P0：建立 MuJoCo delay/applied-control 失败测试；
- P1：建立论文 LaTeX skeleton；
- P1：预约 GPU、实车和场地。

### 第二天

- P0：完成 strict MPPI 最小实现与手算测试；
- P0：修正 command delay 与 applied control 标签；
- P1：启动 clean dynamics benchmark；
- P1：建立自动实验目录与 provenance。

### 第三天

- P0：运行 nominal clean baseline；
- P0：运行低阶 Oracle 与 MuJoCo privileged diagnostic smoke；
- P0：整理第一轮 Foundation Gate 汇报；
- P1：固定 dataset schema；
- P1：启动 structured excitation collector。

P0 为当天优先完成的阻断性任务；P1 可并行启动，但不以牺牲 P0 的正确性为代价。

---

## 十九、恳请老师重点指导的事项

为了避免在有限时间内偏离主线，恳请老师重点指导以下问题：

1. 我的初步倾向是将 cross-layer horizon uncertainty 同时调节 residual trust 与 RL proposal trust 作为唯一核心方法，职责分解和实车迁移只作为系统设计与验证；恳请老师判断这一创新边界是否足够清晰。
2. 我的初步倾向是将 RL 严格限定为 MPPI sampling prior，并优先采用 PPO，以利用并行 MuJoCo、连续低维 control knots 和较成熟的训练流程；只有在环境、reward、观测和动作接口均已通过，而 48 小时优化仍无法形成稳定趋势时，才切换一次 SAC 并随即冻结，恳请老师判断该选择是否妥当。
3. 我的初步倾向是将论文定位在 robot learning 与 model-based control 的交叉，而不将其包装为新的全局导航算法；恳请老师指导更合适的投稿关键词和审稿方向。
4. 考虑到 66 天时间限制，我倾向于以完整 MuJoCo 多 seed/OOD/实时性作为刚性证据，实车 offline/shadow 作为优先加分项，低速闭环服从安全要求与预设停止条件；恳请老师判断实车证据的最低要求。
5. 如果 8 月 10 日前协同假设未获得充分支持，我倾向于保留证据更成熟、边界更清楚的 gated residual 主线，并将 RL 作为诚实消融；恳请老师判断该缩域顺序是否合理。

---

## 二十、总结

本计划希望形成如下证据链：

~~~text
MPPI数学正确
→ MuJoCo物理可信
→ 数据和residual标签可信
→ Oracle证明动力学修正有价值
→ ICODE-style residual改善多步预测
→ RL prior改善少样本搜索
→ 组合证明独立协同
→ Dual gating改善OOD退化
→ 实时预算得到验证
→ 多seed MuJoCo结果成立
→ 可用的实车证据成立
→ 论文、代码和结果可复现
~~~

我会按照 2026 年 9 月 15 日的固定截稿日期推进，并尽可能提前完成关键任务。同时，我也会以实验事实为准，谨慎控制论文主张，避免将尚未验证的设想表述为已经成立的结论。

以上是我目前拟定的研究计划。方案中可能仍有考虑不充分之处，尤其是创新点边界、RL 训练设计、uncertainty 方法和实车实验范围，恳请老师批评指正。
