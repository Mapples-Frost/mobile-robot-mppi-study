# L214 最终论文级 Point-goal Benchmark 预注册

日期：2026-07-19  
状态：在读取 L214 正式 seeds 结果前冻结  
配置：`configs/research/final_paper_benchmark_point_goal_l214.yaml`

## 1. 目的与证据层级

本实验不再用于挑选结构或调节超参数。目标是用未参与方法开发的随机种子，回答两个不同层级的问题：

1. **工程性能问题**：最终方法相对传统 MPPI、ICODE-MPPI、RL-driven MPPI 和简单组合是否改善闭环控制；
2. **机制归因问题**：价值对齐 ICODE 与 reliability-adaptive hybrid sampling（HSS）分别贡献了什么，二者交互是协同、相加还是次加性。

七个方法为：

| Arm | ICODE | RL guidance | value-aligned ICODE | adaptive HSS | 角色 |
|---|---:|---:|---:|---:|---|
| Traditional MPPI | 否 | 否 | 否 | 否 | 上下文基线 |
| ICODE-MPPI | 是 | 否 | 否 | 否 | 上下文基线 |
| RL-driven MPPI | 否 | 是 | 否 | 否 | 上下文基线 |
| Simple combination | 是 | 是 | 否 | 否 | 2×2 对照 |
| Value fixed | 是 | 是 | 是 | 否 | 价值对齐单因素 |
| Ordinary adaptive | 是 | 是 | 否 | 是 | HSS 单因素 |
| Full proposed | 是 | 是 | 是 | 是 | 完整方法 |

前三个方法用于解释实际收益来源；后四个方法构成主要的随机完整区组 2×2 因子实验。不得把前三个上下文比较误写成同一个 2×2 因果设计。

## 2. 冻结因素

- 任务：`clean_single_obstacle` point-goal，目标仍由场景配置给出；
- 物理域：`nominal_seen`、`long_delay_seen`、`combined_unseen`；
- 每个控制周期总 rollout 预算：100；
- RL-driven optimizer iterations：2；
- 最大闭环步数：300；
- HSS terminal guidance radius：1.26 m；
- HSS guided-fraction floor：0.30；
- Memory-Augmented MPPI：关闭；
- Actor、ordinary ICODE ensemble、value-aligned ICODE ensemble 和 reliability calibration 的路径均写入冻结 YAML；
- 正式 seed：101–110；schedule seed：20260719。

正式运行开始后，不得根据中间结果改变以上值。任何后来改变均建立新的实验编号，并作为探索性结果标注。

## 3. 实验单位、区组与随机化

独立实验单位是 **simulation seed**。每个 seed 在相同场景、相同物理域下运行全部七个方法；scene × physics-domain 是该 seed 内的重复分层，不是额外独立样本。

每个 seed × scene × physics-domain 构成一个完整区组，七个方法的执行次序由冻结的 schedule seed 随机打乱。统计 bootstrap 只重采样 seed cluster，绝不重采样控制 timestep 或把同一 seed 下的多个域当成独立样本。

这避免两类伪重复：

- 把数百个控制周期当成数百次独立试验；
- 把同一个 seed 下的 seen/unseen 物理域当成互不相关的独立试验。

## 4. 指标与判定顺序

### 4.1 共同主要指标

1. success rate（越高越好）；
2. final goal distance（越低越好）。

### 4.2 安全约束指标

- collision rate 不得升高；
- minimum clearance 不得出现不可接受退化；
- stuck steps、spin steps 不得系统性恶化。

### 4.3 效率与平滑性指标

- mean / p95 planner compute time；
- control jerk；
- 实际 rollout 预算审计。

所有指标报告 seed-cluster paired effect、95% percentile bootstrap CI 和每个 seed 的差异。success 是配对成功率差，不把单个 episode 当独立 Bernoulli 样本。

## 5. 预先规定的比较

### Confirmatory comparisons

1. Full Proposed vs Simple Combination；
2. Value Fixed vs Simple Combination；
3. Ordinary Adaptive vs Simple Combination；
4. 2×2 中 value-alignment main effect；
5. 2×2 中 adaptive-HSS main effect；
6. value-alignment × adaptive-HSS interaction。

### Contextual secondary comparisons

1. ICODE-MPPI vs Traditional MPPI；
2. RL-driven MPPI vs Traditional MPPI；
3. Simple Combination vs Traditional / ICODE-only / RL-only；
4. Full Proposed vs ICODE-only / RL-only。

正文不以“所有次要比较均显著”为成功条件，也不因某个次要指标不显著而重新调参。

## 6. 可证伪条件

以下任一结果都必须如实报告：

- Full Proposed 在 final goal distance 上不优于 Simple Combination；
- 任一耦合因素在主要指标上为稳定负向；
- success 提升来自碰撞、安全间距或 jerk 的明显牺牲；
- nominal 域提升、unseen 域退化，说明泛化主张不成立；
- interaction 为负，说明两个模块不能宣称超加性“协同”。

如果 interaction 继续呈次加性，论文表述限定为“两种互补耦合机制均有效，但收益存在重叠”，不得使用 super-additive synergy。

## 7. 样本量依据与停止规则

L211 五个开发外确认 seed 中，Full vs Simple 的 final-distance seed-level paired effect 为：均值 0.18343 m、标准差 0.04936 m、配对 Cohen's dz = 3.72。为降低小样本 winner's curse，规划时仅采用其一半，即 dz = 1.86；同时将 dz = 1.0 作为更保守敏感性边界。

最终计划固定为 10 个未用于开发的正式 seeds。对两侧配对检验，n=10 对 dz≈1.0 约提供 80% 量级功效，对收缩后的 dz≈1.86 则明显高于该水平。由于 success 为离散且交互项通常需要更大样本，本研究不以单一 p 值作为证据；重点报告 seed-cluster effect 与置信区间。若 10 seeds 的 CI 较宽，可追加一个**预先编号的 replication cohort**，但首批结果不得被删除或与开发数据混合。

停止规则：必须完成 seeds 101–110；不得在看到 2、5 或 8 个 seed 的中间趋势后提前停止。程序崩溃仅重跑同一 arm/seed/block，不替换 seed。

## 8. Qualification 与正式数据边界

在正式运行前允许使用不属于 101–110 的 seed 做 pipeline qualification，只验证：

- 七方法均能启动并产生完整 schema；
- 每个区组的七方法恰好各出现一次；
- provenance、checkpoint SHA256、schedule、trajectory 和 metrics 可追踪；
- 2×2 分析只纳入四个 confirmatory arms；
- nominal / ICODE / RL 开关没有串扰。

Qualification 数据永不用于论文效果估计。通过后先提交代码、配置和本预注册，再启动正式 seeds。

## 9. 后续外部有效性实验

L214 只冻结 point-goal clean dynamics 主实验。复杂静态障碍、动态障碍和 polyline path-tracking 使用独立编号与独立预注册；它们验证外部有效性，但不会反过来修改 L214 配置。实车 offline 与 shadow-mode 结果也单独报告，scan_guard 始终具有最高控制权限。
