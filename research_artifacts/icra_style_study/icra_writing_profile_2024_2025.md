# ICRA 2024–2025 写作画像：供本项目论文初稿使用

## 1. 研究对象与证据边界

本画像学习的是 ICRA 已录用论文原文，而不是二手报告本身。

- 录用论文母集：DBLP 收录的 ICRA 2024–2025 论文 3,366 篇。
- 全量摘要层：3,336 篇有摘要，覆盖率 99.1%。所有摘要均纳入程序化语言统计。
- Introduction 层：673 篇可从 arXiv/ar5iv 合法解析的 Introduction，覆盖率 20.0%。该层存在明显的开放获取偏差，不能冒充全体论文。
- 领域近邻层：逐篇阅读 16 篇涉及 MPPI/path-integral control 的摘要，并逐段精读 14 篇导航、规划控制、强化学习、SLAM、触觉操作、HRI 与机器人设计论文的 Introduction。
- 结论性质：这里能识别“已录用论文的常见写作惯例”和“较强论文的论证做法”，不能在没有拒稿语料和审稿意见的情况下宣称这些特征因果性地代表审稿人偏好。

语料与审计文件位于 `tmp/icra_style_corpus/data/`。母集由 DOI 对齐；Introduction 主要来自 OpenAlex 标出的 arXiv 版本。自动标题补配结果为空，因此没有把模糊题名匹配计入 Introduction 覆盖率。

## 2. 全体摘要的语言轮廓

### 2.1 长度与节奏

- 摘要中位数为 170 词、8 句，平均每句约 22.7 词。
- 归一化到五个位置后，平均句长依次约为 21.7、23.3、23.9、23.3、21.2 词。常见节奏是：开头和结尾较短，中间的方法解释较密。
- 不宜把 170 词当作硬模板；应优先保持一条可追踪的论证链，通常落在 7–9 句。

### 2.2 作者主体与动词

- 89.0% 的摘要使用 `we/our`。ICRA 并不回避主动语态。
- 高频作者动作依次包括 `we propose`、`we introduce`、`we present`、`we demonstrate`、`we show`、`we evaluate` 和 `we validate`。
- 写作时应让动词与证据类型对应：提出方法用 `propose/introduce`，执行实验用 `evaluate/compare`，得到可复核结果用 `show/demonstrate`。不要用 `prove` 描述经验结果。

### 2.3 对比、缺口与证据

- 48.9% 的摘要含有明确的 challenge/limitation/failure 词汇。
- `however` 出现在 33.6%，`while/although` 出现在 31.6%。高质量用法不是形式上的转折，而是指出已有方法为何在目标工况下失效。
- 85.1% 的摘要在最后两句出现实验、评估、验证、改进或超越基线等证据动作。
- 19.5% 给出百分比、倍数等直接数值结果。数字不是必选项，但只要主要结论稳定且口径清楚，定量结果比形容词更有说服力。
- 31.6% 明确提及真实硬件/真实世界，24.0% 提及仿真。不能把仅有仿真的工作写成“real-world ready”。

### 2.4 开头方式

粗分类显示：55.3% 先陈述任务或领域事实，19.1% 直接以问题/挑战开场，17.1% 直接进入方法，8.5% 从重要性或应用价值切入。

适合本项目的开头不是宽泛地说“mobile robots are increasingly important”，而是尽快给出运行条件与控制矛盾，例如：采样预算、模型失配、动态障碍、实时性、安全与机动性的冲突。

## 3. Introduction 的结构画像

### 3.1 尺度

- 673 篇 Introduction 的中位数约 577 词、6 段、25 句，平均每句约 23.6 词。
- 中位引用标记数为 9。引用承担的是界定方法族和支撑事实，不应把 Introduction 写成小型 Related Work。
- 明确的 paper-organization 段只占约 4.6%；篇幅紧张时可以省略。

### 3.2 最稳定的论证链

1. 具体机器人任务及其目标工况。
2. 现有方法族提供了什么能力。
3. 在目标工况下出现什么可解释的失效机制。
4. 本文组件如何一一对应这些失效机制。
5. 用什么实验、对照和指标验证每项主张。
6. 必要时列贡献；是否使用项目符号取决于贡献是否真正可分离。

强 Introduction 的关键不是段落数量，而是“问题—机制—设计—证据”的可追踪性。每个方法组件都必须回答一个已经写清楚的问题；每个贡献都必须能在实验表格或图中找到对应证据。

### 3.3 语气校准

- 97.0% 的 Introduction 使用 `we/our`，主动表达是常态。
- 76.5% 使用 `however`，81.6% 包含 challenge/limitation 等词，说明差异化通常通过边界条件和失败机制建立。
- 58.4% 使用某种限定或缓和表达，84.8% 同时使用展示、验证、有效、稳健、显著等增强表达。成熟写法不是一味保守，而是让主张强度跟证据强度匹配。
- `novel` 在 47.8% 的开放 Introduction 中出现，但高频不等于高质量。除非新颖性能够由清楚的差异和相关工作支持，否则优先直接说设计差异。
- `to our knowledge` 类首创声明只应在系统检索后使用。

## 4. MPPI 近邻语料的专门画像

16 篇 2024–2025 MPPI/path-integral 相关摘要的中位长度为 160.5 词，平均每句约 22.9 词。

- 68.8% 明确写出挑战或限制。
- 50.0% 包含硬件/真实世界验证，43.8% 包含仿真。
- 25.0% 给出百分比或倍数结果。
- 100% 至少包含一个 evidence/booster 动作，但强论文会把增强词绑定到具体指标、平台和基线。

这一子领域最常见、也最适合本项目的摘要逻辑为：

1. 标准 MPPI 或现有混合控制方法在某个明确条件下受限。
2. 解释限制来自采样分布、模型失配、多模态解、噪声、约束或实时预算中的哪一种机制。
3. 给出一个可命名的干预，并说明它改变控制链的哪个位置。
4. 说明比较对象、场景类型和验证层级。
5. 报告主要收益，同时明确代价或未改善项。

领域近邻论文的标题常采用“方法名 + 解决的功能问题”，而不是把所有组件缩写塞进标题。摘要中也很少需要完整贡献清单；更常见的是在连续叙述中完成 gap、method 和 evidence 三次移动。

## 5. 本项目后续写作的硬约束

以下约束优先于任何通用 ICRA 句式：

- 不能沿用旧叙事，把 RL、Change-Aware 与 ICODE 写成已经被完整验证的跨层协同机制。
- L217 支持的是 HSS 在静态复杂导航中的效率贡献，不足以支持完整跨层因果链。
- Dynamic Stage 4/5 的 RL–HSS gates 失败，不能隐藏或改写成正结果。
- 正式 single-obstacle v4 支持碰撞减少与 clearance 增益，不支持 success-rate 提升，并暴露安全—机动性权衡。
- “robust”“generalizable”“real-time”“safe”均需绑定评估域、指标和运行条件。若只在特定场景成立，应使用 `under the evaluated conditions`、`in the tested scenes` 或同等级限定。
- 负结果和权衡不是写作缺陷。对本项目而言，准确表述安全收益、效率收益及其边界，比制造统一胜利叙事更接近高质量 ICRA 论文。

## 6. 拟采用的 Abstract 骨架

建议用 7–8 句完成，而非照抄固定句式：

1. 一句界定任务和目标工况。
2. 一句指出现有方法的具体不足。
3. 一句解释造成不足的机制。
4. 一句给出本文总体方法及核心干预位置。
5. 一句解释最关键组件为何对应上述机制。
6. 一句交代实验域、基线、预算和指标。
7. 一句给出最可靠的主要定量结果。
8. 如有必要，一句坦率说明权衡、边界或资源开放情况。

应避免的写法：连续三句都以 `We propose` 开头；把 `novel/robust/efficient` 当作证据；未给出比较口径就说 `significantly outperforms`; 在结果不支持时声称 success、generalization 或完整机制得到验证。

## 7. 拟采用的 Introduction 骨架

- 第 1 段：从目标运行条件切入，而不是从机器人学的宏大价值切入。
- 第 2 段：组织最相关的两到三个方法族，说明各自能力和适用前提。
- 第 3 段：把本项目观察到的失败或瓶颈写成机制性 gap，并明确其评估后果。
- 第 4 段：介绍方法总览，按“组件—问题”对应关系展开。
- 第 5 段：列出贡献，每条同时包含对象、差异、证据和边界。
- 第 6 段可选：用一句话概括主要结果或论文结构；若无信息增量则省略。

## 8. 逐段精读样本

与本项目最接近：

- *Improving the Generalization of Unseen Crowd Behaviors for Reinforcement Learning based Local Motion Planners* (ICRA 2024, arXiv:2410.12232)
- *Learning Barrier-Certified Polynomial Dynamical Systems for Obstacle Avoidance with Robots* (ICRA 2024, arXiv:2403.08178)
- *Stranger Danger! Identifying and Avoiding Unpredictable Pedestrians in RL-based Social Robot Navigation* (ICRA 2024, arXiv:2407.06056)
- *ADMM-MCBF-LCA: A Layered Control Architecture for Safe Real-Time Navigation* (ICRA 2025, arXiv:2503.02208)
- *Actor-Critic Cooperative Compensation to Model Predictive Control for Off-Road Autonomous Vehicles Under Unknown Dynamics* (ICRA 2025, arXiv:2503.00577)
- *Safe Multi-Agent Navigation Guided by Goal-Conditioned Safe Reinforcement Learning* (ICRA 2025, arXiv:2502.17813)
- *TANGO: Traversability-Aware Navigation with Local Metric Control for Topological Goals* (ICRA 2025, arXiv:2509.08699)
- *Exploring Adversarial Obstacle Attacks in Search-Based Path Planning for Autonomous Mobile Robots* (ICRA 2025, arXiv:2504.06154)

跨子领域校准：

- *NPC: Neural Predictive Control for Fuel-Efficient Autonomous Trucks* (ICRA 2024, arXiv:2412.13618)
- *Globally Stable Neural Imitation Policies* (ICRA 2024, arXiv:2403.04118)
- *RGB-Only Gaussian Splatting SLAM for Unbounded Outdoor Scenes* (ICRA 2025, arXiv:2502.15633)
- *Tactile-Informed Action Primitives Mitigate Jamming in Dense Clutter* (ICRA 2024, arXiv:2402.09564)
- *Gaze-based Human-Robot Interaction System for Infrastructure Inspections* (ICRA 2024, arXiv:2403.08061)
- *A Bio-Inspired Sand-Rolling Robot: Effect of Body Shape on Sand Rolling Performance* (ICRA 2025, arXiv:2503.13919)

这些样本并非全部都是语言范文。精读目的同时包括识别有效结构和排除录用论文中仍然存在的语法错误、重复贡献、过强首创声明及空泛形容词。

## 9. 后续初稿的执行标准

写每一段前先写出该段唯一功能；写完后逐句标记为 context、gap、mechanism、method、evidence 或 boundary。若一句无法归类，通常应删除或移至 Related Work。整篇初稿完成后执行三次独立审计：

1. 证据审计：每个主张对应哪项实验、指标和置信范围。
2. 边界审计：哪些域、场景、预算或对照尚未覆盖。
3. 语体审计：主动动词、句长节奏、转折逻辑、术语一致性和不必要的夸张。

最终目标不是写得“像所有 ICRA 论文的平均值”，而是在 ICRA 熟悉的论证结构内，以本项目真实证据为中心，写出更清楚、更克制、也更容易核验的论文。
