# ICRA 联合 K/H 与公平 RL-H 评价结论（2026-09-10）

> 审核修订：下文“机制证据”“精度相近”和停止实验的叙事建议不能视为已证实结论。当前是两类固定路径上的性能证据，尚未隔离复杂度的因果作用或证明 K 维度超过 H-only 的增量价值。区间跨零不证明等效；尚不能认定达到 ICRA 投稿标准。260 回合没有 K-only，该对照来自此前 210 回合，不能合并成同一批配对结果。8 个 909130x spec 只能证明这些文件当前的配置，仍需追踪实际评价使用的 12901001/13101001/13101002 checkpoint 配置及训练来源，才能完成 compute-price 来源审计。后续应优先补这个溯源检查和有界的增量机制实验，而不是只在发现程序错误时才做实验。

## 研究问题

本阶段检验的不是“任何场景都优于所有基线”，而是一个更窄、可证伪的问题：在计算受限的移动机器人 MPPI 中，学习策略是否能够根据路径复杂度在采样数 K 与预测时域 H 之间进行联合预算分配，从而改善计算量与跟踪误差的折中。评价保留了固定预算、H-only、K-only 和 RL-H 对照，避免把单轴调节或更快但更不准确的名义模型误认为联合方法收益。

## 评价设计与审计

- 路径：`single_turn` 与更复杂的 `reverse_turns`。
- 新场景：每条路径 5 个共享场景种子（13620001--13620005）。
- 策略初始化：joint、H-only、residual RL-H、nominal RL-H 各 3 个初始化；固定对照使用相同场景。
- 方法总数：26；总回合数：260；成功完成 260/260；碰撞 0 次；物理周期 71,846。
- 分析单位：先在每个场景内平均 3 个策略初始化，再跨 5 个场景汇总。不能把 3 个初始化乘 5 个场景当成 15 个独立场景。
- 原 joint SAC：预先固定的独立 spec 使用 `compute_price=0.05`、`hold=5`、固定 K/H=`128/32`，8 个 spec（H-only、K-only、joint、masked，两个初始化）均保持该值。
- 公平 RL-H：nominal/residual 各 3 个初始化，均重新训练超过 10,000 个物理周期；训练审计通过奖励重构、19D 观测与动作映射、物理延迟、checkpoint/resume、有限 SAC 更新和源码 hash 检查。

## 全部均值

单位：RMSE 为 mm，compute 为每回合累计规划计算秒数；均值已经在场景内平均初始化。

| 路径 | 方法 | RMSE | compute | mean K | mean H |
|---|---|---:|---:|---:|---:|
| single_turn | joint | 10.21 | 3.19 | 120.1 | 24.9 |
| single_turn | H-only | 9.65 | 3.49 | 128.0 | 27.0 |
| single_turn | residual K128/H32 | 9.41 | 3.98 | 128.0 | 32.0 |
| single_turn | nominal K128/H32 | 15.24 | 1.00 | 128.0 | 32.0 |
| single_turn | residual RL-H | 10.36 | 3.21 | 100.0 | 29.4 |
| single_turn | residual RL-H fixed H16 | 17.15 | 2.02 | 100.0 | 16.0 |
| single_turn | residual RL-H fixed H32 | 11.54 | 3.29 | 100.0 | 32.0 |
| single_turn | nominal RL-H | 14.68 | 1.19 | 100.0 | 28.8 |
| single_turn | nominal RL-H fixed H16 | 18.46 | 0.87 | 100.0 | 16.0 |
| single_turn | nominal RL-H fixed H32 | 15.04 | 1.11 | 100.0 | 32.0 |
| reverse_turns | joint | 12.49 | 4.82 | 151.8 | 22.8 |
| reverse_turns | H-only | 12.99 | 4.63 | 128.0 | 22.7 |
| reverse_turns | residual K128/H32 | 16.37 | 5.93 | 128.0 | 32.0 |
| reverse_turns | nominal K128/H32 | 21.54 | 1.48 | 128.0 | 32.0 |
| reverse_turns | residual RL-H | 16.68 | 4.77 | 100.0 | 29.4 |
| reverse_turns | residual RL-H fixed H16 | 19.70 | 3.08 | 100.0 | 16.0 |
| reverse_turns | residual RL-H fixed H32 | 16.86 | 4.92 | 100.0 | 32.0 |
| reverse_turns | nominal RL-H | 21.54 | 1.87 | 100.0 | 29.0 |
| reverse_turns | nominal RL-H fixed H16 | 22.17 | 1.34 | 100.0 | 16.0 |
| reverse_turns | nominal RL-H fixed H32 | 20.94 | 1.70 | 100.0 | 32.0 |

## 关键配对证据

差值定义为候选方法减去对照方法；RMSE 和 compute 均为负时表示候选更好。

### Joint 对固定 residual K128/H32

| 路径 | RMSE 差值 | 95% CI | compute 差值 | 95% CI | 五场景共同更好 |
|---|---:|---|---:|---|---:|
| single_turn | +0.80 mm | [-0.89, 2.49] | -0.79 s | [-0.93, -0.65] | 1/5 |
| reverse_turns | -3.88 mm | [-5.50, -2.27] | -1.11 s | [-1.30, -0.93] | 5/5 |

`reverse_turns` 的逐场景差值（joint - fixed residual K128/H32）为：

| 场景种子 | RMSE mm | compute s |
|---:|---:|---:|
| 13620001 | -5.66 | -1.17 |
| 13620002 | -3.30 | -0.87 |
| 13620003 | -4.60 | -1.16 |
| 13620004 | -2.25 | -1.27 |
| 13620005 | -3.60 | -1.09 |

### Joint 对 H-only

`single_turn` 的平均差值为 RMSE +0.55 mm、compute -0.30 s；`reverse_turns` 为 RMSE -0.50 mm、compute +0.19 s。两个路径的 3 个 joint 初始化方向均不一致，且 95% CI 对 RMSE 都跨零。因此不能写成 joint 普遍优于 H-only；更准确的表述是联合策略提供了路径相关的折中，复杂路径中更偏向精度，简单路径中更偏向计算节省。

### Joint 对公平 residual RL-H

在 `reverse_turns` 中，joint 的 RMSE 比 residual RL-H 低 4.20 mm，95% CI 为 [-4.69, -3.70]；compute 差值 +0.05 s，95% CI 为 [-0.20, 0.30]。5/5 场景 joint 更准确，3/5 场景同时 Pareto 更优。这个结果支持“在相同物理延迟执行接口下，joint 在复杂路径上取得更好的跟踪精度且计算近似相同”，不支持“严格相同优化目标下击败 RL-H”。

### RL-H 自身的 horizon 诊断

Residual RL-H 相对同一 value 网络的 fixed H32，在 `reverse_turns` 中 RMSE 平均改善约 0.18 mm、compute 平均减少约 0.16 s，区间跨零，初始化方向不完全一致。相对 fixed H16，三次初始化都显著改善 RMSE，但计算量增加。这说明局部 H 自适应确实改变了精度/计算折中，但当前样本不足以证明稳定的全局优势。Nominal RL-H 的收益更不稳定。

## 统计解释

每条路径只有 5 个独立新场景，因此双侧精确 Wilcoxon 的最小可达 p 值为 0.0625。现有 95% CI 是场景配对差值均值的 Student-t 区间（自由度 4），不是 bootstrap 区间；它依赖差值分布假设，且不覆盖重新训练新策略的全部不确定性。报告以效应量、逐场景方向和初始化一致性为主。两类路径之间的差异是路径相关证据，不能单独识别复杂度的因果作用；区间跨零也不证明等效或非劣。

## 可以写进论文的主张

1. 本文提出复杂度感知的联合 K/H 预算分配，用同一决策接口动态改变采样数和预测时域。
2. 在复杂的反向连续转弯路径中，联合策略相对强固定 residual K128/H32 同时降低跟踪 RMSE 约 3.88 mm、减少计算约 1.11 s，5 个共享新场景均观察到同方向改进。
3. 相对公平重训的 residual RL-H，联合策略在该复杂路径上 RMSE 低约 4.20 mm，而计算时间近似相同。
4. 在单转弯路径中，联合策略主要表现为约 0.79 s 的计算节省，跟踪误差与固定 residual 基线相近；这构成路径复杂度改变收益形态的边界条件。
5. 所有 260 个评价回合均无碰撞，但这只能报告为本测试集观察结果，不能声称已经证明安全鲁棒性。

## 不能写的主张

- 不能说 joint 普遍优于 H-only；该比较在两个路径和三个初始化下都不稳定。
- 不能说动力学描述本身带来稳定收益；nominal 更快但误差明显更高，masked/ residual 的方向随路径和初始化变化。
- 不能说 RL-H 是文献作者代码的精确复现；本实验是重新训练的 matched-readiness 执行接口对照。
- 不能把 RL-H 的局部奖励与 joint 的 `compute_price=0.05` 奖励称为完全 reward-matched。
- 不能把 5 个场景乘 3 个初始化当作 n=15 的独立场景，也不能用零碰撞宣称安全性。
- 不能从这两类路径外推普适鲁棒性；需要在论文中把场景范围和小样本限制明确写出。

## 投稿叙事与停止规则

主线应集中在“complexity-aware joint K/H budget allocation for compute-tracking trade-offs”。`reverse_turns` 是主机制结果，`single_turn` 是边界条件，RL-H 是公平执行接口对照和局部 horizon 诊断。现有证据已经足以形成一个可复现的实验故事；继续做大规模 K/H 网格不会增加主要论证强度。下一步工作的高价值顺序是：冻结上述数字，生成论文主图与方法示意，写实验与限制，逐条核对原始 JSON/审计 hash；只有在发现数据或接口错误时才启动唯一的确认性实验。

## 数据与代码位置

- 评价原始数据与审计：`research_artifacts/icode_sac_compute_2026-09-08/rl_h_ready_evaluation_2026-09-10/`
- 评价分析：`research_artifacts/icode_sac_compute_2026-09-08/rl_h_ready_evaluation_2026-09-10/analysis/context_level_results.json`
- RL-H 训练审计：`research_artifacts/icode_sac_compute_2026-09-08/rl_h_ready_training_2026-09-10/training_audit.json`
- 训练协议：`docs/protocols/icra_rl_h_ready_training_2026-09-10.md`
- 评价协议：`docs/protocols/icra_rl_h_ready_evaluation_2026-09-10.md`
