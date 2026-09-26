# L54/L55 实际效应审计与高动态固定物理基准转向（2026-07-16）

## 1. 为什么不继续堆同类 seeds

L54 与 L55 的完整性均通过，但两轮预注册主要 Gate 均失败。按实际量级审计：

| Round | Endpoint | Nominal mean | Gated reduction | Relative reduction |
|---|---|---:|---:|---:|
| L54 | issued jerk | 0.09734 | 0.000267 | 0.275% |
| L54 | applied jerk | 0.07535 | 0.000250 | 0.332% |
| L54 | path length | 3.8347 m | 0.04029 m | 1.051% |
| L55 | issued jerk | 0.09740 | 0.000406 | 0.417% |
| L55 | applied jerk | 0.07593 | 0.000398 | 0.524% |
| L55 | path length | 3.7876 m | 0.00616 m | 0.163% |

将 5% jerk reduction 作为会影响系统选择的保守最小实际效应（SESOI）时，当前点效应
低一个数量级。L55 applied-jerk reduction 的 95% CI 上界为 0.000767，仍仅约 nominal
均值的 1.01%，已排除 5% 量级改善。此时继续增加同类 seeds 主要会使“效应很小”的结论
更精确，而不是产生有竞争力的控制贡献。

这不是 post-hoc observed power。按统计功效规范，本审计使用原始效应及置信区间判断
SESOI，并进行设计灵敏度判断；没有把已观测 p 值重新换算为“功效”。

## 2. 小效应的机制原因

当前实验同时具备：

- 低速上限约 0.35 m/s；
- 平滑、无障碍路径；
- MPPI 已知 command delay，并在 rollout 中显式做区间平均补偿；
- 强安全与重规划机制；
- ICODE 只修正 `[v, omega]`，不改变几何运动学。

因此 baseline 本身已很强，残差即使把 H=36 预测 RMSE 降低约 50%，也只会带来很小的
闭环动作差异。离线预测改善不能自动推出有实际意义的控制改善。

## 3. 新基准的研究问题

下一阶段建立**固定小车物理、隐藏动力学失配、高动态、无感知混杂**的路径跟踪层：

- 同一个 MuJoCo diff-drive plant 固定质量、惯量、轮地摩擦和执行器参数；
- nominal predictor 保持原 dynamic-unicycle 时间常数，不读取这些真实 plant 参数；
- command delay 仍显式建模，避免把已知延迟冒充 learned residual 贡献；
- 提高速度、角速度、加减速和曲率需求，使质量/摩擦/执行器滞后真正影响闭环；
- 无障碍场景先隔离动力学贡献，scan guard 与安全链仍保持开启；
- 按完整 episode 的 `(scene, seed, physics domain)` 单元分割数据；
- 留出一条完整未见路径作为 unseen-scene test，不随机拆 timestep；
- 只在固定 plant 上训练，符合“同一辆真实小车采集数据后迁移到新任务”的实际故事。

## 4. 进入条件

在训练 ICODE 前必须先验证：

1. nominal 在全部路径上无碰撞且至少大部分成功，避免把全局规划失败混入动力学问题；
2. nominal 的速度/角速度一步与多步预测误差显著高于低动态基准；
3. 路径仍有明确可行走廊，且不依赖障碍物真值作为 planner 输入；
4. 动作和状态覆盖足以训练 `[v, omega]` residual；
5. 训练/验证/测试/unseen-scene episode 严格不重叠。

若高动态 nominal 本身完全失败，应降低速度或曲率直到形成可比较而非退化的 baseline；
若 nominal 与 plant 仍几乎一致，则不应训练网络，应继续修正 benchmark 的动力学激励。

