# L74 封存确认结果

日期：2026-07-17  
状态：封存 seeds 已消费；主要门槛未通过。

## 1. 完整性

- 3 个模型块；
- 每块 3 个场景 × 5 个封存 seeds × 2 个锁定条件；
- 90/90 闭环 episodes；
- 45 组 BC—30k SAC 配对；
- episode 矩阵完整，无重复、缺失或 NaN/Inf；
- 冻结 BC correction gate alpha 严格为 0；
- SAC checkpoint 未重新选择；
- 零新增碰撞。

产物：

- `l74_deployment_selection_summary.json`：主要门槛；
- `l74_confirmation_descriptive.json`：预声明的描述性次要指标；
- `l74_confirmation_pairs.csv`：45 组配对数据；
- `l74_confirmation_by_block.csv`；
- `l74_confirmation_by_scene.csv`；
- 每个 episode 的 metrics、trajectory、checkpoint hash 与随机调度。

## 2. 主要门槛

L74 要求每个合格模型块的成功损失为 0，且至少 2/3 模型块合格。

| 模型块 | BC 成功 / 15 | SAC 成功 / 15 | gains | losses | 净成功变化 | 新增碰撞 | 平均终点距离改善 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 13 | 12 | 1 | 2 | -1 | 0 | +0.1432 m |
| 1 | 14 | 14 | 1 | 1 | 0 | 0 | +0.0069 m |
| 2 | 9 | 11 | 3 | 1 | +2 | 0 | +0.4514 m |

三个块均出现至少一次成功损失，因此 0/3 模型块满足严格非劣门槛。L74 **主要门槛失败**。

这否定的是“锁定的 30k SAC 在每个模型块上都能做到零成功回退”的强假设，不是否定所有 RL 修正价值。

## 3. 汇总描述

45 个配对 episodes 中：

- BC：36/45 成功；
- SAC：37/45 成功；
- gains：5；
- losses：4；
- 净成功变化：+1；
- 平均终点距离改善：+0.2005 m；
- 平均 minimum clearance 变化：+0.0161 m；
- 新增碰撞：0；
- mean planner compute time 变化：+0.142 ms；
- mean stuck steps 变化：-20.13；
- mean safety interventions 变化：-14.89；
- mean control jerk 变化：-0.00073。

总体方向偏正，但净成功增益很小，且模型块间不稳定，不能用总体平均掩盖主要门槛失败。

## 4. 分场景结果

| 场景 | BC 成功 / 15 | SAC 成功 / 15 | gains | losses | 净成功变化 | 平均距离改善 | stuck steps 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean single obstacle | 13 | 13 | 1 | 1 | 0 | -0.0044 m | -10.47 |
| narrow corridor | 8 | 11 | 4 | 1 | +3 | +0.6055 m | -44.80 |
| U-trap long board | 15 | 13 | 0 | 2 | -2 | +0.0004 m | -5.13 |

最明确的正向信号来自窄通道：RL 降低了卡住现象并将多个失败轨迹转为成功。简单单障碍场景总体中性。U-trap 中 BC 已达到 15/15，常开 RL 反而造成两个成功损失。

U-trap 的两个 SAC 失败最终距离分别约为 0.3197 m 与 0.3127 m，仅略高于 0.30 m 成功阈值。这一事实解释了二元成功指标的敏感性，但不改变预注册失败判定。

另一个较大的成功损失发生在模型块 2 的 clean single obstacle（SAC 最终距离约 2.512 m），说明不能将所有 losses 都归因于阈值边界。

## 5. 科研解释

L74 不支持“RL prior 在所有静态场景常开且严格不回退”的版本。数据更支持一个更窄、更符合导师建议的技术主张：

> 传统 MPPI/BC 在简单或已饱和场景保持主导；RL correction 只在可识别的困难、受限或卡住风险高的状态下启用。

这与窄通道的正向增益、U-trap 饱和基线的负迁移同时一致。后续研究重点应从“继续增加 SAC 训练轮数”转向“学习或校准 competence-aware activation”，即识别何时 RL 有增益、何时应退化到传统 MPPI/BC。

## 6. 后续规则

1. `22170811`–`22170815` 已消费，不得用于后续调参或模型选择。
2. 不回到 L74 结果上重选 5k–25k checkpoint。
3. 下一阶段使用全新开发 seeds 研究按场景状态/卡住风险触发的 RL activation。
4. 新 gate 必须先在开发数据上预注册，再使用新的封存确认 seeds。
5. 动态障碍、感知噪声和实车结果仍需单独验证。

## 7. 解释边界

episode 级总体数据是描述性的；同一训练块中的多个 episode 不是独立训练重复。当前只有 3 个模型块，不声称统计显著性、理论稳定性或所有环境泛化。结果支持继续研究“选择性 RL”，而不是支持“无条件常开 RL”。
