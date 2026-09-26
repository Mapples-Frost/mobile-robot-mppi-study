# L28：Active-gate 推理剖析与行为保持优化预注册

日期：2026-07-15  
性质：L26 计算 Gate 失败、L27 零贡献 fast-path 通过后的独立开发实验；运行前冻结。

## 1. 问题

L26 已支持 complexity-gated RL prior 的样本效率，但 `K=100` 相对 traditional `K=400` 的 planner 时间比例为 0.945。L27 在最终 gate alpha 严格为零时跳过全部 learned inference，并保持 26,208 个配对控制步完全相同，但 blocking 场景仍有约 68.4% 的控制步需要 active learned inference。

代码审计发现，active correction gate 使用 `critic_source=target`，但当前实现仍计算：

1. online critic 1/2 对 base 与 candidate 的四次前向传播；
2. target critic 1/2 对 base 与 candidate 的四次前向传播；
3. actor 内已经得到 frozen-BC base action 后，advantage filter 又重新计算一次 base actor。

其中 online critics 只生成诊断字段，不参与 target-LCB gate；重复 base actor 也不改变数学结果。

## 2. 预注册优化因子

固定 `K=100`，比较三种条件：

| 条件 | 未选 critic 诊断 | 重用 actor 已计算的 base action |
|---|---|---|
| `reference_full_diagnostics` | 计算 online + target | 否 |
| `selected_critic_only` | 只计算 gate 选择的 target critics | 否 |
| `selected_critic_base_reuse` | 只计算 target critics | 是 |

所有条件均启用 L27 的零复杂度 fast-path。优化条件不修改 actor/critic 权重、target-LCB 公式、beta、complexity thresholds、MPPI 参数、场景或 safety arbitration。

未选 online-critic 数值在优化条件中标记为“未计算”，不得用零值冒充实际 critic 输出。论文性能主实验不依赖这些诊断；需要 critic 研究时保留 reference 模式。

## 3. 设计

- 独立训练 checkpoint：`20260721/22/23`；
- 场景：clean、single obstacle、narrow corridor、U-trap；
- 新开发 episode seeds：`20282001–20282010`；
- L28 sealed seeds：`20282011–20282030`；
- 条件数：3；固定 `K=100`；
- 总计划：3 checkpoints × 4 scenes × 10 seeds × 3 conditions = 360 episodes；
- checkpoint-scene-episode 内使用相同随机种子配对；
- 每个 checkpoint block 内随机化 scene-condition-episode 运行顺序；
- L25/L26/L27/L28 sealed seeds 全部禁止使用。

独立重复单位是训练 checkpoint；episode seed 嵌套于 checkpoint。逐控制步只用于行为等价和组件耗时描述，不当作独立训练重复。

## 4. 组件计时

Prior 内记录：

- fallback prior；
- observation encode + normalize；
- goal/scene-complexity features；
- actor；
- advantage critics；
- decoder；
- outer gate；
- prior total。

MPPI 内记录：

- state/reference；
- prior；
- sampling；
- batch rollout；
- trajectory/control cost；
- weighting/update；
- final nominal rollout；
- planner total。

组件计时只用于定位与相对比较。三个条件都启用相同 profiler，避免把计时开销只加到某一条件。

## 5. 首要 Gate：行为完全相同

每个优化条件分别与 reference 配对，要求：

1. 每一步 executed `v/omega`、goal distance、outer gate alpha 最大绝对差为 0；
2. selected target consensus-LCB、correction gate alpha 的判定不变；
3. collision、safety override 和轨迹长度不一致数为 0；
4. episode success/collision 不一致数为 0；
5. 所有结果有限，组合完整，无重复主键。

任何控制或安全差异都使对应优化失败，不以“误差很小”为理由放宽。

## 6. 计算 Gate

仅在行为 Gate 通过后评估：

- `selected_critic_only` active prior 时间下降至少 15%；
- `selected_critic_base_reuse` active prior 时间下降至少 20%；
- combined 优化在 blocking 场景的总 planner 时间下降至少 5%；
- paired blocking planner 时间下降的 20,000 次两阶段 bootstrap 95% CI 下界大于 0。

两阶段 bootstrap 先重采样 checkpoint，再在 checkpoint 内重采样 episode seed。报告均值、95% CI、逐组件占比和逐 checkpoint 结果，不把 step 当成独立样本做显著性检验。

## 7. 后续决策

- combined 条件通过全部 Gate：冻结为 L29 候选实现，在全新 seeds 上重新运行 `K=50/100/200/400`；
- selected-only 通过、base reuse 不通过行为或时间 Gate：只保留 selected-only；
- 两者均无稳定收益：保留 L27，停止这条优化，不扫描阈值；
- sealed seeds 无论结果如何都不自动打开。

L28 不能事后改写 L26 的失败结论。只有 L29 新的完整跨 K 实验才能重新评估端到端计算效率。
