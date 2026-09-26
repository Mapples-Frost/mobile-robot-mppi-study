# L27：零复杂度门控的行为保持推理 Fast-path 结果

日期：2026-07-15  
性质：预注册开发集实验；不是封存测试集结论。

## 1. 一句话结论

L27 **通过全部预注册开发 Gate**：当外层复杂度门控已经能够严格判定 learned contribution 最终为零时，跳过 actor、target twin critics 和 decoder，可在不改变任何控制行为的前提下，将 clean 场景平均 planner 时间降低 23.6%，将三个 blocking 场景合并后的平均 planner 时间降低 9.4%。

这支持的是一个有限但重要的工程结论：

> 门控不仅可以决定 RL prior 是否参与控制，还可以在其贡献被严格置零时避免无效神经网络推理，从而降低计算开销。

L27 不改变 L26 的原始失败判定，也不能据此声称当前方法已经达到端到端计算高效。

## 2. Fast-path 的适用条件

只有以下条件同时成立时，才允许提前返回 conventional prior：

1. gate mode 为 scene complexity；
2. complexity、near-goal 或 safety 条件已经严格保证最终 gate alpha 为零；
3. `learn_covariance=false`，即神经网络不会继续影响采样协方差；
4. 使用者显式启用 `skip_zero_complexity_inference`。

不满足上述条件时，程序仍执行原来的 actor、target twin critics 和 decoder。该优化没有修改策略参数、门控阈值、MPPI 参数、场景或安全仲裁逻辑。

## 3. 实验设计与完整性

- 条件：`standard_complexity` 与 `zero_complexity_fastpath`；
- 固定 MPPI 样本数：`K=100`；
- 场景：clean、single obstacle、narrow corridor、U-trap；
- 独立 checkpoint seeds：`20260721/22/23`；
- 新开发 episode seeds：`20281901–20281910`；
- 每个 scene-condition 单元：30 episodes；
- 总计：240 episodes、26,208 个配对控制步；
- 期望 episodes：240；实际 episodes：240；
- 重复 episode 主键：0；
- L25/L26/L27 protected seeds 使用数：0；
- 条件在 checkpoint block 内随机化，并在相同 checkpoint-scene-episode seed 内配对。

## 4. 行为等价性

行为等价是首要 Gate，任何非零差异都会使 L27 失败。

| 检查项 | 结果 |
|---|---:|
| 配对控制步 | 26,208 |
| executed `v` 最大绝对差 | 0 |
| executed `omega` 最大绝对差 | 0 |
| goal distance 最大绝对差 | 0 |
| gate alpha 最大绝对差 | 0 |
| step 行为不一致 | 0 |
| step 长度不一致组 | 0 |
| episode success 不一致 | 0 |
| episode collision 不一致 | 0 |

因此，当前证据支持该 fast-path 是行为保持优化，而不是近似推理或控制策略改动。

## 5. 推理跳过率与规划耗时

| 场景角色 | Standard (ms/step) | Fast-path (ms/step) | 时间下降 | 下降的 95% CI | Fast-path skip fraction |
|---|---:|---:|---:|---:|---:|
| clean control | 8.870 | 6.777 | 23.6% | [19.9%, 27.1%] | 1.000 |
| blocking（3 场景合并） | 8.594 | 7.784 | 9.4% | [7.6%, 11.5%] | 0.316 |

置信区间使用 20,000 次配对两阶段 bootstrap：先重采样三个独立 checkpoint，再在 checkpoint 内重采样 episode seed；同一 episode 内的多个 scene/step 不被错误视为独立训练重复。

clean 场景的 skip fraction 为 1，是因为复杂度门控始终不需要 RL prior。blocking 场景的 skip fraction 为 0.316，说明 fast-path 只在门控确实为零的控制步触发；其余约 68.4% 的步骤仍正常执行 learned inference。

## 6. 预注册 Gate

| 检查项 | 阈值 | 结果 |
|---|---:|---:|
| 240/240 完整实验 | 必须完整 | 通过 |
| protected seeds 未使用 | 0 | 通过 |
| step 控制行为差异 | 0 | 通过：0 |
| episode success/collision 差异 | 0 | 通过：0/0 |
| clean skip fraction | ≥0.99 | 通过：1.000 |
| blocking skip fraction | ≥0.10 | 通过：0.316 |
| clean planner 时间下降 | ≥15% | 通过：23.6% |
| blocking planner 时间下降 | ≥5% | 通过：9.4% |

总判定：**L27 development Gate passed**。

## 7. 与 L26 的关系

L26 已经证明 learned sampling prior 在 `K=100` 时能够在三个阻塞场景中取得高于 traditional MPPI `K=400` 的成功率，但其原始实现的 planner 时间比例为 0.945，未达到预注册的 `≤0.60`。

L27 解释并修复了其中一部分固定推理开销，但：

- L27 是相同 `K=100` 下优化前后的独立实验；
- 它没有重新运行 L26 的完整跨 K 因子设计；
- 它不允许事后覆盖 L26 的失败 Gate；
- 当前 9.4% 的 blocking 加速仍不足以支持“样本减少四倍，墙钟时间也近似减少四倍”的表述。

因此目前最严谨的结论仍是：**样本质量优势已经得到较强开发集证据，计算效率有所改善，但端到端计算效率仍需继续优化和独立验证。**

## 8. 当前限制

1. 结论来自开发 seeds，封存测试 seeds 仍未打开；
2. 规划耗时为同一平台上的墙钟指标，仍可能受系统调度影响；
3. 只有三个独立训练 checkpoint，置信区间采用层级重采样但不能替代更多独立训练；
4. 只验证了 `learn_covariance=false`；学习协方差时 fast-path 被明确禁止；
5. 尚未比较 TorchScript、ONNX 或批处理推理；
6. 尚未在实车计算平台上验证延迟与抖动。

## 9. 产物

- `development_gate.json`：预注册 Gate 与行为等价检查；
- `audit.json`：完整性和 protected seed 审计；
- `timing_analysis.json`：20,000 次层级 bootstrap 结果；
- `timing_pairs.csv`：checkpoint-scene-episode 配对耗时；
- `episodes.csv`、`gate_steps.csv`：episode 与逐步原始结果；
- `fig_l27_fastpath_timing.pdf/png`：论文格式图。

![L27 behavior-preserving fast-path timing](../../results/research_platform/rl/l27_zero_complexity_fastpath_development_multiseed_20260715_v1/fig_l27_fastpath_timing.png)

## 10. 下一步

下一阶段优先分析 active-gate 控制步中的固定 actor/critic 开销，并在任何实现改动前预注册新的行为等价与延迟 Gate。候选路线包括 TorchScript/推理图优化、减少重复 critic 计算和批处理；不得通过降低 critic 数量、放宽 gate 或改变采样分布来伪造加速。
