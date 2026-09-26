# L26：场景复杂度门控 RL–MPPI 的采样效率结果

日期：2026-07-15  
性质：预注册开发集结果；不是封存测试集结论。

## 1. 一句话结论

Complexity-Gated RL 在三个局部阻塞场景中用 `K=100` 明显超过 traditional MPPI 的 `K=400`，并且没有碰撞或成功损失；但预注册的总 Gate 仍判定为失败，因为当前 CPU actor/critic 推理抵消了大部分 MPPI 采样节省，墙钟时间比例 `0.945` 未达到事先要求的 `≤0.60`。

因此本轮支持的是：

> learned sampling prior 显著提高了候选轨迹的样本质量，但当前实现尚未把样本效率等比例转化为计算效率。

不得把本轮写成“低 K 同时更准且显著更快”。封存种子继续关闭。

## 2. 数据完整性

- 三个独立训练 checkpoint：`20260721/22/23`；
- 十个新开发 episode seeds：`20281801–20281810`；
- 四个场景、两种方法、四个 `K`；
- 期望 episode：960；实际 episode：960；
- 控制 step：250,676；
- 重复主键：0；缺失组合：0；非有限指标：0；
- L25 sealed seeds 使用数：0；
- L26 sealed seeds 使用数：0；
- traditional 和 gated 的碰撞数均为 0。

空旷场景中，在四个 K、30 个 checkpoint–episode 配对上共比较 9,108 个控制 step。`v`、`omega`、goal distance、collision、safety override 的最大差异全部为 0，gate alpha 最大绝对值也为 0。

## 3. 成功率

每个单元为三个 checkpoint × 十个 episode seeds，即 30 个 episode。

| 场景 | 方法 | K=50 | K=100 | K=200 | K=400 |
|---|---|---:|---:|---:|---:|
| clean dynamics | Traditional | 30/30 | 30/30 | 30/30 | 30/30 |
| clean dynamics | Gated RL | 30/30 | 30/30 | 30/30 | 30/30 |
| single obstacle | Traditional | 0/30 | 0/30 | 0/30 | 6/30 |
| single obstacle | Gated RL | 7/30 | 20/30 | 29/30 | 29/30 |
| narrow corridor | Traditional | 0/30 | 0/30 | 0/30 | 0/30 |
| narrow corridor | Gated RL | 18/30 | 19/30 | 21/30 | 12/30 |
| U-trap | Traditional | 0/30 | 0/30 | 0/30 | 0/30 |
| U-trap | Gated RL | 16/30 | 20/30 | 17/30 | 10/30 |

三个阻塞场景合并后：

| K | Traditional | Gated RL |
|---:|---:|---:|
| 50 | 0/90 | 41/90 |
| 100 | 0/90 | 59/90 |
| 200 | 0/90 | 67/90 |
| 400 | 6/90 | 51/90 |

`K=50/100` 的预注册低预算集合中，gated 相对 traditional 共增加 100 次成功、损失 0 次，净增益 100，远高于预注册阈值 18。

## 4. 配对统计

成功率采用 checkpoint–scene–episode–K 内配对；每个 scene–K 使用精确 McNemar 检验，并对全部 16 个 scene–K 比较进行 Holm 校正。效应区间使用 20,000 次两阶段层级 bootstrap：先重采样 checkpoint，再在 checkpoint 内重采样 episode seed。控制 step 不作为独立样本。

除 single-obstacle `K=50` 外，所有阻塞场景比较在 Holm 校正后均达到 `p<0.05`。该例的配对变化为 `+7/−0`，效应为 `+0.233`，层级 bootstrap 95% CI `[0.067, 0.433]`，但 Holm-adjusted exact `p=0.078`，因此按预定多重比较规则不称为显著。

重要配对效应：

| 场景 | K | 成功增加/损失 | 成功率差 | 95% 层级 bootstrap CI | Holm p |
|---|---:|---:|---:|---:|---:|
| single obstacle | 100 | +20/−0 | +0.667 | [0.433, 0.867] | <0.001 |
| narrow corridor | 100 | +19/−0 | +0.633 | [0.400, 0.833] | <0.001 |
| U-trap | 100 | +20/−0 | +0.667 | [0.433, 0.867] | <0.001 |
| single obstacle | 400 | +23/−0 | +0.767 | [0.600, 0.900] | <0.001 |
| narrow corridor | 400 | +12/−0 | +0.400 | [0.200, 0.600] | 0.003 |
| U-trap | 400 | +10/−0 | +0.333 | [0.133, 0.567] | 0.012 |

## 5. `Gated K=100` 与 `Traditional K=400`

这是 L26 最直接的样本效率比较：

| 场景 | 配对成功率差 | 95% CI | planner 时间比例 |
|---|---:|---:|---:|
| single obstacle | +0.467 | [0.200, 0.733] | 0.982 |
| narrow corridor | +0.633 | [0.400, 0.833] | 0.931 |
| U-trap | +0.667 | [0.433, 0.867] | 0.923 |

三个场景都满足“Gated `K=100` 不劣于 Traditional `K=400` 超过 5 个百分点”的预注册成功率条件，而且实际上都显著更好。

但三场景合并的平均 planner 时间为：

- Gated `K=100`：8.663 ms/step；
- Traditional `K=400`：9.168 ms/step；
- 比例：0.945。

因此只节省约 5.5% 墙钟时间，而不是预注册要求的至少 40%。神经 actor 与 target twin critics 的固定推理开销抵消了大部分采样减少。

## 6. 非单调 K 现象

Gated RL 的合并成功率从 `K=50` 的 0.456 上升至 `K=200` 的 0.744，但在 `K=400` 降至 0.567。下降主要来自 narrow corridor 和 U-trap。

这是事后描述性观察，不改变 L26 Gate。它说明“更多随机样本必然让当前闭环更好”并不成立；MPPI 样本数还会与 prior、温度、噪声、代价地形和安全仲裁共同作用。后续若研究这一点，必须单独预注册，不能用本轮数据事后调温度或噪声。

## 7. 预注册 Gate

| 检查项 | 结果 |
|---|---|
| 960/960 完整全因子 | 通过 |
| 无重复、无缺失、无 NaN/Inf | 通过 |
| sealed seeds 未使用 | 通过 |
| 空旷逐控制步精确回退 | 通过 |
| 碰撞回归为 0 | 通过 |
| 低预算净成功增益 ≥18 | 通过：100 |
| K100 至少两场景匹配 Traditional K400 | 通过：3/3 |
| K100/K400 planner 时间比例 ≤0.60 | **失败：0.945** |

总判定：**L26 development Gate failed**。L25/L26 的 sealed test seeds 继续关闭。

## 8. 下一步

先做行为保持的计算优化，而不是重新调 Gate：

1. complexity alpha 必然为零时，跳过 actor、target critics 和 decoder；
2. 用逐 step common-seed 回归证明优化前后控制完全相同；
3. 单独比较优化前后 planner 时间，不能覆盖 L26 的原始失败结果；
4. 若轻量 fast-path 仍不足，再评估 TorchScript/ONNX 或独立批处理推理；
5. 只有预注册的计算优化实验通过，才讨论一次性封存测试。

## 9. 产物

- `results/research_platform/rl/l26_sample_efficiency_development_multiseed_20260715_v1/analysis.json`
- `paired_inference.csv`
- `cross_budget_comparisons.csv`
- `bootstrap_curves.csv`
- `fig_l26_scene_success_vs_k.pdf/png`
- `fig_l26_success_compute_tradeoff.pdf/png`

![L26 scene success curves](../../results/research_platform/rl/l26_sample_efficiency_development_multiseed_20260715_v1/fig_l26_scene_success_vs_k.png)

![L26 success and compute tradeoff](../../results/research_platform/rl/l26_sample_efficiency_development_multiseed_20260715_v1/fig_l26_success_compute_tradeoff.png)
