# L49/L50 结果：任务数据显著改善预测，控制平滑性在强失配下稳定获益

日期：2026-07-16

## 结论先行

L49/L50 得到了一个明确但有边界的正向结论：

1. 使用独立路径跟踪数据迭代训练后，3/3 ICODE 模型在 held-out test 和
   unseen-domain 两个 split 上均显著改善 H=36 多步预测；
2. 在更强的 `combined_long_delay` 物理失配域，ICODE 同时显著降低了规划命令
   jerk 和 MuJoCo 实际施加控制 jerk；
3. 在较弱的 `combined_matched_delay` 域，收益接近零，说明 learned residual
   不应被假设为在所有简单/低失配条件下都有同等价值；
4. L50 的全域联合 gate 仍然失败，主要因为 path-length 的分层 95% 区间跨零，
   且 issued-control jerk 的全域区间下界略低于零。因此不能声称 ICODE 已在所有
   路径和物理域上全面优于 nominal MPPI。

这组结果支持后续“按在线失配证据选择启用残差”的研究路线，而不支持“残差始终
全强度开启”的宽泛主张。

## L49：任务定向数据与训练

数据来自 L48 中 `traditional_nominal` 控制器实际执行的 MuJoCo transitions，使用
plant interval-average applied control 作为训练控制语义。数据按 episode/seed/physics
domain 划分，训练归一化统计只由 train split 计算：

- train：36 episodes，5626 transitions；
- validation：6 episodes，914 transitions；
- test：6 episodes，910 transitions；
- unseen-domain：12 episodes，2221 transitions。

三个模型使用相同配置但不同初始化 seed（20261101、20261102、20261103）。残差
输出继续受结构 mask 约束，只学习动态状态 `[v, omega]` 的修正；位置和航向导数
仍由 nominal kinematics 决定。训练采用 H=36、RK4 和多步 rollout loss，并由
validation multi-step RMSE 选择 checkpoint。

### 冻结离线门槛

L49 的门槛要求：

- 3/3 checkpoint 在 test 和 unseen 上均降低 H=36 总体 rollout RMSE；
- 每个 split 至少 2/3 checkpoint 同时降低 position 与 heading RMSE；
- 所有输出有限且 artifact 完整。

实际结果为 3/3 checkpoint 在两个 split 的总体、位置和航向指标全部改善：

| Seed | Split | Rollout reduction | Position reduction | Heading reduction |
|---:|---|---:|---:|---:|
| 20261101 | test | 49.71% | 64.86% | 69.14% |
| 20261101 | unseen | 50.45% | 68.79% | 72.07% |
| 20261102 | test | 49.83% | 65.82% | 70.26% |
| 20261102 | unseen | 50.90% | 70.31% | 73.70% |
| 20261103 | test | 49.61% | 65.71% | 69.92% |
| 20261103 | unseen | 50.62% | 70.24% | 72.74% |

因此 L49 离线 gate 通过。该结论只证明 frozen trajectories 上的预测改进，不能直接
替代闭环控制证据。

## L50：独立闭环确认设计

L50 在读取结果前冻结，并完整继承 L48 的严格门槛，没有降低阈值：

- 3 independent ICODE initialisation blocks；
- 3 paths：gentle S、double turn、slalom；
- 2 physics domains：matched delay、long delay；
- 10 个全新 paired seeds：21460731–21460740；
- 2 methods：nominal 与 ICODE；
- 共 360 episodes、180 paired comparisons；
- sealed seeds 21460741–21460750 未使用；
- nominal/ICODE 使用相同 MPPI budget、路径、物理、感知与安全仲裁设置；
- memory 和 RL prior 均关闭，避免混杂。

artifact audit 完整：360/360 episodes，0 missing，0 unexpected，未使用 protected 或
sealed seed，无非有限指标。

## L50 全域联合结果

| Endpoint | Mean reduction | Hierarchical 95% CI | Interpretation |
|---|---:|---:|---|
| Path length | 0.01247 m | [-0.01730, 0.04160] | 不显著 |
| Issued-control jerk | 0.000885 | [-0.000028, 0.001875] | 下界略跨零 |
| Applied-control jerk | 0.000732 | [0.000077, 0.001505] | 显著改善 |

补充安全/任务指标：

- success difference：0；
- collision increase：0；
- completion-ratio difference：+0.000264；
- cross-track RMSE 相对增加：4.20%，仍在预注册的 5% non-inferiority 界内；
- ICODE 平均 planner compute：40.43 ms，小于 50 ms 门槛；
- 3/3 model blocks 的平均 path-length reduction 为正；
- 3/3 model blocks 的平均 issued-control jerk reduction 为正。

由于 path-length 均值未达到 0.03 m、其区间跨零，且 issued-control jerk 全域区间
下界略小于零，`confirmation_gate.passed = false`。不得用 applied jerk 的单项显著性
把联合 gate 改写为通过。

## 预先定义物理域上的失败归因

L50 使用的两个物理域在配置中已冻结。分域分析表明效果与 mismatch severity 有明确
交互：

| Physics domain | Issued jerk reduction (95% CI) | Applied jerk reduction (95% CI) |
|---|---:|---:|
| Matched delay | 0.000576 [-0.000371, 0.001660] | 0.000183 [-0.000235, 0.000640] |
| Long delay | 0.001194 [0.000065, 0.002344] | 0.001282 [0.000264, 0.002365] |

在 long-delay 域，两项 jerk 指标区间均严格大于零，而且三个 model blocks 的
path-length mean 均为正（0.04045、0.01936、0.03120 m），success/collision 无退化。
在 matched-delay 域，均值较小且区间跨零。

这不是证明“delay 越大 ICODE 必然越好”；它是当前冻结场景中的经验交互，需要用
新的 seed 和更多 mismatch 类型确认。但它给出了一个可证伪的下一步：在线 gate
应根据最近 transitions 中 nominal-vs-residual 的预测证据决定残差强度，在低失配
时退化为 nominal，在高失配时启用 ICODE。

## 下一步研究决策

下一阶段不再盲目增加训练 epoch，也不通过删掉表现不好的路径制造正结果。应先实现
和测试一个因果在线 residual-reliability gate：

1. 只用已经发生的 transition 计算 nominal 与 ICODE one-step innovation；
2. 使用带遗忘因子的证据累计和冷启动期；
3. 当残差没有可靠改善预测时令 residual scale 接近 0；
4. 当残差持续改善预测时平滑提高 scale；
5. gate 决策不得访问未来状态、场景标签或 ground-truth domain name；
6. 先做 oracle-domain upper-bound ablation，再做在线 gate，最后用新的 sealed seeds
   独立确认。

该路线与导师提出的“简单环境下减少 learned module 作用、复杂/失配区域再启用”的
建议一致，同时保持 story 聚焦在技术方法和可验证 ablation，而不是增加新的理论保证。

## 可复现 artifact

- L49 offline summary：
  `results/research_platform/l49_icode_iterative_path_offline_gate_v1/summary.json`
- L50 paired effects：
  `results/research_platform/rl/l50_icode_iterative_path_confirmation_20260716_v1/efficiency_paired_effects.csv`
- L50 complete summary：
  `results/research_platform/rl/l50_icode_iterative_path_confirmation_20260716_v1/efficiency_confirmation_summary.json`
- Figure：
  `results/research_platform/rl/l50_icode_iterative_path_confirmation_20260716_v1/figures/l49_l50_prediction_and_control.{pdf,png}`

所有结果均来自实际运行；没有生成或补写正式实验数据。
