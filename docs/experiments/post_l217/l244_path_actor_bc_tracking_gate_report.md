# L244 BC-Anchored Path Actor Tracking Gate 报告

日期：2026-07-21

性质：development / qualification，单一 development seed，不是论文确认性结果

## 1. 完整性与来源

- 完成 3 scenes × 2 arms = 6 个唯一 MuJoCo 回合；
- 新运行方法：`icode_mppi`、`full_proposed`；
- 场景：L239 `W=4.0D` 的 Hairpin、S-Chicane、Infinity；
- physics：`nominal_seen`；seed：`923301001`；qualification=`1`；
- Git SHA：`edf9e1a`；MuJoCo Python package：3.2.3；
- L243 selected Actor SHA256：
  `e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c`；
- 每回合 rollout budget 均为 100：Full 使用 50 candidates × 2 iterations，ICODE 使用
  一次 100-sample batch；
- 每回合均存在 resolved config、trajectory、metrics 与 provenance；
- Windows 日志未发现 Traceback、Exception、NaN 或 Inf。

只读 L242 Full 使用相同 seed、地图、物理、cost、安全链和 rollout 预算。L244 模型只由独立
validation 字典序选择，L234/L239/L242 outcome 未参与 checkpoint 选择。

## 2. 结果

| scene | method | success | collision | completion | cross-track RMSE | boundary steps | min boundary margin (m) | safety interventions | proposal authority | guided elites | planner mean (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hairpin | L242 Full | 0 | 0 | 0.1077 | 0.4222 | 0 | 0.0215 | 111 | 0.0922 | 259 | 508.5 |
| Hairpin | L244 BC Full | 0 | 0 | **0.3422** | 0.4333 | 0 | 0.0072 | 26 | 0.1759 | 30 | 390.5 |
| Hairpin | L244 ICODE | 0 | 0 | 0.1118 | 0.0784 | 0 | 0.5630 | 442 | — | 0 | 131.3 |
| S-Chicane | L242 Full | 0 | 0 | 0.1758 | 0.2527 | 1 | -0.0057 | 27 | 0.0837 | 239 | 437.4 |
| S-Chicane | L244 BC Full | 0 | 0 | **0.1877** | 0.4117 | 1 | -0.0027 | 90 | 0.0755 | 25 | 400.0 |
| S-Chicane | L244 ICODE | 0 | 0 | 0.1827 | 0.1288 | 0 | 0.5618 | 437 | — | 0 | 129.9 |
| Infinity | L242 Full | 0 | 0 | 0.1025 | 0.3083 | 0 | 0.0998 | 166 | 0.0610 | 110 | 494.8 |
| Infinity | L244 BC Full | 0 | 0 | **0.1055** | 0.2823 | 1 | -0.0072 | 85 | 0.0961 | 20 | 391.0 |
| Infinity | L244 ICODE | 0 | 0 | 0.1084 | 0.0883 | 0 | 0.5683 | 416 | — | 0 | 128.1 |

汇总：

| method | mean completion | successes | collisions | boundary steps | safety interventions | planner mean (ms) |
|---|---:|---:|---:|---:|---:|---:|
| L242 selected-initial Full | 0.1287 | 0/3 | 0/3 | 1 | 304 | 480.2 |
| L244 BC-anchored Full | **0.2118** | 0/3 | 0/3 | **2** | 201 | 393.8 |
| L244 ICODE-MPPI | 0.1343 | 0/3 | 0/3 | 0 | 1,295 | 129.8 |

L244 Full 相对 L242 Full 的 completion 变化为：

- Hairpin：`+0.2345`；
- S-Chicane：`+0.0119`；
- Infinity：`+0.0030`；
- 三场景平均：`+0.0832`。

proposal authority 平均为 0.1158，guided elites 合计 75，因此 RL prior 被非平凡使用。Full
平均 planner 时间较 L242 Full 减少约 86.4 ms/step，但仍比 ICODE 高约 264.1 ms/step。

## 3. 严格 Gate 判定

**L244 Gate 未全部通过。**

已通过：

- 三场景零碰撞；
- 平均 completion 增益 `+0.0832 >= +0.02`；
- 无任何场景回退；
- RL proposal authority 与 guided-elite contribution 非平凡。

未通过：

- boundary violation 总步数从 L242 的 1 增为 L244 的 2；
- S-Chicane 和 Infinity 的最小 footprint boundary margin 分别为 -2.7 mm 与 -7.2 mm；
- 三场景均未在 700 步内到达终点，且平均 cross-track RMSE 没有同步改善。

因此当前证据支持的准确表述是：

> BC anchor 明显提高了 residual-conditioned Actor 在 Hairpin 上的闭环推进能力，并使三场景
> 平均完成度显著高于旧 Actor 与 ICODE；但该收益尚未满足预注册的边界安全约束，不能登记为
> 全面通过，也不能直接进入 Tracking sealed benchmark。

## 4. 下一项单变量开发

负边界裕度只有毫米量级，而核心 Actor 能力已出现明确增益。下一项 L245 只把所有候选共同使用的
`path_boundary_buffer` 从 0.05 m 增至 0.075 m；Actor、ICODE、HSS、MPPI cost 权重、地图、
rollout budget 和安全链保持不变。该调整是增强规划边界预留，不是放松安全条件。

L245 必须在同一 development seed 上先验证零碰撞、零 boundary violation，并检查 L244 的
completion 增益是否保留；不得使用 sealed seeds，也不得覆盖本报告中的失败。

## 5. 工件

- 原始结果：`results/research_platform/rl/tracking_l244_bc_anchor_*_seed923301001/`；
- 只读 L242 对照：`results/research_platform/rl/tracking_l242_selected_actor_*_seed923301001/`；
- 数值表：`docs/experiments/post_l217/tables/l244_tracking_gate_summary.csv`；
- 图：`docs/experiments/post_l217/figures/fig_l244_tracking_gate.pdf`；
- 预注册协议：`docs/experiments/post_l217/l244_path_actor_bc_tracking_gate_protocol.md`。
