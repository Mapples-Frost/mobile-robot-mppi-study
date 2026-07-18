# L83 covariance oracle 结果与 L84 covariance-only SAC 训练预注册

日期：2026-07-17  
结论：**L83 training-upgrade Gate passed；允许进入 L84**

## 1. L83 完整结果

有效工件：

`results/research_platform/rl/l83_covariance_oracle_pilot_20260717_v2/`

`v1` 因 terminal-phase 语义缺失主动中止并保留为 aborted 工件；`v2` 补齐现有 MPPI
terminal constraints 后完整运行。没有改变 anchor、尺度、候选数、场景或模型块。

`v2` 包含：

- 3 个冻结 ICODE 模型块；
- 4 个 seen/unseen 高动态场景；
- 每场景 4 个固定 anchor；
- 每 anchor 5 个冻结协方差动作；
- 48 个物理状态、240 条 ICODE-MPPI weighted sequence 的真实 MuJoCo rollout；
- 无障碍、无 collision、memory/RL 关闭；
- 所有 cost、weights、ESS、hash 和 snapshot 完整有限。

## 2. 主要数值

| 指标 | 结果 |
|---|---:|
| baseline scale 平均真实成本 | `35.403712` |
| best fixed scale | `broad=(1.75,1.75)` |
| best fixed 平均真实成本 | `26.036877` |
| best fixed 相对 baseline 改善 | `9.366834`（`26.46%`） |
| context oracle 相对 best fixed 额外改善 | `1.077164`（约 `4.14%`） |
| baseline 相对 context oracle regret | `10.443998` |
| unseen context oracle improvement | `1.134168` |
| positive model blocks | `3/3` |

各 block 的 context oracle improvement：

| block | improvement |
|---:|---:|
| 0 | `1.037670` |
| 1 | `1.338455` |
| 2 | `0.855367` |

最优尺度分布：

- `speed_explore=(1.75,0.75)`：29/48，`60.42%`；
- `broad=(1.75,1.75)`：19/48，`39.58%`；
- baseline/narrow/turn_explore：0/48。

四个场景的 context gain 均为正：

| 场景 | context gain |
|---|---:|
| accel_turn | `0.835281` |
| chicane | `1.205037` |
| hairpin_unseen | `0.846198` |
| reverse_s_unseen | `1.422138` |

## 3. 严格解释

L83 证明的不是“RL 已经提升控制”，而是：

1. 当前固定 `noise_sigma` 明显偏保守；
2. 单纯改为 fixed broad 已能获得大部分收益，必须作为强基线；
3. 最优尺度随状态在 `speed_explore` 与 `broad` 间切换；
4. 一个状态相关 selector 相对最佳固定尺度仍有约 4.14% 的 oracle 空间；
5. 该空间跨 3 个 ICODE checkpoints 和 unseen 路径保持正方向。

因此使用 RL 的必要性不是“把方差调大”，而是学习**何时只扩大速度探索，何时同时扩大转向探索**。
若训练后策略退化成恒定 broad，则只能报告为固定 MPPI 调参，不构成 RL 贡献。

## 4. L84 方法

新增 `covariance_only` prior parameterization：

- actor action dimension 固定为 2；
- actor 不输出 mean/control knots/subgoal；
- traditional + previous-sequence mean 逐元素保持；
- actor 输出经对数映射到 `[0.5,2.0]`；
- actor 零输出对应 scale `(1.0,1.0)`；
- gate alpha 为 0 时 covariance 精确退化为 `planner.noise_sigma`；
- ICODE checkpoint 固定，不与 RL 同时训练；
- MPPI、LaserScan、scan_guard、safety arbitration 不变。

## 5. L84 独立训练块

| block | SAC seed | ICODE checkpoint |
|---:|---:|---|
| 0 | `20268801` | L57 seed `20261201` |
| 1 | `20268802` | L57 seed `20261202` |
| 2 | `20268803` | L57 seed `20261203` |

每块 20,000 environment steps；checkpoint 间隔 2,500 steps。训练 seen paths：

- accel_straight；
- accel_turn；
- chicane；
- sweep。

validation paths：accel_turn、chicane、hairpin_unseen、reverse_s_unseen。训练初期不加入障碍、memory、
physics randomization 或 residual fine-tuning，以隔离 covariance learning。

## 6. 奖励与观测

观测使用 3-frame history、relative local target、(v,\omega)、previous action 和 safety state。无障碍路径中
LaserScan 保留但不提供场景真值。

奖励固定为：

- final-goal distance-delta progress；
- squared cross-track error penalty；
- control effort/rate；
- step/stuck penalty；
- success bonus 与 collision penalty。

新增 cross-track 项是因为只按终点距离奖励会鼓励切弯，无法对应 ICODE-MPPI 的 path-tracking 目标。其默认权重为
0，只有 L84 配置显式开启，因此旧实验可复现。

## 7. 训练完整性 Gate

每块必须满足：

1. 20k steps 完整，无 interrupted episode；
2. action dimension=2，`kind=covariance_only`，`learn_covariance=true`；
3. actor 输出、scale、Q、reward、gradient 全部有限；
4. scale 始终位于 `[0.5,2.0]`；
5. checkpoint、optimizer、normalizer、config、seed、git SHA、CSV 完整；
6. replay 四个 seen scene 均非空；
7. frozen ICODE checkpoint hash 不变；
8. memory disabled，RL mean difference 恒为 0。

## 8. 升级到闭环 deployment 比较的 Gate

至少 2/3 blocks 必须选择非初始 checkpoint，并同时满足：

- validation success 不下降；
- collision 不增加；
- mean cross-track RMSE 不高于 best fixed broad；
- 至少 1 个 unseen path 的 cross-track RMSE 改善；
- learned covariance 在不同状态下有非零变化，且不是恒定 broad；
- mean planner compute time 不增加超过 5%（同一 K 下）。

后续闭环必须比较：fixed baseline、fixed speed_explore、fixed broad、learned covariance、oracle diagnostic upper
bound。若 learned 只胜 baseline 而不胜 best fixed broad，不能作为 RL 正贡献。

## 9. 边界

- L83 是 oracle pilot，不是 RL 性能结果；
- L84 smoke 只验证代码，不计入 Gate；
- 不在看到训练结果后改变 scale bounds、reward 或 checkpoint rule；
- L84 先解决高动态采样，不同时加入动态障碍；
- 动态障碍与复杂/OOD gate 只在 covariance-only 贡献通过后加入。
