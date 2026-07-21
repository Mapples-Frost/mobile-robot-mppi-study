# L251 空间路径走廊投影单变量验证协议

日期：2026-07-21  
状态：预注册 development probe

## 唯一修复

L250 证明候选过滤能够消除实际 boundary violation，但 S-Chicane 中存在 90.8% 的
no-feasible 决策。对冻结轨迹进行反事实复算发现：旧实现把候选状态与按固定速度向前移动的
预览点逐时刻配对，导致静止候选在弯道处被错误判为越界。一个真实 footprint margin 为
`+0.3026 m` 的冻结状态，旧预测为 `-0.5716 m`。

L251 只把边界约束改为真正的空间走廊判定：每个预测状态投影到最近的可接受路径段，候选内
进度单调，走廊宽度在投影进度处求值。相同冻结状态修复后预测为 `+0.3003 m`。不修改
Actor、ICODE、MPPI cost 权重、rollout 预算、地图、安全链或终止条件。

## 冻结探针

- 场景：`tracking_grand_s_chicane_l234`；
- 方法：`icode_mppi`、`full_proposed`；
- development seed：`923301001`；
- 物理域：`nominal_seen` MuJoCo；
- 每决策 100 rollouts，最大 1405 步；
- checkpoint、所有其他配置继承 L250；禁止使用 sealed seeds。

## Gate

1. 两回合来源、配置、诊断完整，filter enabled fraction=1；
2. 零碰撞、零 boundary violation、实际最小 footprint margin≥0；
3. Full no-feasible decision fraction 从 L250 的 `0.9082` 降至不高于 `0.10`；
4. Full path completion 不低于 L250 的 `0.1319`；
5. RL proposal authority 非零。

本轮仅验证根因修复。即使通过，也不构成论文或多 seed 结论。负向结果只保留原始工件和
Gate 状态，不制作图表或详细结果报告。

