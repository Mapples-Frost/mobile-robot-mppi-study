# L252 确定性 Warm-Start 候选保留协议

日期：2026-07-21  
状态：预注册 development probe

## 问题与单变量修复

L251 已验证空间路径走廊投影修复：S-Chicane 中 Full Proposed 的
`path_boundary_no_feasible_decision_fraction` 从 L250 的约 0.908 降为 0，完成度从
0.1319 提升为 0.1581，且零碰撞、零边界越界。但 ICODE 与 Full 仍在局部推进后
输出近零控制，未完成全路径。

代码审计发现，基础采样器本来把候选 0 固定为无随机扰动的 prior mean；启用候选级
边界约束后，候选 0 被确定性制动序列覆盖。Paper optimizer 在 guided allocation 为 0
时也会发生相同问题。在 36 步、2 维控制的 72 维序列空间内，100 个独立高斯扰动
不能保证保留一条连贯 warm start；因此该覆盖会造成采样分布退化并放大停车吸引域。

L252 只修改候选槽位分配：候选 0 仍为制动序列，候选 1 保留原候选 0 的确定性
warm start。总候选数 K、rollout 预算、Actor、ICODE、cost、地图、安全链、终止条件和
所有其他配置保持不变。

## 冻结探针

- 场景：`tracking_grand_s_chicane_l234`
- 方法：`icode_mppi`、`full_proposed`
- development seed：`923301001`
- 物理域：`nominal_seen` MuJoCo
- 每决策 100 rollouts；最大 1405 步
- 配置继承 L251；禁止使用 sealed seeds

## 预注册 Gate

以 L251 同 seed、同地图、同预算结果为只读配对基线：

1. 两回合 provenance、配置、checkpoint、逐回合工件完整；
2. 两方法均零碰撞、零 boundary violation、实际最小 footprint margin 不小于 0；
3. 两方法 no-feasible decision fraction 均不高于 0.10；
4. 两方法平均 path completion 相对 L251 至少提高 0.02；
5. 任一方法相对 L251 的 path completion 回退不得超过 0.01；
6. Full Proposed 的 RL proposal authority 必须非零；
7. 报告 stuck steps、有效样本数与 planner time，确认改善不是预算变化造成。

通过后才扩大到三张 Tracking 地图。若失败，只保留原始数据与简短 Gate 状态，不制作
图或详细报告；不得筛 seed、删除失败、改变核心方法或把 development 结果写成正式结论。

