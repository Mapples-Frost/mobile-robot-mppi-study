# L253 路径递推 Warm-Start 单变量协议

日期：2026-07-21
状态：预注册 development probe

## 根因与唯一处理

L252 证明同时保留制动候选与确定性 prior 候选仍未提高完成度。冻结轨迹进一步显示：
两方法在前方净空充足、没有 safety override、候选可行率高的情况下输出近零控制。现有
`GoalWarmStartPrior` 只根据当前 lookahead target 计算一次 `(v, omega)`，随后把同一个
控制复制 36 次。它不是一条随弯道演化的控制序列；在 72 维序列空间中，100 个独立
高斯扰动也难以稳定生成“先转弯、再沿新切线前进”的连贯候选。

L253 唯一处理是：当 reference 提供只读 polyline preview 时，从当前观测出发用相同
`v_gain`、`yaw_gain`、动作上下界和速率约束递推 36 步 virtual unicycle；每一步重新投影
路径进度并计算下一 lookahead 控制，形成 horizon-aware warm start。该先验由 ICODE-MPPI
及 Full Proposed 的传统 fallback 公平共享。Actor、ICODE、MPPI cost、K、迭代数、地图、
安全链、终止条件及 checkpoint 全部冻结。

## 冻结探针

- 场景：`tracking_grand_s_chicane_l234`
- 方法：`icode_mppi`、`full_proposed`
- development seed：`923301001`
- 物理域：`nominal_seen` MuJoCo
- 每决策 100 rollouts；最大 1405 步
- 唯一配置差异：`planner.prior_path_rollout_enabled=true`
- 禁止使用 sealed seeds

## Gate

以 L252 同 seed、同地图、同预算结果为只读配对基线：

1. provenance、配置、checkpoint、MuJoCo 版本及逐回合工件完整；
2. 两方法均零碰撞、零 boundary violation、最小 footprint margin 不小于 0；
3. 两方法 no-feasible decision fraction 均不高于 0.10；
4. 两方法平均 path completion 相对 L252 至少提高 0.05；
5. 任一方法的 path completion 回退不得超过 0.01；
6. 平均 stuck steps 相对 L252 至少下降 20%；
7. Full Proposed 的 RL proposal authority 非零，rollout 预算保持 100。

通过后才扩大到 Hairpin、S-Chicane、Infinity 三地图 development Gate。失败时只保留
原始数据和简短 Gate 状态，不制作图或详细报告；不得筛 seed、删除失败、使用 sealed
seeds、修改核心 RL+ICODE 耦合或把 development 结果称作正式结论。

