# L250 候选边界约束诊断链修复重跑协议

日期：2026-07-21  
状态：预注册 development execution；尚未读取 L249/L250 方法效果

## 1. 唯一变化

L249 因 `EpisodeMetrics.update()` 未持久化 planner 的候选边界诊断而未通过执行完整性 Gate。
L250 只补齐下列诊断的 planner → step record → episode summary 传播：

- filter enabled；
- candidate feasible fraction 与 minimum；
- no-feasible decision/iteration；
- weighted-update feasibility；
- fallback 与候选索引；
- final predicted footprint margin。

控制器、候选过滤算法、ICODE、Actor/value checkpoint、地图、cost、安全链和随机种子均不变。

## 2. 冻结实验条件

- development seed：`923301001`；禁止使用 sealed seeds；
- 场景：L239 的 Hairpin、S-Chicane、Infinity；
- 物理域：`nominal_seen`，MuJoCo；
- 方法：`icode_mppi` 与 `full_proposed`；
- 每次决策总 rollout 预算均为 100：ICODE 为 100×1，Full 为 50×2；
- 最大步数：2210 / 1405 / 2030；
- Actor SHA：`e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c`；
- 所有其他条件继承 L249，禁止调参、筛 seed、覆盖失败或删除结果。

## 3. 读取效果前的完整性 Gate

必须同时满足：

1. 3 场景 × 2 arms = 6 个唯一回合；
2. qualification=1、Git/config/checkpoint/MuJoCo/provenance 与逐回合工件完整；
3. resolved config 的 filter 为 true；
4. 六个 metrics 的 filter enabled fraction 均为 1；
5. candidate feasible、no-feasible、weighted-update、fallback、final margin 字段均存在且有限；
6. 日志无 Traceback、Exception、NaN 和 Inf。

任一不满足，L250 冻结为 engineering-invalid，不读取性能效果。

## 4. 性能 Gate

完整性通过后，沿用 L249 在读取结果前冻结的门槛：

1. 两 arms 均零碰撞；
2. Full 三场景均零 boundary violation，最小实际 footprint margin ≥ 0；
3. Full 相对只读 L247 的平均 completion 回退不超过 0.01；
4. Full 任一场景 completion 回退不超过 0.02；
5. Full 保持非平凡 RL proposal authority；
6. 完整报告可行率、fallback、planner time、安全负担与实际/预测边界余量。

本轮仍只有一个 development seed。通过后只能进入多-seed qualification，不能写成论文确认性结论。

