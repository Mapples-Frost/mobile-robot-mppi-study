# L248 候选级 Footprint 约束执行审计

日期：2026-07-21  
状态：ENGINEERING INVALID / 保留的负向执行记录

## 1. 结论

L248 完成了 3 场景 × 2 arms = 6 个唯一 MuJoCo qualification 回合，Git、Actor、
checkpoint、MuJoCo 3.2.3、rollout budget、episode budget、配置和逐回合工件全部通过
provenance 核验。然而 treatment fidelity Gate 失败，因此本轮不能用于判断候选边界约束是否有效。

失败证据是：Full Proposed 在 Hairpin、S-Chicane、Infinity 的终止步、完成度、边界步和
最小 footprint margin 与 L247 逐项完全相同；三个回合仍各发生 1 次 boundary violation。
与此同时，ICODE-MPPI 对照发生了变化。源码审计确认：标准 `MppiController._solve_plan`
执行了新过滤器，但 `PaperRLDrivenMppiController._solve_plan` 使用独立的 guided/Gaussian
elite optimizer，绕过了标准求解器中的过滤逻辑。

因此 L248 是实现覆盖不完整的工程负向结果，不是“方法无效”的科研负向结果。

## 2. 已观察结果

| 场景 | 方法 | 终止 | 完成度 | 边界步 | 最小 footprint margin |
|---|---|---|---:|---:|---:|
| Hairpin | ICODE-MPPI | max_steps | 0.0924 | 0 | +0.6339 |
| Hairpin | Full Proposed | boundary_violation | 0.3526 | 1 | -0.0033 |
| S-Chicane | ICODE-MPPI | max_steps | 0.1535 | 0 | +0.6415 |
| S-Chicane | Full Proposed | boundary_violation | 0.1877 | 1 | -0.0027 |
| Infinity | ICODE-MPPI | max_steps | 0.0873 | 0 | +0.6255 |
| Infinity | Full Proposed | boundary_violation | 0.1055 | 1 | -0.0072 |

总计 0/6 success、0/6 collision；Full Proposed 为 3/3 boundary termination。

## 3. 第二项执行缺口

虽然求解器生成了候选可行率、无可行候选、加权结果可行性和 fallback 诊断，但 L248 启动时
`EpisodeMetrics` 尚未聚合这些新增字段，因而逐回合 `metrics.json` 中缺少预注册要求的约束机制
统计。这一缺口同样要求新建执行批次，不能在完成后臆造或重构历史内部状态。

## 4. 修复边界

L249 只修复两项执行完整性：

1. 将同一候选级 footprint filter 接入 Paper RL-Driven MPPI 的 guided/Gaussian elite 路径；
2. 将约束诊断持久化到 episode metrics 和 progress CSV。

不改变 RL、ICODE、Actor、地图、cost 权重、episode budget、rollout budget 或安全链。
L248 原始结果、日志和失败均保留，L249 使用新输出目录和新 Git SHA。

