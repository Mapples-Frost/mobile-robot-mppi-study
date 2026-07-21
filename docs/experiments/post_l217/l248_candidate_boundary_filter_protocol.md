# L248 候选级 Footprint 边界约束 Development Protocol

日期：2026-07-21  
状态：预注册 development protocol；尚未读取任何 L248 outcome

## 1. 研究问题

L247 排除了旧 `700` 步上限这一混杂因素，但 Full Proposed 在三张 Tracking 地图中均因
单步 footprint boundary violation 终止。代码审计发现，L244–L247 的边界机制仍是软代价：

1. footprint 越界候选只被加罚，仍可能获得非零 MPPI 权重；
2. 即使参与加权的候选分别可行，它们的加权控制序列经过非线性动力学后仍可能不可行；
3. 采样候选的首控制此前没有在 rollout 前应用与真实输出相同的 slew-rate 约束。

L248 只回答：

> 在不改变 RL、ICODE、MPPI cost、地图或安全链的前提下，将同一个 footprint corridor
> contract 用作所有 MPPI arms 的候选可行性过滤，能否消除这类边界终止且不牺牲推进能力？

## 2. 唯一 treatment

启用：

```yaml
path_boundary_candidate_filter_enabled: true
```

约束语义固定为：

1. 在固定的 `K` 个候选中保留一个确定性制动序列，不增加 rollout 数量；
2. 候选 rollout 的首控制先应用真实 actuator slew-rate contract；
3. footprint 最小 corridor margin 小于零的候选权重严格置零；
4. 若加权控制序列的 rollout 再次不可行，则回退到成本最低的可行候选；
5. 若没有可行候选，则采用预留制动序列并显式记录 fail-closed 诊断；
6. 以上规则对 `icode_mppi` 和 `full_proposed` 完全相同。

这不是修改 MPPI 目标权重，也不是降低环境 boundary termination 或 scan_guard 安全要求。

## 3. 冻结条件

- 核心方法：Value-Consistent ICODE + Path/Residual-Conditioned RL Prior +
  role-aware Reliability-Weighted Value/HSS + MPPI；
- Actor SHA256：`e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c`；
- scenes：L239 `W=4.0D` Hairpin、S-Chicane、Infinity；
- development seed：`923301001`；
- physics：`nominal_seen`；
- arms：`icode_mppi`、`full_proposed`；
- rollout budget：每次决策 100 条；Full 为 50×2 iterations，ICODE 为 100×1；
- episode budgets：Hairpin/S-Chicane/Infinity 分别为 2210/1405/2030；
- 继承 L247 的地图、障碍、path preview、全部 cost 权重、MuJoCo、LaserScan、scan_guard、
  safety arbitration 与终止判据；
- qualification=`1`，禁止使用 sealed seeds。

## 4. 完整性 Gate

读取效果前必须确认：

1. 三场景 × 两 arms = 6 个唯一 MuJoCo 回合；
2. Git SHA、配置、Actor/ICODE/value checkpoint、MuJoCo 版本与逐回合 provenance 完整；
3. 逐场景 max steps 与 rollout 数量和 L247 相同；
4. 每个回合含 resolved config、trajectory、metrics、provenance；
5. 无 Traceback、Exception、NaN、Inf，且不存在重复实验键；
6. L244–L247 负向结果只读保留。

## 5. 预注册性能 Gate

L248 只在同时满足下列条件时通过：

1. 两个 arms 的碰撞数均为零；
2. Full Proposed 三场景 boundary violation 均为零，且最小 footprint margin 不小于零；
3. Full Proposed 相对 L247 的平均 completion 回退不超过 `0.01`；
4. Full Proposed 任一场景 completion 回退不超过 `0.02`；
5. Full Proposed 保持非平凡 RL authority；
6. 报告候选可行率、无可行候选比例、加权结果回退次数、planner time 和安全干预。

即使 Gate 通过，本轮单 development seed 也不能作为论文确认性结论；它只允许进入新的
多-seed development qualification。若失败，必须保留全部结果并分类为“约束可行但推进退化”
或“约束仍未闭环”，不得扫描 cost 权重美化结果。

## 6. 禁止事项

- 不重新训练或更换 Actor/ICODE；
- 不修改地图、障碍、corridor 宽度、path preview、cost 权重或 episode budget；
- 不放松 boundary termination、LaserScan、scan_guard 或安全仲裁；
- 不使用 sealed seeds，不筛 seed、不删失败、不重跑挑结果；
- 不把 qualification 写成论文正式结论。

