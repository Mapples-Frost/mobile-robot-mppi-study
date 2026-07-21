# L245 BC Actor 的 7.5 cm Boundary Buffer 单变量协议

日期：2026-07-21

状态：预注册 development protocol，尚未读取 L245 outcome

## 1. 触发原因

L244 BC Full 的平均 path completion 相对 L242 Full 提升 0.0832，Hairpin 提升 0.2345，
且三场景均零碰撞；但 S-Chicane 和 Infinity 分别出现 1 步 footprint boundary violation，最小
裕度仅为 -2.7 mm 与 -7.2 mm，导致严格 Gate 未通过。

## 2. 唯一变量

```text
path_boundary_buffer: 0.050 m -> 0.075 m
```

增加 2.5 cm 是基于 L244 最大 7.2 mm 负裕度再加入离散控制与状态估计余量的事前安全修复。
它只使所有 MPPI 候选更早承担边界代价，不放松边界判定或 scan_guard。

以下内容全部保持 L244 不变：

- L243 selected Actor 及 SHA；
- Value-Consistent ICODE、Path/Residual-Conditioned RL Prior、role-aware HSS；
- 其余 MPPI cost 权重、K=100、50×2 iterations、700 步；
- 三张 L239 `W=4.0D` 地图、MuJoCo 物理、LaserScan 与安全链；
- development seed `923301001` 与 `nominal_seen` 物理域。

## 3. 最小资格探针

只运行 `full_proposed` × Hairpin/S-Chicane/Infinity，共 3 个 MuJoCo 回合。L244 Full 作为
同 seed、同模型、同预算的只读 buffer=0.05 对照；不重跑 ICODE 或 L242。

## 4. 读取前 Gate

结果读取前确认：3 个唯一回合、qualification=1、Git/config/checkpoint/provenance、每回合工件、
MuJoCo 3.2.3、K100、2 iterations、700 步及日志无异常。

## 5. 通过条件

1. 三场景零碰撞、零 boundary violation、最小 footprint boundary margin 均不小于 0；
2. 相对 L242 Full 的平均 completion 增益仍至少为 0.02；
3. 相对 L244 Full，任一场景 completion 回退不超过 0.02；
4. proposal authority 或 guided elites 仍非平凡；
5. 同时报告 cross-track、安全干预与 planner time。

若失败，保留结果并停止后续 sealed 注册；不得继续在同一 seed 上无界扫描 buffer，也不得改变
核心方法、地图或安全链。
