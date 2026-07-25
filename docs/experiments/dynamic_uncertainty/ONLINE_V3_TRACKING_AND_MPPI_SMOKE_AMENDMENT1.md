# Online V3 Tracking and MPPI Smoke — Amendment 1

日期：2026-07-23  
发生阶段：闭环控制结果产生前的路线资格检查  
原因：原始起点—终点没有满足预注册的重复穿越 Gate

## 发现

原预注册走廊为 `(-3.6, 0.0) → (3.6, 0.0)`。对冻结的三个 development
seed 做纯障碍物路线资格检查后，有效走廊穿越次数为：

| Seed | 原走廊穿越次数 |
|---|---:|
| 730100101 | 3 |
| 730100102 | 2 |
| 730100103 | 1 |

因此原场景违反“每个 episode 至少三次穿越机器人必经路线”的冻结 Gate。尤其 seed
103 的障碍仍在 recurrent patrol，但部分穿越发生于原机器人线段左侧，不能算作有效
交互。

## 修订

只修改机器人任务几何：

- 起点：`(-4.6, -1.5)`；
- 终点：`(4.2, -1.5)`；
- 必经路线：`y=-1.5`。

相同三个 seed 的穿越次数变为：

| Seed | Amendment 1 穿越次数 |
|---|---:|
| 730100101 | 3 |
| 730100102 | 4 |
| 730100103 | 3 |

## 未修改内容

- V3 obstacle generator；
- `hybrid_patrol` 过程及参数；
- development seeds；
- Change-Aware IMM；
- Collision Risk V1；
- paired arms；
- smoke Gate；
- MPPI 参数。

这是闭环结果产生前、由预注册路线 Gate 触发的任务几何修订。后续所有 smoke 使用
`mujoco_v3_probabilistic_crossing_smoke_amendment1.yaml`，原配置保留作为完整审计
记录。

