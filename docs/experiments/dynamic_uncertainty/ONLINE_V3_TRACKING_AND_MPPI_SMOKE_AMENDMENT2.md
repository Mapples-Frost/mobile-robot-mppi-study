# Online V3 Tracking and MPPI Smoke — Amendment 2

日期：2026-07-23  
发生阶段：任何闭环控制结果产生之前的 episode 时域复核  
原因：Amendment 1 误用完整 45 s 生成轨迹计数，而预注册 episode 只有 40 s

## 发现

Amendment 1 的路线为 `(-4.6, -1.5) → (4.2, -1.5)`。重新把障碍物
穿越审计严格截断在控制 episode 的 `[0, 40] s` 后，三个 development seed
的有效穿越次数为：

| Seed | 45 s 旧计数 | 40 s 正确计数 |
|---|---:|---:|
| 730100101 | 3 | 3 |
| 730100102 | 4 | 2 |
| 730100103 | 3 | 3 |

seed 730100102 的后两次穿越发生在 40 s 以后，控制器在本次实验中不可能
经历它们，因此不能计入“每个 episode 至少 3 次穿越”的 Gate。

## 修订

只修改机器人任务几何：

- 起点：`(-4.6, -2.1)`；
- 终点：`(4.2, -1.7)`；
- 必经路线：连接上述两点的直线段；
- 穿越审计时域：`[0, 40] s`。

在严格 40 s 时域内，三个 seed 的路线穿越次数均为 3。

## 未修改内容

- 冻结 V3 obstacle generator；
- `hybrid_patrol` 过程及全部参数；
- development seeds；
- Change-Aware IMM；
- Collision Risk V1；
- paired arms；
- smoke Gate；
- MPPI 参数。

本修订发生在任何 risk-disabled / risk-enabled 闭环结果产生之前。后续
smoke 统一使用
`mujoco_v3_probabilistic_crossing_smoke_amendment2.yaml`，旧配置保留作
完整审计记录。

