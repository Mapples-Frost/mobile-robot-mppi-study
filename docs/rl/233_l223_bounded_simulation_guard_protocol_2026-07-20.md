# L223 有边界的 MuJoCo scan_guard 标定协议（开发集）

## 1. 目的

L222 已证明安全可行参考路径能够让圆柱螺旋场景成功，但蛇形与嵌套 U 在原始
scan_guard 参数下长期低速爬行。本轮检验一个可证伪的工程假设：

> 在不关闭安全仲裁、仍保留车体外几何余量的条件下，降低仅用于 MuJoCo
> 复杂地图的保守阈值，能否恢复 ICODE-MPPI 的基本可达性。

本轮不是论文正式实验，不形成算法优越性结论。

## 2. 冻结项

保持不变：

- 核心方法：Value-Consistent ICODE + Residual-Conditioned RL Prior +
  role-aware Reliability-Weighted Value/HSS + MPPI；
- MuJoCo 物理模型、六张地图、L222 安全参考路径；
- ICODE checkpoint、RK4、horizon=36；
- MPPI cost、K=30、1 iteration；
- LaserScan → scan_guard → local_obstacle_layer → planner obstacles；
- 碰撞半径 0.25 m、碰撞惩罚和碰撞终止；
- seed 91001，物理域 `nominal_seen`；
- 不使用 sealed seeds，不筛选、不删除失败回合。

## 3. 唯一处理因素

仅在 `configs/research/expanded_navigation_relaxed_guard_l223.yaml` 中统一覆盖：

| 参数 | L222 | L223 |
|---|---:|---:|
| near-body stop radius | 0.38 m | 0.32 m |
| side stop distance | 0.38 m | 0.32 m |
| hard/front stop distance | 0.40 m | 0.34 m |
| front soft-block distance | 0.55 m | 0.48 m |
| soft-block max speed | 0.04 m/s | 0.06 m/s |
| front slow distance | 1.10 m | 0.85 m |
| front slow minimum scale | 0.35 | 0.45 |

0.32 m 的近车硬停半径仍比 0.25 m 碰撞车体多 0.07 m。scan_guard 保持启用，
线速度被抑制时仍允许角速度转向。上述覆盖禁止进入 ROS/实车配置。

## 4. Gate A

运行三张困难地图，每图一个 ICODE-MPPI 回合：

- `l222_serpentine_safe`
- `l222_nested_u_safe`
- `l222_cylinder_spiral_safe`

通过条件在运行前冻结为：

1. 3 回合均零碰撞；
2. 至少 2/3 回合成功；
3. 不得出现 NaN、Inf、缺失工件或 provenance 不一致；
4. 若仍有未成功回合，其最后 200 步路径进度不得比 L222 同场景更低。

## 5. 后续 Gate

- Gate A 通过：再以相同参数运行 seed 91002、91003 的三方法开发矩阵；
- 三方法为 ICODE-MPPI、Simple combination、Full proposed；
- 只有共享强 baseline 稳定后才训练带 path-preview 输入的新 Actor；
- 只有三 seed、六地图 Development Gate 通过后才允许预注册全新 sealed seeds；
- Gate A 失败：保留所有数据，单独预注册障碍点密度不变性修复，不继续叠加调参。

## 6. 科研边界

本轮阈值是仿真平台标定，不得描述为 RL 或 ICODE 的贡献。后续所有方法必须共用
相同安全参数，报告中同时给出碰撞率、最小净空、安全干预次数与完成时间，避免用
“放宽安全”换取不可接受的风险。

