# L238 Tracking Boundary Buffer Qualification（15 cm）

## 研究问题

在 L237 的 `W=3.5D` 走廊中，将 footprint 边界软缓冲从 5 cm 提前到 15 cm，是否能在不放松物理碰撞、边界终止和安全链的前提下，让 Full proposed 安全绕过三张地图的首障碍？

## 唯一变量

- L237：`path_boundary_buffer = 0.05 m`；
- L238：`path_boundary_buffer = 0.15 m`。

`path_boundary_weight = 400` 和 `path_boundary_violation_penalty = 10000` 保持不变。因此，本轮只改变软代价开始生效的安全余量，不改变硬边界、地图几何或失败定义。

## 固定项

- 三张 L234 路径中心线与障碍物；
- `W=3.5D=1.75 m`，corridor half width `0.875 m`；
- Value-Consistent ICODE + Residual-Conditioned RL Prior + role-aware Reliability-Weighted Value/HSS + MPPI；
- MuJoCo、LaserScan、scan_guard、local obstacle layer 和 safety arbitration；
- 其他 planner/cost 参数、模型 checkpoints；
- `K=100`、2 iterations、700 steps；
- development seed `923301001`、`nominal_seen`。

## Gate

Full proposed × 三场景各运行一次。每个场景要求：

- collision = false；
- boundary_violation_steps = 0；
- minimum footprint boundary margin ≥ 0；
- Hairpin/S-Chicane/Infinity 完成度分别 ≥0.145/0.205/0.135。

全部通过后才能编写四方法短资格矩阵协议。失败则保留结果、进行失败分类，并继续遵循单变量 development 原则。禁止使用 sealed seeds、筛 seed、删除失败、放松终止条件或修改核心方法。
