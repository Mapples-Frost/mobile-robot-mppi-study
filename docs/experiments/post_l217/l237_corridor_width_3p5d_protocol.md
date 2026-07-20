# L237 Tracking Geometry Qualification：W=3.5D

## 研究问题

在 boundary-aware MPPI 已生效的条件下，将走廊宽度从当前约 `3.2D` 调整为预定敏感性水平 `3.5D`，是否足以让 Full proposed 安全绕过每张地图的首障碍？

## 唯一变量

- MuJoCo collision diameter：`D = 0.50 m`；
- corridor full width：`W = 3.5D = 1.75 m`；
- corridor half width：`0.875 m`。

S-Chicane 的宽度 profile 在本探针中固定为 `0.875 m`，避免同时引入局部窄化。

## 固定项

- 路径中心线和障碍物位置/尺寸；
- Value-Consistent ICODE + Residual-Conditioned RL Prior + role-aware Reliability-Weighted Value/HSS + MPPI；
- L236 footprint-aware boundary cost；
- MuJoCo、LaserScan、scan_guard、local obstacle layer 和 safety arbitration；
- `K=100`、2 iterations、700 steps；
- development seed `923301001`、`nominal_seen`；
- checkpoints 和其他 cost。

## Gate

Full proposed × 三场景各运行一次。每个场景要求：

- collision = false；
- boundary_violation_steps = 0；
- minimum footprint boundary margin ≥ 0；
- Hairpin/S-Chicane/Infinity 完成度分别 ≥0.145/0.205/0.135。

全部通过后才进入四方法短资格矩阵；失败则保留并分类。禁止 sealed seeds、筛 seed、删除失败或修改核心方法。

