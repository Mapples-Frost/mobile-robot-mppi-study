# L239 Tracking Geometry Qualification（W=4.0D）

## 研究问题

在 L237 的 boundary-aware MPPI 与 5 cm 软缓冲条件下，仅将走廊从 `W=3.5D` 扩大到 `W=4.0D`，是否能够解除中心线障碍物、scan_guard 安全距离与 footprint 边界之间的旁路冲突？

## 唯一变量（相对 L237）

- MuJoCo collision diameter：`D=0.50 m`；
- L237 corridor full width：`W=3.5D=1.75 m`；
- L239 corridor full width：`W=4.0D=2.00 m`；
- L239 corridor half width：`1.00 m`。

S-Chicane 的宽度 profile 固定为全程 `1.00 m`。`path_boundary_buffer` 恢复并固定为 L237 的 `0.05 m`，因此本轮与 L237 之间只有走廊宽度这一项变化。

## 固定项

- 三张 L234 路径中心线、障碍物位置与尺寸；
- Value-Consistent ICODE + Residual-Conditioned RL Prior + role-aware Reliability-Weighted Value/HSS + MPPI；
- L236 footprint-aware boundary cost 及全部权重；
- MuJoCo、LaserScan、scan_guard、local obstacle layer、safety arbitration 和边界终止定义；
- checkpoints、其他 planner/cost 参数；
- `K=100`、2 iterations、700 steps；
- development seed `923301001`、`nominal_seen`。

## Gate

Full proposed × 三场景各运行一次。每个场景要求：

- collision = false；
- boundary_violation_steps = 0；
- minimum footprint boundary margin ≥ 0；
- Hairpin/S-Chicane/Infinity 完成度分别 ≥0.145/0.205/0.135。

全部通过后才允许建立四方法短资格矩阵。失败结果必须保留和分类。禁止 sealed seeds、筛 seed、删除失败、移动障碍、放松安全链或修改核心方法。
