# L240 Tracking Path-Preview Qualification（0.75 m/s）

## 研究问题

在 L239 的 `W=4.0D` 几何与 boundary-aware MPPI 上，只增加固定 horizon 内的参考路径预览距离，能否让 Full proposed 评价完整的“绕过首障碍并回归路径”动作链，从而摆脱局部最优？

## 唯一变量（相对 L239）

- L239：`path_preview_speed_mps = 0.45`；
- L240：`path_preview_speed_mps = 0.75`。

该参数只决定 rollout cost 在固定预测步上对应多远的参考路径点；不改变 MuJoCo 实际速度限制、控制周期、horizon、动力学或安全链。

## 固定项

- 三张 L234 路径中心线、障碍物位置与尺寸；
- `W=4.0D=2.0 m`、corridor half width `1.0 m`；
- Value-Consistent ICODE + Residual-Conditioned RL Prior + role-aware Reliability-Weighted Value/HSS + MPPI；
- footprint-aware boundary cost、5 cm buffer 和全部其他 cost 权重；
- MuJoCo、LaserScan、scan_guard、local obstacle layer、safety arbitration 与终止定义；
- checkpoints、horizon 36、`dt=0.1 s`；
- `K=100`、2 iterations、700 steps；
- development seed `923301001`、`nominal_seen`。

## Gate

Full proposed × 三场景各运行一次。每个场景要求：

- collision = false；
- boundary_violation_steps = 0；
- minimum footprint boundary margin ≥ 0；
- Hairpin/S-Chicane/Infinity 完成度分别 ≥0.145/0.205/0.135。

全部通过后才建立四方法短资格矩阵。失败结果必须保留和分类。禁止 sealed seeds、筛 seed、删除失败、移动障碍、放松安全链或修改核心方法。
