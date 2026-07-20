# L238 Tracking Boundary Buffer Qualification（15 cm）结果

## 结果定位

L238 的三个 MuJoCo 3.2.3 development 回合均完整结束并通过来源核验。固定条件为 `full_proposed`、`nominal_seen`、seed `923301001`、`K=100`（50 candidates × 2 iterations）、700 步上限、走廊半宽 `0.875 m`、Git SHA `7ca3e707535d61b2a5e1d64c50a97705279d6846`。唯一 treatment `path_boundary_buffer=0.15 m` 在三个 resolved config 中均已生效，逐回合工件齐全。

L238 Gate **未通过**：

| 场景 | 步数 | 终止原因 | 碰撞 | 越界步数 | 最小 footprint 余量 (m) | 完成度 | Gate 阈值 |
|---|---:|---|---:|---:|---:|---:|---:|
| Hairpin | 700 | max_steps | 0 | 0 | 0.0533 | 0.1064 | ≥0.145 |
| S-Chicane | 181 | boundary_violation | 0 | 1 | -0.0013 | 0.1805 | ≥0.205 |
| Infinity | 324 | boundary_violation | 0 | 1 | -0.0019 | 0.1024 | ≥0.135 |

## 失败分类

15 cm 软缓冲确实使 Hairpin 全程保持正边界余量，并把 S-Chicane 的越界量从 L237 的约 7 mm 降至约 1.3 mm；但三个场景均未达到首障碍完成度 Gate，Infinity 还更早终止。

原因是软缓冲只能把轨迹推向中心线，而首障碍恰好位于或贴近中心线。缓冲增大后，MPPI 在“远离障碍”和“远离走廊边界”之间的可用旁路进一步缩小。Hairpin 的无越界但长时间停滞正是这种约束冲突的表现。因此，本轮否定“仅提前边界软代价即可解决首障碍”的假设。

## 下一项单变量修复

下一轮回到 L237 的 5 cm buffer 分支，只将走廊从 `W=3.5D` 增加到 `W=4.0D`（全宽 `2.0 m`、半宽 `1.0 m`）。这相对于 L237 是严格单变量几何敏感性实验；L238 作为独立 buffer 敏感性负向分支保留。

`W=4.0D` 为机器人、中心线障碍物和 scan_guard 同时提供额外旁路空间，但不移动路径或障碍、不放松碰撞和边界定义，也不改变核心 RL+ICODE 方法、安全链、cost 权重或采样预算。
