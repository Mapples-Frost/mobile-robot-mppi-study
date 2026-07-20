# L237 Tracking Geometry Qualification（W=3.5D）结果

## 结果定位

L237 的三个 MuJoCo 3.2.3 development 回合均完整结束。实验固定使用 `full_proposed`、`nominal_seen`、seed `923301001`、`K=100`（每轮 50 条候选、2 次迭代）、700 步上限和 Git SHA `34dc4e1ecd7acac90f2cb681e1372ad29303d7e6`。三个回合的 resolved config 均确认：走廊半宽为 `0.875 m`，L236 footprint-aware boundary cost 已启用，逐回合配置、轨迹、指标和 provenance 齐全。

L237 Gate **未通过**：

| 场景 | 步数 | 终止原因 | 碰撞 | 越界步数 | 最小 footprint 余量 (m) | 完成度 | Gate 阈值 |
|---|---:|---|---:|---:|---:|---:|---:|
| Hairpin | 429 | boundary_violation | 0 | 1 | -0.0105 | 0.1126 | ≥0.145 |
| S-Chicane | 163 | boundary_violation | 0 | 1 | -0.0071 | 0.1719 | ≥0.205 |
| Infinity | 526 | boundary_violation | 0 | 1 | -0.0003 | 0.1087 | ≥0.135 |

## 失败分类

三个回合均未发生物理碰撞，但都以毫米到厘米级 footprint 越界终止。与 L236 attempt 2 相比，扩大走廊使 Hairpin 和 Infinity 的完成度略有提高，Infinity 的越界显著推迟；S-Chicane 没有改善。因此，`W=3.5D` 只扩大几何可行域，并未让控制器主动保留足够的执行余量。

该结果反驳“只要把走廊加宽到 3.5D 就能解决首障碍失败”的假设。当前更具体的问题是：边界软代价只在剩余 `0.05 m` 时开始作用，真实 MuJoCo 执行相对 rollout 的小偏差仍可能把轨迹推过边界。继续扩大地图会混淆任务难度；直接放松终止判据会掩盖失败。

## 下一项单变量修复

L238 固定 L237 的 `W=3.5D`、路径、障碍、核心方法、所有权重、硬越界惩罚、K、迭代次数、安全链和 seed，只把 `path_boundary_buffer` 从 `0.05 m` 调整为 `0.15 m`。其作用是让 MPPI 在距离 footprint 边界还有 15 cm 时开始付出平滑代价，为预测—执行偏差预留约 10 cm 的额外控制余量。

L236 attempt 1、L236 attempt 2 和 L237 的全部负向结果均保留；不使用 sealed seeds。
