# L239 Tracking Geometry Qualification（W=4.0D）结果

## 结果定位

L239 的三个 MuJoCo 3.2.3 development 回合均完整结束并通过来源核验。固定条件为 `full_proposed`、`nominal_seen`、seed `923301001`、`K=100`（50 candidates × 2 iterations）、700 步上限、5 cm boundary buffer、走廊半宽 `1.0 m`、Git SHA `d972de4`。三个逐回合目录均包含 resolved config、metrics、trajectory 与 provenance。

L239 Gate **未通过**：

| 场景 | 步数 | 终止原因 | 碰撞 | 越界步数 | 最小 footprint 余量 (m) | 完成度 | Gate 阈值 |
|---|---:|---|---:|---:|---:|---:|---:|
| Hairpin | 700 | max_steps | 0 | 0 | 0.0685 | 0.1061 | ≥0.145 |
| S-Chicane | 595 | boundary_violation | 0 | 1 | -0.0084 | 0.1805 | ≥0.205 |
| Infinity | 653 | boundary_violation | 0 | 1 | -0.0027 | 0.1057 | ≥0.135 |

## 失败分类

与 L237 的 `W=3.5D` 相比，`W=4.0D` 将 S-Chicane 的终止从 163 步推迟到 595 步，将 Infinity 从 526 步推迟到 653 步，Hairpin 则在正边界余量下运行至上限。几何裕量因此确实改善了安全存活时间，但三场景的路径完成度仍停留在首障碍附近。

轨迹审计显示，车辆已经开始从正确一侧绕障，却无法及时看到并评价“通过障碍后重新汇入中心线”的长期收益：当前 horizon 为 36、`dt=0.1 s`、`path_preview_speed_mps=0.45`，固定 horizon 内路径预览只向前推进约 1.62 m（再加初始 lookahead 约 0.45 m）。对于带安全膨胀的首障碍绕行，该距离不足以稳定覆盖完整的“偏离—通过—回归”动作链。因此，单纯继续加宽走廊不会解决局部最优。

## 下一项单变量修复

L240 固定 L239 的地图、`W=4.0D`、边界代价、核心方法、horizon、K、迭代次数与安全链，只将 `path_preview_speed_mps` 从 `0.45` 调整为 `0.75 m/s`。这不会改变 MuJoCo 执行速度，而只是让同一 36 步 rollout 的参考路径采样覆盖到约 2.70 m 的前向弧长，使候选轨迹能够看到障碍后的回归收益。

L236–L239 的全部负向结果继续保留；不使用 sealed seeds。
