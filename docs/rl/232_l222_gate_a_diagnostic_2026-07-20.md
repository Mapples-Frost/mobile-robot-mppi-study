# L222 困难地图 Gate A：结果与失败诊断

## 1. 结论

L222 Gate A 未达到预注册通过条件。三张困难地图均无碰撞，但只有
`l222_cylinder_spiral_safe` 到达终点；`l222_serpentine_safe` 与
`l222_nested_u_safe` 在 1400 步达到上限前仍在低速前进。

本结果是开发集结果，不可作为论文正式性能结论，也未使用 sealed seed。

| 场景 | 成功 | 碰撞 | 步数 | 路径完成度 | 最后 200 步路径进度 | 最后 200 步前向减速占比 |
|---|---:|---:|---:|---:|---:|---:|
| Cylinder spiral | 1 | 0 | 1215 | 96.1% | 2.842 m | 45.0% |
| Nested U | 0 | 0 | 1400 | 44.8% | 0.415 m | 88.0% |
| Serpentine | 0 | 0 | 1400 | 31.7% | 0.655 m | 100.0% |

## 2. 数据完整性

- 运行环境：MuJoCo 3.2.3，`mujoco_diff_drive` plant；
- 预测模型：ICODE residual；
- 开发 seed：91001；
- 预算：每拍 30 rollouts、1 iteration；
- Git SHA：`9cfb5d99e6ae29112643d9d98819f8bf973fffb4`；
- 三个 episode 均包含 `config_resolved.yaml`、`trajectory.csv`、
  `metrics.json` 和 `provenance.json`；
- 两个并行启动进程在写完 episode 工件后未写顶层聚合 CSV，因此本审计直接从
  不可变的逐回合工件重建结果；这一启动器异常不改变逐回合数值，但已作为限制记录。

## 3. 失败模式

L221 的首要问题是旧参考路径进入了 scan_guard 膨胀后的不可行区域。L222
已修复这一点：三条新参考路径对 0.38 m 保护包络均可行。因此，当前失败不是
“参考线穿墙”。

轨迹显示新的主导问题是安全参数对狭长障碍走廊过度保守：

- 蛇形场景在第 400 步后，每个 200 步窗口的 `front_obstacle_slow`
  占比均为 100%；
- 嵌套 U 在大多数后续窗口中该比例为 88%–100%；
- 两者最后 200 步平均执行线速度分别只有约 0.034 m/s 与 0.022 m/s；
- 机器人没有碰撞、没有数值异常，也没有完全静止，而是安全仲裁与保守规划共同造成
  “持续爬行”。

## 4. 下一步

L223 只校准 MuJoCo 开发场景的 scan_guard 阈值，不关闭保护，也不改变 ICODE、
RL、MPPI 采样机制、地图、参考路径或碰撞惩罚。所有对照方法共用相同阈值；实车
配置完全不变。若这一单因素修改仍失败，才进入独立预注册的障碍代价密度审计，避免
一次改动多个因素后无法归因。

可复核数据位于：

- `docs/rl/artifacts/l222/gate_a/gate_a_audit.json`
- `docs/rl/artifacts/l222/gate_a/gate_a_episode_audit.csv`
- `docs/rl/artifacts/l222/gate_a/gate_a_progress_windows.csv`

