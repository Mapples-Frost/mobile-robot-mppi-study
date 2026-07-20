# L218 扩展 MuJoCo 复杂路线开发协议

日期：2026-07-20

状态：development；不得作为封存确认结果。

## 目的

L217 已在三张较小静态地图上确认 role-aware reliability HSS 的效率收益，但地图尺寸与路线多样性不足，而且 residual-conditioned policy context 未实际激活。L218 在不修改 ICODE、RL prior、HSS 和 MPPI 核心语义的前提下，扩大地图与路线难度，并显式激活 `ICODE residual/innovation -> RL policy` 耦合。

## 仿真与感知边界

- environment 必须是 `mujoco_diff_drive`，不得用低阶运动学 plant 替代；
- 动作通过轮毂扭矩 PI、轮地接触、摩擦、执行延迟和 2 ms MuJoCo physics step 执行；
- planner obstacles 只能来自 Synthetic LaserScan、`scan_guard` 和 `local_obstacle_layer`；
- 配置中的障碍真值只用于创建 MuJoCo XML、碰撞判定及离线几何审计，不得直接传给 MPPI；
- 外部 polyline 是任务参考线，等价于上层全局规划器的输出，不是局部障碍真值。

## 六个开发场景

1. 四挡板蛇形路线；
2. 大型 U 型逃逸；
3. 双反向 U 型；
4. 嵌套 U 型；
5. 交错圆柱林；
6. 双环圆柱阵。

所有场景位于 6.5 m × 6.5 m 设计域，保留 0.25 m 机器人碰撞半径，并用离线栅格与参考线连续采样检查可行性。

## 开发顺序

1. 配置、几何、MuJoCo 构建及三步闭环 smoke；
2. 单 development seed、`simple_combination` 与 `full_proposed` 六场景筛查；
3. 若无工程失败，再运行 seeds 91001--91003 的四核心臂或七方法矩阵；
4. 只允许在 development seeds 修复工程实现、路线不可行性或训练稳定性；不得选择性删除失败场景或失败 seed；
5. Development Gate 通过后才能创建新的 sealed manifest 和从未使用的正式 seeds。

## Development Gate

- 所有回合记录 `plant_backend=mujoco_diff_drive`；
- 六个场景均通过起点、终点、连通性与参考线安全审计；
- coupled arms 的 `residual_policy_context_enabled_fraction > 0.95`；
- adaptive arms 的 `reliability_hss_enabled_fraction > 0.95`；
- 每个 scene-domain-seed block 方法齐全且 rollout budget 相等；
- Full 不增加碰撞，并在至少四个场景中相对 Simple 改善预先声明的完成步数、路径长度或 cross-track RMSE；
- 若某些场景为负向结果，必须保留并按场景异质性报告。

## 计算策略

当前 WSL 项目环境为 CPU-only PyTorch，但系统可见 RTX 5060。代码使用 `device: auto`，因此在兼容 CUDA PyTorch 环境中会自动使用 GPU，在当前环境中安全回退 CPU。正式决定 GPU 与 CPU 前先测量每决策耗时；由于 MPPI 主循环、MuJoCo 和 obstacle cost 是 NumPy/CPU，优先使用多进程 scene/seed 分片并行，避免小网络频繁 CPU-GPU 传输反而变慢。
