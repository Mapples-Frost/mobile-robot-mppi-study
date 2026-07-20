# L235 首障碍绕行采样能力探针结果

## 结论

L235 Development Gate **未通过**。将所有方法公平共享的 MPPI 预算从 `K=30, I=1` 增加到 `K=100, I=2` 后，Full proposed 在三张地图都更早发生 corridor boundary violation，而不是成功绕过首障碍。

| 场景 | 步数 | 终止原因 | 碰撞 | 越界步数 | 路径完成度 | Gate 阈值 |
|---|---:|---|---:|---:|---:|---:|
| Hairpin | 138 | boundary_violation | 0 | 1 | 0.0904 | ≥0.145 |
| S-Chicane | 369 | boundary_violation | 0 | 1 | 0.1751 | ≥0.205 |
| Infinity | 42 | boundary_violation | 0 | 1 | 0.0256 | ≥0.135 |

三回合均为 MuJoCo 3.2.3、development seed `923301001`、`nominal_seen`、Git SHA `0a45e5a773020a2fea57c78a526578ba2e0933c3`，并完整保存配置、轨迹、metrics 和 provenance。没有使用 sealed seeds。

## 失败分类

L234 的 `K=30` 结果主要表现为首障碍前冻结；L235 增加采样后变为更早越界。代码审计发现 MPPI 已使用多点 path-preview cost，但 corridor boundary 只存在于离线评价和回合终止逻辑中，没有进入候选 rollout cost。因此，更大的采样集合反而更容易找到“缩短局部目标距离但越过硬边界”的候选。

这不是 ICODE、RL prior 或 HSS 核心耦合失效的证据，而是 Tracking 任务约束尚未进入所有方法共享的优化目标。继续增加 K 或 max_steps 不具备解释力。

## 下一项单变量修复

L236 为所有方法共享的 MPPI cost 增加 footprint-aware corridor constraint：

```text
margin = corridor_half_width - footprint_radius - |lateral_error|
```

接近边界时增加连续软代价，`margin < 0` 时增加与 collision penalty 同量级的硬惩罚。该约束默认关闭，保证原有 point-goal baseline 数值兼容；只在 L236 Tracking development 配置中开启。地图、障碍物、ICODE、Actor、HSS、scan_guard 和其他 cost 保持不变。
