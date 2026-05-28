# Memory-Augmented MPPI Implementation Notes

生成日期：2026-05-28

本文档记录当前 `mobile-robot-mppi-study` 中 Memory-Augmented MPPI 的实现状态、接口、调试字段和运行方式。它是工程说明，不是理论论文。

---

## 1. 当前实现目标

当前实现的目标不是训练 RL policy，也不是把 MPPI core 重写成新的控制器。当前目标是把历史执行失败经验以 memory field 的形式接入 MPPI：

```text
execution history
    -> stuck / spin / near-obstacle / hard-stop / high-curvature detection
    -> MemoryFeature
    -> rollout memory cost
    -> temperature scale
    -> nominal sequence escape bias
    -> MPPI weighted update
```

Memory 不是硬控制器，不直接覆盖 `final_control`。`scan_guard` hard safety 仍然是最高优先级。

---

## 2. 当前实现对应 MA-MPPI 的哪些部分

当前实现覆盖了 Memory-Augmented MPPI 的工程核心：

- 历史状态窗口；
- 局部失败检测；
- feature merge / decay / prune；
- type-specific soft potential；
- trajectory-level memory cost；
- temperature scale；
- escape direction suggestion；
- live sim viewer/debug 字段；
- offline memory on/off 消融字段。

当前实现仍然是轻量工程版本，不包含 learned dynamics、RL-driven sampling policy 或全局记忆地图学习。

---

## 3. Feature 类型

`mppi_hardware_bridge/scripts/mppi_memory_field.py` 中当前支持：

```text
STUCK_LOCAL_MIN
LOW_PROGRESS_CORRIDOR
SPIN_TRAP
NEAR_OBSTACLE_TRAP
HARD_STOP_RECOVERY_TRAP
HIGH_CURVATURE_REGION
```

含义：

- `STUCK_LOCAL_MIN`：位置变化小、goal progress 小，表示局部极小值区域。
- `LOW_PROGRESS_CORRIDOR`：机器人仍有小幅移动，但长期无法有效接近目标。
- `SPIN_TRAP`：平均角速度高、平均线速度低、目标进展小。
- `NEAR_OBSTACLE_TRAP`：近障且处于 creep / soft block 类状态。
- `HARD_STOP_RECOVERY_TRAP`：scan_guard hard stop / recovery 相关区域。
- `HIGH_CURVATURE_REGION`：omega sign 频繁切换、目标进展低，表示反复高曲率回绕区域。

检测规则故意保守，宁可少生成 feature，也避免在一开始产生过多紫色点。

---

## 4. MemoryFeature 字段

当前 `MemoryFeature` 保留并规范了：

```text
position
radius
strength
type / feature_type
last_seen_time
hit_count
escape_direction
successful_escape_direction
decay
heading
```

新 feature 与已有 feature 距离小于 `feature_merge_distance` 时会 merge。merge 会：

- 平滑更新 position；
- 更新 escape_direction；
- 增加 hit_count；
- 增强 strength；
- clamp 到 `feature_strength_max`；
- 必要时保留优先级更高的 feature type。

每次 update 会 decay feature strength。超过 `max_features` 时，会优先删除 strength 低、last_seen_time 更旧、hit_count 更小的 feature。

---

## 5. Memory Cost 组成

旧接口保持兼容：

```python
memory_cost_for_state(state) -> float
memory_cost_for_trajectory(trajectory, stride=3) -> float
cost_for_state(state) -> float
cost_for_trajectory(trajectory, step_stride=3) -> float
```

新增 breakdown：

```python
memory_cost_breakdown_for_state(state) -> dict
memory_cost_breakdown_for_trajectory(trajectory, controls=None, stride=3) -> dict
```

返回字段包括：

```text
total
by_type
nearest_type
nearest_distance
nearest_strength
active_feature_count
temperature_scale
```

type-specific potential：

- `STUCK_LOCAL_MIN` / `SPIN_TRAP`：radial repulsive memory potential。
- `LOW_PROGRESS_CORRIDOR`：radial cost + directional penalty，鼓励沿 escape direction 离开。
- `NEAR_OBSTACLE_TRAP` / `HARD_STOP_RECOVERY_TRAP`：更强近区域 penalty，但 cost radius 略收缩，避免误伤过宽。
- `HIGH_CURVATURE_REGION`：轻量区域 penalty；如果传入 controls，则额外统计 omega sign flip 形成 oscillation penalty。

所有 cost 都是 soft cost，不是硬约束。

---

## 6. Temperature Scale

接口：

```python
temperature_scale_for_state(state)
temperature_scale(state, stuck_trap_active=False)
```

逻辑：

- memory disabled 时返回 `1.0`；
- state 靠近 active memory feature 时 scale 增大；
- scale clamp 到 `temperature_boost_max`；
- live sim / ablation 再额外 clamp 到合理范围。

作用：

当机器人靠近历史失败区域时，让 MPPI 权重分布不要过度集中在当前局部最小附近，从而增加探索性。

---

## 7. Escape Direction Helper

新增：

```python
suggest_escape_direction(state, goal=None)
```

逻辑：

1. 找 nearest active feature；
2. 优先使用 `feature.escape_direction`；
3. 如果没有，则使用从 feature 指向当前 state 的径向方向；
4. 如果传入 goal，则和 goal direction 混合；
5. 返回 normalized direction 和 debug；
6. 不返回 `final_control`，不绕开 `scan_guard`。

live sim 的 `apply_escape_bias_to_nominal_sequence()` 会优先使用这个 helper。如果 helper 不可用，会 fallback 到旧的 nearest-feature radial escape 逻辑。

---

## 8. Live Sim 接入

文件：

```text
experiments/mujoco_memory_mppi_live_sim.py
```

主路径保持不变：

```text
MuJoCo geoms
-> raycast_laserscan
-> scan_guard
-> local_obstacle_layer
-> planner_obstacles
-> MPPI + memory
-> safety arbitration
-> final control
```

新增 CSV / log 字段：

```text
memory_nearest_type
memory_nearest_distance
memory_nearest_strength
memory_cost_total
memory_cost_by_type
temperature_scale
escape_active
repeated_spin
stuck_low_progress
```

注意：

- `memory_cost_total` 来自当前 updated trajectory 的 memory breakdown；
- `memory_cost_by_type` 使用字符串写 CSV；
- `escape_active` 只表示 nominal sequence 被偏置，不表示 memory 直接控制机器人；
- `emergency_stop`、`front_soft_block`、`near_body_hard_stop` 不会被 escape bias 绕过。

运行示例：

```bash
python3 experiments/mujoco_memory_mppi_live_sim.py \
  --scene lab_complex \
  --memory-enable \
  --max-steps 60 \
  --no-viewer
```

如果 viewer 可用：

```bash
python3 experiments/mujoco_memory_mppi_live_sim.py \
  --scene u_trap_long_board \
  --memory-enable \
  --max-steps 120
```

---

## 9. Offline Ablation 接入

文件：

```text
experiments/mujoco_memory_mppi_ablation.py
```

重要边界：

```text
offline ablation uses fallback/global circular obstacles;
live sim uses synthetic LaserScan -> local_obstacle_layer.
```

ablation 适合输出 CSV、PNG、GIF 和 compute-time 指标，不代表 live sim 的完整感知闭环。

新增 trajectory 字段：

```text
memory_nearest_type
memory_nearest_distance
memory_cost_total
memory_cost_by_type
memory_temperature_scale
stuck
spin
avoidance_state
```

新增 summary 字段：

```text
memory_nearest_type_final
memory_total_cost_mean
memory_total_cost_max
memory_temperature_scale_mean
memory_feature_types_final
```

benchmark mode：

```bash
python3 experiments/mujoco_memory_mppi_ablation.py \
  --benchmark \
  --seeds 11,12,13 \
  --max-steps 60 \
  --no-viewer
```

普通 run-both：

```bash
python3 experiments/mujoco_memory_mppi_ablation.py \
  --run-both \
  --episodes 1 \
  --max-steps 60 \
  --no-viewer
```

---

## 10. Planner Bridge 兼容

文件：

```text
mppi_hardware_bridge/scripts/mppi_planner_bridge.py
```

仅做小范围兼容：

- 如果 memory field 支持 `memory_cost_breakdown_for_trajectory()`，planner bridge 使用 breakdown total 作为 memory cost；
- debug dict 中增加 `memory_cost_total`、`memory_cost_by_type`、nearest memory 信息；
- 不修改 `mppi_ros_adapter_skeleton.py` 主逻辑；
- 不改变实车 safety arbitration。

---

## 11. 当前实现与理论版本的差异

当前实现不是完整理论 MA-MPPI：

- 没有训练 RL policy；
- 没有 learned dynamics；
- 没有全局语义地图；
- 没有把 memory 建模为概率图模型；
- 没有对 memory feature 做长期任务级学习；
- 没有用 memory 直接生成控制。

当前实现是面向本项目实车/仿真闭环的工程版本：

- 用执行历史识别局部失败区域；
- 用 soft potential 改变 MPPI rollout cost；
- 用 temperature scale 增加局部探索；
- 用 nominal sequence bias 辅助采样云逃离坏区域；
- 保留 scan_guard hard safety。

---

## 12. 调试建议

观察 memory 是否有效时，不要只看紫色点是否出现。需要同时看：

```text
memory_cost_total
memory_cost_by_type
memory_nearest_type
memory_nearest_distance
temperature_scale
escape_active
proposed_control
final_control
front_stop_mode
scan_reason
planner_obstacle_count
```

如果 memory feature 出现但行为没有改善，优先排查：

1. memory cost 是否太弱；
2. planner obstacles 是否过密；
3. scan_guard 是否长期压制 final_v；
4. escape bias 是否触发；
5. escape direction 是否被前方障碍挡住；
6. feature radius 是否过小或过大。

如果 memory 让行为更差，优先检查：

1. feature 是否生成在唯一通路上；
2. feature radius 是否太大；
3. cost weight 是否过高；
4. low progress / spin 检测是否过敏；
5. decay 是否太慢。
