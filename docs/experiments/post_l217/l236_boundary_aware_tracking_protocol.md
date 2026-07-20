# L236 footprint-aware corridor constraint 开发协议

## 假设

L235 的早期越界由 MPPI rollout cost 缺少 corridor boundary constraint 引起。将论文任务定义中的 footprint 边界条件加入所有方法共享的 MPPI cost，应消除早期越界，同时不改变 RL+ICODE 核心耦合。

## 唯一方法改动

启用共享 Tracking 约束：

```text
margin = w(s) - r - |e_y|
soft_cost = 400 * max(0, 0.05 - margin)^2
hard_cost = 10000 if any margin < 0
```

其中 `w(s)` 支持随路径进度变化，`r` 使用任务配置中的 footprint radius。该功能默认关闭，point-goal 和既有 L217/L223 行为保持不变。

## 固定内容

- L234 三张地图与障碍物不变；
- Full proposed 核心机制、checkpoints、其余 MPPI cost、MuJoCo 和安全链不变；
- `K=100`、2 iterations、700 steps；
- development seed `923301001`；
- 不使用 sealed seeds。

## Gate

第一阶段仍为 Full proposed × 三场景 × nominal_seen × 单 development seed。每个场景均需：

- collision = false；
- boundary_violation_steps = 0；
- 最小 footprint boundary margin ≥ 0；
- Hairpin/S-Chicane/Infinity 完成度分别至少为 0.145/0.205/0.135。

三场景全部通过后，才允许建立四方法短资格矩阵。失败结果必须保留并继续按单变量原则分类。
