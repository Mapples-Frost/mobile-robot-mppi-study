# L221 ICODE 反事实 proposal gate 开发协议

## 动机

L220 证明 source-relative competence 已正确更新，并在 cylinder forest 上保持成功、减少 117 步和 101 次安全干预；但 opposed-U 仍失败。其 Actor-guided elite yield 高于 Gaussian yield，说明短时域 MPPI 精英率把一个长期局部最优误判为高 competence。

## 核心方法不变

L221 仍为：

```text
Value-Consistent ICODE
  + Residual-Conditioned RL Prior
  + role-aware Reliability-Weighted Value/HSS
  + MPPI
```

不修改 ICODE checkpoint/结构、RL checkpoint、MPPI cost、horizon、K、iterations、MuJoCo、地图、LaserScan、local obstacle layer、scan_guard 或 safety arbitration。

## 新增的 HSS 证据

在同一当前状态，使用 planner 已声明的 ICODE dynamics 分别预测：

- residual-conditioned Actor mean；
- trusted GoalWarmStart mean。

只使用 PolylineReference 的只读公开投影计算 horizon 末端路径进度与横向误差。定义：

```text
advantage = actor_progress - baseline_progress
            - 0.5 * max(0, actor_cross_track - baseline_cross_track)
```

当 advantage 不低于 0 时 authority 为 1；在 `[−0.15, 0]` m 内线性衰减；不高于 −0.15 m 时为 0。该 authority 同时约束 Actor sampling center 与 guided sequences，使 RL proposal 成为相对于 trusted baseline 的有界修正。

该比较不读取 simulator future、障碍物真值、scene 标签或 sealed outcome；只使用 planner 本来就允许使用的当前观测、参考路径和 ICODE prediction model。

反事实 authority 只在 adaptive HSS 启用时生效。Simple combination 明确保持 HSS 关闭，不能继承该 gate；这保证 Full 对 Simple 的比较仍隔离所提出的可靠性耦合机制。

## 开发探针

首先仅运行 seed 91001：opposed-U、cylinder forest、serpentine；方法为 ICODE-MPPI 与 Full Proposed；K=30、1 iteration、max 900。

通过条件：

- 全部无碰撞；
- opposed-U Full 恢复成功或至少达到 ICODE 完成度的 95%；
- forest 不丢失成功；
- serpentine 完成度不低于 ICODE；
- counterfactual authority 在回合内发生非平凡变化，且 Actor proposal authority 既非永久 0、也非永久 1。

只有通过后才运行三 development seeds 的六地图 Gate；否则保留结果继续开发，不触碰 sealed seeds。
