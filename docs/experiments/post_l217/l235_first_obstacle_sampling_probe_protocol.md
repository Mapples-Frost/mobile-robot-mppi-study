# L235 首障碍绕行采样能力开发协议

## 目的

L234 smoke 已证明端到端链路完整，但三张大尺度 Tracking 地图均在首个中心线附近静态障碍物处失败。L235 用一个有界探针回答：失败是否主要来自 `K=30`、1 iteration 的采样覆盖不足。

## 固定内容

以下内容不得改变：

- 三张 L234 地图及障碍物；
- Value-Consistent ICODE；
- Residual-Conditioned RL Prior；
- role-aware Reliability-Weighted Value/HSS；
- MPPI cost；
- MuJoCo plant、LaserScan、local obstacle layer、scan_guard 和 safety arbitration；
- Actor/critic/ICODE checkpoints；
- development seed `923301001`。

## 唯一开发改动

- rollout budget：30 → 100；
- MPPI iterations：1 → 2；
- 每个探针最多 700 控制步。

该改动对所有后续方法臂公平共享，不改变论文核心机制。

## 第一阶段矩阵

只运行 Full proposed：

```text
1 method × 3 scenes × 1 nominal_seen domain × 1 development seed
= 3 episodes
```

三个场景可分进程并行，但每个进程固定单线程 BLAS，且不得读取单回合中间效果调参。

## 预注册 Gate

每个场景均需：

- collision = false；
- boundary_violation_steps = 0；
- 轨迹完整输出；
- 进度超过首障碍后预设阈值：
  - Hairpin：`path_completion_ratio >= 0.145`；
  - S-Chicane：`path_completion_ratio >= 0.205`；
  - Infinity：`path_completion_ratio >= 0.135`。

只有三场景全部通过，才允许用相同预算运行四方法资格矩阵。任何失败必须保留并分类，不得筛 seed、删除失败或接触 sealed seeds。
