# L36 动态 benchmark 校准结果：可辨识场景族建立

日期：2026-07-16  
结论等级：benchmark development；预注册 Gate 通过。

## 1. 结果

传统 MPPI 在 8 个候选 × 8 个全新 seed，共 64 个 MuJoCo 闭环 episode 中产生了 5 个
同时包含 success 与 collision 的 eligible 候选。产物完整，无重复或非有限指标。

按预注册的 collision-rate 目标 0.25/0.50/0.75，确定：

| 难度 | 场景 | Success | Collision | Timeout |
|---|---|---:|---:|---:|
| Easy | oblique_mid | 5/8 | 2/8 | 1/8 |
| Moderate | anti_diag_small | 1/8 | 4/8 | 3/8 |
| Hard | oblique_steep | 1/8 | 5/8 | 2/8 |

三个场景的 collision-rate span 为 `0.375`，超过预注册阈值 `0.25`，L36 Gate 通过。

![L36 calibration](../../results/research_platform/rl/l36_dynamic_benchmark_calibration_20260716_v1/fig_l36_dynamic_benchmark_calibration.png)

## 2. 解释边界

本轮没有加载 RL 或残差 checkpoint，因此场景选择没有使用本文方法的结果。它只证明
新的动态场景族比 L35 的“全撞/全不撞”二分更适合比较控制器，并不证明 RL 或 ICODE
有效。三个场景仍属于 development benchmark；正式 confirmation 必须使用新 seed 和未
参与选择的运动参数。

## 3. 下一步

使用新 seed 在三个场景上运行严格 2×2：

```text
traditional / bounded RL  ×  nominal / ICODE
```

训练 seed 作为模型 block，场景—episode 作为 block 内重复测量，以分别估计 RL 主效应、
ICODE 主效应和交互项。

