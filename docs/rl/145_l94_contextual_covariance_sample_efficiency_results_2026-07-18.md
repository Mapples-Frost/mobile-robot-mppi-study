# L94 contextual covariance sample efficiency：主 Gate 通过

日期：2026-07-18  
结论等级：预注册、独立 seeds、held-out geometry、四物理域闭环结果。

## 1. 完整性

- 320/320 个 MuJoCo episode；
- 320 个唯一 `(route, physics, K, condition, seed)` 主键；
- `K=50/100/200/400` 各 80 个 episode；
- learned/fixed 各 160 个 episode；
- 5 个新 seeds，与 L89 拟合和确认均不重叠；
- 320/320 success，0/320 collision；
- 主要数值字段无 NaN/Inf；
- ICODE checkpoint、bandit checkpoint、MPPI horizon/cost 和安全链全部冻结。

## 2. 预注册主比较：learned K=50 vs fixed K=100

40 个 route × physics × seed 配对的 learned-50 minus fixed-100：

| 指标 | 均值差 | 分层 bootstrap 95% CI | Gate |
|---|---:|---:|---|
| Success | 0 | -- | 通过 |
| Collision | 0 | -- | 通过 |
| Cross-track RMSE | -4.108 mm | [-7.009, -0.741] mm | 精度通过 |
| Elapsed time | -1.738 s | [-2.720, -0.760] s | 时间通过 |
| Planner compute / step | -18.291 ms | [-32.905, -2.693] ms | 计算通过 |
| Control jerk | +0.00944 | [+0.00330, +0.01573] | 次要指标变差 |

四项预注册主 Gate 全部通过，`primary_gate_passed=true`。因此当前数据支持：在这两条未
参与 bandit 拟合的路线和四个物理域中，路线几何 contextual bandit 只用一半 MPPI
rollout 数，就能保持安全，并同时提高跟踪精度和到达速度、降低 planner 计算。

![L94 sample efficiency](../../results/research_platform/rl/l94_contextual_covariance_sample_efficiency_20260718_v1/fig_l94_sample_efficiency.png)

需要保留 jerk 的代价：half-budget learned 的控制变化略大，且其区间不跨零。不能把本轮
描述成所有指标全面占优。

## 3. 同预算结果

learned minus fixed 在每个 K 的时间优势均稳定：

| K | Elapsed-time mean | 95% CI | RMSE mean | 95% CI |
|---:|---:|---:|---:|---:|
| 50 | -1.395 s | [-2.400, -0.363] | -1.577 mm | [-3.103, -0.330] mm |
| 100 | -1.422 s | [-2.470, -0.368] | -1.010 mm | [-2.005, -0.199] mm |
| 200 | -1.498 s | [-2.600, -0.378] | +0.034 mm | [-0.291, +0.348] mm |
| 400 | -1.560 s | [-2.720, -0.395] | -0.160 mm | [-0.711, +0.323] mm |

RL 的时间优势没有随着 K 增大而消失；精度优势在低 K 最明显，在 K=200/400 时两种方法
接近。这符合“较少样本时，采样方向更重要”的机制解释，但不是样本复杂度定理。

## 4. 计时审计边界

L94 为提高吞吐量使用多进程并行。condition 与 K 被随机化，因此闭环性能比较有效；但
wall-clock planner timing 可能受跨进程资源竞争影响。虽然 half-budget compute 的区间已
完全低于零，仍单独预注册 L95：单进程、交替 arm、新 seeds 的 80-episode confirmation。
正式论文优先引用 L95 的 compute 数值；L94 用于完整 K-scaling 曲线。

## 5. 论文含义

这比“RL 在相同 K 下稍快”更有实际价值：RL 不替代 MPPI 控制，也不输出轮速，而是把
有限 rollout 投向更有价值的控制扰动方向。ICODE 仍负责每条 rollout 的动力学预测；两者
职责分离，能够做清晰消融。

证据范围仍限制在 clean static path tracking、两条 held-out route 和四个 MuJoCo physics
domain。动态障碍预测、真实小车和更广路线族尚未由本轮证明。

原始结果：
`results/research_platform/rl/l94_contextual_covariance_sample_efficiency_20260718_v1/summary.json`。
