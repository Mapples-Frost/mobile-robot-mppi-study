# L95 half-budget 单进程独立确认：主 Gate 再次通过

日期：2026-07-18  
结论等级：预注册、新 seeds、单进程、arm-order balanced confirmation。

## 1. 完整性

- 80/80 个 MuJoCo episode；
- learned `K=50` 与 fixed `K=100` 各 40；
- 2 条未参与 bandit 拟合的路线 × 4 物理域 × 5 个新 seeds；
- 每个 pair 连续执行，arm 顺序交替；
- 无其他 benchmark worker 并发；
- 80/80 success，0/80 collision；
- ICODE、bandit、MPPI horizon/cost、安全链均冻结。

## 2. 结果

learned-50 minus fixed-100：

| 指标 | 均值差 | 分层 bootstrap 95% CI | Gate |
|---|---:|---:|---|
| Success | 0 | -- | 通过 |
| Collision | 0 | -- | 通过 |
| Cross-track RMSE | -0.273 mm | [-1.816, +1.520] mm | +2 mm 非劣通过 |
| Elapsed time | -1.725 s | [-2.680, -0.770] s | 通过 |
| Planner compute / step | -10.359 ms | [-15.085, -6.908] ms | 通过 |
| Control jerk | +0.01029 | [+0.00390, +0.01658] | 次要代价 |

四项主 Gate 再次全部通过，`primary_gate_passed=true`。与 L94 相比，RMSE 的优势幅度
缩小并跨零，但仍稳健满足预注册的 +2 mm 非劣界；时间和 compute 区间仍完全低于零。

## 3. 当前可写入论文的结论

在冻结 ICODE-MPPI、held-out hairpin/reverse-S 和四个 MuJoCo 物理域下，路线几何
contextual bandit 使用 50 个 samples，相对最强全局固定协方差 100 个 samples：

1. 保持相同的成功与碰撞结果；
2. 跟踪 RMSE 在 2 mm margin 内非劣；
3. 平均提前约 1.7 s 完成；
4. 单进程 planner compute 平均减少约 10.4 ms/step。

因此“RL-guided sampling 能在该任务族中把 MPPI candidate budget 减半”已有两套新 seed
实验支持，其中一套覆盖完整 K scaling，另一套控制 wall-clock 并发。

## 4. 必须同时报告的限制

- half-budget learned 的 control jerk 稳定增加，不能声称所有指标占优；
- 这是经验性的 tested-task sample efficiency，不是理论样本复杂度结论；
- 当前只覆盖 clean static path tracking；
- 动态 crossing 的 L92/L93 结果不支持 covariance adaptation，需明确列为局限；
- 实车结果尚未完成。

原始结果：
`results/research_platform/rl/l95_contextual_covariance_half_budget_confirmation_20260718_v1/summary.json`。
