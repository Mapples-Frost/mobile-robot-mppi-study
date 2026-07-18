# L58 闭环结果与 L59 独立确认预注册（2026-07-16）

## 1. L58 development 结果

L58 完成 120/120 episodes，无缺失、重复、非有限指标或 seed 泄漏。nominal 与 ICODE 各 60
episodes，均为 60/60 success、0 collision。

主要结果：

- nominal mean cross-track RMSE：0.06098 m；
- ICODE mean cross-track RMSE：0.04427 m；
- paired relative reduction：26.62%；
- absolute improvement：0.01671 m，hierarchical-bootstrap 95% CI [0.01478, 0.01835] m；
- 3/3 ICODE training blocks 为正，4/4 scenes 为正；
- unseen reverse-S relative reduction：30.38%，95% CI improvement [0.01782, 0.02072] m；
- mean tangent-heading RMSE improvement：0.02351 rad；
- applied-control jerk 相对降低 4.25%；
- nominal/ICODE mean planner compute time：5.66/40.84 ms。

L58 全部预注册 Gate 通过。这支持“multi-step ICODE prediction improvement 能转化为固定高动态 plant
上的闭环 MPPI tracking improvement”，但 L58 仍是 development evidence。

## 2. L59 独立确认设计

L59 不修改 L56 plant、sensor contract、paths、MPPI cost、ICODE checkpoints、terminal controller、
metric、Gate 或统计方法。唯一变化是使用 L58 预先封存、且从未运行过的 seeds 21860771–21860780。

- 3 model blocks × 4 scenes × 10 seeds × 2 conditions = 240 episodes；
- 使用 `--allow-sealed-confirmation` 显式解封；
- conditions 仍仅为 nominal 与 ICODE，RL/memory 关闭；
- 使用与 L58 完全相同的 eligibility Gate；
- 不根据 L58 effect size 提高或降低阈值，保留原始 5% SESOI；
- L59 启动后，不允许再调整模型、控制器或统计脚本。

若 L59 Gate 通过，可将结果表述为固定隐藏高动态 MuJoCo plant 上的独立 seed confirmation；若失败，
必须同时报告 L58 与 L59，不能只保留 development 结果。该确认仍不等价于不同真实小车、不同地面或
动态障碍场景的外部有效性证据。
