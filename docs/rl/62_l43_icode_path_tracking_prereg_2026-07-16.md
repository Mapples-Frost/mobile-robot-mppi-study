# L43 ICODE clean path-tracking 资格实验预注册

日期：2026-07-16  
状态：指标、路径、seed、模型 checkpoint 和 Gate 在查看 L43 结果前冻结。

## 假设

在没有动态障碍混杂的 MuJoCo 物理平台中，L40 的 on-policy H36 ICODE 相比 nominal dynamic-unicycle model，
应降低真实轨迹到给定 polyline 的横向 RMSE，同时不降低到达率、不增加碰撞，也不降低路径完成度。

## 设计

- residual factor：traditional-nominal / traditional-ICODE；
- model block：3 个独立 L40 ICODE training seed；
- path：gentle-S、double-turn、slalom；
- physics domain：40 ms matched-delay 与 100 ms long-delay；
- development episode seed：21060731–21060735；
- 总计：3 × 3 × 2 × 5 × 2 = 180 episode；
- dynamic obstacles：0；memory：off；RL：off；scan_guard 与完整安全链保持开启；
- sealed confirmation seed：21060736–21060745，development Gate 通过前不可开启。

## 主要指标

对每个真实轨迹点投影到 polyline 的最近线段：

- cross-track RMSE（primary）；
- cross-track P95；
- 相对 RMSE 降幅；
- tangent heading RMSE；
- maximum monotonic path-completion ratio；
- success、collision、final distance、planner compute time。

episode seed 是 model block 内的重复测量；control step 不是独立样本。置信区间采用 model-block/scene/domain/seed 分层重采样。

## 预注册 Gate

必须同时满足：

1. 3 个 model block 中至少 2 个 cross-track RMSE 改善；
2. pooled relative cross-track RMSE reduction ≥ 10%；
3. cross-track improvement 的分层 bootstrap 95% CI 下界 > 0 m；
4. pooled success 不减少、collision 不增加；
5. mean completion-ratio difference ≥ -0.01；
6. ICODE mean planner time ≤ 50 ms；
7. 180 个 episode 完整，无保护/封存 seed 泄漏和非有限指标。

Gate 未通过时不开启 confirmation，也不将 path tracking 声称为正向结果。

