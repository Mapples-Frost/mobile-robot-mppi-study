# L57 离线结果与 L58 高动态闭环预注册（2026-07-16）

## 1. L57 离线结果

3 个独立 ICODE model blocks 全部通过冻结 Gate。相对 nominal predictor 的平均改善为：

- test active derivative RMSE：25.1%；
- unseen active derivative RMSE：28.4%；
- test H36 rollout RMSE：36.3%；
- unseen H36 rollout RMSE：41.8%；
- test H36 endpoint position RMSE：43.7%；
- unseen H36 endpoint position RMSE：42.3%。

上述长期指标在 3/3 model blocks 中同方向。unseen H1/H5 endpoint position 的个别结果略为负，
因此当前结论严格限定为：multi-step prediction 明显改善，但尚未证明 closed-loop MPPI 改善。

## 2. L58 研究问题

在相同固定隐藏 MuJoCo plant 上，L57 的 ICODE prediction 是否能在全新 episode seeds 上降低
MPPI path-tracking error，同时不损害成功率、碰撞率和控制平滑性？

## 3. 设计冻结

- 条件：`traditional_nominal` 与 `traditional_icode`；
- ICODE blocks：training seeds 20261201、20261202、20261203；
- 场景：3 条 seen paths 与整条未参与训练的 reverse-S unseen path；
- development episode seeds：21860761–21860765；
- 每个条件使用相同 plant、相同 seed 和相同 MPPI random seed，进行 paired comparison；
- 总规模：3 blocks × 4 scenes × 5 seeds × 2 conditions = 120 episodes；
- memory、RL prior、RL correction 均关闭；ICODE 不读取 MuJoCo 隐藏参数；
- 当前是 development gate，不使用 21860771–21860780 sealed confirmation seeds。

## 4. 冻结 Gate

Primary metric 为 paired cross-track RMSE reduction。通过条件：

1. artifact/seed/config 完整，无 protected 或 sealed seed 泄漏；
2. 3/3 model blocks 的 mean cross-track improvement 为正；
3. pooled relative cross-track RMSE reduction 不低于 5%；
4. hierarchical-bootstrap cross-track improvement 95% CI lower > 0 m；
5. unseen reverse-S 至少 2/3 blocks 为正，relative reduction 不低于 5%，其 CI lower > 0 m；
6. 四个场景中至少 3 个 mean improvement 为正；
7. net success gain >= 0，net collision increase <= 0；
8. completion ratio difference >= -0.01；
9. relative applied-control jerk increase <= 10%；
10. ICODE mean planner compute time <= 50 ms。

只有 Gate 通过，才允许设计 sealed confirmation；若失败，保留全部结果并先定位是 closed-loop
cost sensitivity、模型偏差还是场景特异性问题，不以离线结果替代闭环证据。
