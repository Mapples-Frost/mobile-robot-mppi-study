# L42 ICODE 支持域门控结果：改善 ungated 成功率，但未达到 nominal 资格线

日期：2026-07-16  
结论等级：development；预注册 Gate 未通过；封存种子未开启。

## 完整性

- 270/270 个 MuJoCo episode 完成；
- 3 个独立 ICODE 训练 seed × 3 个动态场景 × 2 个物理延迟域 × 5 个全新 episode seed × 3 个条件；
- 无缺失、重复键、NaN/Inf、历史保护 seed 或封存 seed 泄漏；
- support gate 的 `soft_z=3.0`、`hard_z=5.0` 在运行前冻结。

## 结果

相对 ungated ICODE，support-gated ICODE 净增加 9 次成功，分层 bootstrap 的 success-rate difference 为
`+0.100`，95% CI `[+0.022, +0.200]`。这说明“超出训练支持时退向 nominal”确实修复了部分错误预测。

但相对真正的 traditional-nominal baseline，gated ICODE 仍净减少 1 次成功、净增加 4 次碰撞，平均终点距离恶化
`0.041 m`。碰撞差的 95% CI 为 `[+0.011, +0.100]`，因此不能将门控后的模型宣布为安全非劣。

## 机制判断

门控只在平均约 7.0% 的 rollout step 上降低 residual，且没有 step 被完全关闭。它能修复部分超支持状态，
却不能解决动态障碍未来运动不可见、残差目标与 MPPI cost ranking 不完全一致等问题。

本轮动态场景的碰撞结果不能用于单独判定 residual dynamics 是否改善机器人自身的路径跟踪，因为碰撞同时受动态障碍预测、
LaserScan、local obstacle layer 和 safety arbitration 支配。下一轮改用无动态障碍、明确曲线参考的 residual-only 资格实验，
以 cross-track RMSE、heading RMSE 和 path completion 作为主要终点。

## 决策

L42 封存种子保持关闭；不在同一批 seed 上继续调 support threshold。L43 使用全新开发 seed，并在运行前冻结路径、物理域、指标和门槛。

