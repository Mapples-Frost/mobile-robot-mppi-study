# L44 structure-preserving ICODE 路径跟踪预注册

日期：2026-07-16  
状态：运行前冻结。

## 方法假设

五维 dynamic-unicycle 的位姿运动学是已知结构，residual 不应直接改写。L44 使用：

```text
residual component mask = [0, 0, 0, 1, 1]
```

即仅学习 `v_dot` 与 `omega_dot` 的误差。normalized-support gate 仍使用 L42 已冻结的 `soft_z=3`、`hard_z=5`，
超支持时连续退向 nominal；它不读取仿真真值、碰撞或任务结果。

## 三个条件

1. traditional-nominal；
2. traditional-ICODE + dynamics-channel mask；
3. traditional-ICODE + dynamics-channel mask + support gate（primary candidate）。

路径、物理域、3 个 checkpoint 与 L43 保持一致，但使用全新 development seed 21160731–21160735，共 270 episode。
RL、memory、动态障碍关闭；LaserScan、scan_guard、local obstacle layer 与 safety arbitration 保持原链路。

## Primary Gate

mask+support 相对 nominal 必须同时满足：至少 2/3 model block 改善 cross-track RMSE、pooled 相对降幅 ≥10%、
分层 bootstrap 95% CI 下界 >0、success 不减少、collision 不增加、completion ratio 差值 ≥-0.01、平均规划时间 ≤50 ms。
此外，support-gated candidate 的 cross-track RMSE 不得差于只加 mask 的版本。

sealed confirmation seed 21160736–21160745 在 development Gate 通过前保持关闭。

