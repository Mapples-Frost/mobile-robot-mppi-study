# L43 clean path-tracking 结果：离线预测优势未稳定转化为控制优势

日期：2026-07-16  
结论等级：development；预注册 Gate 未通过；封存种子未开启。

## 完整性与总体结果

180/180 个 MuJoCo episode 完成，无缺失、非有限值或 seed 泄漏。相对 nominal，当前 full-state ICODE 的 pooled
cross-track RMSE 相对变化为 `-6.17%`（负号表示恶化），平均绝对变化为 `-0.0034 m`，分层 bootstrap 95% CI
`[-0.0256, +0.0107] m`。只有 1/3 个模型 block 的平均横向误差改善；净减少 3 次成功，碰撞不变。

## 关键失效模式

主要负效应集中于 matched-delay 的 slalom 末端。失败 episode 已经推进到最后 polyline target，但越过终点后在目标附近持续转圈。
此时航向和控制进入训练支持边缘，full-state residual 可以直接修改 `x_dot/y_dot/theta_dot`，从而破坏已知的差速车运动学结构并改变
MPPI 轨迹排序。

## 科研决策

L43 否定的是“对五个状态导数全部自由学习的 residual 可以直接用于当前 MPPI”，不是 residual learning 本身。
下一轮将已知运动学固定为：

```text
x_dot = v cos(theta)
y_dot = v sin(theta)
theta_dot = omega
```

仅允许 ICODE 修正未知动力学通道 `v_dot`、`omega_dot`，并将该结构约束与冻结的 normalized-support gate 做消融。
L43 封存 seed 保持关闭。

