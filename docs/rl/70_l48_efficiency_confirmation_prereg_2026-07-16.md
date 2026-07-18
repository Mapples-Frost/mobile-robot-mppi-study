# L48 ICODE 路径效率与控制平滑性独立确认预注册

日期：2026-07-16  
状态：运行前冻结。

## 确认假设

在成功率和安全性不劣、cross-track RMSE 实际非劣的前提下，delay-aware structure-preserving ICODE 相对 delay-aware nominal：

1. 缩短真实 MuJoCo 轨迹长度；
2. 降低发出命令的 control jerk；
3. 降低经过 command delay 后实际施加命令的 jerk。

## 设计

- 条件：traditional-nominal / structure-preserving ICODE；
- 3 个独立 ICODE training seed；
- 3 条固定路径 × 2 个物理延迟域；
- 10 个全新 confirmation seed 21360731–21360740；
- 总计 360 episode；
- RL、memory、动态障碍关闭；所有安全与感知链保持一致；
- L47 与所有历史 episode seed 被列为 protected。

## Gate

必须同时满足：

- 3/3 model block 的平均 path-length reduction >0；
- 3/3 model block 的平均 control-jerk reduction >0；
- pooled mean path-length reduction ≥0.03 m，且分层 bootstrap 95% CI 下界 >0；
- control jerk 与 applied jerk reduction 的 95% CI 下界均 >0；
- cross-track RMSE 相对增幅 ≤5%；
- success 不减少、collision 不增加、completion-ratio difference ≥-0.01；
- candidate mean planner time ≤50 ms；
- 360 个 episode 完整且无历史/封存 seed 泄漏。

任一条件失败即判定确认失败，不选择性报告单一终点。

