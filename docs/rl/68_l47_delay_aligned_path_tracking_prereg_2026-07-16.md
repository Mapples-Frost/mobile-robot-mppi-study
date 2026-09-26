# L47 delay-aware structure-preserving ICODE 路径跟踪预注册

日期：2026-07-16  
状态：运行前冻结。

## 条件

1. delay-aware traditional-nominal；
2. delay-aware structure-preserving ICODE（primary）；
3. 同一 ICODE + frozen normalized-support gate（secondary ablation）。

三种条件都使用相同已知 command delay、上一条 safety-executed command、3 条路径、2 个物理域、LaserScan 与完整安全链。
RL 和 memory 关闭。3 个 ICODE checkpoint 来自独立 seed 20261001–20261003。

development seed 为 21260731–21260735，共 270 episode；sealed confirmation seed 21260736–21260745 在 Gate 通过前关闭。

## Primary Gate

primary 相对 nominal 必须满足：至少 2/3 model block 的 cross-track RMSE 改善、pooled 相对降幅 ≥10%、分层 bootstrap
95% CI 下界 >0、success 不减少、collision 不增加、completion ratio 差值 ≥-0.01、平均规划时间 ≤50 ms。

support-gated 条件只作为预先声明的 secondary ablation，不用于替换 primary Gate。development 通过后才允许打开 sealed confirmation。

