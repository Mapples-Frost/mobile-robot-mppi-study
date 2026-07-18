# L108：ICODE + contextual RL（等预算 K100）独立确认预注册

日期：2026-07-18
状态：在运行任何 L108 episode 前冻结

## 1. 从 L107 得到的边界

L107 的 144/144 回合完整性通过。`ICODE + contextual RL K50` 相对传统 `nominal fixed K100` 的 RMSE 和 elapsed time 均显著改善，且 ICODE 单独相对 nominal 的 RMSE 改善显著；但 contextual K50 相对 ICODE fixed K100 的 RMSE 非劣门槛失败。因此：

- 保留“ICODE + RL 完整包优于传统 nominal”信号；
- 删除“半预算保持 ICODE-K100 精度”的主张；
- L108 将三个方法预算统一为 K100，只检验 RL covariance policy 本身的独立贡献。

这不是放宽 L107 门槛。L107 结论仍为 primary Gate 失败，数据永久保留。L108 使用全新 seeds，回答新的等预算问题。

## 2. 冻结条件

1. nominal dynamics + fixed best covariance + K100；
2. ICODE dynamics + fixed best covariance + K100；
3. ICODE dynamics + frozen contextual-bandit covariance + K100。

三者共享相同 MuJoCo plant、两条 held-out 路线、四个物理域、代价、rate limit、LaserScan、scan_guard 和 safety arbitration。memory、SAC prior/value、anytime budget 全部关闭。

## 3. 独立实验

- seeds：`20272011–20272016`，未用于 L89、L94–L108 任何拟合或已查看结果；
- `2 routes × 4 domains × 6 seeds × 3 arms = 144 episodes`；
- 单进程、随机平衡 condition 顺序；episode 为独立单位。

## 4. 预注册门槛

### H1：完整包 vs nominal fixed K100

- success 不下降、collision 不增加；
- RMSE 95% CI 上界 < 0；
- elapsed time 95% CI 上界 < 0；
- proposed mean planner compute ≤ 50 ms。

### H2：contextual RL 的等预算独立贡献

`proposed - ICODE fixed K100`：

- success 不下降、collision 不增加；
- RMSE 95% CI 上界 < 0；
- elapsed time 95% CI 上界 < 0；
- issued/applied jerk 95% CI 上界分别 ≤ +0.010；
- planner mean compute 95% CI 上界 ≤ +2.0 ms。

2 ms 是部署开销非劣界，不代表要求 RL 显著加速；两种方法的 rollout 数相同，主张是“无实质计算开销”。

### H3：ICODE 独立贡献

`ICODE fixed K100 - nominal fixed K100` 的 RMSE 95% CI 上界 < 0，且安全不退化。

完整性审计与 H1/H2/H3 均通过才称 final package Gate 通过。

## 5. 结论边界

通过只支持：在 clean static path tracking、两条 held-out geometry 和四个 MuJoCo 物理域中，离线 contextual RL 能在相同 K 下为 ICODE-MPPI 选择更好的探索分布；ICODE 与 RL 各自贡献可由强对照分开识别。

它仍不是 RL-Driven MPPI 原论文中 SAC policy warm start + terminal value 的逐算法复现，也不证明动态障碍、实车或理论稳定性。
