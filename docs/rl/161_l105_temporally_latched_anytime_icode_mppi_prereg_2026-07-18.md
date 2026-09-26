# L105：时间锁存式 Anytime ICODE-MPPI 预注册

日期：2026-07-18
状态：在读取 L105 任何闭环结果前冻结

## 1. 对 L104 的公开修正

L104 的单 block smoke test 只用于部署剖析，没有开启完整 development。它发现：即使平均预算约为 K=70，每一拍都先运行 K=50、再按需运行另一个 K=50，会因为两次 ICODE batch launch 的固定开销而不一定快于一次 K=100。该问题不否定预算选择是否正确，但否定了“样本数减少必然等于 wall-clock 减少”的实现假设。

在 L104 smoke 中已经看过的 seed `20271602` 永久排除在 L105 开发与确认之外。L104 原始 smoke artifacts 保留，不覆盖、不混入 L105 数据。

## 2. 等价的控制目标与新的部署机制

bandit、特征、ICODE、contextual covariance、K=50/K=100、任务代价与安全链全部保持冻结。唯一变化是把预算决策看作持续两个控制周期的离散 option：

- 第 1 拍：计算 K=50 诊断并选择 STOP/ADD；
- 第 2 拍：复用上一拍的 STOP/ADD；若为 ADD，直接用一次 vectorized K=100 rollout；
- 第 3 拍重新决策，以此类推。

控制周期为 0.1 s，因此预算上下文以 5 Hz 更新，底层 MPPI 与安全仲裁仍以 10 Hz 更新。该锁存只复用“计算预算”，不复用控制，不跳过 LaserScan、ICODE rollout、MPPI 更新或 scan_guard。

## 3. 实验设计

沿用 L104 的三个条件：fixed K50、fixed K100、latched anytime K50→K100。路线、四个物理域、统计指标和非劣门槛保持不变。

- 新开发 seeds：`20271701–20271703`；
- 若且仅若开发门槛通过，独立确认 seeds：`20271711–20271716`；
- 完整 block 为 scene × physics domain × seed；condition 顺序随机且平衡；
- episode 是独立统计单位。

## 4. 开发集主门槛

1. success 不下降、collision 不增加；
2. mean K ≤ 80；
3. 相对固定 K100，cross-track RMSE 差值的层级 bootstrap 95% CI 上界 ≤ `+0.002 m`；
4. 相对固定 K100，planner mean compute 差值的 95% CI 上界 < 0；
5. elapsed time 差值的 95% CI 上界 ≤ `+0.15 s`；
6. 两条路线和至少两个物理域均观察到 STOP 与 ADD，而不是常数策略；
7. 实际 decision refresh fraction 应接近 0.5，容许因回合终止产生的有限偏差。

## 5. 解释边界

通过门槛只支持“contextual bandit 能在 ICODE-MPPI 中分配在线 rollout 预算，并以较少计算保持 K100 质量”。它不支持：

- SAC 已经成功；
- RL 直接规划了轨迹；
- 任意未知场景都有安全或最优性保证；
- ICODE 与 RL 的所有预想耦合都已完成。

若门槛失败，保留否定结果。任何进一步改变 refresh interval、dual price、特征或 K 值都必须创建新编号、使用新 seeds，并在运行前写明。
