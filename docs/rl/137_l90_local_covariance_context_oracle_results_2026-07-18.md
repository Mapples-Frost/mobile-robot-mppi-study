# L90 局部采样协方差 Context Oracle：未通过 Gate

日期：2026-07-18  
结论等级：预注册 development gate；不能作为论文正向结论。

## 1. 问题

L89 已证明：在完整参考路径层面，根据路径几何选择 MPPI 采样协方差，优于最强的全局
固定协方差。本轮进一步检验更强的假设：是否还需要在同一条路径的不同局部区段频繁切换
采样尺度。

为避免先训练 RL 再寻找解释，本轮先运行不可部署的局部 context oracle。若 oracle 本身都
没有稳定收益，就不应继续训练局部切换策略。

## 2. 预注册设计

- 4 条 support route：straight、sweep、accel-turn、chicane；
- 每条路线 3 个锚点：路径进度 0.10、0.35、0.60；
- 2 个物理域：train-anchor、combined-moderate；
- 2 个候选协方差：`narrow=[0.5,0.5]`、`speed=[1.75,0.75]`；
- 独立 selection seeds 2 个、evaluation seeds 3 个；
- 每个分支执行 30 step（3 s），共 240 个 MuJoCo 分支；
- ICODE checkpoint、MPPI、LaserScan、scan_guard、安全仲裁全部冻结。

选择规则先保证无碰撞，再在局部 RMSE 容差 2 mm 内最大化路径推进量。主 Gate 要求：

1. 至少选择两种动作；
2. 至少 25% 的局部 context 不同于最优全局固定动作；
3. evaluation 中路径推进量差值的分层 bootstrap 95% CI 下界大于 0；
4. RMSE 恶化的 95% CI 上界不超过 2 mm；
5. 不增加碰撞。

完整设计见
[`135_l90_local_covariance_context_oracle_prereg_2026-07-18.md`](135_l90_local_covariance_context_oracle_prereg_2026-07-18.md)。

## 3. 完整性

- 240/240 个预注册分支完成；
- 12 个 route-anchor context；
- selection 与 evaluation seed 分离；
- 没有重复主键或缺失分支；
- planner 未访问 MuJoCo 障碍物真值；
- 本轮未训练或调参 ICODE。

## 4. 结果

全局最优候选为 `speed`。局部 oracle 在 12 个 context 中有 11 个仍选择 `speed`，只有
`chicane@0.35` 选择 `narrow`，即只有 `1/12 = 8.3%` 的 context 不同于全局动作，低于
预注册的 25% 异质性门槛。

相对最优全局固定动作：

- 局部路径推进量：`+0.001385 m`，95% CI `[0, +0.004884] m`；
- 局部 RMSE：`-0.000107 m`，95% CI `[-0.000481, +0.000080] m`；
- 碰撞率差：`0`。

精度和安全约束通过，但推进量 CI 下界没有严格大于 0，异质性 Gate 也未通过。因此主
Gate 未通过。

## 5. 科研解释

本结果不否定 L89 的路线级 RL 价值。它缩小了 RL 的合理作用尺度：

- **有证据支持**：在进入一条路线时，根据整体几何选择采样协方差；
- **暂无证据支持**：在同一路线内按局部短窗口频繁切换协方差。

短窗口高频切换会增加策略抖动、滞回设计、训练样本需求和实车审计难度，而 oracle 只给出
毫米级、置信区间触零的收益。按照预注册停止规则，本阶段不训练局部切换 RL。

## 6. 后续决策

下一 Gate 转向导师明确提出的动态障碍问题：先检验静态/不同动态强度是否真的对应不同的
最优采样行为。仍采用“oracle 先行、策略后行”的顺序；只有存在独立 evaluation 的稳定
headroom，才使用 LaserScan 可观测量训练可部署策略。

原始结果：
`results/research_platform/rl/l90_local_covariance_context_oracle_20260718_v1/summary.json`。
