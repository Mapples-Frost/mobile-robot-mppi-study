# L107：ICODE + contextual-RL MPPI 最终包独立确认预注册

日期：2026-07-18
状态：在运行任何 L107 episode 前冻结
阶段：独立确认，不再进行超参数选择

## 1. 方法冻结

本轮确认一个克制、已经具备独立模块证据的完整包：

1. ICODE control-affine residual 替换 MPPI 的 nominal rollout dynamics；
2. 离线训练的 LinUCB contextual policy 读取参考路径的刚体变换不变量，选择 `narrow=[0.5,0.5]` 或 `speed=[1.75,0.75]` 探索协方差；
3. MPPI 使用 K=50，而强对照使用 K=100；
4. 所有方法使用相同的动作 rate limit、control-rate cost、horizon、任务代价、MuJoCo plant、LaserScan、scan_guard 和 safety arbitration；
5. memory、SAC sequence prior、terminal value、anytime budget 全部关闭。

该 contextual bandit 是一阶段 RL。它不直接输出控制，不偷看 physics label 或 simulator truth。ICODE 负责“每条候选轨迹预测是否准确”，RL 负责“有限样本向哪个控制扰动方向探索”。

## 2. 为什么本轮不继续 SAC/value/anytime

- end-to-end SAC sequence prior 在同优化器对照下没有产生独立正贡献；
- L104–L106 anytime budget 的同状态反事实能力没有稳定转化为闭环 compute superiority，且三拍锁存出现跟踪退化；
- 这些否定结果保留，不混入本轮叙事。

L107 不宣称已经复现 RL-Driven MPPI 论文中的 policy warm start + terminal value 全算法，而是检验从该论文抽取的核心原则：用离线学习策略引导 MPPI 的在线采样计算。ICODE 部分对应 ICODE-MPPI 公开描述的连续时间 control-affine residual；不声称原始 ICODE 的稳定性或收敛保证。

## 3. 三个冻结条件

1. `nominal_fixed_k100`：传统 nominal dynamics + 最强固定 covariance + K100；
2. `icode_fixed_k100`：仅替换为 ICODE dynamics，其余同上；
3. `proposed_icode_contextual_k50`：ICODE + 冻结 contextual RL + K50。

## 4. 场景、物理域和独立样本

- held-out hairpin、held-out reverse-S；
- train anchor、high friction、long command delay、combined moderate；
- 六个全新 seeds：`20271911–20271916`；
- 共 `2 × 4 × 6 × 3 = 144` 个 MuJoCo 闭环 episodes；
- scene × domain × seed 为 block，三个 condition 的运行顺序随机且平衡；
- episode 是独立统计单位，控制 step 不是独立样本。

## 5. 主要假设与门槛

### H1：完整包优于传统 nominal K100

`proposed - nominal_fixed_k100` 必须同时满足：

- success 不下降、collision 不增加；
- cross-track RMSE 配对差的层级 bootstrap 95% CI 上界 < 0；
- elapsed time 95% CI 上界 < 0；
- mean planner compute ≤ 50 ms。

### H2：RL 半预算相对 ICODE K100 保持控制质量并节省资源

`proposed - icode_fixed_k100` 必须同时满足：

- success 不下降、collision 不增加；
- RMSE 95% CI 上界 ≤ +2 mm；
- issued jerk 95% CI 上界 ≤ +0.010；
- physically applied jerk 95% CI 上界 ≤ +0.010；
- elapsed time 95% CI 上界 < 0；
- planner mean compute 95% CI 上界 < 0。

jerk margin 0.010 在查看 L107 前冻结，约为既有 K100 jerk 水平的 10%–15%；它表示允许的小幅工程差异，不等同于证明平滑性优越。

### H3：ICODE 的独立贡献

报告 `icode_fixed_k100 - nominal_fixed_k100` 的完整区间。其 RMSE 95% CI 上界必须 < 0，证明完整包的动力学收益不是由 RL sampler 冒充。

只有完整性审计和 H1/H2/H3 全部通过，才称 L107 final package Gate 通过。

## 6. 统计与报告规则

- 分层 bootstrap 先按 scene × physics domain，再按 seed 重采样；
- 报告均值差、95% CI、每条件描述统计及原始 episode CSV；
- 不删除失败 seed、不换 margin、不改 reward、不在本轮选择新 covariance；
- planner timing 使用单进程顺序运行，禁止并发负载；
- 144/144 成功并不是预设，任何失败或碰撞都原样保留。

## 7. 允许与禁止的结论

若通过，可支持：在两条 held-out clean path-tracking 路线和四个 MuJoCo 物理域中，contextual RL 能让 ICODE-MPPI 用一半 rollout 预算保持 ICODE-K100 的安全、精度和平滑非劣，同时显著缩短时间与计算；完整包优于传统 nominal-K100。

仍禁止宣称：动态障碍已解决、实车已验证、SAC 已成功、理论稳定性已证明、或两篇论文被逐算法完整复现。
