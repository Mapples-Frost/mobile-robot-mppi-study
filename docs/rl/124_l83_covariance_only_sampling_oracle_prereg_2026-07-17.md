# L83：covariance-only RL sampling 可学习性预注册

日期：2026-07-17  
性质：development oracle diagnostic；在 L83 数据产生前冻结

## 1. 背景

L79--L81 表明 always-on RL mean correction 不能跨独立训练块稳定复现；L82 pilot 又表明不能把
ICODE 对固定 36 步序列的排序直接当作优于 MLP 的 teacher。L83 保留 RL + ICODE 和 sampling
prior 主线，但把 RL 的动作空间收缩为两个采样尺度：

\[
a_t^{\mathrm{RL}}=(s_{v,t},s_{\omega,t}),
\]

\[
q(U\mid o_t)=\mathcal N\!\left(
\mu_{\mathrm{traditional}}(o_t),
\operatorname{diag}[(s_t\odot\sigma_0)^2]
\right).
\]

RL 不再修改候选分布均值。ICODE 仍作为 MPPI rollout dynamics，MPPI 仍执行 importance
reweighting，安全仲裁保持不变。

## 2. 先验可证伪问题

在训练任何 covariance-only SAC 前先检验：

> 不同物理状态下，最佳采样协方差尺度是否确实不同；一个知道当前状态的 oracle selector
> 是否优于所有固定尺度？

如果所有 anchor 都偏好同一固定尺度，则没有必要使用 RL；直接把该尺度写入 MPPI 配置更简单。

## 3. 冻结尺度动作

固定五个离散诊断动作，不做事后网格搜索：

| 名称 | (s_v) | (s_\omega) | 含义 |
|---|---:|---:|---|
| narrow | 0.50 | 0.50 | 两维都收缩 |
| baseline | 1.00 | 1.00 | 现有 MPPI |
| turn_explore | 0.75 | 1.75 | 主要扩大转向探索 |
| speed_explore | 1.75 | 0.75 | 主要扩大速度探索 |
| broad | 1.75 | 1.75 | 两维都扩大 |

每个 anchor、每个尺度使用相同标准高斯噪声张量 (Z)，只改变逐维 scale。候选数固定为
`K=64`，horizon 固定 36，sample 0 固定为 traditional mean。

## 4. 评价流程

每个 anchor：

1. 从冻结 ICODE-MPPI 的 traditional/previous-sequence prior 读取同一均值；
2. 保存完整 MuJoCo snapshot；
3. 对每个尺度构造候选并用 ICODE rollout 计算 base cost；
4. 使用与控制器相同的 importance-sampling correction 和温度计算 MPPI 权重；
5. 得到该尺度下的 weighted sequence，并应用同样的第一步 rate limit；
6. 从同一 snapshot 在 MuJoCo 中执行完整 weighted sequence；
7. 用同一 cost contract 得到真实 36 步成本；
8. 恢复 snapshot，继续参考闭环 episode。

场景仍无障碍，counterfactual 分支只用于动力学/采样诊断；正式控制链仍保留 LaserScan、scan
guard 和 safety arbitration。

## 5. smoke 与 pilot

smoke：block 0、`accel_turn`、anchor `20,50`。  
mechanism pilot：3 个冻结 ICODE blocks，场景 `accel_turn`、`chicane`、
`hairpin_unseen`、`reverse_s_unseen`，anchor `20,50,80,110`。

同一 scene 内 anchor 是重复测量，不冒充独立 replicate。模型块是主要工程重复单位。

## 6. 主要指标

令 (C_{i,a}) 为 anchor (i) 采用尺度 (a) 后 weighted sequence 的真实成本。

1. 最佳固定尺度：
   \[
   a_{\mathrm{fixed}}^*=\arg\min_a\frac1N\sum_i C_{i,a}.
   \]
2. 状态 oracle：
   \[
   C_i^{\mathrm{oracle}}=\min_a C_{i,a}.
   \]
3. oracle 相对最佳固定尺度的平均收益：
   \[
   \Delta_{\mathrm{context}}=
   \frac1N\sum_i(C_{i,a_{\mathrm{fixed}}^*}-C_i^{\mathrm{oracle}}).
   \]
4. baseline 相对 oracle 的平均 regret；
5. 最佳动作的分布、熵和 baseline-optimal fraction；
6. 每尺度 ICODE predicted ESS、true cost、weighted sequence hash；
7. seen/unseen 和 block 分层结果。

## 7. 升级 Gate

只有同时满足以下条件，才训练 covariance-only SAC：

- `context oracle improvement > 0`；
- 至少 2/3 blocks 的 context oracle improvement 为正；
- 至少两个不同尺度分别在不少于 10% anchor 上成为最优，排除单一固定尺度；
- unseen 子集的 context oracle improvement 不为负；
- baseline 没有在超过 90% anchor 上最优；
- 所有快照、轨迹、cost、weights、ESS 和 hash 完整有限。

pilot 只决定是否值得进入训练，不构成论文性能结论。若 Gate 失败，停止 covariance-only RL，下一候选是
受限的 `K/horizon` budget allocator 或 RL active residual collection；不得通过扩大 scale grid 事后挽救。

## 8. RL 训练边界（仅在 Gate 通过后）

- policy action 只有两个协方差尺度，不含 mean correction；
- continuous action 经对数映射限制在 `[0.5, 2.0]`；
- actor 零输出精确对应 scale 1.0；
- gate authority 为零时协方差精确退化到 `planner.noise_sigma`；
- 初期固定 ICODE checkpoint，不同时微调 residual；
- memory 关闭；
- 训练、validation、deployment、sealed seeds 分离；
- 先比较固定 narrow/baseline/broad，再比较 RL covariance；
- 不因训练 seed 不利而改 Gate。

## 9. Pilot 执行前提修正记录

首次矩阵执行目录 `l83_covariance_oracle_pilot_20260717_v1` 在 block 0 的
`reverse_s_unseen` step 110 主动中止。原因是该 anchor 已进入 `terminal` phase，而首版
诊断器为了避免与控制器语义不一致，禁止处理 terminal anchor。该目录是 aborted 工件，不进入
Gate，也不删除。

本修正不改变预注册 anchor、尺度、候选数、模型块或场景。诊断器补齐现有
`MppiController._solve_plan` 已冻结的终点语义：

- terminal translation speed cap；
- current-step heading translation gate；
- terminal alignment yaw command；
- first-action rate limit；
- importance-sampling correction。

修正后的完整矩阵写入新 `v2` 目录。该修正属于实现完整性修复，不根据已观察成本改变任何实验因素。
