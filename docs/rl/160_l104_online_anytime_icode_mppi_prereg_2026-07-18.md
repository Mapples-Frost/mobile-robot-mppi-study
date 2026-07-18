# L104：在线 Anytime ICODE-MPPI 整回合预注册

日期：2026-07-18
状态：在读取任何 L104 整回合结果之前冻结
性质：开发集筛选 + 独立确认；不是论文正式结论的替代品

## 1. 要回答的问题

此前 L103 只在同一物理状态的反事实 K=50/K=100 候选上证明：低维上下文 bandit 能识别“追加 50 条 rollout 值得”的状态。L104 将其真正放入闭环控制，检验：

> 在 ICODE 预测模型与上下文协方差采样器不变时，RL 是否能按状态决定停止于 K=50 或追加到 K=100，从而以低于固定 K=100 的平均计算量，保持其整回合安全与路径跟踪质量？

这里的 RL 是带计算约束的 contextual bandit。它不直接输出车轮控制；它在看到前 50 条 ICODE rollout 的统计量及局部路径几何后，执行离散动作 `STOP` 或 `ADD`。最终控制仍由 MPPI 加权更新、scan_guard 与 safety arbitration 决定。

## 2. 冻结对象

- ICODE checkpoint：`l57_icode_high_dynamic_h36_seed20261201_v1/best.pt`；
- covariance bandit：`l89_contextual_covariance_bandit_20260718_v1/checkpoint.json`；
- budget bandit：`l103_calibrated_budget_bandit_20260718_v1/bandit_summary.json`；
- 控制频率、horizon、代价、动作限幅、rate limit、MuJoCo plant、传感器、scan_guard 全部固定；
- memory 关闭；SAC policy/value 不进入该实验；
- bandit 部署时冻结，不在线更新参数或 dual price。

## 3. 对照条件

1. `icode_contextual_k50`：固定 50 条 rollout；
2. `icode_contextual_k100`：固定 100 条 rollout；
3. `icode_anytime_k50_100`：先评估前 50 条；bandit 选择 STOP 或再评估同一嵌套样本集的后 50 条。

三者使用同一 ICODE、同一 covariance bandit、同一代价和安全链。对于相同 seed，anytime 的前 50 条控制候选必须与固定 K=50 字节一致；若选择 ADD，其完整 100 条必须与固定 K=100 字节一致。

## 4. 场景与物理域

- 路线：held-out hairpin、held-out reverse-S；
- 物理域：train anchor、high friction、long delay、combined moderate；
- 开发 seeds：`20271601–20271603`；
- 独立确认 seeds：`20271611–20271616`；
- 每个 scene × domain × seed × condition 是一个独立实验单元；控制周期不能伪装成独立样本。

## 5. 指标

安全与任务：success、collision、final goal distance、cross-track RMSE、elapsed time。
控制质量：command/applied jerk、mean absolute omega、safety override。
计算：planner mean/p95/max compute、mean K、ADD fraction、deadline misses。
机制：predicted advantage、confidence width、局部路径几何、ADD 的场景/物理域/进度分布。

## 6. 预注册门槛

开发集仅用于发现实现错误及决定是否值得开启确认，不用于形成论文结论。开启独立确认须同时满足：

1. 无完整性错误、NaN、碰撞恶化或 success 降低；
2. anytime 平均 K 不高于 80；
3. 相对 K=100，cross-track RMSE 配对差值的层级 bootstrap 95% CI 上界不超过 `+0.002 m`；
4. 相对 K=100，planner mean compute 的配对差值 95% CI 上界小于 0；
5. anytime 在至少两个物理域、两条路线中都同时出现 STOP 与 ADD，避免退化为常数预算。

独立确认的主门槛沿用 1–5，并增加：

6. elapsed time 差值 95% CI 上界不超过 `+0.15 s`；
7. mean K 的 95% CI 上界不高于 80；
8. 报告相对 K=50 的所有结果，即使其否定“追加预算带来任务收益”。

## 7. 统计方法

- condition 顺序在每个 scene/domain/seed block 内随机化；
- 采用 common random numbers 和配对差；
- bootstrap 先重采样 scene × domain 分层，再重采样 seed；
- 置信区间、效应量和原始 episode CSV 全部保存；
- 不用单侧挑选指标，不删除失败 seed，不把 smoke test 混入正式数据。

## 8. 可证伪条件

以下任一情况将否定当前在线预算机制，而不是通过改写叙事规避：

- bandit 几乎总 STOP 或总 ADD；
- 节省计算但破坏安全、成功率或跟踪精度；
- 平均 K 没有实质低于 100；
- 效果只存在于单一路线或单一物理域；
- 与 K=50 相比追加预算没有任何可复现的任务价值，且与 K=100 相比也没有稳定计算收益。

若开发门槛失败，可以在新的、有明确编号的开发实验中修正特征/奖励/部署阈值，但不得回写本预注册；之后必须使用未见过的新 seeds 重新确认。
