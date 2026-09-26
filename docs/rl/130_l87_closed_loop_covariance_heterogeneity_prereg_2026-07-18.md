# L87：闭环协方差上下文异质性上界预注册

日期：2026-07-18  
状态：任何 L87 结果生成前冻结

## 问题

L85 自由策略获得速度但损失精度；L86 锚定策略恢复精度但相对强固定锚点没有显著时间优势。下一步不继续调 SAC，而先检验 RL 要解决的问题是否真实存在：不同可观测场景与物理条件是否需要不同的 MPPI 采样协方差。

## 设计

使用冻结的 ICODE checkpoint、MPPI cost、K=100、H=36、MuJoCo plant 和 safety chain。唯一干预是固定 covariance scale：

- narrow `(0.5,0.5)`；
- baseline `(1.0,1.0)`；
- turn `(0.75,1.75)`；
- speed `(1.75,0.75)`；
- broad `(1.75,1.75)`。

上下文为 `scene × physics_domain`：4 条 clean path 与 4 个冻结物理域，共 16 个上下文。物理域只取已经由 L63 method-blind calibration 证明可行且可分辨的 anchor、high-friction、long-delay 和 moderate-combined。

随机化单位是一次完整 MuJoCo episode。相同上下文与 seed 下五个 candidate 配对；执行顺序由固定 schedule seed 打乱。episode 是上下文内重复测量，不得将其伪装为独立上下文。

## 严格的选择/评估分离

- selection seeds：3 个，只用于选择每个上下文的 oracle candidate 和一个全局 fixed candidate；
- evaluation seeds：3 个全新 seed，只用于最终比较；
- evaluation 结果不得反向修改映射或候选集合。

每个上下文先排除发生碰撞或成功率不足 100% 的 candidate；在剩余 candidate 中，以最小 RMSE 加 `0.002 m` 为精度可行集合，再选择到达时间最短者。全局 fixed candidate 使用相同规则，但在所有上下文聚合后只选一个 scale。

## 可证伪假设

H1：上下文 oracle 在 evaluation seeds 上相对 best global fixed 同时满足：

1. success 不下降、collision 不上升；
2. cross-track RMSE 差的 95% 分层 bootstrap 上界不超过 `+0.002 m`；
3. time-to-goal 差的 95% 分层 bootstrap 上界小于 0；
4. 至少两个不同 candidate 被上下文映射选中；
5. 至少 25% 上下文选择不同于 global fixed。

bootstrap 先重采样上下文，再在上下文内重采样配对 evaluation seed。

## 决策

- H1 通过：证明存在可学习的上下文 headroom，下一步训练 context-conditioned、锚定的 RL residual policy。
- 精度通过但时间失败：协方差上下文适应的实用收益不足，停止该 RL 职责。
- 候选映射近似常数：固定调参已足够，RL 转向采样预算分配或探索数据采集。
- oracle 明显通过但 RL 再次失败：问题在策略表示、观测或训练，而不是任务没有 headroom。

L87 是不可部署的 oracle 上界诊断，不作为最终方法，也不允许使用 simulator domain label 训练最终策略。
