# L78 训练结果与 L79 新 seed 部署预注册

日期：2026-07-17  
状态：L78 completed；L79 development deployment 在任何 L79 episode 运行前冻结

## 1. L78 训练结果

三个独立模型块均完成 30,000 environment steps。审计结果保存在：

`results/research_platform/rl/l78_static_conservative_training_audit_20260717_v1/`

| block | SAC seed | selected step | success gain/loss | collision regression | mean final-distance improvement |
|---:|---:|---:|---:|---:|---:|
| 0 | `20260774` | `25k` | `1 / 0` | `0` | `+0.28487 m` |
| 1 | `20260775` | `0` | `0 / 0` | `0` | `0.00000 m` |
| 2 | `20260776` | `25k` | `0 / 0` | `0` | `+0.18641 m` |

L78 训练 Gate 通过：

- 2/3 模型块选择非零 checkpoint；
- 两个非零选择都没有 success loss 或 collision regression；
- 三块均完成 30k，interrupted episode 为 0；
- 三场景 replay 均非空，尾部 minibatch 的场景比例与 `1/3` 最大偏差小于 `6e-5`；
- tail minibatch success fraction 约为 `0.496`–`0.497`；
- initial、5k–30k、latest、selected checkpoints 全部可重新加载；
- residual actor 在 5k 与初始状态逐参数相同，10k 才开始变化；
- frozen BC actor hash 在全部 checkpoint 中保持不变。

这是**训练内 development 结果**。它只支持继续运行 L79，不能作为论文中的最终 RL 性能结论。

## 2. L79 可证伪问题

> L78 的复合保守训练日程所选 checkpoint，能否在从未用于训练或模型选择的新 episode seeds 上，
> 相对各自冻结 BC 基线产生跨模型块的正向控制贡献，而不引入成功损失或碰撞回归？

L79 不比较训练集 reward，不重新挑 checkpoint。它直接检验训练内正信号是否能迁移到新闭环轨迹。

## 3. 固定设计

- 独立模型块：3；
- 场景：clean single obstacle、narrow corridor、U-trap；
- 新 development seeds：`22201101`–`22201105`；
- 封存 seeds：`22201111`–`22201115`，本阶段禁止使用；
- 条件：
  1. `complexity_bc_icode`：对应 block 的冻结 BC + ICODE；
  2. `gated_lcb_icode`：对应 block 的 L78-selected policy + ICODE；
- 总 episode：`3 blocks × 3 scenes × 5 seeds × 2 conditions = 90`；
- 同一 block/scene/seed 下严格配对；
- block 内执行顺序由 `2026071779 + block` 固定随机化。

checkpoint 映射严格固定为：

| block | base | candidate |
|---:|---|---|
| 0 | L78 seed `20260774` initial | L78 seed `20260774` 25k |
| 1 | L78 seed `20260775` initial | 同一个 initial |
| 2 | L78 seed `20260776` initial | L78 seed `20260776` 25k |

block 1 的训练规则选择了 step 0。L79 保留它作为显式 no-op block；不得因为它没有 learned correction
而删除这个模型块，这可避免只汇报两个成功训练 seed 的选择偏差。

## 4. 不变系统

L79 保持 L74/L77 的控制栈：

- high-dynamic MuJoCo plant、0.04 s command delay；
- 每块独立的 L57 ICODE checkpoint；
- target twin-critic conservative LCB；
- LaserScan complexity outer gate 和 near-goal fallback；
- MPPI `K=100`、horizon、cost 和 control bounds 不变；
- LaserScan → local obstacle layer → planner obstacles；
- scan guard 与 safety arbitration 始终启用；
- memory、progress gate、correction-support gate 关闭；
- 不使用全局障碍物真值作为 planner 输入。

## 5. 预注册 Gate

单 block 合格要求：

1. success loss = 0；
2. collision regression = 0；
3. mean final goal distance 不恶化；
4. learned block 的 correction advantage gate alpha ≥ 0.01；
5. 至少 1 个 success gain，或 mean final goal distance 改善 ≥ 0.005 m。

总体 Gate：

- 至少 2/3 blocks 合格；
- success gains ≥ 3；
- success losses ≤ 1；
- net success gain ≥ 2；
- collision regressions = 0；
- overall mean final goal distance 不恶化；
- narrow corridor net success gain ≥ 2；
- clean single obstacle success loss = 0；
- U-trap success loss = 0；
- frozen base correction 必须精确为 0；
- 90 个 episode 完整、唯一、有限且没有旧 seed 泄漏。

任一条失败都不打开封存 seeds。禁止根据 L79 结果修改 gate 后重算通过。

## 6. 解释边界

若通过，L79 只支持进入一次封存确认，不能直接写成 RL 已被证实优于 BC/MPPI。若失败，则说明即使保守
训练改善了训练内稳定性，跨 seed 部署仍不可靠；应转向 critic 算法/校准或缩减 RL 在论文中的角色，
而不是继续添加部署门控。

