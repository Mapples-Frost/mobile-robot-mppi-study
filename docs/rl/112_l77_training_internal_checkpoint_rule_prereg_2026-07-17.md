# L77：训练内 checkpoint 规则的新 seed 验证预注册

日期：2026-07-17  
性质：development validation；运行任何 L77 episode 前冻结

## 1. 动机

L72 的训练内预注册规则在各自固定的 15 个验证 episode 上选择：

- block 0：25k；
- block 1：25k；
- block 2：30k。

随后 L73 只用 3 个新部署 seed 重新选择，把三块都改成 30k。L74–L76 显示 30k
correction 的跨 seed 稳定性不足，尤其 block 0 多次形成负贡献。

L77 不根据 L74–L76 结果重新扫描 checkpoint，而是回到**已经在 L72 结果公布时
固定记录的选择**，检验一个可证伪问题：

> 训练内的零成功损失约束，是否比小样本部署重选更能保留跨 seed 的非劣性？

## 2. 冻结设计

条件：

1. `complexity_bc_icode`：对应 block 的 `initial.pt`；
2. `gated_lcb_icode`：对应 block 的 `25k/25k/30k`。

没有 L75 progress gate，没有 L76 support gate，没有新阈值。其余保持 L74：

- target twin-critic LCB；
- LaserScan complexity outer gate；
- near-goal fallback；
- ICODE checkpoint；
- high-dynamic MuJoCo plant；
- MPPI 与 cost；
- scan guard、local obstacle layer 与最终安全仲裁；
- memory 关闭。

## 3. 区组、随机化与样本

- 独立训练模型块：3；
- 场景：single obstacle、narrow corridor、U-trap；
- 新开发 seed：`22201001–22201005`；
- 封存 seed：`22201011–22201015`；
- 每组条件严格配对；
- 每块内部执行顺序由 `2026071777 + block` 随机打乱；
- episode：`3 × 3 × 5 × 2 = 90`；
- step 是 episode 内重复测量，不是独立样本。

L74–L76 及更早 seed 均由 runner 保护。

## 4. 预注册 Gate

单个模型块合格需要：

1. success loss 为 0；
2. collision regression 为 0；
3. 平均最终距离不恶化；
4. correction gate alpha 至少 0.01；
5. 至少 1 个 success gain，或平均最终距离改善至少 0.005 m。

总 Gate 还要求：

- 至少 2/3 模型块合格；
- 总 success gain 至少 3、loss 不超过 1、净 gain 至少 2；
- collision regression 为 0；
- 总平均最终距离不恶化；
- narrow corridor 净 gain 至少 2；
- single obstacle 与 U-trap 的 success loss 均为 0；
- 冻结 BC correction 必须精确为 0；
- 90 个 episode 完整、唯一、有限且无 seed 泄漏。

任一项失败都停止该 checkpoint 规则，不打开封存 seed。

## 5. 解释边界

通过最多说明：在当前静态高动力学 benchmark 中，较保守的训练内 checkpoint
selection 比 L73 的小样本重选更稳定。它不证明 RL 在所有场景优于 BC，也不证明
理论稳定性或实车有效性。

失败则说明部署不稳定不能仅归因于 30k 过训练；后续必须回到训练目标、critic
校准或更大规模模型集成，而不是继续增加部署门控。
