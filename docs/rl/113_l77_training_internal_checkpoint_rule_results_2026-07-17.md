# L77：训练内 checkpoint 规则的新 seed 验证结果

日期：2026-07-17  
性质：预注册 development validation；封存 seed 未打开

## 1. 结论

L77 Gate **未通过**。恢复 L72 训练内选择的 `25k/25k/30k` 不能消除跨 seed
成功损失，因此此前不稳定不能仅归因于 L73 的小样本 checkpoint 重选。

该结果触发预注册停止规则：停止继续叠加 progress、OOD、限幅或 checkpoint
deployment patch，回到 SAC correction 的训练目标与 critic 校准。

## 2. 完整性

- 期望/实际 episode：`90/90`；
- 配对主键重复：0；
- 保护或封存 seed：0；
- NaN/Inf：0；
- collision regression：0；
- 冻结 BC correction alpha：0；
- checkpoint：block 0/1 为 25k，block 2 为 30k，与预注册一致。

## 3. 主要结果

相对冻结 BC：

| block | checkpoint | gain | loss | 净变化 | 平均最终距离改善 |
|---:|---:|---:|---:|---:|---:|
| 0 | 25k | 3 | 1 | +2 | +0.015 m |
| 1 | 25k | 0 | 3 | -3 | +0.009 m |
| 2 | 30k | 1 | 3 | -2 | -0.494 m |
| 合计 | — | 4 | 7 | -3 | -0.157 m |

按场景：

- single obstacle：`1 gain / 3 loss`，净 `-2`；
- narrow corridor：`1 gain / 2 loss`，净 `-1`；
- U-trap：`2 gain / 2 loss`，净 0；
- 全部 45 对无新增碰撞。

block 0 的 25k 有正净效果，说明 correction 并非原则上无效；但 block 1/2 的负结果
表明训练随机性仍主导部署表现。只选择“某个看起来好的 block”会构成事后挑选，不能
作为论文结论。

## 4. 排除的解释

L75–L77 依次排除：

1. 只在停滞时激活整个 learned prior；
2. 用单一 normalizer OOD 分数衰减 correction；
3. 仅靠恢复较早 checkpoint 解决不稳定。

因此当前主要矛盾不是“缺一个更聪明的部署开关”，而是：

- correction actor 在不同训练 seed 下学到的策略不一致；
- twin-critic LCB 仍可能接受闭环有害但 critic 估值为正的修正；
- 当前 correction penalty 与 actor 更新节奏不足以稳定地保持 BC 非劣性。

## 5. 下一步

L78 转到训练层，保持风险均衡 replay 和所有环境因素不变，仅进行预注册的保守
correction 训练：

- 增强 correction magnitude penalty；
- 降低 actor learning rate；
- 延后 actor 更新，使 critic 获得更长预热；
- 冻结 BC 参数与 checksum 不变；
- 仍用 3 个独立训练 seed；
- 用全新部署 seed 评价，不复用 L74–L77 的效果数据。

若 L78 仍不能在至少 2/3 训练块中保持无成功损失，则应重新考虑 critic 算法或将
RL 降级为探索/数据采集模块，而不是主部署 prior。

## 6. 产物

- 配置：`configs/rl/training_internal_checkpoint_rule_l77.yaml`
- 预注册：`docs/rl/112_l77_training_internal_checkpoint_rule_prereg_2026-07-17.md`
- 汇总：
  `results/research_platform/rl/l77_training_internal_checkpoint_rule_20260717_v1/summary/l77_summary.json`
- 配对：
  `results/research_platform/rl/l77_training_internal_checkpoint_rule_20260717_v1/summary/l77_paired_effects.csv`
