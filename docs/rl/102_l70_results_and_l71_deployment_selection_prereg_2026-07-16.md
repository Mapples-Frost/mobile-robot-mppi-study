# L70 多几何训练结果与 L71 部署一致性选择预注册（2026-07-16）

## L70 的结论

L70 的三组独立训练均完整运行到 30,000 步，三个训练场景均进入
scene-balanced replay，且没有中断 episode。预注册的训练内选择器只在
seed `20260742` 接受了 step 20,000；seed `20260741` 和 `20260743` 均回退到
step 0 的冻结 BC 基线。因此，L70 未通过“至少 2/3 个训练种子选中非零
SAC checkpoint”的资格门槛。

该失败必须保留。后续实验不得将 L70 宣称为稳定的多几何训练成功，也不得使用
L69 的 sealed seeds 为 L70 调参。

## 诊断

L70 训练内验证的是没有部署外层 Gate 的策略，而项目最终方法使用：

1. LaserScan scene-complexity Gate 决定 RL 的外层权限；
2. target-twin-critic LCB 决定 SAC correction 是否优于同一冻结 BC 基座；
3. ICODE 仅修正 MPPI rollout dynamics；
4. scan guard 仍拥有最终安全优先级。

所以 L70 的选择目标与最终部署链路并不完全一致。这个事实不能改变 L70 的失败，
但可以形成一个新的、独立可证伪的问题：在新的数据上，按真实部署链路选择
checkpoint，是否能得到跨训练种子稳定的 SAC 增益？

## L71 预注册问题

对每个 L70 训练 block，固定其 BC 基座、ICODE checkpoint、MuJoCo plant、MPPI
`K=100`、感知和安全链，比较：

- comparator：`complexity_bc_icode`，用 `base` 模式精确恢复 checkpoint 内嵌的
  冻结 BC 动作；
- candidate：steps 5k、10k、15k、20k、25k、30k 的
  `gated_lcb_icode`。

每个 candidate 都在三个静态几何场景和三个全新 episode seeds 上进行配对评估。
本阶段只用于 checkpoint selection；sealed confirmation seeds 在选择完成前保持封存。

## 选择门槛

候选 checkpoint 必须同时满足：

- 相对精确 BC 基座，成功 episode 损失数为 0；
- 碰撞回归数为 0；
- 平均终点距离不得增加；
- 至少增加 1 个成功 episode，或平均终点距离改善不少于 0.005 m；
- 被选 SAC correction 的平均 LCB 接受权重不少于 0.01，排除名义上的非零、
  实际从不生效的策略。

同一 block 中按净成功增益、平均距离改善、平均 return 和更早 checkpoint 的顺序
选择。若无候选通过，则该 block 回退到 BC，不得人为指定非零 checkpoint。

L71 只有在至少 2/3 个独立训练 blocks 选中非零 SAC checkpoint 时才通过资格门槛。
通过后才允许在未见 confirmation seeds 上执行完整的 traditional / BC / SAC 因果
对照；否则应继续改进训练方法，而不是打开 sealed seeds。

