# L69 归因结果与 L70 多几何 SAC correction 训练预注册

日期：2026-07-16

## L69 固定 Gate 结果

L69 完成 240/240 个 episode，产物、seed、逐 step clean fallback 和冻结 BC
对照全部通过。主要阻塞场景比较得到：

- complexity-gated SAC+ICODE 相对 traditional ICODE：净增加 31 次成功，零碰撞增加；
- complexity-gated BC+ICODE 相对 traditional ICODE：净增加 24 次成功，零碰撞增加；
- SAC correction 相对同门控 BC：净增加 7 次成功，三个场景均为正，零碰撞增加；
- SAC correction 接受比例为 35.68%。

但是预注册 Gate 因训练 block 稳定性失败：三个 block 的 SAC 相对 gated BC 成功变化为
`+8/-1/0`，未达到“至少两个 block 严格为正”。因此 L69 判定为
`development_fail`，封存种子未打开。不能把 pooled 正结果写成确认性 RL 结论。

## L70 假设

L17 correction 只在 U-trap 上训练，而 L69 同时评估 single obstacle、corridor 和
U-trap。其训练分布不足可以解释 pooled 正向但训练 seed 不稳定。L70 不改变 prior
接口、门控阈值、SAC 网络、correction 上限或 reward，只改变训练环境覆盖：

> 使用 scene-balanced replay 在三个静态阻塞几何上训练，能否让 frozen-BC 上的
> SAC correction 在多个独立训练种子中稳定获得验证收益？

## 冻结训练设计

- 三个独立 SAC training seeds：20260741/42/43；
- 每个 seed 使用对应的独立 BC base actor；
- 每个 seed 与一个已通过 L57--L68 的 ICODE checkpoint 配对；
- MuJoCo plant 固定为 L56 高动态 anchor；
- pose/twist 为 ground truth；障碍仍只经 LaserScan 进入 planner；
- training/validation 场景均为 single obstacle、narrow corridor 和 U-trap；
- replay 为 scene-balanced，并要求三场景全部已有样本后才开始平衡采样；
- 30k steps，5k 一次验证；每个验证点为 3 scenes × 5 fixed seeds；
- frozen BC normalizer、base actor 和 action bounds 不变；
- actor 从 5k 后更新，correction penalty 与 L17 相同；
- memory off、scan guard 与安全仲裁不变。

checkpoint 使用 fail-closed paired selector：相对 step-zero BC 不允许丢失任何成功，
不允许增加碰撞，并且至少增加一次成功或达到冻结的连续距离改善门槛。没有 checkpoint
合格时必须保留 step zero，不得手选训练曲线中看起来漂亮的点。

## 下一 Gate

L70 训练结束后先审计：

1. 三个 run 的 base actor hash 与各自 BC 初始化一致；
2. scene-balanced replay、验证 seed 和 ICODE checkpoint provenance 完整；
3. 至少两个训练 seed 选择非零 checkpoint；
4. 被选 checkpoint 在固定 validation 上不损失成功、不增加碰撞；
5. 只有满足以上条件，才建立使用全新 episode seeds 的 L71 development attribution；
6. L69 sealed seeds 不自动转为 L71 seeds，仍保持关闭。

L70 是训练资格阶段，不是论文控制结果。

