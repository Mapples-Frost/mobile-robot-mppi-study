# L80：场景组鲁棒 residual actor 训练结果

日期：2026-07-17  
结论：training Gate failed；未运行 L80 deployment

## 1. 完整性

三个模型块均满足：

- 完成 30,000 environment steps；
- interrupted episode 为 0；
- initial、5k–30k、latest 和 selected checkpoint 可加载；
- 5k actor 与 initial 逐参数相同，10k 才开始更新；
- frozen BC actor hash 全程不变；
- replay 的三个 scene groups 全部非空；
- 尾部 minibatch scene fraction 与 `1/3` 的最大偏差小于 `3e-5`；
- group-robust enabled 恒为 1，group count 恒为 3；
- smooth worst-group loss、group min/max loss 全部有限。

结果目录：

`results/research_platform/rl/l80_static_group_robust_training_audit_20260717_v1/`

## 2. 训练内选择结果

| block | SAC seed | selected step | success gain/loss | collision regression | mean distance improvement |
|---:|---:|---:|---:|---:|---:|
| 0 | `20260777` | `20k` | `0 / 0` | `0` | `+0.03296 m` |
| 1 | `20260778` | `0` | `0 / 0` | `0` | `0.00000 m` |
| 2 | `20260779` | `0` | `0 / 0` | `0` | `0.00000 m` |

预注册要求至少 2/3 blocks 选择非零、非劣 checkpoint，实际为 1/3，因此 Gate 失败。

## 3. 机制诊断

L80 尾部 5,000 次更新中，三个 block 的平均 scene-group loss max–min gap 分别为：

- block 0：`8.976`；
- block 1：`5.775`；
- block 2：`4.197`。

group-robust actor loss 相对普通 mean SAC actor loss 的平均提升分别为 `4.866`、`2.852`、`2.158`。
这说明 smooth worst-group 不是空操作：训练确实长期由最差 scene group 主导。

与 L78 相比，block 0 的平均 correction magnitude 从约 `0.0657` 降为 `0.0568`；block 1/2 仍约
`0.0662/0.0649`。最终只有 block 0 在训练内得到小幅、非劣距离改善，另外两个 block 的全部 learned
checkpoints 都被选择器拒绝。

## 4. 科学解释

L80 证明了“跨场景最坏组优化”可以抑制部分 correction，但没有产生跨初始化的一致正收益。它更像把
不稳定 actor 变得保守，而不是解决 critic 对“哪个 correction 真正改善闭环成功”的估值问题。

这与 L79 的证据共同支持：

1. 当前 ICODE + frozen BC + MPPI 基线本身较强；
2. shared SAC correction 的可改善空间有限且具有 seed/scene 异质性；
3. 继续扫描 group temperature 或部署阈值会增加研究者自由度，却不能解决 critic 校准问题；
4. 当前证据不支持把 always-on RL prior 作为主论文已验证结论。

## 5. 冻结决策

- 不运行 `22201201`–`22201205` deployment seeds；
- 不打开 `22201211`–`22201215` 封存 seeds；
- 不对 `tau=0.10` 做事后网格搜索；
- 保留 L80 为结构消融负结果；
- 下一次 RL 方法改动必须进入 critic/return modeling 层，例如 distributional critic、quantile/CVaR
  critic 或显式校准的 ensemble，而不是再叠加一个 gate；
- 若 critic 级方法仍不能在独立模型块上通过，则把 RL 在论文中的角色收缩为复杂/OOD 区域探索与
  residual 数据采集，控制主线保留 ICODE + conventional MPPI + safety arbitration。

这个结论不否定 ICODE 残差动力学结果，也不影响 nominal MPPI、MuJoCo、LaserScan 或实车安全链。

