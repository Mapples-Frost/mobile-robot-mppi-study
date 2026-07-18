# L80：场景组鲁棒 residual actor 训练预注册

日期：2026-07-17  
性质：development training；在任何 L80 transition 产生前冻结

## 1. 假设

L79 的 90 个新 seed episode 显示：L78 candidate 在总体连续距离上改善，但 block 2 把 clean scene
的一次成功增益交换成 corridor 和 U-trap 的两次成功损失。L80 检验：

> 在保持 critic、replay、reward、BC correction、ICODE 和 MPPI 全部不变时，把 residual actor
> 的普通样本均值目标改为平滑 worst-scene 目标，能否减少跨场景的成功交换，并在至少 2/3 独立
> 模型块保留非零、非劣 checkpoint？

## 2. 方法

对 minibatch 中每个 transition 的 actor loss：

\[
\ell_i =
\alpha \log \pi(a_i\mid s_i)
- \min(Q_1,Q_2)(s_i,a_i)
+ \lambda_c\left\|\frac{\Delta a_i}{\Delta a_{\max}}\right\|_2^2 .
\]

先在每个 MuJoCo scene group 内求均值：

\[
\bar{\ell}_g = \frac{1}{|B_g|}\sum_{i\in B_g}\ell_i .
\]

再使用归一化 smooth maximum：

\[
\mathcal{L}_{\mathrm{actor}}^{\mathrm{group}}
=
\tau\left[
\log\sum_{g=1}^{G}\exp\left(\frac{\bar{\ell}_g}{\tau}\right)
-\log G
\right],
\qquad \tau=0.10 .
\]

它把更大梯度权重分配给当前 actor loss 最差的场景组。`scene_outcome_balanced` replay 已保证三个
场景在 minibatch 中近似各占 `1/3`；group objective 不使用测试场景标签，不修改 observation，
也不会在部署时增加计算。

## 3. 唯一方法变化

L80 继承 L78 的全部配置，只增加：

```yaml
rl:
  sac:
    actor_group_robust_enabled: true
    actor_group_robust_temperature: 0.10
```

以下均不变：

- actor LR `5e-5`、correction penalty `1.0`、actor update after 10k；
- twin critic、SAC target、entropy、reward；
- 30k steps 与 5k checkpoint interval；
- scene/outcome-balanced replay；
- 三个静态高动力学场景；
- frozen BC 与 correction scale；
- ICODE、MuJoCo plant、MPPI、perception 和安全链；
- memory 与所有部署 gate 关闭。

L80 是相对 L78 的单因素训练消融。

## 4. 独立模型块与数据边界

| block | SAC seed | frozen BC | ICODE |
|---:|---:|---|---|
| 0 | `20260777` | BC seed `20260721` | L57 seed `20261201` |
| 1 | `20260778` | BC seed `20260722` | L57 seed `20261202` |
| 2 | `20260779` | BC seed `20260723` | L57 seed `20261203` |

- 训练内 validation seeds：`22310801`–`22310805`；
- 若训练 Gate 通过，新的 deployment seeds：`22201201`–`22201205`；
- 对应封存 seeds：`22201211`–`22201215`；
- L79 及更早 seeds 禁止用于模型选择。

## 5. Gate

完整性要求与 L78 相同，并额外检查：

- 每个 actor update 都包含 3 个 scene groups；
- `actor_group_robust_enabled` 日志恒为 1；
- group loss 全部有限；
- 5k actor 仍与 initial 精确相同，10k 才开始变化；
- frozen BC hash 全程不变。

继续到新 seed deployment 至少需要：

- 2/3 blocks 选择非零 checkpoint；
- 每个被选 block success loss = 0、collision regression = 0；
- mean final distance 不恶化；
- 至少 1 个 success gain，或 mean final distance 改善 ≥ 0.005 m。

若少于 2/3 blocks 通过，停止 L80，不运行 deployment。若训练通过而新 seed deployment 再次失败，
则不继续微调 group temperature，应考虑替换 critic 算法，或把 RL 限定为探索/数据采集角色。

## 6. 解释边界

即使 L80 通过，也只说明 smooth worst-scene actor objective 在当前三个静态场景上提高了训练内稳定性。
它不构成分布鲁棒性理论保证，不代表动态障碍或实车有效，也不能省略后续新 seed 和封存确认。

