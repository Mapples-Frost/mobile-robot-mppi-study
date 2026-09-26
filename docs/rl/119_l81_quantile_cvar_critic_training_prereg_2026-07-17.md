# L81：Quantile/CVaR critic 训练预注册

日期：2026-07-17  
性质：development training；在任何 L81 transition 产生前冻结

## 1. 背景与可证伪假设

L78 的 scalar-Q SAC 在训练内通过，但 L79 新 seed 部署出现跨场景成功交换。L80 的 worst-scene
actor objective 能抑制 correction，却只有 1/3 模型块保留非零收益。这说明仅改变样本/场景平均方式
没有解决 critic 只表示期望回报的问题。

L81 检验：

> 学习完整的回报分位数分布，并用下尾 20% CVaR 评价 residual correction，能否避免“平均回报看似
> 有利、但少数闭环轨迹失败”的 correction，并在至少 2/3 独立模型块产生非零、非劣 checkpoint？

如果少于 2/3 模型块通过训练内选择，L81 直接判失败，不运行新 deployment seeds。

## 2. Quantile critic

每个 twin critic 不再输出一个标量，而是输出 25 个回报分位数：

\[
Z_{\psi_k}(s,a)
=
\left[z_{\psi_k,1}(s,a),\ldots,z_{\psi_k,25}(s,a)\right],
\qquad k\in\{1,2\}.
\]

目标分布为：

\[
y_j = r + \gamma(1-d)
\left[
\min\left(z'_{1,j},z'_{2,j}\right)
-\alpha\log\pi(a'\mid s')
\right].
\]

critic 使用 pairwise quantile Huber loss，分位点固定为：

\[
\tau_i=\frac{i-0.5}{25},\qquad i=1,\ldots,25,
\]

Huber 参数固定为 \(\kappa=1.0\)。

## 3. CVaR actor 与部署 advantage

对预测分位数升序排列，取最低的 20%，即 5 个分位数：

\[
Q_{\mathrm{CVaR}_{0.2}}(s,a)
=
\frac{1}{5}\sum_{i=1}^{5} z_{(i)}(s,a).
\]

actor objective 中的 \(Q\) 以及部署时 candidate-vs-BC conservative advantage 都使用 twin critics
中更小的下尾 CVaR。它关注坏结果尾部，不声称是校准概率或理论安全保证。

## 4. 唯一方法变化

L81 继承 L78，只设置：

```yaml
rl:
  sac:
    critic_distribution: quantile
    critic_num_quantiles: 25
    critic_quantile_huber_kappa: 1.0
    actor_cvar_fraction: 0.20
    actor_group_robust_enabled: false
```

保持不变：

- actor LR `5e-5`、critic LR `2e-4`；
- correction penalty `1.0`、actor update after 10k；
- SAC entropy、reward、30k steps；
- scene/outcome-balanced replay；
- frozen BC correction 和 correction bounds；
- 三个静态高动力学场景；
- ICODE checkpoint、MuJoCo plant、MPPI、LaserScan、local obstacle layer 和安全仲裁；
- memory、progress gate、support gate、group-robust actor 全部关闭。

因此 L81 是“scalar mean-Q”与“quantile lower-tail CVaR-Q”的单方法对比，不与 L80 混合。

## 5. 实验单位与随机化

独立单位仍是完整训练模型块：

| block | SAC seed | frozen BC | ICODE |
|---:|---:|---|---|
| 0 | `20260784` | BC seed `20260721` | L57 seed `20261201` |
| 1 | `20260785` | BC seed `20260722` | L57 seed `20261202` |
| 2 | `20260786` | BC seed `20260723` | L57 seed `20261203` |

- 训练内 validation seeds：`22320801`–`22320805`；
- 若训练 Gate 通过，deployment seeds：`22201301`–`22201305`；
- 封存 seeds：`22201311`–`22201315`；
- transition、gradient update 和 validation episode 都不是独立 replicate。

## 6. 完整性与训练 Gate

完整性要求：

1. 每块完成 30,000 steps，interrupted episode = 0；
2. 0–30k 的 5k validation/checkpoint 矩阵完整；
3. 三个 replay scene groups 均非空；
4. quantile enabled 恒为 1、quantile count 恒为 25、CVaR fraction 恒为 0.20；
5. quantile spread 为有限正数，Q expectation/CVaR 全部有限；
6. 5k actor 与 initial 精确相同，10k 才开始变化；
7. frozen BC hash 不变，全部 checkpoint 可加载；
8. config、seed、git SHA、update CSV 和 validation CSV 完整。

继续到 deployment 至少需要：

- 2/3 blocks 选择非零 checkpoint；
- 被选 block success loss = 0；
- collision regression = 0；
- mean final distance 不恶化；
- 至少 1 个 success gain，或 mean final distance 改善 ≥ 0.005 m。

checkpoint 仍由 L78 已冻结的 initial-noninferiority 规则选择，不能因 quantile 方法而放宽。

## 7. 后续决策

若训练 Gate 通过：

- 每块锁定一个 selected checkpoint；
- 在 `22201301`–`22201305` 上做 90 个严格配对 MuJoCo episodes；
- 使用与 L79 相同的总体 Gate；
- development 通过后才可打开封存 seeds。

若训练 Gate 失败：

- 不扫描 CVaR fraction、quantile count 或 Huber kappa；
- 不运行 deployment；
- 当前 shared RL prior 主线停止；
- RL 转为复杂/OOD 区域探索与 residual 数据采集候选，控制主线保留 ICODE + conventional MPPI。

## 8. 解释边界

即使通过，L81 也只说明当前 benchmark 中的下尾回报建模值得进入新 seed 部署验证。它不证明 CVaR
校准、不提供碰撞概率界、不构成稳定性定理，也不等同于动态障碍或实车结论。

