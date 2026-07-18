# L34 有界动态修正：开发实验预注册

日期：2026-07-16  
状态：在查看 L34 结果前冻结。

## 1. 触发原因

L33 在两条固定未见动态路径和 `combined_unseen` MuJoCo 物理域上完成了
step 0、5k、10k、15k、20k、25k、30k 共七次验证。初始化策略为
`1/6` success、`1/6` collision；此后所有训练检查点均为 `0/6` success，
且没有检查点优于初始化策略。训练的 87 个完整 episode 也没有成功样本。

这支持一个具体且可证伪的诊断：在当前困难度下，令 RL 直接替换整个 MPPI
sampling prior，会使早期探索和 actor 更新破坏传统先验已有的可用结构。

## 2. 唯一方法改动

L34 继承 L33 的模型、三帧观察、奖励、优化器、训练步数、场景、物理域、
运动随机化、训练 seed、验证 seed 和初始化 checkpoint。唯一方法改动为：

```yaml
rl:
  training:
    rl_prior_alpha: 0.25
```

执行时的 MPPI sampling mean 为

$$
\mu_{\mathrm{L34}}
=
\mu_{\mathrm{traditional}}
+0.25\left(\mu_{\mathrm{RL}}-\mu_{\mathrm{traditional}}\right).
$$

因此 RL 仍可根据三帧 LaserScan 改变候选轨迹方向，但单步不再拥有完全覆盖
传统 warm-start 的权限。共享 temporal safety、scan guard 和最终控制仲裁保持不变。

## 3. 对照边界

- L33：`rl_prior_alpha = 1.0`，完整 learned prior；
- L34：`rl_prior_alpha = 0.25`，有界 learned correction；
- 两者均从同一 L6 三帧 actor-only checkpoint 初始化；
- critic、optimizer、replay、RNG 和训练计数均重新初始化；
- 验证仍使用相同的两条未见运动路径、相同的 combined-unseen 物理域和六个固定 episode；
- 本轮仍是 development，不接触 sealed seeds，不宣称论文最终结果。

## 4. 预注册判据

只有同时满足以下条件，L34 才进入多训练 seed 扩展：

1. 最佳训练后检查点的验证 collision 数不高于自己的 step-0 checkpoint；
2. 最佳训练后检查点的 success 数高于自己的 step-0 checkpoint；或在 success 持平时，平均终点距离至少改善 `0.05 m`；
3. 最佳训练后检查点优于 L33 的最佳训练后检查点（L33 的训练后 success 为 `0/6`）；
4. 至少一个训练 episode 成功，使 replay 中存在真实正回报闭环轨迹；
5. 无 NaN/Inf，checkpoint、随机种子、随机化运动参数和验证逐 episode 结果完整留档。

若未通过，不继续在同一验证 seed 上搜索混合系数。下一步应转向可学习性课程或
动态教师轨迹，而不是把 `0.25` 事后改成多个阈值并挑选最好结果。

