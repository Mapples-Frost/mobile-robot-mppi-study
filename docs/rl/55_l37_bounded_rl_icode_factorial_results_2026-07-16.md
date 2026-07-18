# L37 有界 RL × ICODE 因子实验结果：当前联合接入不成立

日期：2026-07-16  
结论等级：三模型 block development；预注册 Gate 未通过。

## 1. 主要结论

L37 完成 180/180 个 MuJoCo 闭环 episode，无缺失、重复、NaN/Inf、保护 seed 或 sealed
seed 泄漏。四个条件共享相同的 combined-unseen plant、动态障碍、LaserScan、temporal
safety 和最终仲裁。完整性通过，但效能 Gate 明确失败：

| 条件 | Success | Collision | Final distance | Planner time |
|---|---:|---:|---:|---:|
| Traditional + Nominal | 33.3% | 46.7% | 1.235 m | 4.86 ms |
| Traditional + ICODE | 6.7% | 62.2% | 1.567 m | 34.27 ms |
| Bounded RL + Nominal | 22.2% | 71.1% | 1.669 m | 6.28 ms |
| Bounded RL + ICODE | 17.8% | 75.6% | 1.777 m | 34.86 ms |

联合方法相对 traditional-nominal 净减少 7 次 success、净增加 13 次 collision，平均
final distance 恶化 0.542 m。三套 model block 中，联合方法没有一次在 success 与
collision 上同时最优或并列最优。

![L37 factorial](../../results/research_platform/rl/l37_bounded_rl_icode_factorial_20260716_v1/fig_l37_bounded_rl_icode_factorial.png)

## 2. 主效应

### Bounded RL 主效应

跨 nominal/ICODE 两层汇总，bounded RL 的净 success gain 为 0，但净增加 17 次
collision，平均 final distance 恶化 0.322 m。collision-rate difference 的分层
bootstrap 估计为 `+0.189`，95% CI `[-0.022, 0.411]`；不确定区间较宽，但没有安全改善
证据。

这与 L35 的独立失败一致：L34 的开发集增益不能泛化为“bounded RL 已稳定有效”。

### ICODE 主效应

跨 traditional/bounded-RL 两层汇总，ICODE 净减少 14 次 success、净增加 9 次
collision，平均 final distance 恶化 0.220 m。success-rate difference 为 `-0.156`，
95% CI `[-0.278, -0.033]`。

因此当前证据否定的是“已有 ICODE checkpoint 可以直接改善当前动态 MPPI 闭环”，而不
是否定残差动力学学习本身。此前离线数据仍表明 ICODE 在 unseen split 上将 H20 rollout
RMSE 从 `0.0563` 降到 `0.0460`；本轮证明预测误差的有限下降不会自动转化为控制收益。

## 3. 交互项为什么为正但联合方法仍差

success difference-in-differences 为 `+0.222`，95% CI `[0.022, 0.422]`；final-distance
交互为 `+0.224 m`，但 CI `[-0.287, 0.773]`。正交互意味着：当另一个学习模块已经存在
时，第二个模块造成的额外损失比单独接入时小，或者部分相互抵消。

它不意味着联合方法的绝对性能好。两个主效应从较高的 traditional-nominal baseline
出发均为负，所以即使交互为正，最终组合仍低于 baseline。这是必须在论文叙事中明确的
区别。

## 4. 场景分层与轨迹诊断

传统 nominal 在 easy/moderate/hard 三层的 success 分别为 60%/20%/20%。ICODE 和 RL
的负效应并非只来自一个完全锁死场景：

- easy：traditional-nominal 为 60% success、20% collision；联合为 20%/73.3%；
- moderate：traditional-nominal 为 20%/80%；联合为 0%/86.7%；
- hard：traditional-nominal 为 20%/40%；联合为 33.3%/66.7%。

逐 step 描述性统计显示，traditional-ICODE 的平均绝对角速度约 `0.380 rad/s`，高于
traditional-nominal 的 `0.346 rad/s`，平均线速度则从 `0.194` 降到 `0.172 m/s`；其
表现更接近延长转向/绕行而非更快撞击。该现象与 learned rollout 改变可达性和终端代价
排序一致，但尚不能单独证明具体因果机制。

## 5. 当前最可能的技术原因

下列为由现有证据支持、但仍需专门实验验证的诊断假设：

1. **时域不匹配**：离线模型报告到 H20（2.0 s），而当前 MPPI 使用 H36（3.6 s）。
   学习误差在更长 rollout 中累积，MPPI 还可能主动选择利用模型偏差的控制序列。
2. **训练分布不足**：每个 ICODE model 当前只使用较小的 seen-domain 数据集训练；动态
   benchmark 中实际由 MPPI 和安全仲裁产生的状态—控制分布并未充分覆盖。
3. **目标错配**：最小化平均 rollout RMSE 不等于保持 MPPI 中候选轨迹的 cost ranking。
   很小的系统性偏差也可能改变 soft-min 权重和最终控制。
4. **动态障碍信息缺失**：ICODE 只修正小车动力学，不预测障碍未来运动。更准确的小车
   模型无法替代 dynamic-obstacle prediction；当前系统仍主要依赖反应式 LaserScan。

动作上下界、五状态定义、RK4 和 control dt 已核对一致，因此“控制维度或动作范围直接
不匹配”不是当前首要解释。

## 6. 科研决策

L37 sealed confirmation seeds 保持关闭。下一步不继续堆 RL，而先建立 ICODE 的
closed-loop eligibility gate：

1. 从实际 MPPI rollout 和 MuJoCo 执行中采集 on-policy residual transitions；
2. 在 H1/H5/H10/H20/H36 全时域比较 nominal、现有 ICODE 与新 ICODE；
3. 增加 cost-ranking preservation 指标，而不只看 state RMSE；
4. 先要求 ICODE 在 traditional MPPI 下改善且不增加碰撞；
5. 只有 ICODE 单模块 Gate 通过，才重新与 RL 组合。

这一顺序保留“ICODE 修正模型、RL 引导采样”的研究方向，但拒绝把当前负结果包装成成功。

## 7. 复现

```bash
python3 experiments/rl/run_bounded_rl_icode_factorial.py \
  --config configs/rl/bounded_rl_icode_factorial_l37.yaml \
  --output-dir results/research_platform/rl/l37_bounded_rl_icode_factorial_20260716_v1

python3 experiments/rl/summarize_bounded_rl_icode_factorial.py \
  --config configs/rl/bounded_rl_icode_factorial_l37.yaml \
  --input-dir results/research_platform/rl/l37_bounded_rl_icode_factorial_20260716_v1
```

