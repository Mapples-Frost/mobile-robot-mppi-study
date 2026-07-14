# RL Sampling-Prior Learnability Gate（2026-07-13）

## 1. 目的与边界

本 Gate 只验证 SAC 能否学习对 MPPI 有价值的 sampling prior，不验证 ICODE，
也不混入 Memory、OOD gate 或物理域随机化。

固定设置：

```text
prediction dynamics: nominal
sampling prior: SAC delta prior
MPPI K: 100
MPPI horizon: 36
control dt: 0.1 s
training steps: 30,000
training scenes: clean_single_obstacle + u_trap_long_board
validation: every 5,000 steps, 5 fixed seeds per scene
```

配置：`configs/rl/sac_mppi_learnability_gate.yaml`。

## 2. 实际训练结果

| Step | Clean success | Clean final distance | U-trap success | U-trap final distance | Collision |
|---:|---:|---:|---:|---:|---:|
| 5k | 0/5 | 0.899 m | 0/5 | 0.832 m | 0/10 |
| 10k | 5/5 | 0.291 m | 0/5 | 1.270 m | 0/10 |
| 15k | 3/5 | 0.519 m | 0/5 | 3.904 m | 0/10 |
| 20k | 3/5 | 0.623 m | 0/5 | 3.910 m | 0/10 |
| 25k | 3/5 | 0.676 m | 0/5 | 2.041 m | 0/10 |
| 30k | 5/5 | 0.295 m | 0/5 | 1.099 m | 0/10 |

综合 `best.pt` 位于 10k。15k–25k 的下降表明多场景混合训练存在明显振荡，
不能用训练末尾 checkpoint 代替验证集模型选择。

## 3. 独立固定-seed控制对照

训练结束后重新使用 seeds `41–45` 独立运行40个MuJoCo回合。所有方法使用同一
`K=100`、同一场景、同一安全链。

### Clean single obstacle

| Method | Success | Collision | Final distance | Safety interventions | Planner time |
|---|---:|---:|---:|---:|---:|
| MPPI | 1/5 | 0/5 | 1.840 m | 135.6 | 4.06 ms |
| Untrained RL | 1/5 | 0/5 | 1.428 m | 111.4 | 4.34 ms |
| RL 10k | 4/5 | 0/5 | 0.547 m | 0.0 | 4.25 ms |
| RL 30k | 4/5 | 0/5 | 0.735 m | 56.6 | 4.21 ms |

### U-trap long board

| Method | Success | Collision | Final distance | Safety interventions | Planner time |
|---|---:|---:|---:|---:|---:|
| MPPI | 0/5 | 0/5 | 3.434 m | 264.6 | 3.89 ms |
| Untrained RL | 0/5 | 0/5 | 3.441 m | 259.2 | 4.38 ms |
| RL 10k | 0/5 | 0/5 | 1.278 m | 22.4 | 4.53 ms |
| RL 30k | 0/5 | 0/5 | 1.491 m | 74.6 | 4.70 ms |

在U-trap中，10k模型相对MPPI把平均最终距离降低约62.8%，安全干预降低约91.5%，
但成功率仍是0。代表性轨迹显示RL确实学会了离开传统MPPI的局部停滞区域，并绕到
障碍群外侧；它尚未学会稳定回到终点容差内。

将最大回合从360步延长至600步后，RL 10k仍为0/5成功，平均最终距离1.016 m，
因此失败不只是时间不够。

## 4. Gate 判定

分开判定，避免把“有改善”和“任务成功”混为一谈：

- **L1-A 基础可学习性：通过。** 训练策略显著改变控制行为；未训练策略没有同等增益；
  简单场景成功率和低K采样效率均有改善。
- **L1-B 复杂拓扑任务成功：未通过。** U-trap没有一次达到0.30 m终点容差。
- **安全 smoke：通过。** 独立40个对照回合全部无碰撞，但样本量不足以证明安全性。
- **进入 ICODE+RL 正式消融：暂缓。** 必须先解决U-trap目标捕获和训练稳定性。

## 5. 证据支持的下一步

下一轮不盲目增加训练步数，优先做三个可证伪修改：

1. **课程学习：** 先用clean场景建立稳定goal capture，再逐步提高U-trap采样比例，
   避免两个难度差异大的场景互相遗忘。
2. **探索保持消融：** 比较固定/下限约束entropy coefficient和当前自动alpha；本轮
   alpha从0.2下降到约0.013，复杂场景探索可能过早收缩。
3. **近目标回退：** RL负责远距离探索；进入可直接到达的终点邻域后，逐渐把prior
   交回goal warm-start MPPI。该机制符合“简单区域传统MPPI更可靠”的研究叙事。

只有其中至少一个配置在固定5 seeds的U-trap中出现成功，才扩展至10–20 seeds、
unseen physics、ICODE+RL和OOD gate。

## 6. 可追溯结果

```text
results/research_platform/rl/learnability_gate_20260713/
├── checkpoints/
├── episodes.csv
├── updates.csv
├── validation_episodes.csv
├── training_curves.png
├── fixed_seed_benchmark/
│   ├── episodes.csv
│   ├── summary.csv
│   ├── metrics.json
│   ├── learnability_gate_summary.png
│   └── learnability_gate_trajectories.png
└── u_trap_extended_horizon/
```

这些结果是30k-step研究开发Gate，不是最终论文统计结论。
