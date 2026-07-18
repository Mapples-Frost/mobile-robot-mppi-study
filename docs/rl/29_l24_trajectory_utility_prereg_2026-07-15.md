# L24：配对 MPPI 候选轨迹效用门控预注册

日期：2026-07-15  
状态：冻结于生成任何 L24 episode outcome 之前

## 1. 研究问题

L23 已经排除了“仅凭当前 67 维状态特征预测未来 10-step correction burst 效用”的简单方案。L24 检验：

> 在相同机器人状态、相同感知输入和相同 MPPI 随机数下，同时预览 frozen BC prior 与 SAC candidate prior 产生的候选轨迹，是否能够可靠预测未来 40 步中 10-step gated-SAC burst 相对 frozen BC 的目标距离改善？

主要目标保持不变：

\[
y = d_{40}^{\mathrm{BC}} - d_{40}^{\mathrm{burst}}.
\]

正值表示 correction burst 更接近最终目标，负值表示 correction burst 更差。

## 2. 固定处理与控制条件

本轮只改变效用模型的输入表示，不改变 L23 的处理：

```text
Baseline: 40 steps frozen BC

Candidate:
  first 10 steps = deterministic SAC correction
                   -> target twin-critic consensus LCB, beta=2
                   -> rejection falls back to frozen BC
  next 30 steps = frozen BC
```

- branch steps 固定为 `10, 30, 50, 70, 90, 110`；
- 三个 actor checkpoints 仍为训练种子 `20260721/22/23` 的 `step_000020000.pt`；
- MuJoCo、LaserScan、local obstacle layer、scan_guard 和 safety arbitration 始终启用；
- Memory、ICODE、动态障碍与 OOD 不混入该表征可学性实验；
- 只有 branch 第一步通过原有 beta=2 critic gate 的样本进入数据集。

## 3. 配对候选轨迹预览

在执行任何 branch 动作前，分别把 BC latent action 与 SAC latent action送入同一个 MPPI controller。两次预览：

1. 使用完全相同的物理状态、LaserScan 和 local obstacles；
2. 使用克隆的 MPPI RNG state，形成 common random numbers；
3. 不消耗真实 controller RNG；
4. 不修改 previous sequence、previous action、reference progress 或 safety state；
5. 不执行控制，因此不访问未来 ground truth 或 branch outcome。

这是一项离线研究接口，不改变正式 `plan()` 的默认行为。

## 4. 冻结特征

前 67 维完全沿用 L23：normalized observation、BC/candidate action、action difference、online/target critic diagnostics、correction magnitude 与 branch step。

每条候选轨迹固定提取 20 个物理量：

1. 预测局部目标进展；
2. 预测最终目标进展；
3. 末端局部目标距离；
4. 末端最终目标距离；
5. 轨迹长度；
6. 最小障碍净空；
7. 预测碰撞点比例；
8. 障碍接近代价；
9. 控制能量；
10. 控制变化 RMS；
11. 控制 jerk RMS；
12. 平均绝对角速度命令；
13. 最大绝对角速度命令；
14. 最小 rollout cost；
15. rollout cost 10% 分位数；
16. rollout cost 中位数；
17. rollout cost 90% 分位数；
18. rollout cost 标准差；
19. 归一化 effective sample size；
20. 采样控制饱和比例。

最终输入固定为：

```text
67 state features
+ 20 BC trajectory metrics
+ 20 SAC trajectory metrics
+ 20 (SAC - BC) trajectory differences
= 127 features
```

禁止根据 L24 outcome 删除、增加或筛选特征。

## 5. 独立单位与数据隔离

独立实验单位仍为：

```text
(training checkpoint seed, episode seed)
```

同一 episode 中最多六个 branch rows 是重复测量，不能作为六个独立 replicate。

```text
Train seeds       = 20281601--20281620  -> 60 groups
Model selection   = 20281621--20281628  -> 24 groups
Calibration       = 20281629--20281636  -> 24 groups
Sealed test       = 20281311--20281315
                    20281537--20281548  -> 51 groups
```

L23 development seeds 不复用。L20 closed-loop selection seeds `20281101--20281115` 继续封存。

## 6. 样本量依据

L23 的 108 个独立开发 groups 中，group-mean utility 的标准差为 `2.499 cm`。以会改变 correction 启用决策的 `1 cm` 作为 SESOI：

\[
d_z = \frac{1.0}{2.499} \approx 0.400.
\]

在双侧 `alpha=0.05`、power `0.80` 的正态近似下，约需 50--52 个独立 groups 检出该尺度的平均效应。L24 训练阶段包含 60 groups；最终封存测试保留 51 groups。该计算只约束 group-level 效应敏感度，不是神经网络泛化保证。模型可学性仍由独立 selection 与 calibration gates 判断。

## 7. 固定模型与直接消融

主模型保持 L23 结构和训练配置：5-member group-bootstrap MLP ensemble，hidden sizes `64,32`，SiLU、Huber、Adam、train-only normalization、selection early stopping 和 group conformal calibration。

在完全相同的新数据、split、随机种子和训练程序上同步训练：

1. **State-only ensemble**：只使用前 67 维；
2. **Trajectory-augmented ensemble**：使用全部 127 维；
3. **Zero predictor**；
4. **Full-feature ridge predictor**。

这项直接消融用于判断改善是否来自轨迹信息，而不是换数据或换训练过程。

## 8. Development Gate

只有同时满足以下条件才允许打开 sealed test：

### 数据合同

- replay error 不超过 `1e-10`；
- preview 不改变后续正式 MPPI trajectory；
- feature dimension 精确为 `127`；
- train/selection/calibration rows 分别至少为 `180/60/60`；
- selection 与 calibration 中 `|y|>=3 cm` 的正负 effect groups 各至少 4 个；
- 三个 actor checkpoints 均出现在全部 split。

### 预测可学性

- trajectory ensemble selection RMSE 至少优于 zero predictor 5%；
- trajectory ensemble selection RMSE 至少优于 state-only ensemble 5%；
- trajectory ensemble RMSE 不超过 ridge 的 `1.02` 倍；
- 对 `|y|>=3 cm` 样本的 sign balanced accuracy 至少为 `0.60`。

### 校准可用性

- group-conformal LCB 接受比例至少为 5%；
- accepted rows 的真实平均 utility 至少为 `1 cm`；
- accepted rows 中不得出现 `y<=-3 cm`。

任一条件失败：保留 BC fallback、禁止部署 checkpoint、保持全部 test 与 closed-loop selection seeds 封存。禁止通过扫 hidden size、改 feature list、降 effect threshold 或反复调整 conformal multiplier 补救。

## 9. 可证伪解释

- trajectory model 不优于 state-only：配对单步候选轨迹仍不足以表征 10-step macro-policy outcome；
- 预测改善但校准零接受：模型有排序信息但不能形成安全门控；
- development 通过但 sealed test 失败：表征存在开发集过拟合或分布不稳定；
- sealed test 通过：只支持当前静态 U-trap、当前 actor checkpoints 和当前 MuJoCo 分布内的短期效用预测，不支持 OOD、动态障碍或实车保证。
