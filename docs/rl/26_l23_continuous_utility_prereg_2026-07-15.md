# L23：分组反事实连续效用集成与保守下界预注册

日期：2026-07-15  
状态：冻结于生成任何 L23 episode 数据之前  
前置证据：L22 证明 10-step burst 可产生约 -9.4 cm 至 +18.5 cm 的连续距离效应，但 40-step return 与距离高度冗余，固定三分类标签没有 class coverage。

## 1. 研究问题

L23 检验：

> 仅使用 correction 发生前可获得的 observation、BC/candidate action 和 twin-critic diagnostics，能否预测未来 40 步中 10-step gated-SAC burst 相对 frozen BC 的目标距离改善，并通过独立 calibration groups 构造保守的效用下界？

本轮不再把连续结果事先硬切成 harmful/neutral/beneficial。主要监督目标为：

\[
y = d_{40}^{\mathrm{BC}} - d_{40}^{\mathrm{burst}}.
\]

正值代表 burst 更接近目标，负值代表 burst 更远离目标。return、clearance、collision、success 和 safety override 保留为辅助审计结果，不进入主要监督标签。

## 2. 为什么选择连续距离效用

L22 的探索性分析中，return delta 与距离改善的 Pearson correlation 为 train `0.9868`、validation `0.9962`。在当前 40-step horizon 下，联合阈值并没有提供近似独立的双重证据，反而由 return 的尺度单独阻断全部标签。

因此 L23 只改变学习目标，不改变 intervention、horizon、scene、actor、critic 或 Gate。这避免同时改变多个因素而无法判断收益来自哪里。

## 3. 固定反事实处理

沿用 L22：

```text
Baseline:
  40 steps frozen BC

Candidate:
  first 10 steps = deterministic SAC proposal
                   -> target twin-critic consensus LCB beta 2
                   -> accepted: correction
                   -> rejected: frozen BC
  next 30 steps = frozen BC
```

- branch first step 必须被 β=2 Gate 接受；
- branch points 固定为 `10, 30, 50, 70, 90, 110`；
- MuJoCo、LaserScan、MPPI、local obstacle layer、scan_guard 与安全仲裁保持开启；
- ICODE、Memory、OOD 与动态障碍不在该可学性实验中混入；
- 三个 candidate checkpoints 固定为训练种子 `20260721/22/23` 的 `step_000020000.pt`。

## 4. 独立单位与数据隔离

branch rows 是同一 episode 内的重复测量。独立 group 定义为：

```text
(training_checkpoint_seed, episode_seed)
```

四段数据严格隔离：

### Train

```text
episode seeds = 20281501--20281520
groups = 3 checkpoints × 20 seeds = 60
```

只允许 train 计算 feature/target normalization、优化网络参数和 group bootstrap。

### Model selection

```text
episode seeds = 20281521--20281528
groups = 24
```

只允许用于 early stopping、与 zero/ridge baseline 比较和冻结模型权重。

### Calibration

```text
episode seeds = 20281529--20281536
groups = 24
```

只允许在模型冻结后计算 one-sided conformal multiplier；不得再调网络、特征、loss 或阈值。

### Sealed test

```text
legacy sealed = 20281311--20281315
new sealed    = 20281537--20281548
groups = 3 × 17 = 51
```

只有 development Gate 全部通过才允许生成 test branches。L20 closed-loop selection seeds `20281101--20281115` 继续密封，不能被本轮模型开发使用。

## 5. 样本量依据

L22 的 30 个独立 checkpoint/episode groups 中，组平均距离效应标准差约为 `1.844 cm`。将 `1 cm` 设为会改变是否启用 learned correction 的最小有意义效应（SESOI），标准化效应约为：

\[
d = \frac{1.0}{1.844} \approx 0.542.
\]

以双侧 α=0.05、power=0.80 的正态近似，独立组数敏感性为：

| standardized effect | approximate groups required |
|---:|---:|
| 0.40 | 50 |
| 0.542 | 27 |
| 0.80 | 13 |

最终 sealed test 预留 51 groups，覆盖到约 `d=0.4` 的敏感性。该计算只是组平均效用检验的样本量下界，不等价于神经网络泛化保证，也不把 episode 内 branch rows 当成独立样本。

## 6. 固定模型

输入沿用 L21/L22 的 67 维 leakage-free feature：

```text
normalized observation
frozen BC action
candidate action
candidate - BC correction
online/target twin-critic advantage and disagreement
correction magnitude
branch step
```

严禁输入 future distance、return、success、collision 或 branch outcome。

主模型为 5-member MLP ensemble：

```text
67 -> 64 -> 32 -> 1
activation = SiLU
loss = Huber(delta=1.0) on train-standardized target
optimizer = Adam(lr=1e-3, weight_decay=1e-4)
batch size = 64
maximum epochs = 300
gradient clipping = 1.0
early stopping patience = 40
ReduceLROnPlateau patience = 15, factor = 0.5, min lr = 1e-5
```

每个 member 使用一个固定 seed，并以完整 `(checkpoint, episode)` groups 为单位 bootstrap；不得逐 branch row bootstrap。feature 与 target normalizer 仅由原始 train groups 计算，bootstrap 不改变统计量。

## 7. 固定基线

1. **Zero predictor**：始终预测 `0 m`，代表“不知道 correction 有何影响”；
2. **Ridge predictor**：train-only 标准化特征，固定 `lambda=0.01` 的线性 ridge；
3. **MLP ensemble**：上述固定结构。

不扫描 hidden size、activation、ensemble size 或 ridge lambda。model-selection split 只选择每个 member 的训练 epoch，不选择新架构。

## 8. Group conformal 保守下界

冻结 ensemble 后，对每一行得到：

\[
\mu_i = \frac{1}{M}\sum_m \hat y_{im},
\qquad
\sigma_i = \max\left(\operatorname{Std}_m(\hat y_{im}), 0.005\ \mathrm{m}\right).
\]

calibration row nonconformity：

\[
s_i = \frac{\mu_i-y_i}{\sigma_i}.
\]

为避免把同一 episode 的六个 branch 当独立 calibration samples，每个 group 取：

\[
s_g = \max_{i\in g} s_i.
\]

使用 90% one-sided split-conformal 的 finite-sample `higher` quantile，并强制 multiplier 不小于 0：

\[
q = \max\left(0, Q_{\lceil(n_g+1)(1-\alpha)\rceil/n_g}(s_g)\right),
\quad \alpha=0.10.
\]

部署候选下界：

\[
\mathrm{LCB}_i = \mu_i - q\sigma_i.
\]

只有 `LCB > 0` 才允许进入未来 correction Gate；scan_guard 与控制仲裁仍拥有更高优先级。这里的 conformal 只针对同分布 exchangeable groups，不声称覆盖 OOD 或实车分布。

## 9. Development Gate

打开 sealed test 前必须同时满足：

### 数据质量

1. replay error ≤ `1e-10`；
2. burst accounting、CSV/NPZ、schema、split 与 finite-value 检查通过；
3. train ≥ 180 rows，selection ≥ 60 rows，calibration ≥ 60 rows；
4. selection 和 calibration 中 `|y|>=3 cm` 的正、负 effect groups 各至少 4 个；
5. 三个 checkpoints 均贡献所有 split。

### 模型可学性

1. ensemble model-selection RMSE 至少优于 zero predictor 5%；
2. ensemble RMSE 不得超过 ridge RMSE 的 1.02 倍；
3. 对 `|y|>=3 cm` rows 的 sign balanced accuracy ≥ 0.60；
4. 所有指标按 group 同时报 row-level 描述，不把 rows 当独立重复。

### 校准可用性

1. calibration LCB 接受比例 ≥ 5%；
2. 被接受 rows 的真实平均距离改善 ≥ 1 cm；
3. 被接受 rows 中不得出现 `y <= -3 cm`；
4. calibration 只计算 `q` 和 Gate 审计，不反向修改模型。

任一条件失败：保留 BC fallback，不打开 test，不通过降低 effect threshold、换 seed、扫描网络或反复调 `q` 来补救。

## 10. Sealed-test Gate

若 development Gate 通过，冻结 checkpoint SHA256 与全部配置后打开 test。测试通过条件预先固定：

1. ensemble test RMSE 优于 zero predictor；
2. ensemble test RMSE ≤ 1.02 × ridge test RMSE；
3. group-level one-sided LCB empirical coverage ≥ 0.85；
4. LCB 接受比例在 5%–50%；
5. accepted true mean utility ≥ 1 cm；
6. accepted rows 无 `y <= -3 cm`；
7. 三个 actor checkpoints 均至少有一个 accepted row，防止 Gate 只适用于单一训练 seed。

只有 test 通过，才允许把 utility Gate 接入完整 episode，并最终打开仍密封的 closed-loop selection seeds。

## 11. 解释边界

- 通过 development 但未打开 test：只能声称开发可学性；
- test 通过：只能声称静态 U-trap、当前 frozen actors 和当前 MuJoCo 分布上的短时效用可预测；
- 不声称 conformal 覆盖 OOD、动态障碍或实车；
- 不声称 RL 一定优于传统 MPPI；
- 不把 utility predictor 解释为因果安全证明；
- ICODE 与 RL utility Gate 的联合实验必须在各自单模块验证后另行预注册。

