# L23：分组连续效用集成结果与 Development Gate 审计

日期：2026-07-15  
性质：development 数据、模型与校准完成；Development Gate 未通过；sealed test 未打开  
预注册：[`26_l23_continuous_utility_prereg_2026-07-15.md`](26_l23_continuous_utility_prereg_2026-07-15.md)  
数值修正：[`27_l23_zero_variance_scaling_correction_2026-07-15.md`](27_l23_zero_variance_scaling_correction_2026-07-15.md)

## 1. 结论先行

L23 成功建立了 actor-independent continuous-utility 数据、训练、基线、deep ensemble、group conformal 和 fail-closed 加载链路，但当前 67 维“单时刻 observation/action/critic”特征不足以稳定预测未来 40 步 burst 效用：

```json
{
  "development_gate_passed": false,
  "selection_zero_rmse_m": 0.04572,
  "selection_ensemble_rmse_m": 0.04532,
  "relative_improvement": "0.88% < preregistered 5%",
  "meaningful_sign_balanced_accuracy": "56.67% < 60%",
  "calibration_lcb_accept_fraction": "0% < 5%",
  "sealed_test_opened": false
}
```

因此该 checkpoint 被明确标记为不可部署。默认加载接口会拒绝 development Gate 失败的 artifact；只有显式 `require_eligible=False` 才允许离线检查。

## 2. 数据规模与质量

三套冻结 SAC checkpoints 使用相同的 L22 反事实处理：10-step target-LCB-β2 burst，随后 30-step frozen BC。

| split | independent groups | accepted branch rows | positive-effect groups | negative-effect groups |
|---|---:|---:|---:|---:|
| train | 60 | 235 | 24 | 21 |
| model selection | 24 | 96 | 14 | 8 |
| calibration | 24 | 92 | 4 | 8 |
| total development | **108** | **423** | — | — |

effect group 使用 `|distance improvement| >= 3 cm` 作预注册覆盖审计，不作为硬分类训练标签。

数据质量：

- 648 个 scheduled/considered branch points；
- 423 个首步 β=2 accepted branches；
- history replay maximum error = `0.0`；
- burst accounting = passed；
- duplicate composite keys = 0；
- CSV/NPZ missing、NaN、Inf = 0；
- 三个 actor checkpoints 均贡献 train/selection/calibration；
- sealed test 17 episode seeds 全部未运行。

## 3. 连续目标分布

| split | min | median | max | mean ± SD |
|---|---:|---:|---:|---:|
| train | -19.80 cm | +0.13 cm | +21.76 cm | +0.26 ± 4.25 cm |
| selection | -21.66 cm | -0.01 cm | +19.46 cm | +0.44 ± 4.57 cm |
| calibration | -15.97 cm | -0.12 cm | +13.75 cm | -0.42 ± 3.63 cm |

正负样本数量总体接近平衡，但绝大多数局部效应围绕 0，少数 90/110-step branch 出现较大正负变化。这是一个“多数微弱、少数重要”的异方差回归问题。

## 4. 模型合同

- input：67 维 leakage-free current-state feature；
- ensemble：5 × `67 -> 64 -> 32 -> 1` SiLU MLP；
- 每个 member 以完整 checkpoint/episode group bootstrap；
- train-only feature/target normalization；
- Huber loss、Adam、gradient clipping、early stopping；
- zero predictor 与 fixed-lambda ridge 作为预注册基线；
- model-selection split 冻结权重；
- calibration split 只计算 conformal multiplier，不反向调模型。

每个模型有 6,465 个参数。五个 member 的最佳 epoch 分别为 `1, 1, 1, 9, 5`，说明 selection loss 很早停止改善，并非因 300 epoch 预算不足而提前截断。

## 5. Model-selection 结果

### Row-level

| predictor | RMSE | MAE | Pearson | R² |
|---|---:|---:|---:|---:|
| zero | 4.572 cm | 2.509 cm | 0 | -0.010 |
| ridge | 4.865 cm | 3.082 cm | 0.009 | -0.143 |
| ensemble | **4.532 cm** | 2.526 cm | 0.123 | 0.008 |

ensemble 相对 zero RMSE 改善：

\[
1 - \frac{0.045316}{0.045718} = 0.88\%.
\]

预注册要求至少 5%，因此失败。ensemble 明显优于 ridge，但并没有获得足以支持部署的绝对预测增益。

### Meaningful-effect sign

在 `|y|>=3 cm` 的 24 行中：

| sign | rows | accuracy |
|---|---:|---:|
| positive | 15 | 80.0% |
| negative | 9 | 33.3% |
| balanced | 24 | **56.7%** |

模型倾向预测正效用，能识别部分收益，却漏掉大多数明显负效用。对安全 Gate 而言，这种非对称错误不可接受。

## 6. Calibration 与不确定性

数值修正后，ensemble epistemic standard deviation 处于合理量级：

```text
calibration mean   = 0.244 cm
calibration median = 0.177 cm
calibration max    = 0.647 cm
```

但 calibration row RMSE 为 3.664 cm，远大于 member disagreement。换言之，ensemble members 彼此很一致，却共同预测错误；epistemic disagreement 严重低估了未来闭环变化。

group-wise 90% one-sided conformal 使用每个 checkpoint/episode group 的最坏 branch，得到：

```text
q = 32.4849
empirical calibration group coverage = 100%
LCB > 0 accepted rows = 0 / 92
```

这不是 calibration 算法故障。为了覆盖 calibration 中模型未预见的负效用，保守下界必须大幅下移，最终合理地退化为“全部使用 BC”。

## 7. 首轮无效数值运行

第一次训练运行将 train 近常量 feature 除以 `1e-6`，使 calibration feature 放大到约 `-8.11e6`，产生上千米级无效预测。该 artifact 保留但不用于科研结论：

```text
results/research_platform/rl/l23_utility_ensemble_development_20260715_v1/
```

修正采用标准零方差规则 `scale=1`，不改变任何模型或实验 Gate。修正后所有 split 的最大标准化绝对值为：

```text
train       6.30
selection   3.79
calibration 8.11
```

有效结果目录为 `...development_20260715_v2/`。

## 8. 探索性 EDA：为什么 current-state 模型不可学

该部分在 Development Gate 判定后进行，只解释失败，不修改 L23。

### 8.1 单个输入特征缺少稳定信号

train 中与 target 相关性最大的 feature 也只有 `|r|=0.117`。67 个 features 中，没有任何一个同时满足：

```text
|train correlation| >= 0.2
|selection correlation| >= 0.2
same correlation sign
```

部分 train/selection 同方向特征到了 calibration 又反向，说明单时刻相关结构不稳定。

### 8.2 相近状态仍可产生不同未来结果

使用标准化 feature 的最近 train row 直接预测：

| split | nearest-neighbor RMSE | zero RMSE |
|---|---:|---:|
| selection | 5.932 cm | 4.572 cm |
| calibration | 5.183 cm | 3.635 cm |

即使找当前 feature 最接近的历史状态，未来 burst 效用仍可能相反。原因包括后续 MPPI perturbation sequence、后续 Gate 接受序列和轨迹几何演化；这些都没有包含在一个 current-state feature 中。

### 8.3 效应主要出现在轨迹后段

train 中 branch step 50 的效应 SD 仅 0.39 cm，而 step 90/110 分别达到 6.32/6.99 cm；validation 也呈相同的后段高方差趋势。branch step 虽已作为标量输入，但无法描述该时刻候选轨迹会朝障碍物哪一侧发展。

## 9. 科研判断

L23 否定的是下面这个窄假设：

> 只凭当前 observation、当前 BC/candidate action 和 actor 自身 twin-critic diagnostics，就能可靠预测 10-step correction 对未来 40 步轨迹的正负效用。

它没有否定：

- RL correction 能改变 MPPI 轨迹；
- correction 的影响具有状态和时序依赖性；
- 使用独立风险/效用信息进行调控具有研究价值；
- ICODE 改善动力学预测的独立价值。

相反，本轮说明 future trajectory context 是缺失变量。继续增加同构 MLP 层数、扫描学习率或降低 conformal 标准，很可能只会对 development split 过拟合。

## 10. 下一阶段方法建议（尚未实施）

下一阶段应预注册 trajectory-aware utility Gate。候选输入不再只有一个动作，而应包含 planner 在做决定时本来就能获得的短轨迹信息，例如：

- frozen-BC 与 RL-prior nominal rollout 的预测 progress difference；
- predicted minimum clearance difference；
- rollout cost difference 与 cost component vector；
- top-k sampled trajectory cost quantiles；
- curvature、control jerk 和 obstacle-side geometry；
- ICODE/nominal rollout disagreement，作为动力学不确定性；
- 最近若干步 Gate/critic/clearance history。

这样 Gate 回答的是“这两条候选轨迹哪条更值得执行”，而不是要求单时刻网络猜测尚未生成的未来随机轨迹。

在 trajectory-aware development Gate 通过前：

- L23 checkpoint 不接入闭环；
- sealed test 不打开；
- closed-loop selection seeds 不打开；
- BC fallback 与 scan_guard 保持不变。

## 11. 结果位置

```text
results/research_platform/rl/l23_utility_data_multiseed_20260715_v1/
  audit.json
  samples_all_training_seeds.csv
  samples_all_training_seeds.npz

results/research_platform/rl/l23_utility_ensemble_development_20260715_v2/
  utility_ensemble.pt
  development_metrics.json
  training.csv
  config_snapshot.json
```

