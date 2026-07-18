# L24：配对 MPPI 轨迹效用模型结果与研究转向

日期：2026-07-15  
性质：development experiment；Development Gate 未通过；sealed test 未打开  
预注册：[`29_l24_trajectory_utility_prereg_2026-07-15.md`](29_l24_trajectory_utility_prereg_2026-07-15.md)

## 1. 结论先行

L24 完成了 side-effect-free MPPI preview、BC/SAC common-random-number 候选轨迹比较、127 维轨迹效用数据、state-only 直接消融、group-bootstrap ensemble 和 group-conformal calibration。

但结果否定了本轮核心假设：

> 当前时刻的一对 BC/SAC MPPI 候选轨迹，仍不足以可靠预测随后 10-step repeated-replanning correction burst 的 40-step 闭环效用。

```json
{
  "selection_zero_rmse_cm": 5.144,
  "selection_state_only_rmse_cm": 5.024,
  "selection_trajectory_rmse_cm": 5.075,
  "trajectory_improvement_over_zero": "1.32% < 5%",
  "trajectory_change_vs_state_only": "1.03% worse",
  "meaningful_sign_balanced_accuracy": "57.74% < 60%",
  "calibration_lcb_acceptance": "0% < 5%",
  "development_gate_passed": false,
  "sealed_test_opened": false
}
```

因此该 checkpoint 被标记为不可部署；运行链路继续使用 frozen BC / traditional MPPI fallback。

## 2. 实现与因果隔离

新增 `MppiController.preview_plan()`，使用克隆 RNG 做候选轨迹预览，不消耗正式 MPPI RNG，也不修改 previous sequence 或 previous action。`MppiPriorEnv.preview_policy_action()` 同时恢复 latent prior 参数与 reference progress。

正式采集前完成了同 seed 回归：在关闭和打开 trajectory preview 的两套 collector 中，10 条 branch 的所有非 feature 字段逐字段完全一致，包括 baseline/candidate outcome、return、distance、clearance、safety 与 intervention accounting。这证明 preview 没有改变被测处理。

每个样本固定包含：

```text
67 L23 state/action/critic features
+ 20 BC trajectory metrics
+ 20 SAC trajectory metrics
+ 20 SAC-minus-BC trajectory deltas
= 127 features
```

轨迹指标覆盖 progress、clearance、collision fraction、obstacle proximity、control effort/rate/jerk、turning、cost quantiles、effective sample size 与 saturation。

## 3. 数据质量

三个独立 actor checkpoints 在全新的 episode seeds 上产生 409 条 accepted branches：

| split | independent groups | rows | positive-effect groups | negative-effect groups |
|---|---:|---:|---:|---:|
| train | 60 | 222 | 19 | 21 |
| model selection | 24 | 94 | 12 | 7 |
| calibration | 24 | 93 | 11 | 5 |
| total | **108** | **409** | — | — |

质量合同：

- 648 个 considered branch points，409 个首步 beta=2 accepted branches；
- replay maximum error `0.0`；
- burst accounting passed；
- duplicate composite keys `0`；
- CSV/NPZ missing、NaN、Inf 均为 `0`；
- feature dimension 精确为 `127`；
- candidate-minus-base feature block 精确复现；
- 三个 checkpoints 覆盖全部 split；
- 17 个 sealed test episode seeds 全部未运行。

连续 utility 范围：

| split | minimum | median | maximum |
|---|---:|---:|---:|
| train | -13.43 cm | -0.03 cm | +28.90 cm |
| selection | -26.30 cm | +0.21 cm | +19.72 cm |
| calibration | -7.37 cm | +0.28 cm | +11.70 cm |

数据量与正负效应覆盖通过预注册门，因此失败不是由于缺少 rows 或单边标签。

## 4. Model-selection 结果

### Row-level

| predictor | RMSE | MAE | Pearson | R2 |
|---|---:|---:|---:|---:|
| zero | 5.144 cm | 2.693 cm | 0.000 | -0.001 |
| full ridge | 5.732 cm | 3.579 cm | 0.040 | -0.243 |
| state-only ensemble | **5.024 cm** | 2.664 cm | 0.289 | 0.045 |
| trajectory ensemble | 5.075 cm | **2.665 cm** | 0.298 | 0.026 |

轨迹 ensemble 比 zero 只改善 `1.32%`，未达到 5%；同时比完全相同训练程序的 state-only ensemble 差 `1.03%`，未通过直接表征消融。

### Group-mean

| predictor | group RMSE | group Pearson |
|---|---:|---:|
| zero | 2.483 cm | 0.000 |
| state-only ensemble | **2.397 cm** | **0.488** |
| trajectory ensemble | 2.457 cm | 0.337 |

按真正独立的 checkpoint/episode groups 聚合后，trajectory representation 仍没有超过 state-only representation。

### Meaningful-effect sign

在 `|y|>=3 cm` 的 selection rows 中：

| model | negative accuracy | positive accuracy | balanced accuracy |
|---|---:|---:|---:|
| state-only ensemble | 71.43% | 58.33% | **64.88%** |
| trajectory ensemble | 57.14% | 58.33% | **57.74%** |

加入当前候选轨迹后，明显负效用识别反而下降。

## 5. Calibration

trajectory ensemble 在 calibration 上：

```text
row RMSE              = 3.094 cm
zero predictor RMSE   = 3.084 cm
epistemic std median  = 0.253 cm
conformal multiplier  = 14.4376
empirical group cover = 100%
LCB > 0 accepted      = 0 / 93
```

ensemble disagreement 仍显著小于真实误差。group conformal 为覆盖每个 episode group 的最坏 row，需要把不确定度放大约 14.4 倍，最终合理地退化为全部拒绝 learned correction。

## 6. Gate 后解释性 EDA

完整报告位于：

```text
results/research_platform/rl/l24_trajectory_utility_data_multiseed_20260715_v1/trajectory_eda.md
```

主要发现：

1. `SAC - BC` 预测最终进展与真实 40-step utility 的相关系数分别为 train `-0.001`、selection `-0.178`、calibration `+0.032`；方向不稳定。
2. 127 个 feature 中，没有任何一个在 train、selection、calibration 的 group-mean 层面同时达到 `|r|>=0.2` 且方向一致。
3. selection 中 cost delta 和 clearance delta 有局部相关性，但在 train/calibration 不复现。
4. full-feature nearest-neighbour 相对 state-only 在 selection 略好，却在 calibration 明显更差，说明轨迹特征引入了 split-specific 邻域结构。
5. 大效应继续集中于 branch step 90/110；早期单步 preview 无法描述随后多次重新感知、重新采样和重新 gate 后的轨迹分叉。

## 7. 科研判断

L24 不是“神经网络训练轮数不够”。十个 ensemble members（full 与 state ablation）多数在早期 selection epoch 达到最佳，且 ridge、nearest-neighbour 与相关性诊断给出一致结论：当前输入与未来 macro-policy outcome 的映射不稳定。

根本原因是 treatment 与 feature horizon 不一致：

```text
feature  = 当前一次 MPPI preview
outcome  = 未来 10 次重新感知、重新采样、重新规划、重新 gate 的闭环序列
```

后续 outcome 取决于未来 LaserScan、MPPI perturbations、accepted/rejected gate sequence 和安全仲裁；这些变量在当前 preview 时尚未发生。继续扩大 MLP、扫描超参数或降低 conformal 标准，很可能只会对 development split 过拟合。

## 8. 研究转向

L18--L24 已经系统验证并排除了 raw critic threshold、twin-critic consensus、自监督风险分类、state-only utility 和 one-preview trajectory utility。为避免论文 story 继续膨胀，下一阶段不再把“预测整个未来 correction burst 效用的 learned Gate”作为主线。

更合理的主线是：

1. 传统 MPPI/BC 作为简单、低扰动、in-distribution 场景的强基线；
2. 使用可审计的 OOD/场景复杂度信号决定是否开放 RL sampling prior；
3. RL 只负责在复杂或 OOD 区域提供更有方向的候选分布；
4. MPPI 仍优化轨迹，scan_guard 仍拥有最高安全优先级；
5. ICODE 作为独立动力学预测模块单独消融，固定环境中允许冻结或低频更新；
6. 主要证据来自静态/动态障碍、seen/OOD、无 residual/不同 residual 的完整 factorial ablation，而不是再增加一个难以校准的 learned utility network。

该转向与导师提出的“简单场景传统 MPPI 可能更好、OOD 强调 RL 探索功能、story 不要有过多元素”一致。

## 9. 结果位置

```text
results/research_platform/rl/l24_trajectory_utility_data_multiseed_20260715_v1/
  audit.json
  samples_all_training_seeds.csv
  samples_all_training_seeds.npz
  trajectory_eda.json
  trajectory_eda.md

results/research_platform/rl/l24_trajectory_utility_ensemble_development_20260715_v1/
  utility_ensemble.pt
  development_metrics.json
  training.csv
  config_snapshot.json
```

有效 checkpoint SHA256：

```text
68974e89b6d6f23f8092ca11945b587cfc6fafd886cae7ab832485bb3fa1e850
```

“有效”仅表示文件与数值合同有效；该 checkpoint 的 Development Gate 为失败，默认 loader 必须拒绝部署。
