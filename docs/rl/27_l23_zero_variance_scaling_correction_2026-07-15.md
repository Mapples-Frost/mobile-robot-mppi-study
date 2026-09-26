# L23 数值实现修正记录：近零方差特征不得除以 epsilon

日期：2026-07-15  
状态：首轮 development model 输出判为无效后、任何 sealed test 打开之前  
性质：数值正确性修复，不改变 L23 数据、模型结构、loss、超参数、split、Gate 或结论标准

## 1. 发现的问题

L23 第一次训练运行使用：

```python
scale = max(train_std, 1e-6)
z = (x - train_mean) / scale
```

train 中有 15 个 feature 的标准差低于 `1e-6`。calibration 某行的 feature 11 从 train 常值约 `0.0536` 变化到 `-8.0584`，被除以 `1e-6` 后变成约 `-8.11e6`。这使一个 ensemble member 外推到上千米，导致：

```text
calibration epistemic_std_max > 1000 m
conformal multiplier > 32
```

该预测超出任务尺度数个数量级，属于标准化实现错误，不能作为方法失败或成功的证据。

## 2. 固定修正

采用标准零方差处理：

```python
if train_std < 1e-6:
    scale = 1.0
else:
    scale = train_std
```

即近常量 feature 只减去 train mean，不通过 epsilon 人为放大。该规则只由 train statistics 决定，不查看 target，不裁剪 holdout outcome，也不改变任何统计 Gate。

修正后预期该 calibration feature 的标准化绝对值约为 `8.11`，仍明确保留其分布偏移信息，但不会制造百万倍数值放大。

## 3. 重跑合同

以下内容完全不变：

- 423 个 L23 development branches；
- train/model-selection/calibration episode groups；
- 5 个 ensemble seeds；
- `67 -> 64 -> 32 -> 1` SiLU 网络；
- optimizer、Huber loss、epochs、early stopping；
- ridge/zero baseline；
- group conformal 公式；
- development Gate 的全部阈值；
- sealed test `20281311--20281315, 20281537--20281548` 仍未打开。

第一次数值无效输出保留在：

```text
results/research_platform/rl/l23_utility_ensemble_development_20260715_v1/
```

修正后的重跑使用新目录和新 checkpoint hash，不覆盖该审计痕迹。

