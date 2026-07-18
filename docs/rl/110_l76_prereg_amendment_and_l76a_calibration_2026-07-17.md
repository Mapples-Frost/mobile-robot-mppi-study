# L76 预注册修订：逐 step 支持度尺度与 L76a 校准

日期：2026-07-17  
状态：在任何 L76 正式开发 episode 运行前冻结

## 1. 修订原因

初始 L76 预注册把 L75 的 episode 平均 OOD 诊断直接映射为逐 step 阈值
`soft=3.0, hard=4.5`。随后只使用一个未注册 seed、每个条件每个场景 30 step 的
instrumentation smoke 检查日志字段和极限退化。该 smoke 显示 narrow corridor
早期逐 step OOD 已超过 4.5，导致 correction support 恒为零。

这是统计尺度错误：episode 平均值不能直接作为逐 step 硬上界。该 smoke 不报告
success、final distance 或方法胜负，也不作为效果证据。

## 2. 允许的开发校准

为避免使用新的 L76 seed 调参，L76a 只复用已开放的 L75 开发 seed
`22200801–22200805`，并保持 L75 的冻结 BC 与原 SAC-LCB 轨迹作为配对基线。
L76a 只复跑 soft support 条件，阈值为：

\[
s_{\mathrm{soft}}=3.0,\qquad s_{\mathrm{hard}}=7.0.
\]

L76a 配置明确标记 `calibration_only: true`。它不是确认实验，不能进入论文最终
独立验证统计。

## 3. L76a 结果

相对冻结 BC：

| 方法 | success gain | success loss | 净变化 | 平均最终距离改善 |
|---|---:|---:|---:|---:|
| 原 SAC-LCB | 4 | 4 | 0 | +0.176 m |
| soft support SAC-LCB | 3 | 1 | +2 | +0.157 m |

soft support 按场景：

- single obstacle：`0 gain / 0 loss`；
- narrow corridor：`3 gain / 1 loss`，净 `+2`；
- U-trap：`0 gain / 0 loss`。

按独立模型块的净成功变化为 `+2、-1、+1`。平均 support confidence 为 `0.849`；
有效 correction gate alpha 相对原 SAC-LCB 降低 `22.2%`。

该结果只说明 `3.0/7.0` 不是恒开或恒关，并具备进入新 seed 开发验证的资格。

## 4. 唯一修订

正式 L76 将：

- 保持 `soft=3.0`；
- 把 `hard` 从 `4.5` 修订为 `7.0`；
- 保持三个 checkpoint、场景、全新 seed、随机调度和所有预注册 Gate 不变；
- 不再扫描任何 support threshold；
- 不使用 L76a 结果作为独立确认结果；
- 若 L76 Gate 失败，不打开 L76 封存 seed。

正式 L76 seed 为 `22200901–22200905`，在本修订冻结时尚未运行。
