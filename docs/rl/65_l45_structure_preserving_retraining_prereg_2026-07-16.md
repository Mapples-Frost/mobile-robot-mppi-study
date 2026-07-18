# L45 structure-preserving ICODE 重训练预注册

日期：2026-07-16  
状态：训练前冻结。

## 结构约束

沿用 L40 的 MPPI on-policy command dataset、H36、RK4、multi-step state loss 和所有超参数，只改变一个预先声明的结构：

```text
residual_output_mask = [0, 0, 0, 1, 1]
```

网络仍输出 control-affine residual，但仅 `v_dot` 与 `omega_dot` 通道参与 residual derivative loss 并进入 combined dynamics；
`x_dot=v cos(theta)`、`y_dot=v sin(theta)`、`theta_dot=omega` 始终由 nominal model 提供。

## 独立训练

- training seeds：20260901、20260902、20260903；
- dataset split 与 L40 相同，归一化只由 train split 计算；
- best checkpoint 仍按 validation H36 multi-step rollout RMSE 选择；
- test 与 unseen split 报告 H=1/5/10/20/36；
- 不读取 L44 正式开发 seed，也不打开 L44/L45 confirmation seed。

## 离线准入

三个 checkpoint 在 test 与 unseen 的 H36 rollout RMSE 均不得比 nominal 更差；至少 2/3 checkpoint 的 H36 position RMSE
和 heading RMSE 同时改善。通过后，L44 配置中的 checkpoint 才替换为 L45 模型并以新版本号运行；否则停止闭环实验。

