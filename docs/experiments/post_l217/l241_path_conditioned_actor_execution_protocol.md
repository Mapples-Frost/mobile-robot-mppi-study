# L241 Path-Conditioned Residual Actor 执行协议

## 继承关系

本轮执行 `docs/rl/231_l222_safe_reference_path_preview_protocol_2026-07-20.md` 中尚未运行的 Gate C，不重新定义方法。训练配置固定为：

`configs/rl/residual_conditioned_path_preview_l222.yaml`

## 科学问题

在 residual/innovation context 之外加入因果、无障碍物真值泄漏的局部路径预览后，Residual-Conditioned Actor 是否能在隔离 validation seeds 上学到安全且具路径进度收益的 proposal，为后续 L234 Tracking development Gate 提供非平凡候选？

## 冻结训练设计

- 输入新增四个车体坐标系路径预览点，距离 `[0.4, 0.8, 1.2, 1.6] m`；
- residual/innovation/reliability context 保持不变；
- 初始化 Actor：L219 选中 checkpoint `seed20262193/step_000010000.pt`；
- 新增输入列零初始化，critics、optimizers、replay 和计数器重新初始化；
- 三个独立训练 seed：`20262221, 20262222, 20262223`；
- 各训练 30,000 environment steps；
- 三个候选的 validation seed bases 分别固定为 `20262821, 20262822, 20262823`，并与训练 seed、development seed `923301001` 隔离；
- 训练/验证场景沿用 L222 的六张 safe-reference MuJoCo 地图；
- seen/unseen physics 均覆盖，scene-balanced replay；
- 不使用任何 Tracking sealed seed。

## 工程 Gate

每个候选必须满足：

1. `updates.csv` 存在且 update records > 0；
2. 无 Traceback、NaN 或 Inf；
3. 30,000 steps、checkpoints、evaluations、config snapshot 和 provenance 完整；
4. path-preview observation dimension 与 checkpoint metadata 一致；
5. 训练与验证 seed 不和 L234 development/sealed seeds 重叠。

## 候选选择

只使用固定 validation episodes，按 L222/L219 已冻结的字典序选择：

1. 最少碰撞；
2. 最大成功数；
3. 最大平均路径完成度；
4. 最小平均 cross-track RMSE；
5. 最小平均目标距离；
6. 最大平均 return；
7. seed 和 step 只用于完全平局。

选中 checkpoint 后，先在 development seed `923301001` 上重新运行三场景短 Gate，并与 ICODE-MPPI、旧 Actor Full 作等预算比较。只有 development Gate 通过后才允许设计新的 sealed benchmark。

## 防止结果美化

- 三个训练候选和所有失败均保留；
- 不根据 L234 development outcome 回头选择训练 checkpoint；
- 不筛 seed、不删除失败、不接触 sealed seeds；
- 不把训练或 qualification 写成论文正式结论；
- CUDA 仅在运行时可用且不会改变数值协议时使用。当前 WSL PyTorch 为 CPU-only，MuJoCo 环境步仍以 CPU 为主，因此本轮记录为 CPU execution，不伪称 CUDA 加速。
