# L241 Path-Conditioned Residual Actor 训练与冻结选择报告

日期：2026-07-21  
性质：development / qualification，不是论文确认性结果

## 1. 执行结论

三个预注册候选均完成 30,000 environment steps，工程 Gate 通过：

| train seed | validation base | update records | validation rows | checkpoints | 异常 |
|---:|---:|---:|---:|---:|---|
| 20262221 | 20262821 | 29,001 | 210 | 9 | 无 |
| 20262222 | 20262822 | 29,001 | 210 | 9 | 无 |
| 20262223 | 20262823 | 29,001 | 210 | 9 | 无 |

三组训练均记录 Git SHA
`0ab321ae348b851c89f156158ae98cc9d9d2f00f`，观测维度为 69，包含四个
车体坐标系 path-preview 点，训练与验证 seed 未使用 L234 development seed。
日志未发现 Traceback、Exception、NaN 或 Inf。当前 WSL PyTorch 是 CPU-only，
本轮没有使用 CUDA。

## 2. 冻结选择

使用 `experiments/rl/select_actor_checkpoint_lexicographic.py` 对 3 个训练 seed、
每个 7 个评估时刻、每个时刻 30 个固定验证回合进行选择，共 21 个候选块。
排序严格遵循 L222/L241 预注册顺序：

1. 最少碰撞；
2. 最大成功数；
3. 最大平均路径完成度；
4. 最小平均 cross-track RMSE；
5. 最小平均目标距离；
6. 最大平均 return；
7. train seed 与 global step 仅用于完全平局。

冻结选择结果为：

| train seed | step | collision | success | completion | cross-track RMSE | goal distance | mean return |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **20262222** | **0** | **0/30** | **2/30** | 0.5858 | 1.6712 | 4.5599 | -20,711.66 |

选中 checkpoint：

```text
results/research_platform/rl/l241_path_preview_actor_seed20262222_30k_v1/checkpoints/initial.pt
SHA256: 150a2d7ba1e4ab125fcb94e614810acb33e86f549b3eaa310e6120cd02440982
```

## 3. 科学解释

这是一个必须保留的负向训练结论。虽然三个 SAC 训练过程都真实发生且没有工程
异常，但冻结的安全优先验证规则选择了 step 0，而不是任何训练后的 checkpoint。
step 0 是由 L219 Actor 扩展到 69 维输入后的初始策略；新增 path-preview 输入列
为零初始化。因此，本轮数据**没有证明训练后的 Actor 学会了可泛化的路径条件
引导**。

选择 step 0 不是人为挑选。seed 20262222 的 step 0 在固定验证集上取得 2/30
成功，所有训练后候选最多只有 1/30 成功；在“碰撞优先、成功第二”的冻结规则下，
后续较低 RMSE 或较高 return 不能越过成功数排序。

这个结果不否定 ICODE、MPPI 或论文核心耦合方向；它只否定了当前这次 30k
path-conditioned SAC 训练已经优于初始化策略的说法。接下来仍按协议用选中模型
做 L242 development Gate，检查其在 L234 Tracking 地图上的真实闭环表现，并把
L239 旧 Actor Full 作为同预算、同 seed、同地图的不可变历史对照。

## 4. 可复现工件

- 原始训练目录：`results/research_platform/rl/l241_path_preview_actor_seed*_30k_v1/`
- 冻结排序：`results/research_platform/rl/l241_path_preview_actor_selection_v1/candidate_ranking.csv`
- 选择记录：`results/research_platform/rl/l241_path_preview_actor_selection_v1/selection.json`
- 选择脚本：`experiments/rl/select_actor_checkpoint_lexicographic.py`
- 训练协议：`docs/experiments/post_l217/l241_path_conditioned_actor_execution_protocol.md`

全部负向候选均保留；未读取 sealed seeds，未依据 L234 outcome 回头改选 checkpoint。
