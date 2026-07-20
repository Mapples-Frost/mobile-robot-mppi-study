# L243 Path/Residual-Conditioned Actor 的 BC Anchor 训练报告

日期：2026-07-21
性质：development training / validation-only selection，不是论文确认性结果

## 1. 工程与数据完整性

三个预注册候选均完成 30,000 个 environment steps：

| train seed | validation base | update records | validation rows | checkpoints | NaN/Inf |
|---:|---:|---:|---:|---:|---:|
| 20262331 | 20262931 | 29,001 | 210 | 9 | 0 |
| 20262332 | 20262932 | 29,001 | 210 | 9 | 0 |
| 20262333 | 20262933 | 29,001 | 210 | 9 | 0 |

共同来源约束：

- 训练 Git SHA：`263b58bbe14a797656a8433be453726cf255eafa`；
- source Actor：`l219_expanded_actor_seed20262193_30k_v2/checkpoints/step_000010000.pt`，SHA256 为
  `bf26a67ebac313930d63760db931e5d50704cdd9181923afd7f66d16159356b4`；
- BC 数据集 manifest fingerprint：
  `918712d513763fd272db25d7a428c6fcf563e6ca1b98c0f089cb46f52cc91fbb`；
- student observation/action contract：69D / 2D `direct_control`，不包含 absolute pose；
- observation normalizer 冻结；Actor、critic 与 SAC 更新保持启用；
- 当前 WSL PyTorch 为 CPU-only，本轮未使用 CUDA。

完整教师数据包含六张 L222 safe-reference 地图、90 个 MuJoCo 回合。60 个零碰撞、无
boundary violation 的成功回合进入 train/validation/test shards，共 41,932 个样本；30 个失败
回合完整保留在 privileged audit 中，没有删失败或筛 seed。

## 2. 冻结 checkpoint 选择

选择脚本：`experiments/rl/select_actor_checkpoint_lexicographic.py`。每个训练 seed 包含
step 0、5k、10k、15k、20k、25k、30k 七个 validation block，每个 block 为 30 个固定
validation 回合，共 21 个候选。排序规则与 L241 完全相同：

1. 最少碰撞；
2. 最大成功数；
3. 最大平均路径完成度；
4. 最小平均 cross-track RMSE；
5. 最小平均目标距离；
6. 最大平均 return；
7. train seed 与 global step 只用于完全平局。

冻结选中结果：

| train seed | step | collision | success | completion | cross-track RMSE | goal distance | mean return |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **20262332** | **30,000** | **0/30** | **6/30** | 0.3843 | 0.2901 | 3.6355 | -1075.13 |

选中 checkpoint：

```text
results/research_platform/rl/l243_path_preview_bc_anchor_seed20262332_30k_v1/checkpoints/step_000030000.pt
SHA256: e8e446cbe9a9520a4053012a28995e63238a85dae4f0462e5574e2a0c0746c3c
```

该结果满足 L243 训练 Gate：没有退回 step 0，validation 碰撞仍为 0，成功数由 L241
selected initial 的 2/30 提高到 6/30。它是进入闭环 development Gate 的资格证据，不能直接
写成 Tracking 控制效果或论文正式结论。

## 3. 可复现工件

- 三个训练目录：`results/research_platform/rl/l243_path_preview_bc_anchor_seed*_30k_v1/`；
- 冻结排序：`results/research_platform/rl/l243_path_preview_bc_anchor_selection_v1/candidate_ranking.csv`；
- 选择记录：`results/research_platform/rl/l243_path_preview_bc_anchor_selection_v1/selection.json`；
- 教师数据：`results/research_platform/rl/l243_direct_control_teacher_dataset_v1/`；
- 训练协议：`docs/experiments/post_l217/l243_path_actor_bc_anchor_protocol.md`。

下一步只允许按 L244 预注册协议，在 development seed `923301001` 上进行闭环比较。
