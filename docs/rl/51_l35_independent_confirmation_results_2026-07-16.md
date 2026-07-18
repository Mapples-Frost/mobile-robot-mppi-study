# L35 独立确认结果：L34 开发集增益未复现

日期：2026-07-16  
结论等级：独立 confirmation；预注册 Gate 未通过。

## 1. 结论摘要

L35 完成了 60 个 MuJoCo 闭环 episode，所有运行均使用全新的 episode seed、冻结的
`initial.pt`/`best.pt`、固定 `alpha=0.25` 和 nominal rollout dynamics。产物完整性通过，
但效能 Gate 未通过：

| 指标 | Initial | Best trained | 变化 |
|---|---:|---:|---:|
| Success rate | 43.3% | 40.0% | -3.3 pp |
| Collision rate | 50.0% | 50.0% | 0.0 pp |
| Mean final-distance improvement | — | 0.005 m | 远低于 0.10 m Gate |

分层 bootstrap 的 success-rate difference 为 `-0.033`，95% CI
`[-0.233, 0.167]`；final-distance improvement 为 `0.005 m`，95% CI
`[-0.052, 0.072]`。这些区间同时包含明显退化和有限改善，不支持稳定提升。

因此必须撤回“L34 已在新 seed 上稳定有效”的推断。L34 的 `4/18 → 10/18` 仍是有效的
开发集观察，但不能升级为独立复现证据。

## 2. 三个训练 seed

| Training seed | Success initial→best | Collision initial→best | Final-distance improvement |
|---:|---:|---:|---:|
| 20260731 | 4/10 → 4/10 | 5/10 → 5/10 | +0.028 m |
| 20260732 | 4/10 → 5/10 | 5/10 → 5/10 | +0.009 m |
| 20260733 | 5/10 → 3/10 | 5/10 → 5/10 | -0.021 m |

只有 `1/3` 个训练 seed 提升 success；pooled success gain/loss 为 `2/3`。碰撞没有任何
discordant pair，说明在这组场景中 checkpoint 改变没有改变碰撞结局。

## 3. 失效模式

两个 held-out motion path 出现完全场景分离：

- diagonal：30/30 次运行碰撞，success 为 0%；
- offset：30/30 次运行不碰撞，success 为 83.3%。

diagonal 动态障碍沿机器人直达目标的对角路径反向穿越。当前 MPPI 没有未来动态障碍
预测，只能使用逐帧 LaserScan、局部障碍层和 temporal safety 做反应式处理；机器人即使
减速，也可能被继续运动的障碍物撞上。offset 场景则没有产生相同的碰撞压力。因此本轮
同时暴露了两件事：

1. L34 checkpoint 对新 seed 的泛化不足；
2. 当前 confirmation 场景缺少处于可改善区间的中等难度动态交互，碰撞指标出现地板/
   天花板效应。

第二点不能用于否定第一点，也不能事后删除 diagonal 场景。预注册 Gate 仍按原计划判定
失败。它只决定下一轮如何形成新的、独立的开发假设。

## 4. RL 确实进入 MPPI

所有 trajectory 的 `prior_type=rl_sac`、`rl_gate_mode=fixed`，平均
`rl_gate_alpha=0.25`。`rl_subgoal_distance` 和 `rl_subgoal_bearing` 均为非零，并随
initial/best checkpoint 发生变化。因此失败不是“忘记打开 RL”。

trajectory 中 `rl_applied_correction_abs_* = 0` 是诊断字段语义造成的：这些字段描述
correction-policy 参数化；L34 使用的是 direct local-subgoal 参数化，应查看
`rl_subgoal_*`。后续报告不得再用前一组字段判断 L34 是否工作。

![L35 independent confirmation](../../results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1/fig_l35_independent_confirmation.png)

## 5. 科研决策

按照预注册规则，本轮不进入 Traditional/RL × Nominal/ICODE 的正式 2×2 实验，也不在
L35 seed 上搜索新的 alpha、奖励或 checkpoint。

下一阶段应先使用全新的 development seeds，以传统 MPPI 为唯一校准器，建立一个具有
明确可行路径且 baseline success/collision 不处于 0% 或 100% 的动态交互场景族。场景
难度校准不得读取 RL 结果。只有 benchmark 可辨识性通过后，才提出新的 RL 训练假设并
重新预注册独立确认。

## 6. 复现入口

```bash
python3 experiments/rl/run_l34_independent_confirmation.py \
  --config configs/rl/l34_independent_confirmation_l35.yaml \
  --output-dir results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1 \
  --resume

python3 experiments/rl/summarize_l34_independent_confirmation.py \
  --config configs/rl/l34_independent_confirmation_l35.yaml \
  --input-dir results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1

python3 experiments/rl/analyze_l35_confirmation_failures.py \
  --input-dir results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1
```

