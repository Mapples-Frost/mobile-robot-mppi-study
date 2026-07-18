# L33–L34 动态障碍训练结果：RL 应作为有界修正，而不是完全接管 MPPI 先验

日期：2026-07-16  
结论等级：多训练 seed development 结果；尚未使用新的 confirmation episode seeds。

## 1. 研究问题

本阶段检验的不是 ICODE 残差模型本身，而是 RL sampling prior 在动态障碍下的
接入方式。两个实验均使用三帧 LaserScan、相同 SAC 网络、相同随机化动态障碍、
相同 MuJoCo 物理域和相同安全链：

- **L33**：RL 输出完全替换传统 MPPI sampling prior，`alpha = 1.0`；
- **L34**：RL 只对传统 prior 做 25% 的有界修正，`alpha = 0.25`。

预测动力学在本阶段固定为 nominal，以隔离 RL 接入方式；因此本结果不能单独声称
“RL+ICODE 已在新动态任务上完成最终验证”。ICODE 的残差建模证据和 L32 跨层实验
仍保留，下一阶段才将 L34 checkpoint 与 nominal/ICODE 组成正交因子实验。

## 2. 实验设计

### 2.1 观察与训练域

- observation：三帧 36-sector LaserScan，加目标、速度、上一动作和安全状态；
- 训练运动族：anchor、reverse、fast、large-early；
- 每个 episode 随机化运动相位、周期缩放和端点位置；
- 训练物理域：nominal、high-mass、low-friction、long-delay；
- 验证运动路径：diagonal 与 offset，训练中未出现；
- 验证物理域：combined-unseen；
- 每个检查点：2 条路径 × 3 个固定 episode seed，共 6 个闭环 episode；
- 检查点：step 0、5k、10k、15k、20k、25k、30k。

planner 只接收 LaserScan → local obstacle layer 的结果，不读取 MuJoCo 障碍物位置、
运动速度或未来轨迹。temporal scan guard、原 scan guard 和最终控制仲裁始终启用。

### 2.2 多训练 seed

L34 在开发 seed `20260731` 通过事前 Gate 后，按预注册扩展到：

- `20260731`；
- `20260732`；
- `20260733`。

每个训练 seed 从同一 L6 三帧 actor checkpoint 做 actor-only 初始化；critic、target
critic、optimizer、replay、entropy 状态和计数器均重新初始化。

## 3. L33：完全 learned prior 失败

L33 的 step 0 为 `1/6` success、`1/6` collision；30k 训练后的所有检查点均为
`0/6` success。训练过程没有产生任何完整成功 episode，最佳训练后检查点虽然为
`0/6` collision，但平均终点距离达到 `3.701 m`，表现为规避碰撞却远离目标。

因此，失败不是“网络还不够大”的直接证据，而是当前难度下完全接管 prior 破坏了
传统 MPPI 已有结构，SAC 缺乏可用于保持任务能力的成功闭环信号。

## 4. L34：有界修正的开发 seed

训练 seed `20260731` 的最佳检查点位于 30k：

| 指标 | step 0 | best trained | 变化 |
|---|---:|---:|---:|
| success | 1/6 | 4/6 | +3 |
| collision | 4/6 | 2/6 | -2 |
| 平均终点距离 | 1.628 m | 0.927 m | -0.701 m |

6 个逐 episode 配对中有 3 个 success gain、0 个 success loss、2 个 collision
improvement、0 个 collision regression。训练本身产生 54 个成功 episode，说明
replay 中不再只有失败或超时轨迹。

## 5. 三训练 seed 复现

| training seed | best step | success 初始→最佳 | collision 初始→最佳 | 终点距离初始→最佳 | success gain/loss |
|---:|---:|---:|---:|---:|---:|
| 20260731 | 30k | 1/6 → 4/6 | 4/6 → 2/6 | 1.628 → 0.927 m | 3 / 0 |
| 20260732 | 30k | 2/6 → 3/6 | 3/6 → 2/6 | 1.349 → 1.014 m | 1 / 0 |
| 20260733 | 15k | 1/6 → 3/6 | 5/6 → 3/6 | 1.902 → 1.299 m | 2 / 0 |

汇总 18 个逐 episode 配对：

- success：`4/18 → 10/18`，净增加 6 个；
- collision：`12/18 → 7/18`，净减少 5 个；
- success gains / losses：`6 / 0`；
- collision regressions / improvements：`0 / 5`；
- 三个训练 seed 的平均终点距离改善：`0.547 m`；
- 3/3 训练 seed 均为正 success 净增益；
- 3/3 训练 seed 的 collision 均不高于各自 step 0。

多训练 seed 预注册 Gate 因此通过。

![L34 multiseed result](../../results/research_platform/rl/l34_bounded_dynamic_multiseed_development_20260716_v1/fig_l34_dynamic_multiseed.png)

L33 与 L34 的完整训练曲线如下。图中保留所有中间退化点，没有只截取最优点。

![L33 versus L34](../../results/research_platform/rl/dynamic_history_bounded_l34_seed20260731_30k_20260716_v1/fig_l33_l34_dynamic_training.png)

## 6. 产物与完整性审计

三个 L34 run 均满足：

- 预期/实际 validation rows：`42/42`；
- 重复 validation key：0；
- NaN/Inf 行：0；
- `config_snapshot.json` 存在；
- `initial.pt`、`best.pt`、`latest.pt` 存在；
- 训练、验证场景和物理域完整展开到 snapshot；
- 每个训练 seed 的 episode、update、validation CSV 均保留。

seed `20260731` 曾因交互中断从 15k checkpoint 恢复。trainer 明确记录
`interrupted_episodes = 1`，重新开始未序列化的半截 MuJoCo episode；已完成 episode、
replay、模型、optimizer 和 RNG 状态从 checkpoint 恢复，没有把中断片段伪装为连续轨迹。

## 7. 额外发现与修复

多 seed 审计发现，旧执行顺序会先把第一个训练 reset observation 加入 online
normalizer，再做 step-0 验证。这会使不同训练 seed 的 step-0 reference 产生一个很小
但可能被闭环放大的差异。本轮已经按每个训练 seed 相对自己的 step 0 做严格配对，
没有跨 seed 强行共用 reference。

代码现已修复：如果 actor-only checkpoint 已带有 fitted normalizer，则先保存并评估
完全相同的 warm-start policy，再允许训练数据更新 normalizer。该修复只影响未来运行，
没有事后改写本轮 CSV。

同时，标准 checkpoint evaluator 已补齐：

- 显式 MuJoCo physics-domain 选择；
- 训练时 temporal perception overlay；
- `complexity_confidence` gate 模式；
- evaluation config snapshot、episode seeds 和命令行覆盖记录。

## 8. 可以和不可以得出的结论

当前证据支持：

> 在本项目的随机化动态障碍开发域中，RL sampling prior 作为传统 MPPI 的有界修正，
> 比完全替换传统 prior 更容易产生成功经验，并在三个独立训练 seed 上同时改善成功、
> 碰撞和终点距离。

当前证据仍不支持：

- 不能宣称已完成论文最终 test；固定 validation cells 同时参与 checkpoint 选择；
- 不能宣称动态障碍下 collision 已经足够低，最佳模型仍有 `7/18` collision；
- 不能把 18 个 episode 当作最终统计功效；
- 不能声称 improvement 来自 ICODE，本阶段 prediction mode 固定为 nominal；
- 不能声称有稳定性、收敛性或安全理论保证。

## 9. 下一项冻结实验

在新的 episode seeds 上，对每个训练 seed 的 `initial.pt` 与 `best.pt` 做顺序、独立的
闭环确认；推理固定使用与训练一致的 `fixed alpha = 0.25`。通过后再做 2×2 正交实验：

```text
traditional prior × nominal dynamics
traditional prior × ICODE dynamics
bounded RL prior × nominal dynamics
bounded RL prior × ICODE dynamics
```

该顺序可以分别回答“有界 RL 是否有效”“ICODE 是否有效”“二者是否互补”，避免再次
把策略、残差动力学和安全机制的作用混在同一个平均数里。

## 10. 复现入口

```bash
python3 experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_dynamic_history_bounded_l34.yaml \
  --output-dir results/research_platform/rl/dynamic_history_bounded_l34_seed20260731_30k_20260716_v1 \
  --initialize-actor-from results/research_platform/rl/subgoal_history_l6_20260713/checkpoints/best.pt \
  --seed 20260731

python3 experiments/rl/summarize_dynamic_history_multiseed.py \
  --run-dir results/research_platform/rl/dynamic_history_bounded_l34_seed20260731_30k_20260716_v1 \
  --run-dir results/research_platform/rl/dynamic_history_bounded_l34_seed20260732_30k_20260716_v1 \
  --run-dir results/research_platform/rl/dynamic_history_bounded_l34_seed20260733_30k_20260716_v1 \
  --output-dir results/research_platform/rl/l34_bounded_dynamic_multiseed_development_20260716_v1
```

