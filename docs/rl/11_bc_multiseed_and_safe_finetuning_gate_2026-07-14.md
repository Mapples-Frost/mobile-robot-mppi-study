# L13--L15：BC 多训练种子复现与 SAC 安全微调审计

日期：2026-07-14  
结论：接受 BC 作为受控条件下的稳定启动底座；否决当前 full-actor SAC
微调为“安全改进”；暂不与 ICODE/Memory 组合。

## 1. 数据和闭环条件

- scene：`u_trap_long_board`；
- teacher：privileged offline polyline，仅在采集阶段可见；
- student input：goal-relative Odom、LaserScan、twist、previous action、safety state；
- `include_absolute_pose=false`；
- MPPI samples `K=100`，horizon 与 L11/L12 一致；
- prediction：nominal；Memory、ICODE、intrinsic bonus 全部关闭；
- controlled localization：ground-truth pose/twist；
- held-out evaluation seeds：101--110；
- near-goal fallback：0.35 m 完全回退，0.80 m 完全采用 learned prior；
- scan_guard、local obstacle layer、safety arbitration 始终开启。

数据集：

```text
requested teacher episodes   15
successful / audit-only      14 / 1
train episodes / samples      8 / 2059
validation episodes / samples 3 / 737
test episodes / samples       3 / 891
failed episodes in shards      0
```

## 2. BC-only 三训练种子结果

| BC seed | offline test action RMSE | held-out success | collision | mean final distance (m) |
|---:|---:|---:|---:|---:|
| 20260721 | 0.0923 | 9/10 | 0/10 | 0.364 |
| 20260722 | 0.0767 | 10/10 | 0/10 | 0.296 |
| 20260723 | 0.0922 | 10/10 | 0/10 | 0.296 |
| pooled | — | **29/30 (96.7%)** | **0/30** | seed mean 0.319 |

训练种子 success-rate 样本标准差为 0.0577。相同 held-out seeds 的
SAC-uniform 三训练种子为 16/30（53.3%），标准差 0.4509；BC 的 safety
interventions seed mean 为 21.5，SAC-uniform 为 29.4。

这不是同等信息预算下“BC 优于 RL”的一般性结论：BC 使用了 privileged
teacher route，而 SAC-uniform 依靠自探索。它支持的窄结论是：当前困难场景中，
成功行为蒸馏远比从零探索稳定，适合作为后续 RL 的启动策略。

聚合产物：

```text
results/research_platform/rl/bc_vs_sac_multitraining_seed_20260714/
```

## 3. 无锚 SAC 微调：明确失败

L13 从 BC actor 初始化，但 critic、target critic、alpha、optimizer、replay、
global step 全部重新初始化；normalizer 冻结。结果：

| SAC step | fixed validation success | mean goal distance (m) |
|---:|---:|---:|
| step 0（事后同种子检查） | 4/5 | 0.425 |
| 5k | 0/5 | 3.862 |
| 10k | 0/5 | 2.539 |
| 15k | 0/5 | 4.040 |
| 20k | 0/5 | 4.431 |
| 25k--40k | 0/5 | 4.12--9.96 |

训练 episode 给出了进一步诊断：前 2k、actor 尚未更新时，easy suffix reset
为 22/22 成功；20k 后 original-start episode 共 41 次，0 次成功。该现象与
actor 开始接受 SAC 更新后的 catastrophic policy forgetting 一致；“critic 尚未
成熟”“训练分布偏向路线后半段”是待分离的候选解释，不是本轮已经证明的单一因果。

## 4. L14：BC anchor 只能减慢退化

L14 在 SAC actor loss 中加入 train-split-only 辅助项：

\[
\mathcal L_{\mathrm{actor}}
=
\mathcal L_{\mathrm{SAC}}
+5\mathcal L_{\mathrm{BC}}
+10^{-3}\mathcal L_{\log\sigma}.
\]

step 0 为 5/5；5k 为 2/5；10k 为 0/5。固定系数 BC anchor 减慢了遗忘，
没有建立 safe policy improvement。不能通过只报告 best=step 0 来声称 SAC 有效。

## 5. L15：不中断 actor-update onset 诊断

L15 前 10k 只更新 critic，actor 和 entropy temperature 逐参数冻结。早期
10k→20k 结果由旧格式 checkpoint 恢复得到，进程重启与开放 actor 更新同时发生，
且旧 checkpoint 未保存 Torch RNG/MuJoCo episode 状态，因此该组 15k/20k 数字
只能保留作审计，不能用于因果判断：

| step | actor update | success | mean distance (m) |
|---:|---:|---:|---:|
| 0 | off | 5/5 | 0.2950 |
| 5k | off | 5/5 | 0.2950 |
| 10k | off | 5/5 | 0.2950 |
| 15k | on + BC anchor | 2/5 | 0.6339 |
| 20k | on + BC anchor | 2/5 | 0.8241 |

为消除该混杂，补跑了三个 BC/SAC 训练种子从 step 0 到 20k **不中断** 的实验。
训练期间只使用固定 validation seeds；完成训练后，再用未参与 checkpoint 选择的
held-out seeds 101--110 复核 step 0 与 step 20k。固定 5-seed validation 曲线为：

| training seed | step 0 | step 5k | step 10k | step 15k | step 20k |
|---:|---:|---:|---:|---:|---:|
| 20260721 | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 |
| 20260722 | 5/5 | 5/5 | 5/5 | 3/5 | 5/5 |
| 20260723 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |

step 0、5k 和 10k 的 actor 完全相同；其中 step 5k/10k 不重复做 held-out，是因为
critic-only burn-in 期间 actor 逐参数冻结，数值检查也确认 step 0 与 step 10k 的
actor 参数最大差值为 0。独立 held-out 结果如下：

| training seed | BC step 0 success | SAC step 20k success | collision 0→20k | mean final distance 0→20k (m) |
|---:|---:|---:|---:|---:|
| 20260721 | 9/10 | 9/10 | 0/10 → 0/10 | 0.3636 → 0.3836 |
| 20260722 | 10/10 | 10/10 | 0/10 → 0/10 | 0.2956 → 0.2932 |
| 20260723 | 10/10 | 10/10 | 0/10 → 0/10 | 0.2963 → 0.2956 |
| pooled / seed mean | **29/30** | **29/30** | **0/30 → 0/30** | **0.3185 → 0.3241** |

三训练种子聚合后，20k SAC 没有提高成功率；平均终点距离略高，平均最小净空从
0.3521 m 降至 0.3366 m，平均 control jerk 从 0.1594 升至 0.1620，平均 planner
计算时间从 4.93 ms 升至 5.43 ms。只有 safety interventions 从 21.5 降至 19.87，
但在成功率、净空和终点误差没有同步改善时，不能把单项下降解释为整体控制提升。
当前只有三个训练种子，因此这些数字是受控工程 gate，不是充分统计功效的论文结论。

中间 checkpoint 还揭示了非单调风险：seed 20260721 的 step 15k 在固定验证上仍为
4/5，但在独立 10-seed 检查上为 0/10、碰撞 0/10、平均终点距离 1.2650 m；到 20k
又恢复为 9/10。这不是“训练越久越好”，而是 full actor 更新可先破坏、再偶然恢复
已有能力。其他两个训练种子的 15k 退化程度不同，说明风险对随机种子敏感。现有实验
仍不能把该现象唯一归因于 critic 估计、SAC 目标、BC anchor 冲突或状态分布偏移。

连续运行与聚合产物：

```text
results/research_platform/rl/bc_anchor_burnin_l15_seed20260721_20k_continuous_20260714_v2/
results/research_platform/rl/bc_anchor_burnin_l15_seed20260722_20k_continuous_20260714_v2/
results/research_platform/rl/bc_anchor_burnin_l15_seed20260723_20k_continuous_20260714_v2/
results/research_platform/rl/bc_anchor_burnin_l15_continuous_15k_heldout101_110_20260714_v2/
results/research_platform/rl/bc_vs_l15_continuous_multitraining_seed_20260714_v2/
```

代码现在把 step 0 纳入 checkpoint selection，因此不会因为只看 5k 以后节点而漏掉
已验证的初始化策略。但 model selection 仍只能依据预注册 validation seeds；不能在
看到 held-out 101--110 后反向挑 checkpoint。三训练种子的 20k 结果只回到与 step 0
相同的 29/30，且若干二级指标变差，因此仍不支持“RL 改善了 BC”。

## 6. 配置层错误与作废结果

早期一次 BC-only 调用没有显式传 `--scene-config`，实际运行的是训练配置继承的
`clean_single_obstacle`，得到 0/10。该结果与 U-trap 问题无关，必须作废。

评估 CLI 现已提供显式双配置：

```bash
--config configs/rl/sac_mppi_utrap_bc_l13.yaml
--scene-config configs/research/mujoco_u_trap_long_board.yaml
```

输出 `summary.json` 也强制记录 `scene`。以后不能再根据输出目录名称猜测场景。

## 7. 定位边界

选取 BC seed 20260722，在相同 U-trap seeds 101--110 改为 wheel odometry：

```text
success 2/10
collision 0/10
mean final goal distance 0.673 m
```

这说明 controlled-localization 结论不能直接迁移到当前 wheel-odometry 链。安全链
仍阻止了碰撞，但定位/观测分布鲁棒性必须作为独立 gate 处理。

## 8. 科研决策

1. 接受 BC actor 作为当前 U-trap、controlled-localization 条件下的受控启动基线；
2. 保留 full-actor SAC、BC anchor、critic burn-in 作为可复现实验消融；
3. 否决“当前 SAC 微调改善了 BC”的表述；
4. 不调大一个系数后挑最好结果，不在 held-out seeds 上做超参数搜索；
5. 下一方法门应冻结已验证的 BC base，只让 RL 学习有界 policy-correction head，
   并由 OOD/场景复杂度 gate 控制修正幅度；zero correction 必须严格退化为 BC；
6. policy correction 与 ICODE dynamics residual 必须命名和代码层分离；
7. 在该门通过三训练种子前，不加入 ICODE、Memory 或动态障碍联合实验。

该决策保持导师建议的主线：简单/已知区域保留传统或已验证行为，RL 只承担复杂、
OOD 条件下的探索修正；不把多个未经证实的模块同时写进 story。

后续状态：上述 frozen-BC correction 已在 L16 实现并完成三训练种子 Gate。它避免了
full-actor 的严重中间崩溃，但未超过 BC；详见
`docs/rl/12_frozen_bc_bounded_correction_gate_2026-07-14.md`。

## 9. 可复现命令

以下命令统一使用仓库虚拟环境。数据采集与 BC 训练见文档 10；这里给出三种
BC→SAC 诊断的完整入口。`DATASET` 与 `BC_CKPT` 必须来自同一 demonstration
manifest，代码会比较 fingerprint，不能混用。

```bash
DATASET=results/research_platform/rl/bc_datasets/utrap_l13
BC_CKPT=results/research_platform/rl/bc_l13_seed20260721/checkpoints/best.pt

# L13：无 BC anchor，40k steps
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_bc_l13.yaml \
  --seed 20260721 --steps 40000 \
  --initialize-actor-from "$BC_CKPT" \
  --output-dir results/research_platform/rl/bc_to_sac_l13_seed20260721_40k

# L14：固定系数、train-split-only BC anchor，10k steps
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_bc_anchor_l14.yaml \
  --seed 20260721 --steps 10000 \
  --bc-anchor-dataset "$DATASET" \
  --initialize-actor-from "$BC_CKPT" \
  --output-dir results/research_platform/rl/bc_anchor_l14_seed20260721_10k

# L15：不中断跑满 20k；前 10k critic-only burn-in
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_bc_anchor_burnin_l15.yaml \
  --seed 20260721 --steps 20000 \
  --bc-anchor-dataset "$DATASET" \
  --initialize-actor-from "$BC_CKPT" \
  --output-dir results/research_platform/rl/bc_anchor_burnin_l15_seed20260721_20k_continuous

# resume 仅用于验证恢复能力；不可替代上面的不中断因果对照
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_bc_anchor_burnin_l15.yaml \
  --seed 20260721 --steps 25000 \
  --bc-anchor-dataset "$DATASET" \
  --resume results/research_platform/rl/bc_anchor_burnin_l15_seed20260721_20k_continuous/checkpoints/latest.pt \
  --output-dir results/research_platform/rl/bc_anchor_burnin_l15_seed20260721_20k_continuous
```

SAC checkpoint 的中途续训会恢复模型、优化器、replay、normalizer、NumPy/Torch
RNG 和实验 contract。MuJoCo 的 episode 内部状态不序列化；若 checkpoint 位于
episode 中间，续训会显式记录一次 `episode_restart_applied=true` 并从新 episode
开始，不能把它描述成与不中断运行逐 bit 相同。旧格式 checkpoint 默认拒绝续训；
只有审计历史结果时才可显式使用 `--allow-legacy-resume`。
