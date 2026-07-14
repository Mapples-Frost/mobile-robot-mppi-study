# RL 训练、评估与实验开关

## 1. 最短命令链

先运行真实 MuJoCo smoke：

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_prior_smoke.yaml \
  --output-dir results/research_platform/rl/smoke --smoke
```

正式训练：

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_prior.yaml \
  --output-dir results/research_platform/rl/formal_v1
```

断点恢复（`total_steps` 是目标总步数，而不是追加步数）：

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_prior.yaml \
  --output-dir results/research_platform/rl/formal_v1 \
  --resume results/research_platform/rl/formal_v1/checkpoints/latest.pt
```

确定性多 seed 评估：

```bash
.venv/bin/python experiments/rl/evaluate_rl_sampling_prior.py \
  --config configs/research/mujoco_u_trap_long_board.yaml \
  --checkpoint results/research_platform/rl/formal_v1/checkpoints/best.pt \
  --output-dir results/research_platform/rl/eval_u_trap \
  --seeds 11,12,13,14,15 --gate-mode none
```

完整方法消融：

```bash
.venv/bin/python experiments/rl/run_rl_mppi_ablation.py \
  --configs configs/research/mujoco_strong_mppi_baseline.yaml,configs/research/mujoco_lab_complex.yaml,configs/research/mujoco_narrow_corridor.yaml,configs/research/mujoco_u_trap_long_board.yaml \
  --rl-checkpoint results/research_platform/rl/formal_v1/checkpoints/best.pt \
  --icode-checkpoint results/research_platform/checkpoints/icode_v1/best.pt \
  --output-dir results/research_platform/rl/ablation_v1 \
  --seeds 11,12,13,14,15 --samples 50,100,200,400
```

## 2. 必须保留的实验开关

| 研究问题 | 配置开关 |
|---|---|
| 不使用 RL | `planner.sampling_prior: goal_warm_start`, `rl.enabled: false` |
| 使用 RL | `planner.sampling_prior: rl`, `rl.enabled: true` |
| RL 完全接管 prior | `rl.gate.mode: none` |
| 固定混合 | `rl.gate.mode: fixed`, `fixed_alpha` |
| OOD 回退 | `rl.gate.mode: ood` |
| 绝对/增量 prior | `rl.prior.mode: absolute/delta` |
| RL 同时调采样方差 | `rl.prior.learn_covariance: true` |
| ICODE 开关 | `planner.prediction_mode: nominal/icode_residual` |
| Memory 开关 | `memory.enable`；主 RL/ICODE 消融默认 false |
| 采样效率 | `planner.num_samples: 50/100/200/400` |
| 训练物理域 | `training.physics_domain_config` 与 domain roles |
| 安全后立即回退 | `gate.fallback_after_safety_override` |

## 3. 正式实验最小矩阵

第一层证明传统方法边界：

1. MPPI；
2. ICODE-MPPI；
3. RL-MPPI；
4. gated RL-MPPI；
5. ICODE + RL-MPPI；
6. ICODE + gated RL-MPPI。

第二层结构消融：`absolute/delta`、mean-only/mean+covariance、固定 alpha/OOD gate、
有无 Critic disagreement、不同 knot 数量。

第三层泛化：seen/unseen physics、simple/complex/U-trap/narrow corridor、静态/动态障碍
（动态障碍属于后续场景实现与评价轴，不应被写成当前 RL 算法创新）。

每个正式配置建议至少 10 seeds；早期 smoke 使用 1 seed，开发 gate 使用 5 seeds。

## 4. 输出与判读

训练输出：

- `episodes.csv`：episode return、成功、碰撞、最终距离；
- `updates.csv`：Actor/Critic loss、熵、alpha、Q、梯度范数；
- `validation_episodes.csv`：固定验证 seed 结果；
- `best.pt/latest.pt`：完整可恢复 checkpoint；
- `run_metadata.json`：Git SHA、维度、encoder 和 parameterization。

控制评估另外输出 resolved config、provenance、trajectory CSV、metrics JSON。轨迹记录
`rl_gate_alpha`、`rl_ood_score` 和 `rl_critic_disagreement`，因此可以复核每次回退发生在何处。

## 5. 当前 smoke 不能说明什么

60-step smoke 只能证明 MuJoCo、Replay、SAC 更新、checkpoint、自动加载与 ablation
入口都可执行。它没有足够交互数据，不能说明 RL 已学会避障，也不能用于论文图表。
