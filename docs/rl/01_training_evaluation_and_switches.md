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

新格式 checkpoint 会锁定训练场景、SAC/encoder/prior、replay 规则、课程、
normalizer 和 BC-anchor fingerprint；续训只允许延长 `total_steps` 或调整
checkpoint 写盘间隔。模型、优化器、replay 和 NumPy/Torch RNG 都会恢复。
MuJoCo episode 内部状态不序列化，因此中途 checkpoint 会显式记为
`restart_interrupted_episode_v1`，恢复后从新 episode 开始，而不是声称逐 bit
等同于未中断运行。无 replay 或旧格式 checkpoint 默认拒绝续训。冻结 BC correction
checkpoint 还保存 base actor SHA-256；加载时校验失败会直接拒绝。

冻结 BC、有界 correction 训练：

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_frozen_bc_correction_l16.yaml \
  --seed 20260721 --steps 20000 \
  --initialize-actor-from \
    results/research_platform/rl/bc_l13_seed20260721_20260714/checkpoints/best.pt \
  --output-dir results/research_platform/rl/frozen_bc_correction_l16_seed20260721
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
| full actor / 冻结 BC correction | `rl.sac.policy_mode: direct/frozen_bc_correction` |
| correction 分维上限 | `rl.sac.correction_scale` |
| 固定 correction 幅度 | `rl.sac.correction_gate_alpha` |
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
`rl_gate_alpha`、`rl_ood_score`、`rl_critic_disagreement`、policy mode 和实际
correction magnitude，因此可以复核每次回退以及 RL 偏离冻结 BC 的幅度。

## 5. 当前 smoke 不能说明什么

60-step smoke 只能证明 MuJoCo、Replay、SAC 更新、checkpoint、自动加载与 ablation
入口都可执行。它没有足够交互数据，不能说明 RL 已学会避障，也不能用于论文图表。

## 6. Twin-critic correction confidence switches

Frozen-BC correction policy 提供三个兼容的推理模式：

```yaml
rl:
  gate:
    correction_advantage_gate_mode: none  # none | hard | lcb
    correction_advantage_critic_source: target
    correction_advantage_threshold: 0.0
    correction_advantage_uncertainty_multiplier: 2.0
```

- `none` 保留 learned correction，只记录诊断；
- `hard` 使用 L18/L19 的 raw conservative-advantage threshold；
- `lcb` 使用 L20 的 `mean(delta_q) - beta * half_disagreement >= 0`。

LCB 是显式 opt-in，beta 必须不小于 1。它是工程上的 critic disagreement
惩罚，不是校准概率或理论安全界。被拒绝的 correction 会在 MPPI decoder 之前
精确恢复 frozen BC latent action，不会绕过 MPPI 或 `scan_guard`。

## 7. LaserScan scene-complexity gate

L25 增加了只读取局部 LaserScan 的场景复杂度门控。它不访问 MuJoCo 障碍物真值，
并把正前方近障、双侧夹窄和近障光束密度三个分量的最大值映射为 RL 混合系数。

```yaml
rl:
  gate:
    mode: complexity
    complexity:
      front_half_angle_deg: 40.0
      side_outer_angle_deg: 125.0
      near_distance_m: 0.45
      far_distance_m: 1.30
      density_distance_m: 1.25
      density_full_fraction: 0.30
      soft_threshold: 0.35
      hard_threshold: 0.70
```

分数不超过 soft threshold 时，门控精确恢复 traditional MPPI prior；超过 hard threshold
时才允许完整 RL correction。扫描缺失时采用 fail-closed 行为，即 `alpha=0`，不会让 RL
因为传感器无效而接管。逐步轨迹会记录总分、三个分量、左右 clearance、scan 有效性和
实际 alpha，以便复核每一次启用与回退。

开发集消融入口：

```bash
.venv/bin/python experiments/rl/run_scene_complexity_gate_ablation.py \
  --config configs/rl/sac_mppi_scene_complexity_l25.yaml \
  --training-seed 20260721 \
  --episode-seeds 20281701,20281702,20281703,20281704,20281705,20281706,20281707,20281708,20281709,20281710 \
  --samples 200 \
  --output-dir results/research_platform/rl/l25_scene_complexity_seed20260721
```

预注册与结果分别见 `docs/rl/31_l25_scene_complexity_gate_prereg_2026-07-15.md` 和
`docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md`。
