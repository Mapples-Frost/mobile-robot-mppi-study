# L16：冻结 BC 的有界 SAC policy-correction Gate

日期：2026-07-14

结论：接受该结构为比 full-actor SAC 更安全、可审计的研究接口；否决“L16 已经改善
BC”的表述。当前 Gate 停在 seen U-trap，不进入 OOD、动态障碍、ICODE 或 Memory 联合实验。

## 1. 研究问题与可证伪条件

L13--L15 表明，BC 在受控 U-trap 上达到 29/30，但允许 SAC 修改完整 actor 会发生
catastrophic forgetting 和非单调恢复。L16 只检验一个变化：冻结 BC，让 SAC 学习
一个幅度受限的 policy correction：

\[
\mathbf a_{\mathrm{prior}}
=
\mathbf a_{\mathrm{BC}}
+
\alpha\,\Delta\mathbf a_{\mathrm{RL}}.
\]

这里的 \(\mathbf a\) 是归一化 local-subgoal 参数，不是最终 `/cmd_vel`。decoder 将其
转换为 MPPI sampling mean；MPPI、LaserScan、local obstacle layer、scan_guard 和
safety arbitration 仍走原链路。

预先规定的 Gate：

1. zero correction 必须逐项退化为 BC；
2. 训练前后 BC 参数必须完全不变；
3. seen U-trap 三训练种子不得低于 BC 的 29/30，且不得增加碰撞；
4. 只有 seen Gate 通过，才进入独立 OOD 场景；
5. held-out seeds 不能用于反向挑 checkpoint 或调超参数。

## 2. 实现边界

配置入口：

```text
configs/rl/sac_mppi_utrap_frozen_bc_correction_l16.yaml
```

核心实现位于 `src/mobile_robot_mppi/rl/sac.py`：

- `policy_mode=direct`：保留原 SAC/BC 行为；
- `policy_mode=frozen_bc_correction`：BC actor 作为不可训练 base actor；
- correction actor 输出经 `tanh` 归一化的残差；
- correction 输出层零初始化，确定性 step 0 严格等于 BC；
- base actor 逐参数 `requires_grad=False`，不加入 optimizer；
- base 使用确定性 BC mean，BC 的旧 log-std 不向 base 注入噪声；
- correction SAC 的 log probability 包含残差到最终动作的分段线性 Jacobian；
- checkpoint 保存 base actor、correction actor、optimizer、critic、replay 和 base SHA-256；
- 加载时重新计算 base SHA-256，不一致则拒绝；
- correction 模式强制 `warmup_policy=actor` 和 `normalizer_update=frozen`；
- correction 模式禁止同时启用 full-actor BC anchor。

本轮归一化修正上限为：

```text
local-subgoal distance parameter: 0.20  -> 最多约 0.10 m
local-subgoal bearing parameter:  0.10  -> 最多约 18 degrees
fixed correction alpha:           1.00
```

算法还会根据 BC action 到 `[-1, 1]` 边界的实际剩余空间缩小修正。因此每一维都满足：

\[
|\Delta a_i|
\le
\alpha c_i,
\qquad
a_{\mathrm{BC},i}+\Delta a_i\in[-1,1].
\]

ICODE dynamics residual 与这里的 policy correction 完全不同：前者修正状态导数，后者
只修正 MPPI 采样先验参数。本轮 `prediction_mode=nominal`，没有混入 ICODE。

## 3. 数据、训练和评估协议

- BC seeds：20260721、20260722、20260723；
- 每个 SAC run 使用对应 BC checkpoint；
- SAC training seeds 与 BC seeds 对齐；
- step 0--10k：只更新 critic，base 与 correction actor 均不更新；
- step 10001--20k：只允许 correction actor 和 entropy temperature 更新；
- 训练连续运行，不用 resume 拼接因果曲线；
- 固定 validation seeds：20270718--20270722；
- 独立 held-out seeds：101--110；
- scene：`u_trap_long_board`；
- pose/twist：ground truth controlled localization；
- MPPI samples：100；
- Memory、ICODE、intrinsic exploration：关闭；
- covariance learning：关闭；
- scan_guard 和安全仲裁：开启。

`seeds 101--110` 已经在本轮用于最终复核，后续 L17 不得再把它们当未见 test set。

## 4. Gate A：zero correction 等价性

seed 20260721 的 L16 step-0 checkpoint 在 held-out 101--110 上得到：

```text
success                  9/10
collision                0/10
mean final distance      0.363625 m
mean trajectory length   6.230893 m
mean minimum clearance   0.358273 m
mean control jerk        0.159745
```

这些任务指标与原 BC seed 20260721 逐项一致。单元测试还直接比较归一化 action tensor，
确认 deterministic zero correction 与 BC 输出完全相同，而不是只在容差内接近。

三个训练种子的 base actor SHA-256 在 initial 与 step 20k 间均保持一致：

| seed | frozen base SHA-256 前缀 | initial = 20k |
|---:|---|---:|
| 20260721 | `8064c99464fc1b07` | true |
| 20260722 | `269a451bc27d75f8` | true |
| 20260723 | `1769f2ee158cb2c7` | true |

## 5. Gate B：三训练种子 seen U-trap

固定 5-seed validation 在 step 0/5k/10k/15k/20k 上全部为 5/5、零碰撞。与 L15
full-actor SAC 相比，L16 没有出现 15k 的明显崩溃。但独立 10-seed 复核给出：

| training seed | BC step 0 | L16 step 15k | L16 step 20k | collision |
|---:|---:|---:|---:|---:|
| 20260721 | 9/10 | 10/10 | 9/10 | 全部 0/10 |
| 20260722 | 10/10 | 10/10 | 9/10 | 全部 0/10 |
| 20260723 | 10/10 | 9/10 | 9/10 | 全部 0/10 |
| pooled | **29/30** | **29/30** | **27/30** | **全部 0/30** |

聚合二级指标：

| method | final distance (m) | minimum clearance (m) | trajectory length (m) | safety interventions |
|---|---:|---:|---:|---:|
| BC step 0 | 0.3185 | 0.3521 | 6.2945 | 21.50 |
| L16 step 15k | 0.3207 | 0.3831 | 6.4908 | 30.53 |
| L16 step 20k | 0.3364 | 0.3601 | 6.3252 | 29.00 |

15k 的成功率与 BC 相同、净空更大，但轨迹更长且 safety interventions 更多；不能称为
整体改进。20k 的成功率和终点误差都退化。六组 15k/20k 评估曾并行运行，因此其
planner wall-clock time 受 CPU 竞争影响，不能与早期串行 BC 计时进行方法比较。

checkpoint selection 只看固定 validation return。它为 seed 20260721、20260723 选择
step 0，为 seed 20260722 选择 step 20k；对应 held-out 合计为 28/30，仍低于 BC 的
29/30。这说明当前 5-seed validation 太小，无法可靠选择轻度退化 checkpoint。

## 6. 科研结论

本轮可以支持：

1. 冻结 BC + 有界 correction 明显降低了 full-actor SAC 的灾难性遗忘风险；
2. zero correction 的安全退化性质已经由单元测试和真实闭环验证；
3. 20k 内没有碰撞，安全链保持有效；
4. correction actor 的训练、保存、恢复和推理链路完整可运行。

本轮不能支持：

1. RL 已经超过 BC；
2. 15k 是“最佳训练步数”；该观察来自已经打开的 held-out set；
3. 当前 reward/critic 能可靠选择 correction；
4. 该结果能迁移到 OOD、动态障碍、wheel odometry 或实车；
5. BC+RL 已经具备论文主结果强度。

因此 L16 是“安全结构通过、性能 Gate 未通过”。不应继续堆 OOD gate、ICODE、Memory
或动态障碍来掩盖基础训练问题。

## 7. 下一轮 L17 的必要修改

L17 应先修正实验选择机制，而不是扩大 correction 上限：

1. 新建与 seeds 101--110 完全分离的更大 validation set；
2. 预注册新的 final-test seeds，训练完成前不查看；
3. 在 episode summary 中记录 correction magnitude，检查失败是否与持续大修正相关；
4. 增加 correction magnitude/rate 消融，判断 reward 是否鼓励无必要偏离 BC；
5. 将 seen non-inferiority 与 OOD improvement 分成两个 checkpoint score；
6. 只有 seen success 不低于 BC 且选择机制稳定，才进入静态 OOD；
7. 动态障碍、ICODE 与 Memory 继续后置。

## 8. 可复现命令与产物

训练单个种子：

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_frozen_bc_correction_l16.yaml \
  --seed 20260721 --steps 20000 \
  --initialize-actor-from \
    results/research_platform/rl/bc_l13_seed20260721_20260714/checkpoints/best.pt \
  --output-dir \
    results/research_platform/rl/frozen_bc_correction_l16_seed20260721_20k_continuous_20260714_v1
```

明确场景的 held-out 评估：

```bash
.venv/bin/python experiments/rl/evaluate_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_frozen_bc_correction_l16.yaml \
  --scene-config configs/research/mujoco_u_trap_long_board.yaml \
  --checkpoint RUN_DIR/checkpoints/step_000020000.pt \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --pose-source ground_truth --twist-source ground_truth \
  --output-dir EVAL_DIR
```

主要产物：

```text
results/research_platform/rl/frozen_bc_correction_l16_seed20260721_20k_continuous_20260714_v1/
results/research_platform/rl/frozen_bc_correction_l16_seed20260722_20k_continuous_20260714_v1/
results/research_platform/rl/frozen_bc_correction_l16_seed20260723_20k_continuous_20260714_v1/
results/research_platform/rl/bc_vs_l16_correction_multitraining_seed_20260714_v1/
```

这些是开发 Gate 的真实结果，不是正式论文统计，也没有伪造成 RL 正向结果。
