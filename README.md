# Mobile Robot MPPI Research Platform

## Current validated research status (2026-07-18)

The frozen control-affine ICODE residual model is integrated into MuJoCo MPPI
rollouts.  The current primary RL mechanism is an interpretable route-context
bandit that selects an MPPI covariance option rather than directly commanding
the robot.  Independent L94, L95 and L97 evaluations show that contextual
`K=50` preserves tracking precision and safety relative to the strongest fixed
`K=100` sampler while reducing completion time and CPU planner compute.

L96/L97 also isolate a fixed yaw-command slew constraint that significantly
reduces both issued and physically applied jerk relative to the raw contextual
controller.  The strict final-package jerk confidence interval versus fixed
`K=100` crosses zero, so the preregistered L97 overall Gate is recorded as
failed; no jerk-superiority claim is made.  See
`docs/rl/149_l97_contextual_covariance_jerk_confirmation_results_2026-07-18.md`.

L98 then evaluated ICODE and contextual sampling in one blocked 2 x 3
factorial. The ICODE contribution and complete-package Gates passed, but the
contextual `K=50` arm did not meet the frozen +2 mm RMSE noninferiority margin
against ICODE fixed `K=100`; consequently L99 remains sealed. The retained next
question is a causal anytime `K=50 -> K=100` stop/continue policy, not a change
of the ICODE + RL research direction. See
`docs/rl/151_l98_cross_layer_factorial_results_and_anytime_pivot_2026-07-18.md`.

L100--L103 resolve that question with nested common-random-number rollouts and
a compute-constrained contextual bandit. L102 is retained as a failed frozen
trial: its terminal online dual price under-used the held-out compute budget.
L103 separates exploration pricing from discovery-only deployment calibration
and passes every preregistered confirmation Gate on 48 new MuJoCo episodes. At
held-out physical states the bandit uses a mean `K=77.86`, improves true branch
cost by `2.400%` over fixed `K=50` (episode-level 95% CI for the raw delta
`[-0.523, -0.114]`), and retains `58.1%` of hindsight-Oracle gain. A secondary
stratified matched-budget randomization test gives `p=0.00020`, indicating that
the learned allocation matters beyond merely spending the same average budget.
This is sampled-state evidence; online closed-loop and ICODE-specific
interaction tests remain the next claim boundary. See
`docs/rl/156_l103_calibrated_budget_bandit_confirmation_results_2026-07-18.md`.

本仓库是一套面向科研实验的移动机器人控制平台，主线包括：

- 可变目标、路径与轨迹参考；
- 维度可配置的 MPPI；
- MuJoCo 差速驱动物理仿真；
- LaserScan → `scan_guard` → `local_obstacle_layer` 感知安全链；
- Nominal、Oracle、MLP、ICODE 残差预测模型；
- Memory-Augmented MPPI；
- 为 RL-driven sampling 保留的稳定接口；
- 统一的数据、指标、配置与多随机种子 benchmark。

当前研究代码位于标准 `src` package：

```text
src/mobile_robot_mppi/
```

原有 planner、MuJoCo live sim 和 ROS Kinetic 实车脚本继续保留，作为兼容基线与实车安全资产，不会被训练依赖或 PyTorch 污染。

## 核心边界

```text
Task / Reference
       ↓
RobotObservation ← Odom / Encoder / IMU / LaserScan
       ↓
scan_guard → local_obstacle_layer
       ↓
MPPI rollout using PredictionDynamics
       ↓
proposed control
       ↓
safety arbitration
       ↓
executed control
       ↓
MuJoCo true plant / real robot
```

必须区分：

```text
PredictionDynamics ≠ environment true plant
RobotObservation ≠ MuJoCo ground truth
Proposed control ≠ executed control
Viewer ≠ simulation environment
```

MuJoCo ground truth 只能进入 logger、metrics 和 residual dataset，不得直接提供给普通 MPPI。

## MuJoCo 物理主线

新 backend：`mujoco_diff_drive`。

```text
6-DoF free chassis
+ left/right wheel hinge
+ caster contacts
+ two wheel actuators
+ wheel-ground friction/contact
+ encoder/actuator/IMU sensors
```

控制链：

```text
(v_cmd, omega_cmd)
→ left/right wheel targets
→ velocity servo or torque PI
→ MuJoCo contact dynamics
→ odometry/LaserScan observation
```

旧的 `legacy_kinematic` backend 继续存在，用于数值回归和历史结果复现。

## 可变 Goal 与扩展状态

Goal 不再是核心代码中的全局常量，支持 `PointGoal`、`PoseGoal`、`WaypointReference` 和 `TimeTrajectoryReference`。旧实验默认仍可配置为 `(3.0, 3.0)`，但可以通过 YAML 或 CLI 修改：

```bash
python experiments/run_research_simulation.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --goal 2.0 1.5 --headless
```

支持的规划状态：

```text
unicycle_3:         [x, y, theta]
dynamic_unicycle_5: [x, y, theta, v, omega]
wheel_augmented_7:  接口已定义，需显式七状态预测模型
```

高层动作默认保持 `[v_cmd, omega_cmd]`。`StateSpec` 与 `ActionSpec` 允许后续增加状态和控制维度，不需要重写 MPPI。

## ICODE

```text
f_res(x, u) = f_theta(x) + G_theta(x) u
f_pred(x, u) = f_nominal(x, u) + f_res(x, u)
```

当前保留两条研究链：

1. `src/dynamics`、`src/learning`：已验证的三状态 ICODE 兼容框架；
2. `src/mobile_robot_mppi/learning`：面向 MuJoCo 五状态数据的新训练入口。

当前实现复现公开的 control-affine residual structure，但没有实现或声称原始 ICODE 的 contraction、稳定性或收敛保证。

## RL-guided MPPI

当前已实现可训练的 SAC sampling prior，而不只是接口占位：

```text
Odom + LaserScan + Goal + Previous Control
                    ↓
    SAC policy (local subgoal / sequence knots)
                    ↓
       bounded MPPI mean / covariance
                    ↓
     MPPI optimization + ICODE (optional)
                    ↓
          scan_guard safety arbitration
                    ↓
                 MuJoCo
```

默认研究配置仍关闭 RL；只有同时设置 `planner.sampling_prior: rl` 和
`rl.enabled: true` 才会加载策略。训练、断点恢复、确定性评估、OOD 门控、
多场景/物理域训练和 baseline/RL/gated-RL/ICODE+RL 消融入口见
[`docs/rl/00_rl_mppi_architecture.md`](docs/rl/00_rl_mppi_architecture.md)。
首轮30k-step可学习性Gate的真实结果与未通过项见
[`docs/rl/03_learnability_gate_2026-07-13.md`](docs/rl/03_learnability_gate_2026-07-13.md)。
L2-L5 的课程学习、奖励修正、低维 prior 与独立种子证据见
[`docs/rl/04_sampling_prior_learnability_iterations_2026-07-13.md`](docs/rl/04_sampling_prior_learnability_iterations_2026-07-13.md)。
L6 的二维局部子目标、三帧历史与严格 Gate 结果见
[`docs/rl/05_local_subgoal_history_gate_2026-07-13.md`](docs/rl/05_local_subgoal_history_gate_2026-07-13.md)。
L7 的人工子目标上界诊断、停止条件与解码器定位结果见
[`docs/rl/06_scripted_subgoal_upper_bound_gate_2026-07-13.md`](docs/rl/06_scripted_subgoal_upper_bound_gate_2026-07-13.md)。
L8 的动态翻译器、分段净空路线、定位漂移审计与独立种子 Gate 见
[`docs/rl/07_dynamic_decoder_and_localization_gate_2026-07-13.md`](docs/rl/07_dynamic_decoder_and_localization_gate_2026-07-13.md)。
L9--L12 的多场景遗忘、探索负消融、成功轨迹回放、held-out 结果与定位边界见
[`docs/rl/08_exploration_and_success_replay_gate_2026-07-13.md`](docs/rl/08_exploration_and_success_replay_gate_2026-07-13.md)。
L12 的三训练种子复现、单种子结论撤回、固定验证种子与跨种子聚合规则见
[`docs/rl/09_multitraining_seed_replication_2026-07-14.md`](docs/rl/09_multitraining_seed_replication_2026-07-14.md)。
L13 的特权教师数据隔离、行为克隆（BC）预训练、BC→SAC 状态边界与预注册决策门见
[`docs/rl/10_behavior_cloning_bootstrap_gate_2026-07-14.md`](docs/rl/10_behavior_cloning_bootstrap_gate_2026-07-14.md)。
BC 三训练种子复现、SAC 非单调策略震荡、legacy-resume 作废审计、
BC anchor/critic burn-in 消融、wheel-odometry 边界与下一步 safe policy-correction 决策见
[`docs/rl/11_bc_multiseed_and_safe_finetuning_gate_2026-07-14.md`](docs/rl/11_bc_multiseed_and_safe_finetuning_gate_2026-07-14.md)。
L16 的冻结 BC、有界 policy correction、zero-correction 回归、三训练种子真实结果与
“安全结构通过但性能 Gate 未通过”的决策见
[`docs/rl/12_frozen_bc_bounded_correction_gate_2026-07-14.md`](docs/rl/12_frozen_bc_bounded_correction_gate_2026-07-14.md)。
L17 的保守 correction 正则、paired fail-closed checkpoint selection、验证 seed/初始状态合同审计与三训练种子 sealed-final 结果见
[`docs/rl/13_conservative_correction_preregistered_gate_2026-07-14.md`](docs/rl/13_conservative_correction_preregistered_gate_2026-07-14.md)、
[`docs/rl/14_validation_seed_audit_and_l17_v2_prereg_2026-07-14.md`](docs/rl/14_validation_seed_audit_and_l17_v2_prereg_2026-07-14.md) 和
[`docs/rl/15_l17_v3_conservative_correction_results_2026-07-14.md`](docs/rl/15_l17_v3_conservative_correction_results_2026-07-14.md)。最终结论是：安全退化链路通过，但当前 SAC correction 没有优于 BC 的证据。
L18 的 twin-critic 相对 BC 优势记录、零阈值 online/target hard gate 预注册、
三训练种子配对结果和数据质量审计见
[`docs/rl/16_critic_advantage_diagnostic_prereg_2026-07-15.md`](docs/rl/16_critic_advantage_diagnostic_prereg_2026-07-15.md) 和
[`docs/rl/17_l18_critic_advantage_diagnostic_results_2026-07-15.md`](docs/rl/17_l18_critic_advantage_diagnostic_results_2026-07-15.md)。当前结论是：critic 含有弱排序信息，但零阈值优势门控仍损失 BC-success episodes，未进入新 final test。
L19 的独立 calibration/selection 数据隔离、跨训练种子全局 raw-advantage margin 校准和 fail-closed 结果见
[`docs/rl/18_l19_advantage_margin_calibration_prereg_2026-07-15.md`](docs/rl/18_l19_advantage_margin_calibration_prereg_2026-07-15.md) 与
[`docs/rl/19_l19_advantage_margin_calibration_results_2026-07-15.md`](docs/rl/19_l19_advantage_margin_calibration_results_2026-07-15.md)。六个 margin 均未满足 paired BC non-inferiority，系统保留 BC fallback，预留 selection seeds 未打开。
L20 的尺度不变 twin-critic 共识 LCB 公式、预注册数据隔离和三训练种子校准结果见
[`docs/rl/20_l20_twin_critic_consensus_lcb_prereg_2026-07-15.md`](docs/rl/20_l20_twin_critic_consensus_lcb_prereg_2026-07-15.md) 与
[`docs/rl/21_l20_twin_critic_consensus_lcb_results_2026-07-15.md`](docs/rl/21_l20_twin_critic_consensus_lcb_results_2026-07-15.md)。beta 2 将成功从 27/36 提高到 31/36，但仍有 1 次 paired BC-success loss，因此保持 BC fallback 且未打开 selection seeds。
L21 的确定性同 seed action replay、单步 correction 反事实分支数据和模型训练充分性 Gate 见
[`docs/rl/22_l21_counterfactual_risk_dataset_prereg_2026-07-15.md`](docs/rl/22_l21_counterfactual_risk_dataset_prereg_2026-07-15.md) 与
[`docs/rl/23_l21_counterfactual_risk_dataset_results_2026-07-15.md`](docs/rl/23_l21_counterfactual_risk_dataset_results_2026-07-15.md)。124 个 accepted branches 全部为 neutral，说明单步 correction 会被 BC 快速恢复；系统按预注册规则不训练风险模型、不打开 test seeds，下一步转向短时 correction-burst 反事实。
L22 的 10-step gated correction burst、嵌套配对设计和完整数据审计见
[`docs/rl/24_l22_counterfactual_burst_prereg_2026-07-15.md`](docs/rl/24_l22_counterfactual_burst_prereg_2026-07-15.md) 与
[`docs/rl/25_l22_counterfactual_burst_results_2026-07-15.md`](docs/rl/25_l22_counterfactual_burst_results_2026-07-15.md)。120 个分支的数据质量与 burst 执行合同通过，轨迹差异扩大到约 -9.4 cm 至 +18.5 cm，但预注册的 harmful/beneficial label 仍不足，因此继续保持 BC fallback、不开 sealed test，也不训练平凡分类器。
L23 的 continuous-utility ensemble、group bootstrap、ridge/zero baselines、group conformal LCB 与 fail-closed checkpoint loader 见
[`docs/rl/26_l23_continuous_utility_prereg_2026-07-15.md`](docs/rl/26_l23_continuous_utility_prereg_2026-07-15.md)、
[`docs/rl/27_l23_zero_variance_scaling_correction_2026-07-15.md`](docs/rl/27_l23_zero_variance_scaling_correction_2026-07-15.md) 与
[`docs/rl/28_l23_continuous_utility_results_2026-07-15.md`](docs/rl/28_l23_continuous_utility_results_2026-07-15.md)。423 个 development branches 的数据质量通过，但 current-state ensemble 只比 zero predictor 改善 0.88%，负效用识别和 conformal acceptance 均未过 Gate；sealed test 保持关闭，下一步转向 trajectory-aware utility features。
L24 的 side-effect-free MPPI preview、127 维配对 BC/SAC 候选轨迹特征、state-only 直接消融与结果见 [`docs/rl/29_l24_trajectory_utility_prereg_2026-07-15.md`](docs/rl/29_l24_trajectory_utility_prereg_2026-07-15.md) 和 [`docs/rl/30_l24_trajectory_utility_results_2026-07-15.md`](docs/rl/30_l24_trajectory_utility_results_2026-07-15.md)。409 条 development branches 的质量合同通过，但 trajectory ensemble 只比 zero 改善 1.32%，并比同数据的 state-only ensemble 差 1.03%；校准后接受率仍为 0%。因此 sealed test 继续关闭，主线不再扩张 learned macro-utility Gate，而转向可审计的 OOD 场景级 RL 激活与简化 factorial ablation。

最小 MuJoCo smoke（仅检查链路，不代表性能）：

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_prior_smoke.yaml \
  --output-dir results/research_platform/rl/smoke --smoke
```

RL 只影响候选序列的抽样先验，不直接输出最终 `/cmd_vel`，也不能绕过
`scan_guard`。`mppi_hardware_bridge/scripts/` 不 import PyTorch。

## 安装

```bash
cd /home/mapples/projects/mobile-robot-mppi-study
python3 -m venv .venv
.venv/bin/pip install -e '.[simulation,learning,dev]'
```

当前物理基线固定 MuJoCo `3.2.3`，避免重构与依赖升级同时改变实验行为。

## 快速运行

Legacy 回归：

```bash
.venv/bin/python experiments/run_research_simulation.py \
  --config configs/research/legacy_kinematic.yaml --headless
```

差速物理 MuJoCo：

```bash
.venv/bin/python experiments/run_research_simulation.py \
  --config configs/research/mujoco_diff_drive.yaml --headless
```

开启 viewer 时将 `--headless` 改为 `--viewer`。

## MuJoCo → ICODE

```bash
.venv/bin/python experiments/collect_mujoco_residual_data.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --output-dir results/research_platform/datasets/mujoco_v1 \
  --episodes 40 --steps 120 --source mixed

.venv/bin/python experiments/train_platform_residual.py \
  --config configs/research/icode_dynamic5.yaml \
  --dataset-dir results/research_platform/datasets/mujoco_v1 \
  --output-dir results/research_platform/checkpoints/icode_v1

.venv/bin/python experiments/run_research_simulation.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --prediction-mode icode_residual \
  --checkpoint results/research_platform/checkpoints/icode_v1/best.pt \
  --headless
```

将训练配置替换为 `configs/research/mlp_dynamic5.yaml` 即可训练 MLP baseline。

## Multi-seed benchmark

```bash
.venv/bin/python experiments/run_research_benchmark.py \
  --config configs/research/mujoco_diff_drive.yaml \
  --output-dir results/research_platform/benchmark_v1 \
  --seeds 11,12,13,14,15 \
  --samples 50,100,200,400 \
  --mlp-checkpoint results/research_platform/checkpoints/mlp_v1/best.pt \
  --icode-checkpoint results/research_platform/checkpoints/icode_v1/best.pt
```

每次运行输出 resolved config、provenance、trajectory CSV 和 metrics JSON；benchmark 额外输出 episode CSV、summary CSV 和聚合 JSON。

## 测试

```bash
.venv/bin/python -m pytest tests/dynamics -q
.venv/bin/python -m pytest tests/learning -q
.venv/bin/python -m pytest tests/planners -q
.venv/bin/python -m pytest tests/platform -q
```

## 实车边界

`mppi_hardware_bridge/scripts/` 仍是 ROS Kinetic/Python 2 兼容区域：

- 不直接 import Torch；
- 不把训练依赖变成启动依赖；
- `/cmd_vel`、odom、LaserScan、scan_guard 和控制仲裁保持不变；
- learned residual 必须先经过仿真、offline 和 shadow mode。

详细说明见 [docs/refactor/00_master_plan.md](docs/refactor/00_master_plan.md)。

## L25: LaserScan scene-complexity gating

The RL sampling prior can now be activated continuously from local LaserScan geometry while
preserving an exact traditional-MPPI fallback in obstacle-free geometry. The development
ablation used 3 independently trained checkpoints, 10 paired episode seeds, 4 scenes and 4
methods (`480` episodes, `124334` control steps, `K=200`). The gated method achieved `30/30`
success in clean dynamics and `20/30`, `24/30`, and `26/30` in the single-obstacle, narrow-corridor,
and U-trap scenes, with zero collisions in all 480 episodes. The preregistered development gate
still failed because the globally labelled “simple” single-obstacle scene required substantial
local activation; sealed test seeds therefore remain unopened.

See the [L25 preregistration](docs/rl/31_l25_scene_complexity_gate_prereg_2026-07-15.md) and
[L25 development results](docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md). These are
development results, not final paper claims.

## L29/L30: cross-layer ICODE--RL--Gate development

The research platform now includes a collidable prescribed dynamic obstacle whose current
MuJoCo geometry is shared by contact, clearance, LaserScan and visualization, while the planner
continues to receive obstacle information only through LaserScan and the local obstacle layer.
The L29 blocked factorial exposed a late-activation failure of the spatial gate for a sparse
lateral crossing. L30 added a scan-only temporal closing-risk signal. Across 60 blocking-scene
development episodes per method, temporal-gate + ICODE achieved `60/60` success, zero collisions
and `0.289 m` mean final distance; its clean-scene trajectory remained stepwise identical to
traditional MPPI. The fixed L30 gate nevertheless remains formally failed because the nominal
temporal-gate ablation had one collision above its matched always-RL comparator in one dynamic
stratum. No confirmation seed was opened.

See the [L29 preregistration](docs/rl/39_l29_cross_layer_factorial_prereg_2026-07-15.md),
[L30 remediation preregistration](docs/rl/40_l30_temporal_closing_gate_prereg_2026-07-15.md), and
[L29/L30 development results](docs/rl/41_l29_l30_cross_layer_results_2026-07-15.md).

## L31: dynamic-motion generalization stress test

L31 froze the L30 temporal thresholds and evaluated four obstacle-motion variants, two physics
domains, five paired methods, three independently trained model blocks and five fresh development
seeds (`600` MuJoCo episodes). The data audit passed, but the preregistered development gate did
not: temporal gate + ICODE reached `64/120` successes with `56/120` collisions. Reverse-direction
and faster crossings exposed that sector-minimum differencing is not obstacle tracking and that a
high hazard signal cannot be treated as high confidence in a static-scene RL prior. All L31
confirmation seeds remain sealed.

See the [L31 preregistration](docs/rl/42_l31_dynamic_variant_generalization_prereg_2026-07-15.md)
and [L31 results and failure analysis](docs/rl/43_l31_dynamic_variant_generalization_results_2026-07-15.md).
