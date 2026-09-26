# L13：特权教师蒸馏与 BC→SAC 启动门

日期：2026-07-14  
状态：L13 BC-only 受控实验已完成；SAC 微调反例与后续门见文档 11。

## 1. 为什么进入这一关

L12 的三训练种子复现实验推翻了“固定比例成功回放足以稳定学习整条
U-trap 路线”的单种子印象。成功回放只会加强策略自己已经偶然走通的片段；
如果从原始起点到终点的完整因果链从未进入 replay，它并不能凭空补出这条链。

本关检验一个更窄、可以被否证的假设：

> 一个仅在训练阶段使用离线无碰路线的教师，能否把“下一段应该往哪里走”
> 蒸馏到只观察 Odom/LaserScan/goal 的 local-subgoal actor；在此基础上，SAC
> 是否比随机初始化更稳定地学习控制回报。

本关不启用 ICODE、Memory 或 RL uncertainty gate。否则无法判断提升来自哪里。

## 2. 信息边界

```text
离线路线与场景真值 ──> ScriptedPolylineSubgoal（仅教师）
                              │
                              │ 归一化 [distance, bearing]
                              ▼
Odom/LaserScan/goal ──> encoded observation ──> BC actor
                                                   │
                                                   ▼
                                         local-subgoal prior
                                                   │
                                                   ▼
LaserScan -> local_obstacle_layer -> MPPI -> scan_guard -> MuJoCo
```

教师路线和真值轨迹只写入 `audit/`。可被训练 loader 打开的 NPZ shard 采用
固定字段白名单，只包含：

- `observation`：`MppiPriorEnv.reset()/step()` 返回的编码观测；
- `teacher_action`：教师输出的两个归一化 local-subgoal 参数；
- `episode_id`、`step`：用于验证 episode 完整性。

actor 不接收路线进度、waypoint、真值位姿、全局障碍物或教师的 cross-track
error。正式在线 MPPI 的障碍仍只能来自 LaserScan 和 local obstacle layer。

新采集数据使用 manifest schema v2，顶层完整记录 observation encoder 与
local-subgoal parameterization；训练前会逐字段核对，不能只凭“维度相同”混用数据。
三个 shard 的实际 `episode_id` 也必须两两不相交。历史 v1 数据只在其全部
`audit/resolved_configs` 完整且语义一致时只读兼容，原 manifest 与 fingerprint
保持不变；缺少审计证据时直接拒绝加载。

## 3. BC 到底监督什么

教师标签不是电机命令，也不是最终的 \((v,\omega)\)。它和 SAC actor 的动作
空间完全一致：

\[
a_E = [a_d,a_\beta]\in[-1,1]^2,
\]

分别编码 body-frame 子目标的距离和方位。BC 监督高斯 actor 的确定性均值：

\[
\mathcal L_{\mathrm{BC}}
=
\left\|
\tanh\!\left(\mu_\theta(\operatorname{Norm}(o))\right)-a_E
\right\|_2^2
+
\lambda_\sigma
\left\|\log\sigma_\theta(o)-\log\sigma_{\mathrm{target}}\right\|_2^2.
\]

第二项只控制 SAC 启动时的探索尺度，不会把 teacher action 当成随机样本。
归一化统计只由 train episodes 计算，validation/test 不得更新统计量。

## 4. 两种 checkpoint 操作必须分开

`--resume-bc` 用于精确继续中断的 BC，恢复 actor optimizer、epoch、RNG 和
训练集 normalizer，并校验数据集、网络、损失和 action contract。新 checkpoint
还携带 best-actor snapshot；旧格式只有显式加入 `--allow-legacy-resume` 才能继续，
且必须在实验记录中标为 legacy continuation。

`--initialize-actor-from` 用于 BC→SAC，只复制 actor 权重和 normalizer。以下
状态必须重新初始化：

- 两个 critic 与 target critic；
- entropy temperature；
- actor/critic/alpha optimizer 状态；
- replay buffer；
- SAC global step、episode count 和 RNG。

这样 BC→SAC 的含义才是“给 SAC 一个行为起点”，而不是暗中恢复另一场 SAC。
BC→SAC 的 warm-up 也必须从该 actor 随机采样；若先执行几千步均匀随机 action，
就会在 actor 第一次控制环境之前用无关 replay 冲掉预训练，实验定义将失去意义。

## 5. 预注册比较

先在 controlled localization（ground-truth pose/twist，相当于实验室 mocap
条件）完成算法判断，再单独进行 wheel-odometry robustness gate。

第一阶段比较：

1. scripted teacher upper bound；
2. BC-only；
3. SAC-only（uniform replay）；
4. BC→SAC（uniform replay）；
5. conventional MPPI。

所有闭环方法使用相同场景、相同 MPPI `K`、相同 held-out evaluation seeds、
相同 safety chain。BC 与 SAC 的模型选择 seed 固定，不随 training seed 改变。

主要指标：success rate、collision rate、final goal distance、minimum clearance、
safety interventions、trajectory length、control jerk、planner compute time。
离线 action RMSE 只是诊断指标，不能替代闭环成功率。

## 6. 决策门

- 若 scripted teacher 本身不能稳定成功：停止训练，先修 route/decoder/MPPI 接口。
- 若 BC 离线误差下降但 BC-only 闭环失败：说明 imitation loss 与闭环分布之间
  存在缺口，不能宣称教师信息已被有效蒸馏。
- 若 BC-only 有效、BC→SAC 反而退化：检查探索方差、normalizer 漂移和 actor
  被 critic 梯度快速冲掉的问题。
- 若 BC→SAC 只在一个 training seed 有效：按 L12 的标准判定为不稳定，不进入
  ICODE/RL 联合阶段。
- 只有三训练种子与独立 held-out seeds 均支持提升，才扩展到 unseen physics、
  wheel odometry 和动态障碍。

## 7. 运行入口

下面是接口设计，实际数据路径以运行产物为准：

```bash
# 1) 采集成功教师 episode；失败 episode 只进入 audit
.venv/bin/python experiments/rl/collect_scripted_subgoal_demonstrations.py \
  --rl-config configs/rl/sac_mppi_utrap_bc_l13.yaml \
  --configs configs/research/mujoco_u_trap_long_board.yaml \
  --seeds 41,42,43,44,45,46,47,48,49,50,51,52,53,54,55 \
  --output-dir results/research_platform/rl/bc_datasets/utrap_l13

# 2) BC
.venv/bin/python experiments/rl/train_behavior_cloning_prior.py \
  --config configs/rl/sac_mppi_utrap_bc_l13.yaml \
  --dataset-dir results/research_platform/rl/bc_datasets/utrap_l13 \
  --output-dir results/research_platform/rl/bc_l13_seed20260721

# 3) BC-only 闭环；仍由 MPPI 和 scan_guard 输出最终控制
.venv/bin/python experiments/rl/evaluate_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_bc_l13.yaml \
  --scene-config configs/research/mujoco_u_trap_long_board.yaml \
  --checkpoint results/research_platform/rl/bc_l13_seed20260721/checkpoints/best.pt \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --pose-source ground_truth --twist-source ground_truth \
  --output-dir results/research_platform/rl/bc_l13_heldout

# 4) 只把 actor + normalizer 交给一场全新的 SAC
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_bc_l13.yaml \
  --initialize-actor-from \
    results/research_platform/rl/bc_l13_seed20260721/checkpoints/best.pt \
  --output-dir results/research_platform/rl/bc_to_sac_l13_seed20260721
```

## 8. 当前禁止的表述

- 不得把 privileged teacher 称为 deployable planner；
- 不得把 BC 离线 MSE 下降称为闭环控制提升；
- 不得把 controlled localization 结果称为 wheel-odometry robustness；
- 不得在三训练种子复现前把单次成功称为稳定改进；
- 不得把本关与 ICODE 或 Memory 的效果混为一谈。

## 9. L13 实际结果摘要

15 个 scripted teacher episode 中 14 个成功、零碰撞；seed 45 以 0.305 m
结束，略高于 0.30 m 成功阈值，按预注册规则只进入 audit、不进入训练 shard。
最终 train/validation/test 分别包含 8/3/3 个完整成功 episode 和
2059/737/891 个 transition。

三个独立 BC 初始化种子在同一 held-out seeds 101--110、U-trap、K=100、
ground-truth pose/twist 条件下得到 9/10、10/10、10/10，总计 29/30，零碰撞。
聚合 success rate 为 96.7%，训练种子间样本标准差为 5.8 个百分点。相同
held-out 条件下的纯 SAC uniform 对照为 16/30（53.3%）。这说明 BC 是稳定
启动底座；并不说明 privileged teacher 是可部署输入，也不说明 SAC 微调有效。

wheel-odometry 单模型边界检查只有 2/10 成功、零碰撞。因此本结果仍属于
controlled-localization 算法条件，不能外推为实车定位鲁棒性。
