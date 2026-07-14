# RL-guided MPPI 交付架构

## 1. 本轮上线范围

本实现把强化学习完整接入 Python 3 / MuJoCo 研究链，但没有把神经网络部署到
ROS Kinetic 实车脚本。RL 的职责被严格限制为 MPPI sampling prior：它决定
“下一轮应该重点从哪些控制序列附近采样”，而不是绕过 MPPI 直接控制小车。

```text
MuJoCo true plant
    ↓ odom / wheel state / LaserScan
scan_guard + local_obstacle_layer
    ↓ observable RobotObservation
RL observation encoder
    ↓ 48-D default feature vector
SAC actor
    ↓ normalized sequence knots (default 12-D)
prior parameterization
    ↓ horizon × control_dim mean, optional covariance
confidence gate + conventional goal prior
    ↓ blended sampling distribution
MPPI rollout (nominal / MLP / ICODE)
    ↓ proposed control
scan_guard safety arbitration
    ↓ executed control
MuJoCo true plant
```

这里必须保持三个区分：

1. RL prior 不等于最终控制；MPPI 仍然比较 `K` 条候选序列。
2. MPPI prediction dynamics 不等于 MuJoCo true plant；ICODE 只修正前者。
3. proposed control 不等于 executed control；最终命令仍由安全仲裁决定。

## 2. 代码职责

| 文件 | 职责 |
|---|---|
| `rl/observation.py` | 将相对目标、速度、上一拍控制、安全状态和 LaserScan 分扇区编码；维护训练统计与 OOD 距离。 |
| `rl/parameterization.py` | 把低维策略输出的控制关键点插值成完整 MPPI prior；支持 `delta/absolute` 和可选 covariance。 |
| `rl/sac.py` | Squashed Gaussian Actor、双 Critic、目标网络、自动熵系数和 SAC 更新。 |
| `rl/replay.py` | 可复现、可恢复的经验回放。 |
| `rl/environment.py` | 执行 RL prior → MPPI → safety → MuJoCo 的训练环境与逐项 reward 日志。 |
| `rl/prior.py` | 训练时外部 prior、检查点推理 prior、传统 prior 混合与 OOD 回退。 |
| `rl/checkpointing.py` | 保存模型、优化器、normalizer、配置、Git SHA、训练状态和可选 replay。 |
| `rl/trainer.py` | 多场景训练、验证、best/latest checkpoint、resume 和 CSV 日志。 |
| `runtime/factories.py` | 根据 YAML 自动构建传统 prior 或 RL checkpoint prior。 |

## 3. RL 动作为什么不是 `(v, omega)`

若 RL 直接给出最终 `(v, omega)`，MPPI 就会退化为旁路，无法保留模型预测、
在线代价比较和传统控制的可靠性。当前 Actor 默认输出 6 个时间关键点，每个关键点
含 `(v, omega)` 两维，共 12 维。关键点插值后形成完整 horizon 的均值序列：

\[
z_t = \pi_\phi(o_t), \qquad
\mu_{0:H-1}^{\mathrm{RL}} = \mathcal{D}(z_t).
\]

稳定默认模式为 `delta`：

\[
\mu^{\mathrm{learned}}
= \operatorname{clip}\left(
\mu^{\mathrm{goal}} + s\,\Delta\mu_\phi,
u_{\min},u_{\max}
\right).
\]

因此零输出等价于传统 goal warm start；`absolute` 模式保留为结构消融。

## 4. 门控与回退

最终 prior 均值为：

\[
\mu = (1-\alpha)\mu^{\mathrm{goal}}
+ \alpha\mu^{\mathrm{learned}}, \qquad \alpha\in[0,1].
\]

可用模式：

- `none`：`alpha=1`，测试纯 RL prior；
- `fixed`：固定混合比例，测试 RL 强度；
- `ood`：依据训练观测的最大标准化距离连续降低 `alpha`；
- `use_critic_disagreement`：可选乘以 twin-Q 分歧因子。

OOD score 和 Critic disagreement 都是可记录的经验诊断量，不是校准概率，不构成
稳定性或安全性证明。真正的硬安全边界仍是下游 `scan_guard`。

## 5. 与 ICODE 的正交组合

ICODE 修正 MPPI rollout 内的预测动力学；RL 修正 MPPI 采样分布。两者接口正交：

```text
planner.prediction_mode: nominal | mlp_residual | icode_residual
planner.sampling_prior: goal_warm_start | rl
```

正式实验必须先分别建立 ICODE-only 和 RL-only 结果，再测试 ICODE+RL，避免把两类
增益混在一起。

## 6. 实车边界

本轮没有修改 `mppi_hardware_bridge/scripts/`，没有让 Python 2 bridge import Torch，
也没有让 RL 直接发布 `/cmd_vel`。后续实车顺序仍应是 offline replay、shadow mode、
低速封闭场测试，最后才允许受 scan_guard 保护的在线推理。
