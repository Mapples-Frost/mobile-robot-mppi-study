# L257 Direct SAC CUDA 高预算训练协议

日期：2026-07-21  
状态：预注册 development protocol；训练结果尚未读取

## 1. 研究问题

L256 显示当前 Actor/critic 的在线权限几乎被可靠性机制压到零。该现象可能来自两类原因：

1. 论文耦合机制本身不合适；
2. 现有 direct SAC 仅训练 30,000 environment steps，尚未获得足够的跨地图控制能力和 critic 一致性。

L257 优先检验第二种、也更简单的解释。不得在得到本轮结果前改变 Actor 的在线角色、混合公式或论文核心机制。

## 2. 冻结处理变量

唯一学习处理变量是训练暴露量：

```text
30,000 steps -> 120,000 steps (4x)
```

执行设备由 CPU 改为 CUDA 仅用于加速数值计算，不作为算法处理因素。以下项目全部冻结为 L243：

- direct-control SAC policy mode；
- 69 维 Path/Residual-Conditioned observation 与 2 维 `(v_cmd, omega_cmd)` action；
- Actor/critic 网络、学习率、熵调节、quantile critic 与 group-robust Actor；
- BC anchor 数据、权重和 frozen normalizer；
- 六张 L222/L218 训练地图与 seen/unseen MuJoCo 物理域；
- reward、ICODE、MPPI cost、scan_guard 与安全仲裁；
- 在线 `Traditional prior + Actor prior` 的比例融合、HSS 和 terminal value 机制。

保存 replay tensors、降低评估/检查点写盘频率只改变可恢复性和测量开销，不改变训练更新。

## 3. 初始化与来源

- 初始 Actor：`results/research_platform/rl/l219_expanded_actor_seed20262193_30k_v2/checkpoints/step_000010000.pt`
- 初始 Actor SHA256：`bf26a67ebac313930d63760db931e5d50704cdd9181923afd7f66d16159356b4`
- BC dataset：`results/research_platform/rl/l243_direct_control_teacher_dataset_v1`
- dataset manifest file SHA256：`221ede77433864094a505149ed83e7f4864f4607f87ed5338ce8ed6d298b82b7`
- dataset contract fingerprint：`918712d513763fd272db25d7a428c6fcf563e6ca1b98c0f089cb46f52cc91fbb`
- policy/action contract：`direct`, observation 69D, action 2D

L243 checkpoint 不包含 replay arrays，因此不得声称 L257 是 L243 的 exact resume。L257 从相同初始 Actor、相同数据、相同 seeds 和相同学习配置重新训练 120k，并在同一 CUDA run 内配对比较 30k 与 120k checkpoint。

## 4. 实验设计

- 独立重复单位：训练 seed；
- seeds：`20262331, 20262332, 20262333`；
- validation bases：`20262931, 20262932, 20262933`；
- block：seed；
- repeated strata：scene × physics domain；
- 主要配对比较：同 seed 的 30k checkpoint 与 120k checkpoint；
- checkpoint：30k、60k、90k、120k；
- validation：每 10k steps；
- 中间运行只允许审计进度、NaN/Inf、update records、GPU 使用与工件完整性，不得根据中间效果调参。

三个训练进程允许并行共享单张 RTX 5060 Laptop GPU，但启动前必须验证显存余量；出现 CUDA OOM 时只允许改为顺序运行，不允许减小 batch、网络或训练量。

## 5. CUDA 合同

独立环境为 `.venv-cuda`，不得替换原 CPU `.venv`。训练启动前必须记录并验证：

- `torch.cuda.is_available() == True`；
- GPU 名称、compute capability、PyTorch 与 CUDA runtime；
- 69→256→256→2 网络真实完成 CUDA forward/backward/Adam step；
- CPU/GPU 同权重前向最大绝对误差 `< 1e-4`；
- MuJoCo 版本为 3.2.3；
- 每个训练 run 的 resolved device 必须为 `cuda`，否则 fail closed。

当前验证环境：PyTorch `2.7.0+cu128`、CUDA runtime `12.8`、RTX 5060 Laptop GPU，compute capability `(12, 0)`。

## 6. 训练后 Gate

首先只使用冻结 validation，不读取 Tracking development outcome 参与 checkpoint 选择。必须同时报告：

1. 三个 seeds 均完成 120k、update records 非空且无 NaN/Inf；
2. 30k 与 120k 的 validation success、collision、return、path completion、cross-track 和 goal distance；
3. critic disagreement、entropy/alpha、Actor action distribution 与 BC anchor error；
4. seed-cluster 聚合及逐 scene/domain strata，不把 environment steps 当独立样本；
5. 120k 是否在至少 2/3 seeds 上优于同 run 的 30k，且不增加碰撞；
6. checkpoint 必须按预注册 validation 字典序选择，不得依据 L234/L247--L256 outcome 改选。

只有 validation Gate 表明更长训练提高 Actor/critic competence，才进入三 Tracking 地图的等预算 development Gate。若失败，保留原始结果，并据此判定“训练不足”解释未获支持；不得美化、筛 seed 或改写为正式论文结论。

## 7. 冻结启动命令

每个 seed 使用对应 L257 config，并统一执行：

```bash
.venv-cuda/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config <seed-config> \
  --initialize-actor-from results/research_platform/rl/l219_expanded_actor_seed20262193_30k_v2/checkpoints/step_000010000.pt \
  --bc-anchor-dataset results/research_platform/rl/l243_direct_control_teacher_dataset_v1 \
  --device cuda
```

不得使用 sealed seeds，不得覆盖 L243/L256 工件，不得删除负向训练。
