# L56 数据结果与 L57 ICODE 离线训练预注册（2026-07-16）

## 1. L56 数据 Gate 结果

固定隐藏 MuJoCo plant 的正式数据共 48 episodes、12,074 control steps。机器 Gate 全部通过：

- 48/48 success，0 collision；四个场景均为 12/12 success；
- applied `v` 的 95% 分位为 0.47386 m/s；
- applied `|omega|` 的 95% 分位为 0.55105 rad/s；
- state `v`/`omega` 标准差分别为 0.15093 m/s 与 0.20823 rad/s；
- issued action 上限饱和比例均为 0，safety override fraction 为 0；
- 48 个 resolved config 共享唯一 plant/action/sensor contract；
- episode/step key 无重复，无 sealed/protected seed 泄漏。

该结果只证明数据采集可复现且激励充分，不证明 residual model 已经有效。

## 2. 冻结数据集

数据集目录为 `results/research_platform/datasets/l56_high_dynamic_fixed_plant_v1`：

- train：24 episodes，6,053 transitions；
- validation：3 episodes，744 transitions；
- test：9 episodes，2,261 transitions；
- unseen：12 episodes，3,016 transitions，全部来自未进入训练的 reverse-S 路径。

四个 split 以完整 `(scene, physics_domain, seed)` 为单位互斥。归一化只使用 train；
checkpoint 选择只使用 validation。test 与 unseen 不参与调参或 early stopping。

冻结 SHA-256：

- train：`5583b183bd60e4db51b49fcef225b6b8fb1e22a0291bd338bc3c027aaff2159c`；
- validation：`c2bc6b75a259782cff0659c8f03dcb8bcee9a753306861820ec7f5d346b54d7d`；
- test：`100dea8847e6e8178a551b84b484cb29cc57ff2780d581542554e4a6bab92c96`；
- unseen：`aeede24d5765920fc9de1c573fb79368f2d92d057d245b0d97932438efb325fc`。

## 3. L57 模型与训练冻结项

训练 3 个独立 ICODE model blocks，seeds 为 20261201、20261202、20261203。模型保持论文公开描述的
control-affine residual：`f_theta(x) + G_theta(x)u`，只修正 dynamic-unicycle 的 `dv/dt` 与
`domega/dt`，不学习已知的 pose kinematics。网络为 64-64 Softplus，RK4，H=36 multi-step loss；
最多 120 epochs，batch size 512，patience 20，按 validation H36 multi-step RMSE 选 best checkpoint。

本实现复现的是 ICODE-MPPI 论文公开的 control-affine residual structure，不声称复现原始 ICODE
论文的全部理论保证，也不声称具备稳定性、收缩性或收敛性证明。

## 4. L57 离线 Gate

每个 checkpoint 在 test 与 unseen 上使用所有 contiguous windows 评估 H=1,5,10,20,36，禁止仅取
前 512 个 windows。训练开始前冻结如下通过条件：

1. 3 个 training seeds、模型类型、residual mask 与数据 SHA-256 完整一致；
2. test/unseen 的 active derivative RMSE 在 3/3 blocks 中均优于 nominal；
3. test 与 unseen 的 H36 rollout RMSE 至少各有 2/3 blocks 优于 nominal；
4. mean test H36 rollout RMSE reduction 不低于 10%；
5. mean unseen H36 rollout RMSE reduction 不低于 10%；
6. mean test/unseen H36 endpoint position RMSE reduction 均不低于 5%；
7. 所有指标有限，评估窗口数与 artifact 完整。

通过该 Gate 只允许进入 closed-loop MPPI 开发实验；它本身不是闭环控制收益证据。若失败，保留结果并
先诊断数据、模型结构或 loss，不直接进入闭环挑选有利 seed。
