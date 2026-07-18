# L17 验证 seed 审计、v1 隔离与 v2 预注册

## 1. 为什么必须重跑

L17 v1 完成后，标准 checkpoint 评估入口在同一组 validation seeds 上没有复现训练器内部验证结果。例如 training seed `20260722` 的 20k checkpoint：

- 训练器内部 validation：`15/15` success；
- 标准 checkpoint evaluator：`12/15` success。

逐层核查发现，`MppiPriorEnv.reset(seed)` 将 episode seed 传给了 MuJoCo plant 和 simulated sensors，却调用无参数 `controller.reset()`，使 MPPI sampling RNG 每个回合都回到 training-config seed。标准 `ExperimentRunner` 则会在构建 controller 前把 evaluation seed 写入 experiment config。因此，旧训练器中的 validation seed 没有完整定义一个实验重复：plant/sensors 使用 evaluation seed，MPPI sampling 使用 training seed。

该问题不改变 LaserScan、local obstacle layer、scan guard、safety arbitration 或 MuJoCo 物理模型，但会破坏训练验证与标准部署评估的一致性，也会使不同 training seeds 之间的验证条件不完全配对。

## 2. 修复内容

`MppiController.reset(seed=None)` 现在支持显式 episode seed：

- 省略 seed 时仍使用配置 seed，保持既有 runtime 行为；
- `MppiPriorEnv.reset(seed)` 显式调用 `controller.reset(seed=seed)`；
- seed 越界、负数会 fail closed；
- 单元测试验证同 seed 重放一致、不同 seed 产生不同采样流。

该修复使一个 episode seed 同时控制：

```text
initial-state noise
MuJoCo plant reset
sensor noise
MPPI perturbation sampling
```

## 3. v1 数据质量审计

三个 v1 training runs 的文件均可读取：每个 `updates.csv` 有 18,001 行、27 列，每个 `validation_episodes.csv` 有 75 行、19 列；目标 correction 字段没有缺失值，未发现重复 update row 或重复 `(global_step, scene, seed)` validation key。

actor 更新阶段的平均诊断为：

| training seed | normalized correction penalty | weighted penalty | sampled applied correction abs mean | sampled applied correction abs max |
|---:|---:|---:|---:|---:|
| 20260721 | 0.28389 | 0.07097 | 0.06905 | 0.19440 |
| 20260722 | 0.29197 | 0.07299 | 0.06996 | 0.19453 |
| 20260723 | 0.29300 | 0.07325 | 0.07078 | 0.19465 |

这说明 correction penalty 确实参与了优化且数值有限，但相对 actor loss 量级仍小；随机策略样本经常接近 correction 上限。该观察只能作为后续诊断，不能据此在 v2 中修改预注册的 `0.25` 权重。

## 4. v1 与首次 final-test 的处理

v1 的 paired selector 对三个 training seeds 都保留 step 0。随后打开的 `30301--30320` 上，BC 与 L17 selected 逐回合一致：pooled success 均为 `48/60`，collision 均为 `0/60`，平均终点距离均为 `0.425310 m`。这证明 zero-correction checkpoint 封装能够安全退回 BC，但不证明 RL 带来收益。

由于上述 seed 传播问题是在这次检查中发现的：

- v1 training/validation 结果标记为工程审计数据；
- seeds `30301--30320` 已经被查看，标记为 consumed audit set；
- 两者均不得作为 L17 v2 的未见最终测试或论文主结果；
- 原始目录保留，不删除、不覆盖，以便追溯问题。

## 5. L17 v2 的冻结方案

v2 在运行前冻结以下内容：

- 配置：`configs/rl/sac_mppi_utrap_conservative_correction_l17_v2.yaml`；
- training seeds：`20260721`、`20260722`、`20260723`；
- validation seeds：`20280701--20280715`；
- 新 sealed final-test seeds：`40301--40320`；
- 每个 seed 训练 `20,000` steps；
- actor 从 step `10,001` 更新；
- correction scale、gate、penalty weight `0.25` 全部不变；
- paired fail-closed checkpoint selection 阈值全部不变；
- BC checkpoints、场景、MPPI cost、MuJoCo plant、感知和安全链全部不变。

v2 相对 v1 唯一的实验语义改变是：episode seed 现在完整控制 MPPI sampling RNG。该改变既作用于训练回合，也作用于内部 validation，因此必须重新训练，不能只重算旧 checkpoint 的 validation。

## 6. v2 执行与停止规则

1. 通过 targeted tests 与 MuJoCo smoke；
2. 独立训练三个 seeds；
3. 由 `20280701--20280715` 冻结各自 selected checkpoint；
4. 核对训练器内部验证与标准 checkpoint evaluator 的同 seed 一致性；
5. 只有一致性通过，才首次打开 `40301--40320`；
6. final test 只比较对应 BC step 0 与 selected checkpoint；
7. final test 不用于重新选模型或调参。

若三组仍选择 step 0，结论保持为“安全退化成立，但当前 RL correction 没有优于 BC 的证据”。若出现候选被选中，则仍须通过新 sealed final test，才能把它视为可继续研究的正向信号。

## 7. 第二次一致性审计与 v3 修订

v2 修复 MPPI RNG 后，标准 evaluator 与训练器内部 validation 仍未完全一致。进一步逐字段核对发现，`MppiPriorEnv` 会将 `rl.training.initial_state_noise` 应用于所有 reset，其中包括 validation；标准 `ExperimentRunner` 则从 benchmark 的精确初始状态开始。因此，两条路径虽然使用相同 seed，却并非同一个初始条件。

为消除这个隐式差异，训练配置新增显式的 `validation_initial_state_noise`：

- training episodes 继续使用原来的小幅 initial-state jitter；
- L17 v3 validation 固定为五维全零 jitter；
- validation seed 继续控制 plant、sensors 与 MPPI sampling；
- standard evaluator 同样从 benchmark 精确初始状态开始。

v3 配置为 `configs/rl/sac_mppi_utrap_conservative_correction_l17_v3.yaml`。方法、loss、correction 上限、training seeds、validation seeds、选择阈值和尚未打开的 final seeds `40301--40320` 均不改变。v2 checkpoints 保留为 seed-contract 调试数据；v2 自动 best 不进入 final test。
