# L81：Quantile/CVaR critic 训练结果

日期：2026-07-17  
结论：**training Gate failed；未运行 deployment；未打开 sealed seeds**

## 1. 本轮问题

L81 检验以下可证伪假设：

> 将 scalar mean-Q critic 替换为 25 分位数 return distribution，并使用下尾
> 20% CVaR 训练 residual actor，能否减少“平均回报有利但少数闭环轨迹失败”的
> correction，从而让至少 2/3 独立模型块选择非零、非劣 checkpoint？

本轮相对 L78 只改变 critic 的回报表示和 actor 使用的风险值。reward、frozen BC
correction、ICODE、MPPI、MuJoCo plant、LaserScan、安全仲裁、训练步数和验证场景均保持不变。

## 2. 工件完整性

三个独立模型块均满足：

- 完成 30,000 environment steps；
- interrupted episode 为 0；
- initial、5k–30k、latest 和 selected checkpoint 均可加载；
- 5k actor 与 initial 精确相同，10k 才开始更新；
- frozen BC actor hash 全程不变；
- 三个 replay scene groups 全部非空；
- 28,001 条 update records 完整，其中 20,001 条包含 actor 更新；
- quantile enabled 恒为 1，quantile count 恒为 25，CVaR fraction 恒为 0.20；
- quantile spread 始终为有限正数；
- Q expectation、Q CVaR 和 policy Q CVaR 均为有限数；
- config snapshot、训练 seed、validation seed、git SHA、CSV 和 checkpoint 元数据完整。

正式审计工件：

`results/research_platform/rl/l81_static_quantile_cvar_training_audit_20260717_v1/`

## 3. 训练 Gate 结果

| block | SAC seed | selected step | success gain/loss | collision regression | mean distance improvement | eligible |
|---:|---:|---:|---:|---:|---:|---|
| 0 | `20260784` | `30k` | `0 / 0` | `0` | `+0.184204 m` | yes |
| 1 | `20260785` | `0` | `0 / 0` | `0` | `0.000000 m` | no |
| 2 | `20260786` | `0` | `0 / 0` | `0` | `0.000000 m` | no |

预注册要求至少 2/3 blocks 选择非零且满足非劣性约束，实际为 **1/3**。因此：

- `minimum_nonzero_selected_blocks = false`；
- `selected_checkpoint_noninferiority = false`；
- `gate_passed = false`。

失败不是工件错误造成的。所有结构、checkpoint、冻结边界和量化诊断检查均通过。

## 4. checkpoint 轨迹

下表给出每个非零 checkpoint 相对 step 0 frozen-BC reference 的固定配对结果：

| seed | step | success delta | collision delta | mean goal-distance improvement |
|---:|---:|---:|---:|---:|
| `20260784` | `10k` | `-1` | `0` | `-0.064643 m` |
| `20260784` | `15k` | `-1` | `0` | `-0.072330 m` |
| `20260784` | `20k` | `0` | `0` | `-0.003444 m` |
| `20260784` | `25k` | `0` | `0` | `-0.106028 m` |
| `20260784` | `30k` | `0` | `0` | `+0.184204 m` |
| `20260785` | `10k` | `-2` | `0` | `-0.243121 m` |
| `20260785` | `15k` | `-2` | `0` | `-0.085884 m` |
| `20260785` | `20k` | `0` | `0` | `-0.115256 m` |
| `20260785` | `25k` | `0` | `0` | `-0.054229 m` |
| `20260785` | `30k` | `+1` | `0` | `-0.070185 m` |
| `20260786` | `10k` | `-3` | `0` | `-0.098768 m` |
| `20260786` | `15k` | `-1` | `0` | `+0.036538 m` |
| `20260786` | `20k` | `-2` | `0` | `+0.022253 m` |
| `20260786` | `25k` | `-2` | `0` | `+0.103201 m` |
| `20260786` | `30k` | `-1` | `0` | `+0.144269 m` |

表现呈现清晰的跨模型块权衡：

- block 0 在 30k 同时保持成功和安全，并改善终点距离；
- block 1 在 30k 增加一个成功，但平均终点距离恶化；
- block 2 在 30k 改善终点距离，但损失一个成功；
- 三个 block 在全部 checkpoint 上均没有碰撞回归。

这不是“完全没有学习”，而是**正收益没有跨独立初始化稳定复现**。

## 5. Quantile/CVaR 机制是否真正工作

量化 critic 没有退化为空操作。三块训练结束时：

| seed | final quantile spread | mean quantile spread | final Q mean | final lower-tail CVaR |
|---:|---:|---:|---:|---:|
| `20260784` | `6.0602` | `2.3903` | `-65.9114` | `-67.5877` |
| `20260785` | `5.9997` | `3.1251` | `-67.3360` | `-69.2542` |
| `20260786` | `2.3113` | `1.3925` | `-69.2397` | `-69.8830` |

CVaR 始终低于对应均值，且回报分布具有非零宽度。因此失败的含义不是实现错误或
分布坍缩，而是：

> 在当前数据、reward 和共享 residual-prior 参数化下，下尾风险建模仍不足以让策略收益
> 跨 BC/ICODE/SAC 初始化稳定复现。

## 6. 科学结论

L81 不否定 ICODE 残差动力学，也不否定 conventional MPPI、安全仲裁或 MuJoCo
研究平台。它否定的是一个更窄的主张：

> 当前 always-on shared SAC correction prior 已具备可复现、可进入部署验证的优势。

结合 L79、L80 和 L81：

1. frozen BC + ICODE + MPPI reference 已较强，共享 correction 的可改善空间有限；
2. scalar critic、worst-scene actor aggregation 和 quantile/CVaR critic 都出现明显的
   seed/scene 异质性；
3. 风险建模改善了一个模型块，但没有达到独立模型块复现要求；
4. 继续扫描 CVaR fraction、quantile count、Huber kappa 或 checkpoint 阈值会扩大
   研究者自由度，不能作为严谨的下一步；
5. 当前证据支持保留 RL 基础设施与负消融，但不支持把 always-on RL prior 作为论文
   已验证的核心收益。

## 7. 冻结决策与下一主线

- 不运行 deployment seeds `22201301`–`22201305`；
- 不打开 sealed seeds `22201311`–`22201315`；
- 不对 CVaR fraction、quantile count 或 Huber kappa 做事后网格搜索；
- 保留 quantile/CVaR critic 作为可复现实验模块和结构消融；
- 停止当前 always-on shared RL correction prior 的连续调参主线；
- RL 下一阶段转为 **OOD/复杂区域探索与 residual 数据采集**；
- 控制主线继续使用 **ICODE + conventional MPPI + safety arbitration**；
- 只有新的探索数据闭环在独立 model-block 和新鲜 evaluation seeds 上通过预注册 Gate，
  才重新讨论 RL 是否进入在线 sampling prior。

该转向与导师建议一致：在简单场景中保留传统 MPPI 的强基线，在复杂/OOD 区域利用 RL
的探索能力，同时把每个模块的作用通过独立消融验证，而不是把过多机制同时堆进控制器。

## 8. 解释边界

- 本轮是 development-training Gate，不是正式论文统计结果；
- 15 条固定验证轨迹用于 checkpoint 选择，不是 sealed test；
- 1/3 正向模型块不能外推为总体收益；
- 未运行 deployment，因此没有新 seed 泛化结论；
- 未测试动态障碍、实车部署或理论稳定性；
- 不声称 CVaR 是校准概率、碰撞概率界或安全保证。
