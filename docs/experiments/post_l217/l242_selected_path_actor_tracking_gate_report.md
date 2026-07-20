# L242 冻结选中 Path Actor 的 Tracking Gate 报告

日期：2026-07-21  
性质：development / qualification，单一 development seed，不是论文确认性结果

## 1. 完整性与来源

- 完成 3 scenes × 2 arms = 6 个唯一 MuJoCo 回合；
- methods：`icode_mppi`、`full_proposed`；
- scene：Hairpin、S-Chicane、Infinity（L239 `W=4.0D` 版本）；
- physics：`nominal_seen`；seed：`923301001`；qualification=`1`；
- Git SHA：`65e03747146c9b6e3c35d14f6ca87e3283ba246a`；
- coupled Actor SHA256：
  `150a2d7ba1e4ab125fcb94e614810acb33e86f549b3eaa310e6120cd02440982`；
- MuJoCo Python package：3.2.3；
- 每回合均存在 `config_resolved.yaml`、`trajectory.csv`、`metrics.json`、
  `provenance.json`；
- Windows 日志未发现 Traceback、Exception、NaN 或 Inf。

每个方法每拍总 rollout 预算均为 100。需要特别说明：paper-style Full 使用
50 candidates × 2 iterations；standard ICODE arm 在 runner 中使用一次 100-sample
batch。因此两者 rollout 数相等，但优化器内部迭代结构不同。这是既有 arm 定义，
不能写成二者都执行 50×2。

## 2. 结果

| scene | method | success | collision | completion | boundary steps | min boundary margin (m) | proposal authority | guided elites | planner mean (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Hairpin | L242 Full | 0 | 0 | 0.1077 | 0 | 0.0215 | 0.0922 | 259 | 508.5 |
| Hairpin | ICODE | 0 | 0 | 0.1118 | 0 | 0.5630 | — | 0 | 186.5 |
| Hairpin | L239 old Full | 0 | 0 | 0.1061 | 0 | 0.0685 | 0.1281 | 0 | 452.3 |
| S-Chicane | L242 Full | 0 | 0 | 0.1758 | 1 | -0.0057 | 0.0837 | 239 | 437.4 |
| S-Chicane | ICODE | 0 | 0 | 0.1827 | 0 | 0.5618 | — | 0 | 195.3 |
| S-Chicane | L239 old Full | 0 | 0 | 0.1805 | 1 | -0.0084 | 0.0680 | 0 | 448.0 |
| Infinity | L242 Full | 0 | 0 | 0.1025 | 0 | 0.0998 | 0.0610 | 110 | 494.8 |
| Infinity | ICODE | 0 | 0 | 0.1084 | 0 | 0.5683 | — | 0 | 188.2 |
| Infinity | L239 old Full | 0 | 0 | 0.1057 | 1 | -0.0027 | 0.1149 | 0 | 451.7 |

汇总：

| method | mean completion | successes | collisions | boundary steps | mean planner ms |
|---|---:|---:|---:|---:|---:|
| L242 selected-path Full | 0.1287 | 0/3 | 0/3 | 1 | 480.2 |
| ICODE-MPPI | 0.1343 | 0/3 | 0/3 | 0 | 190.0 |
| L239 old-Actor Full | 0.1308 | 0/3 | 0/3 | 2 | 450.7 |

## 3. Gate 判定

**L242 Gate 未通过。**

- 新 Full 相对旧 Full 的平均完成度变化为 `-0.0021`，未达到预注册的
  `+0.02`；
- 三个场景的 completion 变化分别为 `+0.0016 / -0.0047 / -0.0032`；
- boundary violations 从 2 降到 1，是局部安全改善；
- proposal authority 平均 0.079，guided elites 合计 608，说明 RL prior 确实被
  非平凡地使用，而不是完全退化为 Gaussian MPPI；
- 但新 Full 的平均完成度仍比 ICODE-MPPI 低 0.0056，并多一个 boundary
  violation；
- 新 Full 比旧 Full 平均慢约 29.6 ms/step，比 ICODE-MPPI 慢约 290.2 ms/step。

因此不能把“Actor 被使用”误写成“Actor 有效”。本轮证据支持的准确表述是：

> 冻结选择得到的初始化 Path Actor 能进入 HSS 并产生 guided elites，但没有
> 转化为 Tracking 完成度或成功率提升；当前训练后的 path-conditioned SAC
> 候选也未通过 validation 选择。

## 4. 对论文主线的影响

该结果不否定 Value-Consistent ICODE；本轮 ICODE arm 在三场景中均保持零碰撞、
零 boundary violation，并且完成度略高于 Full。它否定的是当前 Actor 训练实现
已经构成有效 `Path/Residual-Conditioned RL Prior` 的说法。

因此不得注册新的 Tracking sealed benchmark。下一项 development 修复应直接
提高 Actor 的路径条件能力，而不是继续调 MPPI cost、扩大走廊或放松安全链。
建议只引入一个因素：在 SAC 更新中加入来自安全参考路径控制器的
path-following behavior-cloning anchor，再使用相同的三训练 seed、隔离 validation
和冻结选择规则验证。该因素仍属于 RL Actor 的稳定化训练，不改变 ICODE、HSS、
MPPI 或论文耦合机制。

## 5. 原始工件

- L242：`results/research_platform/rl/tracking_l242_selected_actor_*_seed923301001/`
- L239 只读对照：`results/research_platform/rl/tracking_l239_w4p0d_*_seed923301001/`
- 数值表：`docs/experiments/post_l217/tables/l242_tracking_gate_summary.csv`
- 预注册协议：`docs/experiments/post_l217/l242_selected_path_actor_tracking_gate_protocol.md`

所有失败和边界违规均保留；未使用 sealed seeds，未筛 seed，未修改冻结配置。
