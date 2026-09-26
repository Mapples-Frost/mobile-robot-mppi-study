# L52 失败分析与 L53 跨层可靠性门预注册（2026-07-16）

## 1. L52 的否定性结果

L52 的 36 个预注册候选全部未通过。最优诊断候选在三个独立模型块上的最坏
long/matched 平均 alpha 分离仅为 0.0283，远低于预注册的 0.25。该结果保留为
正式否定性证据，不修改 L52 门槛，也不把 L52 重新标记为成功。

开发集机制诊断显示，两域的一步创新统计高度重合。例如 block 0：

- long-delay nominal normalized MSE 均值 0.00414，matched 为 0.00433；
- long-delay ICODE normalized MSE 均值 0.00337，matched 为 0.00355；
- 相对改进均值分别为 0.0761 与 0.0705。

其余两个模型块呈相同趋势。这说明“残差是否比 nominal 准”可以衡量模型可信度，
但不能单独衡量“残差在当前执行链中是否值得注入 MPPI”。

## 2. L53 的跨层定义

L53 将两个来源严格分开：

1. **模型层证据**：仅由已完成转移计算的 ICODE 相对预测改进下置信界；
2. **执行层上下文**：命令延迟与控制周期的比例
   \(c_{\tau}=\tau_{\mathrm{cmd}}/\Delta t\)。

最终残差系数为

\[
\alpha_t=\alpha_{\mathrm{model},t}\,\alpha_{\mathrm{delay}}(c_{\tau}).
\]

其中 \(\alpha_{\mathrm{delay}}\) 在 \(c_{\tau}\leq0.50\) 时为 0，在
\(c_{\tau}\geq0.80\) 时为 1，中间线性插值。

这里的命令延迟不是 MuJoCo 隐藏标签。当前仿真从控制器已使用的
`planner.command_delay_s` 读取；实车迁移时必须由 `/cmd_vel` 时间戳、驱动反馈和
odom 响应离线标定或在线估计。若无法获得可信延迟估计，应 fail closed，而不是
假定自己处于高延迟域。

## 3. 为什么这一修改不是事后放宽 L52

L52 的科学问题是“模型创新能否单独区分控制受益域”，答案为否。L53 增加一个不同
传感/执行层的可测变量，检验的是新的、明确命名的跨层机制。L52 的失败记录、原网格、
原门槛和结果文件均保留，不覆盖、不删除。

## 4. 冻结校准网格

在执行任何 L53 校准前冻结以下 48 个候选：

- forgetting factor：0.90、0.95；
- minimum samples：8、12；
- confidence z：0.0、0.5；
- evidence off threshold：-0.05、0.0；
- evidence on threshold：0.05、0.10、0.15；
- rise/fall rate：0.25/0.50；
- delay context off/on threshold：0.50/0.80。

数据边界仍为 L51 nominal 开发轨迹；不使用 sealed seeds。

## 5. 预注册门槛

候选必须同时满足：

- 三模型块最坏 long-delay mean alpha ≥ 0.55；
- 三模型块最坏 matched-delay mean alpha ≤ 0.05；
- 三模型块最小 long-minus-matched alpha ≥ 0.50；
- long-delay active fraction ≥ 0.50；
- matched-delay active fraction ≤ 0.05；
- 冷启动违规为 0；
- 输入轨迹、checkpoint、episode key 和 seed 审计完整。

若有多个候选通过，沿用 L52 的冻结排序规则。通过只允许进入新种子闭环开发实验，
不能直接声称控制性能成立。

## 6. 当前方法边界

L53 是“历史模型可靠性 × 已测执行延迟”的可部署工程机制，不是从观测中自动发现任意
OOD 域的通用定理。后续若以学习式 delay estimator 替代标定值，必须另设标定集、
不确定性输出和独立确认，且不得把仿真域标签作为在线输入。

