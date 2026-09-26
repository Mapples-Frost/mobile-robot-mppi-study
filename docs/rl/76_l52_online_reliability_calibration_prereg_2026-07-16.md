# L52：因果在线残差可靠性门校准预注册（2026-07-16）

## 1. 研究问题

L51 已证明：如果控制器预先知道物理域，并仅在长控制延迟域启用 ICODE，
则可以在不降低成功率、不增加碰撞且不过度损害跟踪精度的前提下，显著降低
issued-control jerk 与 applied-control jerk。L51 使用仿真器域标签，属于不可部署的
性能上界，不是最终方法。

L52 回答更严格的问题：只利用**过去已经完成的状态转移**，能否判断残差模型相对
名义模型是否更可信，并据此连续调节残差注入系数？运行时禁止访问域名称、未来状态、
MuJoCo 隐藏状态、任务成败或 oracle residual。

## 2. 在线可用证据

对已完成的区间 \((x_{t-1},u_{t-1},x_t)\)，分别计算名义模型和未门控 ICODE
的一步预测误差。仅在速度通道上计算归一化均方误差：

\[
e_{\mathrm{nom}}=\frac{1}{2}\left[
\left(\frac{\hat v_{\mathrm{nom}}-v_t}{0.25}\right)^2+
\left(\frac{\hat\omega_{\mathrm{nom}}-\omega_t}{0.60}\right)^2
\right],
\]

\[
e_{\mathrm{res}}=\frac{1}{2}\left[
\left(\frac{\hat v_{\mathrm{res}}-v_t}{0.25}\right)^2+
\left(\frac{\hat\omega_{\mathrm{res}}-\omega_t}{0.60}\right)^2
\right].
\]

相对改进证据为

\[
q_t=\operatorname{clip}\left(
\frac{e_{\mathrm{nom}}-e_{\mathrm{res}}}
{e_{\mathrm{nom}}+e_{\mathrm{res}}+\varepsilon},-1,1\right).
\]

门控器维护指数加权均值、方差、有效样本量和均值下置信界，再把下置信界映射为
\(\alpha_t\in[0,1]\)。冷启动阶段严格令 \(\alpha_t=0\)，即退化为 nominal。

## 3. 数据边界与防泄漏约束

- 校准输入：L51 的 `traditional_nominal` 开发轨迹；
- 模型块：3 个独立训练种子的 L49 ICODE checkpoint；
- 场景：3 条预注册路径；
- 物理域：matched delay 与 long delay；
- episode seeds：L51 的 10 个开发种子；
- L52 不使用 L51 闭环控制效果来拟合神经网络；
- 域标签只用于离线阈值选择，不进入运行时门控输入；
- L51 sealed seeds 与后续 L52 sealed seeds 均不得用于本次校准。

控制延迟的区间平均命令只使用已发出的历史命令和控制器已知的延迟配置重建；
不读取 MuJoCo 实际电机力矩或未来动作。

## 4. 冻结的候选网格

候选集合为完整笛卡尔积，共 36 组：

- forgetting factor：0.90、0.95、0.98；
- minimum samples：8、12；
- confidence z：0.0、0.5、1.0；
- off threshold：0.0；
- on threshold：0.15、0.30；
- rise rate：固定 0.25；
- fall rate：固定 0.50；
- active 判据：\(\alpha\geq0.5\)。

## 5. 预注册选择门槛

每个候选必须同时满足：

1. 三个模型块中，long-delay 最低平均 \(\alpha\geq0.55\)；
2. 三个模型块中，matched-delay 最高平均 \(\alpha\leq0.35\)；
3. 三个模型块中，long 与 matched 的最小平均 \(\alpha\) 差至少为 0.25；
4. long-delay 最低 active fraction 至少为 0.50；
5. matched-delay 最高 active fraction 不超过 0.35；
6. 冷启动期不得出现非零 \(\alpha\)。

若多个候选通过，依次选择：最坏模型块分离度最大、matched 平均 alpha 最小、
long 平均 alpha 最大、候选编号最小者。若无候选通过，本轮必须判失败并保持
fail-closed nominal，禁止为了得到正结果临时放宽门槛。

## 6. 解释边界

通过本门槛只说明历史一步创新量具有区分这两个已定义物理域的潜力。它不等于闭环
控制有效，也不等于已泛化到实车或未见扰动。只有冻结阈值后，在全新 episode seeds
上完成在线闭环实验，才能评价 causal online gate 的控制价值。

