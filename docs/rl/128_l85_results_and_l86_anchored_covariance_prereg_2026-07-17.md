# L85 开发结果与 L86 锚定协方差预注册

日期：2026-07-17

## L85 实际结果

L85 单训练块完成 20,000 steps。验证 cross-track RMSE 从 step-zero 的 0.05206 m 降至 0.04072 m，成功率 100%，碰撞率 0%，但未达到预注册训练阈值 0.040 m。

随后在 4 个场景、每场景 5 个全新 seed 上完成 60 个配对回合。相对固定 `speed=(1.75,0.75)`：

- 到达时间差：-1.755 s，95% CI `[-1.945, -1.575]`；
- cross-track RMSE 差：+0.003899 m，95% CI `[+0.001580, +0.006388]`；
- success 差 0，collision 差 0。

预注册非劣界为 +0.002 m，故 L85 H1 失败。结果目录：

`results/research_platform/rl/l85_precision_constrained_development_eval_20260717_v1/`

## 机制判断

L85 证明对偶约束可把早期探索劣化拉回，但自由协方差策略仍需重新发现已知强传统设置，并在训练中产生较大的协方差振荡。继续增大惩罚不能解决参数化的探索效率和退化语义问题。

## L86 唯一改动

冻结强固定协方差为锚点：

$$
s_{\mathrm{anchor}}=(1.75,0.75),
$$

RL 只输出有界对数残差：

$$
s_t=s_{\mathrm{anchor}}\odot\exp(\delta_t),
\qquad
|\delta_{t,j}|\le\log(1.25).
$$

因此 actor 零输出、RL gate 为零或策略关闭时，都必须精确退化到强固定协方差，而不是默认 `(1,1)`。ICODE checkpoint、SAC 网络、观测、reward、对偶精度约束、场景、K、horizon 和 safety chain 均保持不变。

## 可证伪开发门槛

先训练 1 个全新 seed。只有同时满足以下条件才进入 3-block 确认：

1. 训练验证 success 不下降、collision 不上升；
2. best checkpoint 的 mean cross-track RMSE 不高于 0.040 m；
3. 使用不同于 L84/L85 的 4 场景 × 5 episode seeds 配对评估；
4. 相对固定 `speed`，RMSE 差的 95% CI 上界不超过 +0.002 m；
5. 相对固定 `speed`，到达时间差的 95% CI 上界小于 0。

若只满足精度、不满足时间优势，则锚定机制安全但没有 RL 增益；若只满足时间、不满足精度，则仍是 L84/L85 的 Pareto 权衡，不支持研究假设。

## 声明边界

L86 是带传统强基线的有界策略残差，不是安全性 theorem，也不保证约束必然满足。所有主张仍依赖独立训练块、配对物理 seed、碰撞指标和实车前的 shadow-mode 验证。
