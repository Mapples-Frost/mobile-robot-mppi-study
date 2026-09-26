# L54：跨层 ICODE 门控闭环开发预注册（2026-07-16）

## 1. 目的

L53 只在冻结轨迹上校准了门控系数。L54 使用全新 episode seeds 检验：因果在线门
是否能把 L51 的不可部署 oracle-domain 上界转化为真实闭环收益。

## 2. 冻结条件

比较四种条件：

1. `traditional_nominal`：名义动力学；
2. `traditional_icode`：始终启用 ICODE；
3. `traditional_icode_reliability_gate`：L53 冻结的跨层门；
4. `traditional_icode_oracle_domain_gate`：仿真域标签上界，仅供比较。

全部条件关闭 RL prior 和 memory；感知、scan guard、安全仲裁、路径、MPPI 参数和
MuJoCo 物理参数保持一致。三独立 ICODE checkpoint、三条路径、两个物理域、十个
新 seeds，共 720 episodes。开发 seeds 为 21660731–21660740；
21660741–21660750 保持 sealed。

## 3. 冻结在线门参数

L53 candidate 24：forgetting factor 0.95、minimum samples 8、confidence z 0、
evidence off/on 为 -0.05/0.05、rise/fall 为 0.25/0.50；delay-ratio off/on 为
0.50/0.80。不得根据 L54 结果修改这些参数后重跑同一 seeds。

## 4. 完整性检查

- 期望 720 episodes，无缺失、重复或额外 key；
- 不得使用任何 protected/sealed seed；
- matched-delay 中跨层门的状态、控制和安全轨迹必须逐步等同 nominal；
- oracle 分支必须逐步等同其预注册 comparator；
- `samples < 8` 时 alpha 必须为 0；
- matched delay context alpha 必须为 0，long delay 必须为 1；
- 非有限值、checkpoint 身份不一致或轨迹 step 不连续均判完整性失败。

## 5. 主要闭环门槛

跨层门相对 nominal 必须同时满足：

1. issued-control jerk reduction 的层级 bootstrap 95% CI 下界 > 0；
2. applied-control jerk reduction 的层级 bootstrap 95% CI 下界 > 0；
3. applied jerk reduction 在 3/3 模型块中为正；
4. 相对 oracle 上界至少保留 50% applied-jerk reduction，且对应 retention margin
   的 95% CI 下界 > 0；
5. long-delay mean alpha ≥ 0.50，matched-delay mean alpha ≤ 0.01；
6. 成功数不减少，碰撞数不增加；
7. cross-track RMSE 相对增幅不超过 5%；
8. completion ratio 差不低于 -0.01；
9. 平均规划耗时不超过 50 ms。

路径长度为次要诊断指标，不作为 L54 的成败门槛。所有 bootstrap 以模型块为最高
重采样层，episode seed 为实验单位，控制 step 不作为独立样本。

## 6. 解释边界

L54 是开发实验。即使通过，也必须使用 sealed seeds 完成独立确认；不能据此声称实车
有效。命令延迟比例当前由控制器的已标定参数提供，后续实车必须给出独立的时间戳测量
流程及误差范围。

