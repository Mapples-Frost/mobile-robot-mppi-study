# L37：有界 RL × ICODE 正交因子实验预注册

日期：2026-07-16  
状态：在运行 L37 前冻结。

## 1. 研究问题

L37 在 L36 校准通过的动态 benchmark 上分别检验：

1. bounded RL 是否改善 MPPI sampling prior；
2. ICODE 是否改善 MPPI rollout dynamics；
3. 两者是否存在正交互补，而不是一个模块掩盖另一个模块。

四个条件为：traditional-nominal、traditional-ICODE、bounded-RL-nominal、
bounded-RL-ICODE。RL 固定 `alpha=0.25`，不启用事后 confidence/LCB gate；memory 关闭，
temporal scan guard、scan guard、local obstacle layer 与安全仲裁全部开启。

## 2. 实验单元

- 3 个 RL training seed 与 3 个 ICODE training seed 一一组成 model block；
- 3 个经传统 baseline 校准的动态场景；
- 1 个 combined-unseen MuJoCo 物理域；
- 5 个全新 development episode seed：`20660731`–`20660735`；
- 4 个正交条件。

总计 `3 × 3 × 1 × 5 × 4 = 180` 个闭环 episode。每个 model block 内使用 common
random numbers 配对，运行顺序随机化。控制 step 不是独立重复。

## 3. 主要估计量

- RL 主效应：在 nominal 与 ICODE 两层内分别计算 bounded-RL − traditional；
- ICODE 主效应：在 traditional 与 bounded-RL 两层内分别计算 ICODE − nominal；
- 组合相对 baseline：bounded-RL-ICODE − traditional-nominal；
- 交互项：

$$
\Delta_{\mathrm{int}}
=
(Y_{\mathrm{RL,ICODE}}-Y_{\mathrm{traditional,ICODE}})
-
(Y_{\mathrm{RL,nominal}}-Y_{\mathrm{traditional,nominal}}).
$$

主要终点依次为 success、collision、final goal distance；同时报告最小间隙、jerk 和规划
耗时。置信区间使用 model-block/episode 两阶段 bootstrap。

## 4. Development Gate

只有同时满足以下条件才允许设计新的独立 confirmation：

1. 180 个 episode 完整，无重复、保护 seed 泄漏或 NaN/Inf；
2. combined 条件在至少 2/3 个 model block 中 success 最优或并列最优，且 collision 最低或并列最低；
3. combined 相对 traditional-nominal 的净 success gain 至少为 3；
4. combined 的 collision regressions 不多于 improvements；
5. bounded RL 跨两种 dynamics 汇总后的净 success gain 不为负；
6. ICODE 跨两种 policy 汇总后的平均 final-distance improvement 不为负；
7. success 交互项不为负；或者 final-distance 的正交互至少为 0.05 m。

若失败，不打开 sealed seed，不把结果表述为“联合方法有效”，而是按主效应/交互项定位
是 RL 泛化、ICODE 模型域还是模块耦合失败。

