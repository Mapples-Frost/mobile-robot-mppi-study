# L34 有界动态修正：多训练种子复现预注册

日期：2026-07-16  
状态：在启动确认训练 seed 之前冻结。

## 1. 进入条件

开发训练 seed `20260731` 已按 L34 预注册方案完整运行 30k step。最佳检查点为
30k：固定未见域从 step 0 的 `1/6` success、`4/6` collision、`1.628 m`
平均终点距离，改善到 `4/6` success、`2/6` collision、`0.927 m`。逐 episode
配对为 3 个 success gain、0 个 success loss，collision 净减少 2；产物完整性审计通过。

该结果只触发复现，不直接升级为论文结论。

## 2. 冻结设计

追加两个独立训练 seed：

- `20260732`；
- `20260733`。

两者均使用：

- `configs/rl/sac_mppi_dynamic_history_bounded_l34.yaml`；
- 相同 L6 三帧 actor-only 初始化 checkpoint；
- 30k SAC 环境步；
- 每 5k 一次固定验证；
- 四个训练运动族、四个 seen MuJoCo 物理域；
- 两个 held-out 运动路径、一个 combined-unseen 物理域；
- 固定验证 seed base `20360731`，以 common random numbers 配对；
- `rl_prior_alpha = 0.25`；
- temporal scan guard、scan guard 与最终安全仲裁保持开启。

每个训练 seed 的 best checkpoint 只能从事先固定的
`{5k, 10k, 15k, 20k, 25k, 30k}` 中，按现有 success/collision/return 标量规则选择。
step 0 只作为配对 reference，不视为训练后候选。

## 3. 多种子开发判据

只有同时满足以下条件，才允许进入新的、尚未使用的 confirmation episode seeds：

1. 至少 `2/3` 个训练 seed 的最佳训练后检查点，相对自身 step 0 具有正的 success 净增益；
2. 至少 `2/3` 个训练 seed 的 collision 数不高于自身 step 0；
3. 汇总 18 个配对 episode 后，success gains 多于 success losses；
4. 汇总 collision regressions 不多于 collision improvements；
5. 三个训练 seed 的平均终点距离改善为正；
6. 三个 run 均有完整的配置快照、42 行固定验证、initial/best/latest checkpoint，且无 NaN/Inf；
7. 不使用 sealed confirmation seeds 调整 `rl_prior_alpha`、奖励权重或 checkpoint。

若失败，保留 seed `20260731` 为 development 结果，不报告为稳定性能。下一阶段改进必须
从训练稳定性、checkpoint 选择或动态课程中提出新的单一假设，并重新预注册。

