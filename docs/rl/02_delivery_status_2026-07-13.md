# RL 全链路交付状态（2026-07-13）

## 已实际验证

- 新增 RL Python 文件全部通过 `py_compile`；
- 全仓测试：`142 passed`；
- MuJoCo SAC smoke：60 environment steps、4 truncated episodes；
- SAC Actor/Critic 实际发生参数更新；
- `best.pt` 和 `latest.pt` 均已保存，含 replay 与 normalizer；
- checkpoint inspect 成功；
- checkpoint 经统一 `ExperimentRunner` 自动加载并完成确定性推理；
- MPPI / RL-MPPI / gated RL-MPPI 三方法 smoke 消融成功输出 CSV/JSON。

Smoke checkpoint 位于（结果目录已被 `.gitignore` 排除）：

```text
results/research_platform/rl/full_stack_smoke_20260713/checkpoints/best.pt
```

## Smoke 数字（不是科研结果）

- 训练步数：60；
- 验证回合长度：15 control steps；
- 验证 success rate：0；
- 验证 collision rate：0；
- 单 seed 推理 final goal distance：约 3.916 m；
- 单 seed推理 RL gate alpha mean：1.0。

15 个控制周期不足以到达 `(3, 3)`，所以 0 成功率不代表训练失败，也不能和之前完整
360-step baseline 数据比较。

## 尚未宣称完成

- 未运行 200k-step 正式训练；
- 未选择最终 reward、网络宽度或 knot 数量；
- 未得到 RL 优于传统 MPPI 的正式统计结论；
- OOD gate 尚未校准，只是已实现可消融的经验距离机制；
- 未部署实车策略；
- 未把动态障碍支持误写为本轮已完成。
