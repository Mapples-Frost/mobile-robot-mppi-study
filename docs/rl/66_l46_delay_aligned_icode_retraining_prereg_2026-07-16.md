# L46 delay-aligned structure-preserving ICODE 重训练预注册

日期：2026-07-16  
状态：训练前冻结。

## 唯一数据语义变化

L40/L45 使用 safety-executed command 作为当前控制输入，但 MuJoCo plant 会在 `command_delay` 后才激活该命令。
L46 将训练输入改为数据中已记录的 `average_applied_control`：即每个 100 ms 物理区间内，经过命令队列延迟后实际送入轮速执行器的平均命令。
随后重新计算 nominal derivative 和 residual target。

这不是把 true plant 暴露给在线 planner。部署时 planner 只使用已知的硬件延迟参数和上一条已执行命令，将候选命令序列转换为相同的区间平均控制：

```text
u_effective[k] = delay/dt * u[k-1] + (1-delay/dt) * u[k]
```

其中第一步的 `u[k-1]` 是经过 safety arbitration 的上一条真实命令。当前实现明确限制 `0 <= delay <= dt`。

## 其余冻结项

- structure mask `[0,0,0,1,1]`；
- H36、RK4、网络、损失权重、训练/验证/test/unseen episode split 全部沿用 L45；
- training seed：20261001、20261002、20261003；
- best checkpoint 仍按 validation multi-step RMSE 选择。

离线准入规则与 L45 相同。只有 3 个 checkpoint 在 test/unseen 的总体 H36 rollout RMSE 均优于 nominal，且至少 2/3 在两套 split
同时改善 H36 position 与 heading RMSE，才允许进入新的 delay-aware 闭环开发实验。

