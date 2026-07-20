# L220 Actor 相对候选胜率修复开发协议

## 固定核心方法

本轮不改变论文主线：

```text
Value-Consistent ICODE
  + Residual-Conditioned RL Prior
  + role-aware Reliability-Weighted Value/HSS
  + MPPI
```

ICODE checkpoint、L219 Actor checkpoint、残差上下文、MPPI cost/rollout、LaserScan、local obstacle layer、scan_guard、MuJoCo plant 与控制仲裁均保持不变。

## 唯一候选修改

启用代码中已有但 L219 漏配的 source-relative competence：

1. 每拍记录 Actor-guided 与 Gaussian 候选进入 elite set 的数量；
2. 用各自 candidate opportunities 归一化为 elite yield；
3. 将 yield ratio 映射成 `[0,1]` competence；
4. 以指数滑动方式更新，并在下一控制拍乘入 Actor authority。

全部信号来自当前拍已经完成的 MPPI 候选评价，不访问 simulator future、障碍真值或 sealed 结果。

## 开发顺序

候选 A 先在 seed 91001 的三张有诊断价值的地图运行：

- opposed-U：L219 的主要错误授权案例；
- cylinder forest：部分回退案例；
- serpentine：Actor 具有非零 authority、但尚未形成明显增益的旧失败地图。

比较 ICODE-MPPI 与 Full Proposed，预算固定为 K=30、1 iteration、max 900。探针通过条件：

- 0 碰撞；
- competence 更新次数大于 0，guided/Gaussian yield 不再同时恒为 0；
- opposed-U 不再出现 `3/3 -> 0/3` 类型的明显回退趋势；
- Full 在至少一张地图不低于 ICODE 的完成度，且 Actor authority 不被永久固定为 0 或 1。

探针只用于工程机制资格确认，不作为正式结果。通过后才运行三个 development seeds 的六地图 Gate；失败则保留结果并继续开发，不使用 sealed seeds。
