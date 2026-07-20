# L220 source-relative competence 三地图探针结果

## 结论

L220 探针部分通过，但不能进入六地图完整 Gate。

- 三张地图、两方法、seed 91001，共 6 个 qualification 回合；
- MuJoCo 3.2.3，K=30，1 iteration，全部 0 碰撞；
- Actor competence 不再恒为 1，guided/Gaussian yield 均产生非零记录，证明遗漏配置已修复；
- cylinder forest：Full 成功，从 ICODE 的 659 步降到 542 步，安全干预从 356 降到 255；
- serpentine：Full 完成度 0.2481，高于 ICODE 的 0.2406，但差值很小，仍未到达；
- opposed-U：Full 完成度 0.1291，远低于 ICODE 的 0.9689，仍未到达。

## 机制诊断

opposed-U 中：

- Full proposal authority 均值为 0.9391；
- Actor guided elite yield 均值为 0.3155；
- Gaussian elite yield 均值为 0.1312；
- 因而 source-relative competence 均值仍为 0.9608。

这不是 competence 没有更新，而是短 horizon 内 Actor 候选确实更容易进入 elite set，但长期闭环路径进入局部最优。由此可证伪“只用短时域 elite yield 即可可靠判断长期 proposal 质量”的假设。

## 决策

保留 L220 作为负向开发结果。L221 增加 ICODE 对 Actor mean 与 trusted baseline mean 的同状态反事实路径进度比较，并用连续 authority 将 Actor proposal 限制为 baseline 周围的有界修正。核心 ICODE + residual-conditioned RL prior + role-aware HSS + MPPI 不变。
