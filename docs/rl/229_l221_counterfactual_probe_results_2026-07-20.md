# L221 ICODE 反事实 proposal gate 三地图探针结果

## 结论

L221 探针通过，允许进入三个 development seeds 的六地图完整 Gate。该结论仍是工程开发资格结论，不是 sealed 论文结论。

| 场景 | 方法 | 成功 | 步数 | 路径完成度 | 安全干预 | 碰撞 |
|---|---|---:|---:|---:|---:|---:|
| opposed-U | ICODE-MPPI | 1 | 677 | 0.9689 | 373 | 0 |
| opposed-U | Full Proposed | 1 | 586 | 0.9682 | 296 | 0 |
| cylinder forest | ICODE-MPPI | 1 | 659 | 0.9633 | 356 | 0 |
| cylinder forest | Full Proposed | 1 | 582 | 0.9638 | 273 | 0 |
| serpentine | ICODE-MPPI | 0 | 900 | 0.2406 | 672 | 0 |
| serpentine | Full Proposed | 0 | 900 | 0.2464 | 641 | 0 |

Full 在两张可达地图保持成功，并分别减少 91/77 个控制步与 77/83 次安全干预；serpentine 尚未到达，但完成度和安全负担方向均改善。三张地图全部零碰撞。

## 机制审计

- counterfactual gate 在三个 Full 回合均 100% 启用；
- counterfactual authority 均值分别为 forest 0.1128、opposed-U 0.3361、serpentine 0.3919，最小值均到 0；
- 最终 proposal authority 均值分别为 0.0213、0.0492、0.0664；
- source-relative competence 仍保留在线更新，并未被替换；
- Actor 没有被永久关闭，也没有继续完全覆盖 baseline。

结果支持：ICODE 的反事实 rollout 可以纠正只看短时域 elite yield 造成的错误授权，并把 residual-conditioned Actor 变成 trusted baseline 周围的有界 proposal。

## 公平性修正

完整 Gate 前补充了强制边界：counterfactual gate 只有 adaptive HSS 启用时才生效，Simple combination 保持 HSS 关闭。测试保证该机制不会泄漏到 Simple 对照组。此修正不改变上述 Full 探针行为。
