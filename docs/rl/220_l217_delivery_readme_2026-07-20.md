# L217 复杂场景科研交付包：请先阅读

本交付包包含 2026-07-20 完成的 L217 封存确认实验全部 420 个原始回合、统计结果、图、配置、代码快照索引和关键 checkpoint。

建议阅读顺序：

1. `report/218_l217_complex_navigation_sealed_results_2026-07-20.md`：主要科研结论；
2. `report/219_l217_data_quality_and_raw_data_guide_2026-07-20.md`：数据结构与限制；
3. `analysis/mechanism_coverage_audit.json`：为什么本轮只能称为部分机制覆盖；
4. `figures/fig1...fig4`：论文级 PDF/PNG；
5. `tables/full_vs_simple_seed_cluster_effects.csv`：主要比较；
6. `tables/core_factorial_seed_cluster_effects.csv`：2×2 消融；
7. `raw_results/`：420 个不可替换原始回合。

最重要的结论：Full 在 60/60 个复杂场景单元中成功、0 碰撞，相对 Simple 平均减少 17.42 步，95% seed-cluster CI 为 `[9.08, 26.05]`；但 clearance 和 jerk 变差。Role-aware HSS 的效率收益得到确认，value-alignment 的独立闭环收益不明确。

最重要的边界：逐拍机制审计表明 residual-conditioned policy context 和 reliability-weighted terminal value 在 L217 中没有激活。L217 是有效的正式基线/HSS 证据，但不是完整论文方法的最终唯一主表；不得改名或隐去该缺口。

统计单位是 10 个 seed cluster，不是 420 个独立样本。场景和物理域是每个 seed 内重复分层。置信区间来自冻结的 10,000 次 seed-cluster bootstrap。

压缩包与文件夹内容相同。`SHA256SUMS.txt` 可用于传输后完整性验证。
