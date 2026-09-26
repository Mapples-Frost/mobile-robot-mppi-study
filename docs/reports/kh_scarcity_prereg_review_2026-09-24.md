# 旧计算稀缺性预登记：接续前勘误

本记录核对 `research_artifacts/kh_compute_scarcity_2026-09-24/prereg.json`
及其关联源码。原预登记、冻结哈希和历史结果继续保留；本记录不重写它们。
检查时该研究目录只有预登记文件，尚无该研究的 profiling、训练和测试产物。
当前优先完成 Bøhn 完整固定 H 网格的留出评估和审计，再登记后续实验。

## 已核实的问题

1. **论文身份和复现状态错误。** 原预登记将论文称为
   “Optimization of the MPC Update Interval Using Reinforcement Learning”，
   并写成已经复现。实际保存论文为 Bøhn 等的
   *Reinforcement Learning of the Prediction Horizon in Model Predictive Control*。
   研究变量是预测时域，不是控制更新间隔。本地工作是作者同期核心代码恢复与配置重建；
   作者原配置、完整训练工程和测试集未找回，不能称为原论文数值复现成功。

2. **“原策略未支付计算成本”不成立。** 已有
   `kh_scene_strata_2026-09-23/deviation_train_deploy_2026-09-23.json`
   记录：部署模型的训练奖励包含 `-0.05 * sum(measured_e2e_s)`。
   环境奖励中 `lambda_compute=0` 不等于 SAC 实际优化的奖励无计算项。
   计算价格较小能提出新假设，不能直接证明旧负结果的原因。

3. **未超截止期不等于没有分配问题。**
   `physical_tradeoff_v2/environment.py` 将小于控制周期的延迟同样传入物理执行，
   训练还存在计算惩罚。因此“所有动作满足截止期，所以增加采样永不付出代价”
   不能从低平均 rho 推出；平均值也不能证明每一步从不超时。

4. **固定成本模型不能产生固定动作的时变超时率。**
   `ScaledComputeModel.predict()` 只依赖 mode、K、H、alpha 和固定推理成本。
   在同一 alpha 下，固定 K/H 的计费延迟为常数，非空回合的截止期违规率
   因而为 0 或 1。若要主张“随路段需求变化在线回退避免超时”，必须区分
   动作切换、外部负载变化和纯粹选中一个可行固定动作，不能预先写成已证实机制。

5. **联合独有收益需要直接对照。**
   “联合对固定显著、H-only 对固定不显著”不等于联合显著优于 H-only。
   旧 ladder 的 `analysis_test.json` 中 K-only 也复现了高难度完成率改善：
   B 轴 level 4，联合相对 H-only 为 +0.70，K-only 相对 H-only 为 +0.80。
   这些是原配对键口径的描述性数字，不能用于证明联合机制。
   后续联合贡献必须直接比较联合、H-only、K-only，并包括调优固定和规则对照。

6. **280 个配对键不是 280 个独立训练重复。**
   原方案复用同一场景、同一固定基线并交叉两个策略种子；场景与训练种子的
   相关性必须保留。对所有键平铺 bootstrap 不能替代按独立场景及训练种子
   处理依赖结构。没有功效或精度依据时不能称样本量“远超需要”。
   “alpha=1 未显著优于”也不证明相等或没有效果，需要预先定义等效范围。

7. **旧测试场景已经曝光。** 新方案直接继承 ladder 的 scene seed 11..20，
   这些场景已有完整测试结果并用于提出新机制。它们可作探索与回归检查，
   不能再作为这一新机制的首次确认性留出集。

## 接续边界

旧 ladder 的 A/B 负结果与 C 的部分支持继续保留，不能被“实验没有真正检验算法”
一句话撤销。训练 hold=5、部署 hold=1 是已知限制，修正后的新实验须另列。
旧 scarcity 方案需要带版本的新协议、新的场景划分及直接消融比较，
不能直接按旧 `freeze_prereg.py` 重新生成并覆盖预登记。

本记录是源码与协议核对，不是新实验结果，也没有证明计算稀缺会使联合 K/H 获胜。
完成 Bøhn 审计后的下一步应先做小规模机制可辨识性检查，再决定是否值得投入完整训练。

## 证据入口

- `research_artifacts/bohn2021_reproduction_2026-09-17/sources/bohn2021.txt`，第一页标题和摘要。
- `docs/reports/bohn2021_grid_interpretation_2026-09-24.md`。
- `research_artifacts/kh_scene_strata_2026-09-23/deviation_train_deploy_2026-09-23.json`。
- `research_artifacts/kh_ladder_rolls_2026-09-23/analysis_test.json`。
- `research_artifacts/kh_compute_scarcity_2026-09-24/prereg.json`。
- `experiments/icra_compute_scarcity/compute_model.py`。
- `src/mobile_robot_mppi/physical_tradeoff_v2/environment.py`。
