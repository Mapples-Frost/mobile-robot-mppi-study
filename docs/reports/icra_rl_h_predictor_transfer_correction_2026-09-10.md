# RL-H 迁移对照的额外配置差异

2026-09-10 08:58 UTC 静态核验发现：旧 rl_geometry_mass/seed12401001/manifest.json 中 planner.prediction_mode=nominal，且没有 compute_allocation 项。旧 run.py 通过 make_components 构建名义预测器。

当前独立评估 evaluate_route_sac_independent.py 的 rl_h_transfer 使用 RouteObservableEnv 和默认 residual dynamics_mode。因此该项不仅经历固定步到计算延迟物理执行的变化，也经历名义预测器到残差预测器的变化。不能将表现差异单独归因于计算延迟，也不能作为公平战胜文献方法的证据。现有数据、配置和脚本不改；此前“transfer diagnostic”标签成立，但原因披露不完整，需要在最终汇总补充两项迁移。

新 ReadyHEnvironment 默认也接入残差预测器。目前资格检查用于接口验证，仍可运行，但后续训练必须显式指定预测器。建议先保留文献启发 RL-H 的 nominal 版本用于外部方法比较，同时训练 residual 版本控制预测器差异；50维H-only仍承担同网络/奖励/预测器的消融。预算和训练分布应在开始前另立协议，不从本轮结果挑选版本。两者都应在相同 measured readiness 执行机制下训练与评估。

证据：旧训练 manifest、experiments/icra_rl_geometry_mass/run.py、src/mobile_robot_mppi/runtime/factories.py 第65行及当前评估中 rl_h_transfer 的配置路径。尚未实施新训练。
