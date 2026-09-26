# Bøhn 2021 分支回报校准扩展

数据审计通过；validation gate：False。此为方法扩展，H 成本是代理，不代表实测加速。

|任务|种子|方法|成本|约束回合|到达回合|求解失败步|
|---|---:|---|---:|---:|---:|---:|
|vehicle|0|actor|45.370696|0|8|0|
|vehicle|0|raw_greedy|7920.755047|1|5|36|
|vehicle|0|calibrated_greedy|358.586785|0|7|30|
|vehicle|0|fixed|19.125811|0|10|0|
|vehicle|1|actor|17.366509|0|10|0|
|vehicle|1|raw_greedy|224.358324|7|3|93|
|vehicle|1|calibrated_greedy|20118.552891|1|6|232|
|vehicle|1|fixed|16.403496|0|10|0|
|vehicle|2|actor|18.724717|0|10|0|
|vehicle|2|raw_greedy|5454.535034|0|9|7|
|vehicle|2|calibrated_greedy|2958.195284|0|9|13|
|vehicle|2|fixed|16.398639|0|10|0|
|pendulum|0|actor|203.730463|1|0|15|
|pendulum|0|raw_greedy|204.093361|1|0|15|
|pendulum|0|calibrated_greedy|212.769161|1|0|15|
|pendulum|0|fixed|203.746944|1|0|15|
|pendulum|1|actor|214.429798|1|0|15|
|pendulum|1|raw_greedy|215.474738|1|0|7|
|pendulum|1|calibrated_greedy|271.846848|1|0|15|
|pendulum|1|fixed|203.643209|1|0|15|
|pendulum|2|actor|232.720606|1|0|15|
|pendulum|2|raw_greedy|313.241035|1|0|13|
|pendulum|2|calibrated_greedy|344.345813|1|0|11|
|pendulum|2|fixed|203.775168|1|0|15|

完整逐场景配对差、检查项和输入散列保存在相应 gate.json。
固定 H 继承 300,000 搜索训练步和 60,000 补充种子训练步；author RL/min-Q 各 90,000 步。校准的额外仿真和拟合预算见 training_audit.json，另保留 smoke 和全部 attempt 记录。
仅三个训练种子；场景和分支噪声不是独立训练重复。两次续跑的标签含学习 target-V 尾项；冻结原 actor 的回报不保证重复贪心部署有效。
数据审计通过不等于核心结论复现。验证不通过时禁止开启测试；Goal 保持 active。
