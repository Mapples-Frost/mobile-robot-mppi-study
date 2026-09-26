# Controlled T1 conditioning diagnostic (development only)

Original T1: four variants, one training seed,2048 transitions each. No variant
passed the prespecified utility/variation gate. Full kept exactly K144/H25 at
deterministic evaluation; its value losses often exceeded300 (half-MSE), with
pre-clipping gradient norms in the tens versus the .5 clip limit. This suggests
poor numerical conditioning/insufficient optimization, not a confirmed scientific
negative. Original data/checkpoints are retained.

One controlled rerun is authorized by the user's explicit instruction to diagnose
and re-experiment. Freeze s=.01 before the rerun; all four variants get2048 steps,
same seed/scenarios/ranges, same beta=.1,eta=1 and task_scale=1. No task reward term
or computation price is changed.

The critic internally predicts s*V. Its initial last layer is multiplied by s,
so decoded initial V and actor are unchanged. Public policy values, logs and GAE
rewards stay in raw units. Critic targets are multiplied by s for MSE; actor
advantages are standardized as before. This is standard fixed reward/value
normalization, preserving relative task/computation utility, not reward tuning.
Checkpoints contain reward_scale and reject incompatible configurations.

The numerical loss values across scales are not directly comparable. Diagnostics
therefore include raw-value RMSE and explained variance. The rerun keeps the SAME
T1 acceptance criteria. No T2 seed expansion occurs unless it produces positive
task/computation and conditional-allocation signals. Improvement in critic fit
alone does not establish useful allocation or ICODE's independent value.
