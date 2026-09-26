# RL-Horizon adaptation based on Bøhn et al.

Source: [author paper](https://torarnj.folk.ntnu.no/lahmpc_nmpc2021_eeb.pdf),
IFAC-PapersOnLine54(6),314–320,2021, DOI10.1016/j.ifacol.2021.08.563.

The original horizon policy uses SAC, a squashed Gaussian mapped/rounded to an
integer horizon, nonlinear MPC, horizon-length computation cost, and learned
terminal value. Tasks are inverted pendulum and collision avoidance. Reported
settings include Nmax50, two32-unit layers and discount.97. Official runnable code
was not established in this bounded source search.

Our comparator retains learned state-dependent H and environment-side rounding.
It uses the common Beta-PPO, navigation scene features, actual latency reward,
fixed K256, H10..40, same task reward and frozen training budget as proposed.
No MPC terminal-value learning, SAC, or linear-H compute proxy is added. This is
deliberately **paper-principle adaptation**, not exact reproduction.

Config: `configs/baselines/compute_rl_horizon_bohn.yaml`. Its mechanism is the
same horizon-only ablation interface; do not double-count it as independent
evidence. Mask/mapping, PPO checkpoint and deterministic evaluation tests cover
the implementation. Training is pending the research gates; no learned comparator
performance is claimed from sanity alone.

2026-09-07 execution update: a full-information H-only fairness variant was trained
alongside full KH under the same PPO and frozen development conditions. It is an
additional common-information control, not the original paper's state definition.
The result report contains the actual development comparisons; T2/T3 were not run.
