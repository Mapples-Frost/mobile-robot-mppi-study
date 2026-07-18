# Table 11: L100-L103 anytime-budget evidence chain

| Study | Policy | Mean K | Gain vs K50 | Oracle retention | Frozen status |
|---|---|---:|---:|---:|---|
| L100 | unconstrained ridge | 83.20 | 3.368% | 65.5% | failed compute Gate |
| L101 | calibrated ridge | 72.40 | 2.664% | 58.7% | passed supervised Gate |
| L102 | uncalibrated bandit | 58.85 | 0.969% | 22.3% | failed retention Gate |
| L103 | calibrated bandit | 77.86 | 2.400% | 58.1% | passed confirmation Gate |

L103 raw episode-mean cost delta versus K50 was `-0.2919`, with hierarchical-
bootstrap 95% CI `[-0.5229, -0.1144]`. Its post-hoc stratified exact-budget
randomization diagnostic used 10,000 draws and gave one-sided `p=0.00020`.

Primary source: `docs/rl/156_l103_calibrated_budget_bandit_confirmation_results_2026-07-18.md`.
