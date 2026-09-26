# L103 calibrated contextual-budget bandit: confirmation results

Date: 2026-07-18

Status: frozen confirmation Gate passed

## Question

At one MPPI control state, first evaluate a byte-identical K50 prefix. Can a
contextual-bandit policy decide whether to stop or add the remaining 50 samples,
improving physical rollout quality without paying the fixed K100 cost at every
state?

ICODE is the rollout dynamics for every arm. The bandit does not output robot
controls: it selects only `STOP` or `ADD50`; MPPI performs the optimization and
the existing safety arbitration remains downstream.

## Protocol and integrity

- two obstacle-free high-dynamic routes;
- four MuJoCo physics domains;
- three discovery and three evaluation seeds per route/domain;
- 48 independent episodes and eight anchors per episode;
- 384 nested K50/K100 counterfactual records;
- episode, rather than anchor, is the inferential unit;
- hierarchical bootstrap is blocked by route and physics domain;
- K50 is a byte-identical prefix of K100 at every anchor;
- both control sequences start from one restored MuJoCo snapshot;
- zero live or counterfactual collisions and no non-finite features.

The L103 evaluation seeds were not used in L100--L102 development. Algorithm,
deployment calibration and primary Gate were frozen in
`155_l102_failure_l103_calibrated_bandit_prereg_2026-07-18.md` before collection.

## Frozen primary results

| Policy | ADD fraction | Mean K | Relative gain vs K50 | Episode-mean raw delta vs K50 | 95% CI |
|---|---:|---:|---:|---:|---:|
| Fixed K50 | 0.000 | 50.00 | 0.000% | 0.000 | [0.000, 0.000] |
| Matched random | 0.542 | 77.08 | 1.315% | -0.160 | [-0.413, -0.017] |
| Contextual bandit | 0.557 | 77.86 | **2.400%** | **-0.292** | **[-0.523, -0.114]** |
| Fixed K100 | 1.000 | 100.00 | 2.257% | -0.274 | [-0.495, -0.087] |
| Hindsight Oracle | 0.438 | 71.88 | 4.133% | -0.503 | [-0.713, -0.359] |

All primary clauses passed:

1. selective ADD rate: `0.557` lies in `[0.10, 0.75]`;
2. superiority to fixed K50: CI upper bound is `-0.114 < 0`;
3. compute feasibility: mean `K=77.86 <= 80`;
4. Oracle utility retention: `58.06% >= 30%`;
5. counterfactual integrity audit passed.

The bandit also achieved a slightly lower mean true cost than fixed K100 while
using 22.14 fewer rollouts per decision. That numerical ordering is reported as
a result, not yet as a formal noninferiority or superiority claim between those
two methods.

## Does context matter, or is this only extra computation?

Against the single frozen matched-random allocation, the paired episode-mean
bandit-minus-random true-cost delta was `-0.132`, with 95% CI
`[-0.235, -0.037]`.

After the primary result was opened, a secondary robustness diagnostic sampled
10,000 policies with exactly the bandit's ADD count inside each route/physics
stratum. The learned policy's statistic was `-0.292`, versus a random null mean
of `-0.171` and null 95% interval `[-0.234, -0.106]`; the conservative one-sided
Monte-Carlo p-value was `0.00020`. Because this diagnostic was added after the
primary L103 result, it is explicitly labelled post-hoc and is not a frozen
Gate.

## Interpretation

The confirmation supports three limited statements:

1. additional MPPI samples have state-dependent physical value;
2. a contextual bandit trained from selected simulated ADD rewards can learn a
   non-random allocation of that compute;
3. deployment-budget calibration is necessary: the uncalibrated L102 policy
   was statistically better than K50 but failed Oracle-retention.

This result does **not** yet establish whole-episode closed-loop improvement,
dynamic-obstacle benefit, sim-to-real online bandit learning, or a statistical
interaction unique to ICODE. The two ICODE reliability features were constant
in this experiment, so they cannot be cited as active explanatory variables.

## Reproduction

```bash
.venv/bin/python experiments/rl/run_anytime_budget_oracle.py \
  --config configs/rl/anytime_budget_bandit_l103.yaml \
  --output-dir results/research_platform/rl/l103_calibrated_budget_bandit_20260718_v1

.venv/bin/python experiments/rl/evaluate_anytime_budget_bandit.py \
  --records results/research_platform/rl/l103_calibrated_budget_bandit_20260718_v1/anchors.csv \
  --config configs/rl/primal_dual_budget_bandit_l103.json \
  --output results/research_platform/rl/l103_calibrated_budget_bandit_20260718_v1/bandit_summary.json
```

The next Gate is an incremental online MPPI implementation that reuses the
first 50 samples, jointly reweights K100 only after `ADD50`, preserves the exact
legacy path when disabled, and evaluates nominal/ICODE x fixed/adaptive in
closed loop.
