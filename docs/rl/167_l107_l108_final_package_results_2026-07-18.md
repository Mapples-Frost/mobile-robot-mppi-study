# L163--L164: final ICODE + contextual RL package results

Date: 2026-07-18
Status: independent confirmation completed; **retained method established with a qualified claim**.

## Retained controller

The retained research controller is:

```text
Frozen control-affine ICODE residual dynamics
        +
Frozen route-context bandit selecting MPPI covariance
        +
K=100 MPPI sampling
        +
unchanged LaserScan / scan_guard / safety arbitration
```

The contextual bandit is a one-step reinforcement-learning policy: it learns
which sampling action yields better closed-loop return in a route/scene
context. It never directly commands the robot and cannot bypass MPPI or safety.

## Experimental design

L108 used a blocked, balanced three-arm design:

1. nominal dynamics + fixed covariance, `K=100`;
2. ICODE dynamics + fixed covariance, `K=100`;
3. ICODE dynamics + learned contextual covariance, `K=100`.

There were 48 matched blocks and 144 completed MuJoCo episodes. Each method
appeared 16 times in every run position. All 144 episodes succeeded and no
collision was recorded.

The two held-out path geometries were a high-dynamic hairpin and reverse-S;
the physical-domain blocks included train-anchor, long delay, high friction,
and combined moderate mismatch conditions.

## Independent-confirmation results

All deltas below are treatment minus reference; negative is better for the
reported error, time, jerk, and compute metrics.

### Complete package versus traditional nominal MPPI

| Metric | Mean paired delta | 95% bootstrap CI | Result |
|---|---:|---:|---|
| Cross-track RMSE | -21.540 mm | [-24.053, -19.240] mm | significant improvement |
| Elapsed time | -1.558 s | [-2.417, -0.698] s | significant improvement |
| Issued-control jerk | -0.01223 | [-0.01849, -0.00587] | significant improvement |
| Applied-control jerk | -0.00921 | [-0.01415, -0.00457] | significant improvement |
| Mean planner compute | +27.280 ms | [+25.897, +28.950] ms | ICODE cost; absolute mean 32.036 ms |
| Success / collision | 0 / 0 delta | -- | parity |

The complete package passed the preregistered package-versus-nominal gate. Its
mean planner time remained below the frozen 50 ms absolute deadline.

### ICODE contribution at fixed policy and budget

ICODE fixed `K=100` versus nominal fixed `K=100` produced:

- RMSE: -21.275 mm, 95% CI [-23.757, -19.126] mm;
- elapsed time: -0.200 s, 95% CI [-0.354, -0.048] s;
- issued jerk: -0.00454, 95% CI [-0.00553, -0.00355];
- applied jerk: -0.00365, 95% CI [-0.00448, -0.00283];
- identical success and collision outcomes.

The independent ICODE contribution gate passed.

### Contextual RL contribution at fixed ICODE and fixed `K=100`

Contextual covariance versus the strongest fixed covariance produced:

| Metric | Mean paired delta | 95% bootstrap CI | Interpretation |
|---|---:|---:|---|
| Cross-track RMSE | -0.265 mm | [-1.088, +0.402] mm | statistical parity; upper loss bound < 0.5 mm |
| Elapsed time | -1.358 s | [-2.356, -0.346] s | significant improvement |
| Issued-control jerk | -0.00770 | [-0.01325, -0.00199] | significant improvement |
| Applied-control jerk | -0.00556 | [-0.00986, -0.00139] | significant improvement |
| Mean planner compute | -0.063 ms | [-1.724, +1.610] ms | no material overhead |
| Success / collision | 0 / 0 delta | -- | parity |

This is the clearest positive RL result obtained in the current study: at equal
rollout dynamics and equal sample budget, the learned sampling action improves
closed-loop time and smoothness without a detectable tracking, safety, or
compute penalty.

## Why the JSON still says `primary_gate_passed: false`

The L108 preregistration required strict RMSE *superiority* for the contextual
policy. Its RMSE confidence interval crossed zero, so that exact clause failed,
even though the mean favored the contextual policy and the upper confidence
limit was only +0.402 mm. The result must therefore be reported as:

> significant time and jerk improvement with tracking parity,

not as statistically superior tracking. The package and ICODE sub-gates both
passed. Keeping this distinction prevents post-hoc relaxation of the frozen
criterion.

## L107 half-budget boundary

L107 tested the contextual policy at `K=50`. The package still improved nominal
MPPI by -17.528 mm RMSE and -1.669 s, but versus ICODE fixed `K=100` it incurred
+4.844 mm RMSE, 95% CI [+1.138, +8.767] mm. Therefore `K=50` is not retained as
a universal final setting. This negative result motivated the equal-budget
L108 test and isolates the RL policy effect from sample-count confounding.

## Claim boundary

Supported now:

- the frozen ICODE residual model materially improves prediction-dependent
  closed-loop control under the tested mismatch domains;
- an offline learned contextual policy can improve how ICODE-MPPI samples,
  producing faster and smoother control at equal `K`;
- the combined package significantly outperforms traditional nominal MPPI on
  tracking, time, and smoothness in the tested held-out path/domain blocks.

Not supported now:

- that SAC policy warm-start or terminal value improves this platform;
- that half-budget `K=50` is universally noninferior;
- that the current anytime STOP/ADD mechanism saves whole-controller compute;
- that the complete SAC-based RL-Driven MPPI paper has been reproduced;
- any stability, contraction, convergence, or general real-robot guarantee.

## Reproduction

```bash
.venv/bin/python experiments/rl/run_final_icode_contextual_package.py \
  --config configs/rl/final_icode_contextual_k100_confirmation_l108.yaml \
  --output-dir results/research_platform/rl/l108_reproduction

.venv/bin/python experiments/rl/plot_final_icode_contextual_package.py \
  --summary results/research_platform/rl/l108_reproduction/summary.json \
  --output-dir results/research_platform/rl/l108_reproduction/figures
```

Canonical outputs:

- `results/research_platform/rl/l108_final_icode_contextual_k100_confirmation_20260718_v1/summary.json`
- `results/research_platform/rl/l108_final_icode_contextual_k100_confirmation_20260718_v1/episodes.csv`
- `results/research_platform/rl/l108_final_icode_contextual_k100_confirmation_20260718_v1/figures/fig_l108_final_package_effects.pdf`
- `results/research_platform/rl/l108_final_icode_contextual_k100_confirmation_20260718_v1/figures/fig_l108_final_package_effects.png`
