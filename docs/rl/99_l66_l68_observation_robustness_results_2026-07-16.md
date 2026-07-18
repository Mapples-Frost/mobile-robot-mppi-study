# L66--L68 observation-robustness results

Date: 2026-07-16

## Outcome

The L68 sealed primary observation-domain gate passed. All 360 expected
episodes and artifacts were present despite the outer orchestration process
reaching its timeout after every block had already written 120/120 complete
runs and metadata.

On the three primary domains (clean state, 100 ms state-observation latency,
and moderate noise plus 100 ms latency), every method achieved 90/90 successes
and zero collisions. The sealed paired results were:

| Contrast | Absolute improvement | Relative RMSE reduction | Hierarchical 95% interval |
|---|---:|---:|---:|
| ICODE vs Nominal | 0.020896 m | 30.17% | [0.019502, 0.022256] m |
| MLP vs Nominal | 0.015408 m | 22.25% | [0.014247, 0.016587] m |
| ICODE vs MLP | 0.005487 m | 10.15% | [0.004243, 0.006799] m |

The ICODE-versus-MLP effect was positive in every primary observation domain:

- clean ground truth: 0.005495 m, 10.91%;
- 100 ms observation latency: 0.005295 m, 9.44%;
- noise plus 100 ms latency: 0.005671 m, 10.10%.

All three independent training blocks were positive: 0.006879, 0.004264 and
0.005319 m. Across the two shifted observation domains, the estimate was
0.005483 m with interval [0.004261, 0.006831] m. On the unseen reverse-S path,
the primary-domain estimate was 0.005806 m with interval
[0.004614, 0.007142] m.

## Calibration result

L66 varied only the state observation while holding the MuJoCo plant fixed.
Two nominal-only calibration attempts found:

- 50--100 ms observation latency changed nominal cross-track RMSE by
  approximately 8--14% while preserving success and collision safety;
- combined bounded noise and latency changed it by approximately 8--14%;
- zero-mean state noise alone changed true tracking RMSE by at most 1.6%, below
  the preregistered 3% resolution floor, even at the strongest tested level;
- raw uncorrected wheel odometry produced 0--16.7% nominal success and roughly
  fourfold larger tracking error, so it was classified as a state-estimation
  stress case rather than a primary residual comparison.

No learned-model result was observed while choosing the primary domains. Pure
noise was removed after the capped second failure instead of being tuned until
a favorable learned result appeared.

## Raw wheel-odometry stress result

The promising L67 development ordering did not replicate:

| Stage | Nominal | MLP | ICODE |
|---|---:|---:|---:|
| Development | 3/18 | 5/18 | 10/18 |
| Sealed confirmation | 18/30 | 13/30 | 12/30 |

In sealed data, ICODE lost six successes relative to nominal and one relative
to MLP, while MLP lost five relative to nominal. All stress-domain conditions
had zero collisions. This result weakens any claim that residual dynamics alone
can solve accumulated wheel-odometry drift. Localization correction must remain
a separate system component.

The stress-domain mean relative ICODE-versus-MLP statistic is not interpreted:
large between-episode scale differences make the mean of per-episode ratios
unstable, while task success directly shows that the development ordering did
not reproduce.

## Compute

Across primary and stress domains, mean planner times were 5.20 ms for Nominal,
31.25 ms for MLP and 36.04 ms for ICODE, below the 50 ms preregistered mean
limit. Large per-step maxima occurred during the heavily contended parallel
run (up to 619 ms for ICODE), so this is not a hard real-time guarantee.

## Claim boundary

L68 supports bounded robustness of the parameter-matched ICODE advantage to
100 ms state-observation latency and moderate observation noise plus latency on
one fixed MuJoCo plant. It does not establish robustness to raw odometry drift,
arbitrary sensor errors, obstacle-rich perception, real-robot localization, or
RL composition. RL and memory remained disabled.

## Reproducibility

- L67 config SHA-256:
  `FA0B2207F8E61610A0145384214544D321BC0565BD8E352ED767CC130BC0F689`
- L68 config SHA-256:
  `8C2A04AB6498D49DF5811DB723ADEB06A14B3303B60F617CEC3BD56E802AAF99`
- Development summary:
  `results/research_platform/rl/l67_residual_structure_observation_domains_20260716_v1/residual_structure_observation_summary.json`
- Confirmation summary:
  `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/residual_structure_observation_summary.json`
- Figure generator:
  `experiments/rl/plot_residual_structure_observation.py`
- Figure:
  `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/figures/fig_l67_l68_observation_robustness.{pdf,png}`
