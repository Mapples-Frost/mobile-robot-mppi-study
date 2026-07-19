# Two Coupling Mechanisms Frozen for the Paper

## Decision

The paper method is frozen around two bidirectional, independently ablated
coupling mechanisms. Neither mechanism changes the safety chain, LaserScan
obstacle source, equal MPPI rollout budget, or MuJoCo plant.

### Mechanism A: RL value calibrates ICODE learning

The frozen SAC Actor/critic supplies a competence-gated value-consistency term
to continuous-time ICODE residual fine-tuning. State derivative, one-step,
multi-step RK4 and checkpoint-anchor supervision remain active. Only critic
contexts supported by training data and above the training-outcome competence
threshold may backpropagate into ICODE. The Actor and critics never update in
this stage.

This is the **RL -> ICODE** direction.

### Mechanism B: ICODE reliability calibrates RL guidance

A three-member ICODE ensemble combines support, disagreement and causal
prediction/execution innovation into a cross-layer reliability score. At fixed
total MPPI rollout budget, this score allocates 0/30/60% of candidates to
persistent Actor-guided sampling. A completion-preserving 30% floor is active
only in the terminal phase. The gate changes candidate allocation, not plant
dynamics or safety arbitration.

This is the **ICODE -> RL/MPPI** direction.

## Independent factorial evidence

The sealed 2x2 experiment used five seed clusters, three MuJoCo physics domains
(nominal seen, long-delay seen and combined unseen), and four arms. The unit of
inference was seed, not control timestep. Every arm used K=100, two MPPI
refinements, a 36-step horizon and the same frozen Actor/critic.

| Contrast against ordinary ICODE + fixed 30% Actor guidance | Success effect | Final-distance improvement | Other constraints |
|---|---:|---:|---|
| Value-aligned ICODE only | +0.3333, 95% CI [0.1333, 0.5333] | +0.17102 m, 95% CI [0.14693, 0.20265] | zero collision regression |
| Reliability-adaptive HSS only | +0.2667, 95% CI [0.1333, 0.3333] | +0.15997 m, 95% CI [0.10000, 0.21993] | zero collision regression; planner time improved 2.83 ms |
| Both mechanisms | +0.3333, 95% CI [0.1333, 0.5333] | +0.18343 m, 95% CI [0.14505, 0.22166] | zero collisions; jerk within the preregistered 5% noninferiority margin |

The two mechanisms have reproducible main effects. Their success interaction is
negative/sub-additive (-0.2667, 95% CI [-0.4667, -0.0667]), so the manuscript
must describe **mutual calibration and complementary functions**, not claim a
super-additive synergy theorem.

## Alternatives screened and rejected in this round

### Always-on residual-conditioned Actor correction

After preserving proposal covariance, L207 improved unseen-domain cross-track
RMSE by 5.56% but worsened nominal-domain RMSE by 16.15%. L209 added a causal
innovation/support authority gate. Across fresh seeds 582--584 it retained
100% success and zero collisions, but nominal RMSE still worsened 10.61%; unseen
RMSE improved only 1.22% while jerk worsened 5.69%. It therefore fails the
preregistered advancement gate. The causal context and fallback implementation
remain behind configuration switches for future work, but are not a paper
contribution.

### Pairwise value-ranked ICODE

L210 attempted to preserve frozen-critic terminal-state ordering instead of
absolute Q scale. The original ICODE already achieved test/unseen rank
correlations of 0.9773/0.9829. Fine-tuning reduced them by 0.0100/0.0019,
worsened test rollout RMSE by 7.88%, and failed the offline gate. This mechanism
is rejected as a ceiling-limited, overfitting-prone objective on the current
dataset.

## Frozen narrative

The manuscript's central claim is now:

> Under dynamics mismatch, a frozen RL value can make continuous-time residual
> learning task-aware when protected by competence gating, while calibrated
> ICODE reliability can decide how much fixed-budget Actor guidance MPPI should
> trust online.

The claim is narrower and better supported than residual-conditioned action
correction or universal ICODE-RL synergy. Formal stability, convergence,
complex-navigation success and real-robot generalization are not claimed.

## Evidence bindings

- `docs/rl/173_gate2_competence_gated_value_alignment_results_2026-07-18.md`
- `docs/rl/176_gate3_reliability_hss_results_2026-07-18.md`
- `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`
- `ara/evidence/tables/table12_l188_l191_value_alignment.md`
- `ara/evidence/tables/table13_gate3_reliability_hss.md`
- `ara/evidence/tables/table14_full_proposed_factorial.md`
- `results/research_platform/rl/full_proposed_confirmation_l211/`
- `results/research_platform/rl/residual_conditioned_closed_loop_l209/`
- `results/research_platform/value_ranked_icode_member1_l210/`
