# Claims

## C01: Zero correction exactly recovers the frozen BC prior
- **Statement**: A `frozen_bc_correction` checkpoint with deterministic zero correction produces the same task trajectory as its source BC actor under the same scene and episode seed.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any paired seed differs in success, collision, final distance, trajectory length, clearance, jerk or safety interventions.
- **Proof**: [`docs/rl/15_l17_v3_conservative_correction_results_2026-07-14.md`, `results/research_platform/rl/bc_vs_l17_v3_conservative_correction_finaltest_20260714_v1/`]
- **Dependencies**: []
- **Tags**: BC, correction, regression

## C02: Current SAC correction improves the frozen BC prior
- **Statement**: With the L17 loss and bounds, SAC produces a checkpoint that passes paired seen-scene non-inferiority and improves success or final distance over step 0.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: All three training seeds retain step 0, or learned checkpoints lose paired successes.
- **Proof**: [`docs/rl/15_l17_v3_conservative_correction_results_2026-07-14.md`]
- **Dependencies**: [C01]
- **Tags**: SAC, negative-result, seen-gate

## C03: Episode seeds now define the full stochastic simulation replicate
- **Statement**: In the corrected RL environment, an episode seed controls initial-state sampling, MuJoCo reset, simulated sensors and MPPI perturbation sampling; with zero validation jitter, trainer validation matches the standard evaluator.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Trainer and standard evaluator disagree on task metrics for the same checkpoint, scene and seed.
- **Proof**: [`docs/rl/14_validation_seed_audit_and_l17_v2_prereg_2026-07-14.md`, `docs/rl/15_l17_v3_conservative_correction_results_2026-07-14.md`]
- **Dependencies**: []
- **Tags**: reproducibility, seed, validation

## C04: Paired fail-closed selection prevents deployment of observed regressions
- **Statement**: A candidate that loses any reference success or introduces a collision is rejected even when aggregate success or mean distance otherwise improves.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: `best.pt` advances to a checkpoint with a prohibited paired regression.
- **Proof**: [`tests/rl/test_sac_and_checkpoint.py`, `docs/rl/15_l17_v3_conservative_correction_results_2026-07-14.md`]
- **Dependencies**: [C01]
- **Tags**: checkpoint-selection, safety, paired-design

## C05: L17 twin-critic advantage contains weak relative ranking information
- **Statement**: Across the fixed L18 validation episodes, larger episode-mean conservative critic advantage is descriptively associated with better paired correction return, but the direction is not stable across every training seed.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Pooled and per-training-seed paired advantage/return associations are uniformly null or negative.
- **Proof**: [`docs/rl/17_l18_critic_advantage_diagnostic_results_2026-07-15.md`, `results/research_platform/rl/l18_advantage_validation_multiseed_20260715_v1/aggregate.json`]
- **Dependencies**: [C02, C03]
- **Tags**: critic, advantage, calibration, diagnostic

## C06: A zero-threshold twin-critic advantage gate protects BC performance
- **Statement**: Applying correction only when online or target conservative advantage is non-negative prevents paired BC-success losses while retaining a nondegenerate fraction of corrections.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Either gate loses any paired BC-success episode or fails to improve pooled success over the ungated correction.
- **Proof**: [`docs/rl/16_critic_advantage_diagnostic_prereg_2026-07-15.md`, `docs/rl/17_l18_critic_advantage_diagnostic_results_2026-07-15.md`]
- **Dependencies**: [C01, C05]
- **Tags**: critic, advantage-gate, negative-result, BC-noninferiority

## C07: One raw target-Q margin generalizes across independent critic seeds
- **Statement**: A single margin selected from the preregistered raw target-Q grid can preserve every paired BC success and provide meaningful benefit for all three independently trained critics.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Every preregistered margin is ineligible because it loses a paired BC success, regresses an independent training seed, collapses the gate, or provides no meaningful benefit.
- **Proof**: [`docs/rl/18_l19_advantage_margin_calibration_prereg_2026-07-15.md`, `docs/rl/19_l19_advantage_margin_calibration_results_2026-07-15.md`, `results/research_platform/rl/l19_margin_calibration_multiseed_20260715_v1/selection.json`]
- **Dependencies**: [C05, C06]
- **Tags**: critic, calibration, margin-gate, negative-result, cross-seed

## C08: Twin-critic consensus alone can safely gate SAC correction
- **Statement**: A single dimensionless disagreement-penalized beta applied to the existing target twin critics can preserve every paired BC success while retaining a meaningful correction benefit across all three independently trained checkpoints.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Every preregistered beta loses at least one paired BC success, introduces a collision, regresses a training seed, collapses the gate, or provides no meaningful benefit.
- **Proof**: [`docs/rl/20_l20_twin_critic_consensus_lcb_prereg_2026-07-15.md`, `docs/rl/21_l20_twin_critic_consensus_lcb_results_2026-07-15.md`, `results/research_platform/rl/l20_lcb_calibration_multiseed_20260715_v1/selection.json`]
- **Dependencies**: [C05, C06, C07]
- **Tags**: critic, disagreement, LCB, negative-result, BC-noninferiority

## C09: Scale-invariant critic consensus retains useful correction signal
- **Statement**: On the preregistered L20 calibration set, beta 2 improves pooled success from 27/36 to 31/36, yields positive mean paired return in every training seed and zero collision, while still losing one paired BC success.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The audited paired rows do not reproduce 5 success gains, 1 success loss, positive per-training-seed return deltas, or zero collision regressions.
- **Proof**: [`docs/rl/21_l20_twin_critic_consensus_lcb_results_2026-07-15.md`, `results/research_platform/rl/l20_lcb_calibration_multiseed_20260715_v1/paired_calibration_rows.csv`]
- **Dependencies**: [C05, C08]
- **Tags**: critic, disagreement, near-miss, descriptive, paired-design

## C10: Same-seed action replay exactly restores the stochastic control state
- **Statement**: Resetting the full environment with the same episode seed and replaying the same frozen-BC latent actions reproduces encoded observations, actions, rewards, goal distance and termination flags exactly before a counterfactual branch.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any development or formal L21 replay exceeds the preregistered `1e-10` observation tolerance or changes a recorded reward, action, distance or termination flag.
- **Proof**: [`docs/rl/22_l21_counterfactual_risk_dataset_prereg_2026-07-15.md`, `docs/rl/23_l21_counterfactual_risk_dataset_results_2026-07-15.md`, `results/research_platform/rl/l21_counterfactual_multiseed_20260715_v1/audit.json`]
- **Dependencies**: [C03]
- **Tags**: counterfactual, replay, reproducibility, MuJoCo, MPPI

## C11: One-step correction intervention provides sufficient independent risk labels
- **Statement**: Replacing one accepted BC latent action with one SAC correction and then recovering BC for a 20-step horizon yields enough harmful and beneficial train/validation groups to train an actor-independent risk estimator.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered dataset sufficiency gate fails because either class coverage, sample coverage, replay quality or training-seed coverage is inadequate.
- **Proof**: [`docs/rl/22_l21_counterfactual_risk_dataset_prereg_2026-07-15.md`, `docs/rl/23_l21_counterfactual_risk_dataset_results_2026-07-15.md`, `results/research_platform/rl/l21_counterfactual_multiseed_20260715_v1/audit.json`]
- **Dependencies**: [C09, C10]
- **Tags**: counterfactual, risk-dataset, intervention, negative-result, credit-assignment

## C12: Ten-step gated correction bursts provide sufficient three-class risk labels
- **Statement**: A ten-step target-consensus-LCB SAC burst followed by thirty frozen-BC steps yields enough harmful and beneficial train/validation groups under the preregistered joint return/distance label to train an independent risk estimator.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered data-sufficiency gate fails because harmful or beneficial coverage is inadequate despite valid replay and burst execution.
- **Proof**: [`docs/rl/24_l22_counterfactual_burst_prereg_2026-07-15.md`, `docs/rl/25_l22_counterfactual_burst_results_2026-07-15.md`, `results/research_platform/rl/l22_burst_multiseed_20260715_v1/audit.json`]
- **Dependencies**: [C10, C11]
- **Tags**: counterfactual, burst, risk-dataset, negative-result, credit-assignment

## C13: A ten-step burst exposes larger continuous effects than a one-step intervention
- **Statement**: In the audited development data, extending the intervention from one accepted correction to a ten-step gated burst expands the observed short-horizon goal-distance effect from approximately -3.8/+5.2 cm to -9.4/+18.5 cm while preserving exact history replay.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Recomputed L21/L22 audits do not reproduce the stated ranges or show nonzero replay drift.
- **Proof**: [`docs/rl/23_l21_counterfactual_risk_dataset_results_2026-07-15.md`, `docs/rl/25_l22_counterfactual_burst_results_2026-07-15.md`]
- **Dependencies**: [C10, C11]
- **Tags**: counterfactual, sequence-effect, continuous-target, descriptive

## C14: Forty-step return delta independently corroborates distance effect
- **Statement**: The preregistered forty-step return delta adds a substantially independent outcome signal beyond goal-distance improvement for classifying burst benefit and harm.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Return/distance correlation is near one, signs nearly always agree, and the return threshold alone prevents all otherwise large distance effects from receiving a non-neutral label.
- **Proof**: [`docs/rl/25_l22_counterfactual_burst_results_2026-07-15.md`, `results/research_platform/rl/l22_burst_multiseed_20260715_v1/samples_all_training_seeds_eda_report.md`]
- **Dependencies**: [C12, C13]
- **Tags**: reward-design, target-redundancy, negative-result, EDA

## C15: Current-state features are sufficient for safe sequence-utility gating
- **Statement**: A group-bootstrap neural ensemble using the frozen 67-dimensional pre-intervention state features predicts forty-step burst utility well enough to improve over the zero predictor, classify meaningful effect signs and admit a nontrivial safe lower-confidence subset.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered development gate fails any of its predictive-accuracy, meaningful-sign or calibrated-acceptance requirements.
- **Proof**: [`docs/rl/26_l23_continuous_utility_prereg_2026-07-15.md`, `docs/rl/28_l23_continuous_utility_results_2026-07-15.md`, `results/research_platform/rl/l23_utility_ensemble_development_20260715_v2/development_metrics.json`]
- **Dependencies**: [C13, C14]
- **Tags**: continuous-utility, sequence-effect, negative-result, development-gate

## C16: Deep-ensemble disagreement calibrates sequence-utility uncertainty
- **Statement**: Disagreement among independently bootstrapped utility networks is large where forty-step utility prediction error is large, allowing a useful one-sided confidence bound after grouped conformal calibration.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Ensemble disagreement remains much smaller than observed errors, conformal scaling becomes extreme, or the calibrated lower bound accepts no development samples.
- **Proof**: [`docs/rl/28_l23_continuous_utility_results_2026-07-15.md`, `results/research_platform/rl/l23_utility_ensemble_development_20260715_v2/development_metrics.json`]
- **Dependencies**: [C15]
- **Tags**: epistemic-uncertainty, conformal, calibration, negative-result

## C17: Ten-step burst utility is trajectory-phase dependent
- **Statement**: In L23 development data, the scale and sign of forty-step utility vary substantially with branch phase, while no single frozen current-state feature shows a stable moderate correlation across train, selection and calibration splits.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Recomputed grouped EDA shows uniform effect variance across branch steps or a stable same-sign feature association of at least absolute 0.2 in all three development splits.
- **Proof**: [`docs/rl/28_l23_continuous_utility_results_2026-07-15.md`, `results/research_platform/rl/l23_utility_data_multiseed_20260715_v1/samples_all_training_seeds_eda_report.md`]
- **Dependencies**: [C13, C15]
- **Tags**: trajectory-context, phase, representation, EDA

## C18: Side-effect-free MPPI preview preserves counterfactual treatment outcomes
- **Statement**: Cloning the MPPI random state and restoring all latent-prior state permits paired BC/SAC candidate-trajectory preview without changing the subsequently executed branch treatment or any recorded non-feature outcome.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Enabling trajectory preview changes any baseline/candidate return, distance, clearance, safety, intervention-accounting or other non-feature field under the same checkpoint, episode seed and branch contract.
- **Proof**: [`docs/rl/29_l24_trajectory_utility_prereg_2026-07-15.md`, `docs/rl/30_l24_trajectory_utility_results_2026-07-15.md`, `results/research_platform/rl/l24_trajectory_utility_smoke_20260715_v1/`, `results/research_platform/rl/l24_preview_regression_no_preview_20260715_v1/`]
- **Dependencies**: [C10]
- **Tags**: MPPI, preview, common-random-numbers, counterfactual, reproducibility

## C19: One paired candidate preview improves macro-utility prediction over current state
- **Statement**: Adding paired BC/SAC MPPI preview metrics and their differences to the frozen state representation improves selection RMSE by at least five percent over a same-data, same-seed state-only ensemble when predicting the same ten-step burst's forty-step utility.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The trajectory-augmented ensemble fails the preregistered five-percent improvement over state-only selection RMSE, degrades independent group-mean prediction, or reduces meaningful-effect sign accuracy.
- **Proof**: [`docs/rl/29_l24_trajectory_utility_prereg_2026-07-15.md`, `docs/rl/30_l24_trajectory_utility_results_2026-07-15.md`, `results/research_platform/rl/l24_trajectory_utility_ensemble_development_20260715_v1/development_metrics.json`]
- **Dependencies**: [C15, C17, C18]
- **Tags**: trajectory-features, utility, ablation, negative-result, development-gate

## C20: Current candidate-trajectory differences have stable utility associations across development splits
- **Statement**: At least one preregistered candidate-minus-baseline trajectory metric has a same-sign group-mean correlation of absolute magnitude at least 0.2 with forty-step utility across train, model-selection and calibration splits.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: No feature reaches the stated magnitude with the same sign in all three splits, and apparent selection-only associations fail to reproduce in calibration.
- **Proof**: [`docs/rl/30_l24_trajectory_utility_results_2026-07-15.md`, `results/research_platform/rl/l24_trajectory_utility_data_multiseed_20260715_v1/trajectory_eda.md`, `results/research_platform/rl/l24_trajectory_utility_data_multiseed_20260715_v1/trajectory_eda.json`]
- **Dependencies**: [C17, C19]
- **Tags**: trajectory-features, stability, split-shift, EDA, negative-result

## C21: LaserScan complexity gating exactly recovers traditional MPPI in obstacle-free geometry
- **Statement**: With the preregistered L25 complexity mapping, an obstacle-free valid LaserScan produces zero RL mixing and exactly reproduces traditional MPPI step by step under paired checkpoint and episode seeds.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any paired clean-dynamics step differs in executed control, goal distance, collision or safety-override state, or any clean-dynamics gate alpha is nonzero.
- **Proof**: [`docs/rl/31_l25_scene_complexity_gate_prereg_2026-07-15.md`, `docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md`, `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/eda.json`]
- **Dependencies**: [C03, C09]
- **Tags**: LaserScan, scene-complexity, exact-fallback, paired-design

## C22: Complexity-gated RL improves traditional MPPI in blocking static scenes at K=200
- **Statement**: On the L25 development set at `K=200`, the fixed LaserScan complexity gate improves traditional MPPI success in the single-obstacle, narrow-corridor and U-trap scenes across three independently trained checkpoints without collision regression.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The audited paired data fail to reproduce gated/traditional success counts of 20/0, 24/0 and 26/0 in the three blocking scenes, or any gated collision regression appears.
- **Proof**: [`docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md`, `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/summary.json`, `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/audit.json`]
- **Dependencies**: [C09, C21]
- **Tags**: RL-prior, MPPI, scene-complexity, development-result, static-obstacles

## C23: A global simple-scene label is a sufficient proxy for low RL need
- **Statement**: Globally simple scenes, including a scene with only one obstacle, should keep the local RL-prior activation below the preregistered simple-scene alpha and active-fraction limits.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: A globally simple scene forms a decisive local blockage, requires sustained gate activation, and benefits from RL-prior activation.
- **Proof**: [`docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md`, `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/development_gate.json`]
- **Dependencies**: [C21, C22]
- **Tags**: scene-label, local-geometry, negative-result, preregistration

## C24: Complexity gating uniformly dominates always-on RL and frozen BC
- **Statement**: The L25 complexity gate preserves simple-scene performance while matching or improving both always-on LCB and frozen-BC priors in every blocking scene.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Gating loses paired successes to either comparator in any blocking scene.
- **Proof**: [`docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md`, `results/research_platform/rl/l25_scene_complexity_development_multiseed_20260715_v1/eda.json`]
- **Dependencies**: [C22]
- **Tags**: gating, ablation, negative-result, non-dominance

## C25: The proposed static real-robot layouts are geometrically traversable under the current envelope assumption
- **Statement**: Each configured static obstacle scene in the 6.5 m field admits a start-to-goal path on a 5 cm occupancy grid when obstacles are inflated by a 0.25 m robot radius and an additional 0.05 m geometry buffer.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The layout regression loses reachability, a configured passage is narrower than 0.85 m, or measurement shows that the real robot swept footprint requires more than the assumed 0.30 m inflation.
- **Proof**: [`tests/platform/test_real_robot_obstacle_layout.py`, `configs/real_robot/obstacle_kit_6p5m.yaml`, `docs/real_robot/obstacle_kit_6p5m/scene_placements.csv`]
- **Dependencies**: [C22, C23]
- **Tags**: real-robot, geometry, reachability, obstacle-kit, conditional-evidence

## C26: Complexity-gated RL improves MPPI sample efficiency in static blocking scenes
- **Statement**: On the L26 development set, complexity-gated RL at `K=100` exceeds traditional MPPI at `K=400` in all three static blocking scenes while causing no collision regression.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The audited paired design fails to reproduce success-rate differences of +0.467, +0.633 and +0.667 for single-obstacle, narrow-corridor and U-trap scenes, or any collision regression appears.
- **Proof**: [`docs/rl/34_l26_sample_efficiency_results_2026-07-15.md`, `results/research_platform/rl/l26_sample_efficiency_development_multiseed_20260715_v1/analysis.json`, `results/research_platform/rl/l26_sample_efficiency_development_multiseed_20260715_v1/development_gate.json`]
- **Dependencies**: [C21, C22]
- **Tags**: sample-efficiency, RL-prior, MPPI, development-result, static-obstacles

## C27: Lower MPPI sample count does not imply proportional compute efficiency
- **Statement**: In the unoptimized L26 CPU implementation, gated RL at `K=100` requires 8.663 ms/step versus 9.168 ms/step for traditional MPPI at `K=400`, so the 0.945 time ratio fails the preregistered `<=0.60` compute-efficiency threshold despite higher success.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Re-analysis of the frozen L26 outputs yields a planner-time ratio at or below 0.60 under the preregistered aggregation, or shows that learned inference was not included in the measured planner time.
- **Proof**: [`docs/rl/34_l26_sample_efficiency_results_2026-07-15.md`, `results/research_platform/rl/l26_sample_efficiency_development_multiseed_20260715_v1/development_gate.json`]
- **Dependencies**: [C26]
- **Tags**: compute-efficiency, negative-result, wall-clock, preregistration

## C28: Zero-contribution learned inference can be skipped exactly
- **Statement**: When the scene-complexity gate provably forces final alpha to zero and learned covariance is disabled, the L27 fast path preserves every paired control outcome while reducing mean planner time by 23.6% in clean geometry and 9.4% in blocking scenes on the development set.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any of the 26208 paired steps differs in executed control, goal distance, gate alpha, collision or safety state, or the hierarchical-bootstrap timing reduction crosses zero.
- **Proof**: [`docs/rl/36_l27_zero_complexity_fastpath_results_2026-07-15.md`, `results/research_platform/rl/l27_zero_complexity_fastpath_development_multiseed_20260715_v1/development_gate.json`, `results/research_platform/rl/l27_zero_complexity_fastpath_development_multiseed_20260715_v1/timing_analysis.json`]
- **Dependencies**: [C21, C27]
- **Tags**: behavior-preserving, inference-fast-path, scene-complexity, development-result

## C29: Unselected critic diagnostics can be removed without changing the selected gate
- **Statement**: In the L28 development implementation, evaluating only the selected target twin critics and optionally reusing the deterministic actor base action exactly preserves executed controls, gate variables, target Q-values, success and collision outcomes while producing a positive blocking-scene timing reduction.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any of the 26578 paired steps per candidate differs in a pre-registered behavior field, any episode success or collision outcome differs, or the hierarchical-bootstrap planner-time reduction includes zero.
- **Proof**: [`docs/rl/38_l28_inference_profile_results_2026-07-15.md`, `results/research_platform/rl/l28_inference_profile_development_multiseed_20260715_v1/development_gate.json`, `results/research_platform/rl/l28_inference_profile_development_multiseed_20260715_v1/profile_steps.csv`]
- **Dependencies**: [C27, C28]
- **Tags**: behavior-preserving, critic-pruning, profiling, development-result

## C30: Exact diagnostic pruning is sufficient to close the active-prior compute gap
- **Statement**: Selected-target-critic inference plus actor-base reuse reduces active-prior time by at least 20% and blocking-scene end-to-end planner time by at least 5% relative to full diagnostics.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Either fixed practical threshold is missed under the preregistered paired aggregation, even if the measured speedup is positive.
- **Proof**: [`docs/rl/37_l28_inference_profile_prereg_2026-07-15.md`, `docs/rl/38_l28_inference_profile_results_2026-07-15.md`, `results/research_platform/rl/l28_inference_profile_development_multiseed_20260715_v1/development_gate.json`]
- **Dependencies**: [C27, C29]
- **Tags**: compute-efficiency, practical-threshold, negative-result, preregistration

## C31: Scan-gated cross-layer control exactly recovers traditional MPPI in clean geometry
- **Statement**: In the L29/L30 blocked development design, both the spatial and temporal gates produce zero learned-prior contribution and exactly match residual-matched traditional MPPI step by step in obstacle-free geometry.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any of the 7,806 paired clean steps differs in executed control, goal distance, collision, safety override or gate alpha, or any paired step is missing.
- **Proof**: [`docs/rl/41_l29_l30_cross_layer_results_2026-07-15.md`, `results/research_platform/rl/l30_temporal_gate_development_multiblock_20260715_v1/development_gate.json`]
- **Dependencies**: [C21, C28]
- **Tags**: exact-fallback, cross-layer, LaserScan, development-result

## C32: Spatial LaserScan complexity alone is sufficient for lateral dynamic crossing
- **Statement**: Front proximity, bilateral constriction and density activate the RL sampling prior early enough to prevent collision with a sparse obstacle crossing laterally.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Spatial-gated methods collide frequently while always-on RL remains safer because the gate alpha stays low until the one-sided obstacle is already close.
- **Proof**: [`docs/rl/39_l29_cross_layer_factorial_prereg_2026-07-15.md`, `docs/rl/41_l29_l30_cross_layer_results_2026-07-15.md`, `results/research_platform/rl/l29_cross_layer_development_multiblock_20260715_v1/condition_summary.csv`]
- **Dependencies**: [C21, C22]
- **Tags**: dynamic-obstacle, temporal-risk, negative-result, gating

## C33: Temporal-gate plus ICODE is robust across the L30 development blocking strata
- **Statement**: With the fixed scan-temporal closing rule, the combined temporal-gate + target-LCB RL + ICODE method succeeds without collision across all L30 static/dynamic and seen/unseen development cells while retaining exact clean fallback.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Re-analysis fails to reproduce 60/60 blocking successes, 0/60 blocking collisions, 30/30 clean successes or exact paired clean fallback across three model blocks.
- **Proof**: [`docs/rl/41_l29_l30_cross_layer_results_2026-07-15.md`, `results/research_platform/rl/l30_temporal_gate_development_multiblock_20260715_v1/condition_summary.csv`, `results/research_platform/rl/l30_temporal_gate_development_multiblock_20260715_v1/fig_l29_l30_cross_layer_development.pdf`]
- **Dependencies**: [C31, C32]
- **Tags**: ICODE, RL-prior, temporal-gate, dynamic-obstacle, development-result

## C34: L30 satisfies its preregistered development eligibility gate
- **Statement**: Both temporal-gate residual levels are collision-noninferior to residual-matched always-on RL in each dynamic physics stratum, permitting confirmation.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Any dynamic stratum has a positive aggregate collision difference relative to always-on RL under the frozen L30 rule.
- **Proof**: [`docs/rl/40_l30_temporal_closing_gate_prereg_2026-07-15.md`, `results/research_platform/rl/l30_temporal_gate_development_multiblock_20260715_v1/development_gate.json`]
- **Dependencies**: [C33]
- **Tags**: preregistration, safety-gate, negative-result, confirmation-sealed

## C35: L30 temporal-gate plus ICODE generalizes across obstacle motion variants
- **Statement**: With L30 thresholds frozen, temporal-gate + ICODE remains collision-free and at least 90% successful when crossing direction, speed, phase and obstacle size change on fresh development seeds.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The primary method has any collision, fewer than 108/120 successes, or fewer than 12/15 successes in any preregistered dynamic-variant/physics stratum.
- **Proof**: [`docs/rl/42_l31_dynamic_variant_generalization_prereg_2026-07-15.md`, `docs/rl/43_l31_dynamic_variant_generalization_results_2026-07-15.md`, `results/research_platform/rl/l31_dynamic_variant_development_multiblock_20260715_v1/development_gate.json`]
- **Dependencies**: [C33, C34]
- **Tags**: dynamic-generalization, ICODE, RL-prior, negative-result, preregistration

## C36: Consecutive sector minima are a physically valid obstacle closing-speed estimator
- **Statement**: Differencing the minimum LaserScan clearance in fixed angular sectors tracks obstacle-relative closing speed closely enough to support dynamic controller gating across direction changes.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Fresh motion variants produce nonphysical rate spikes from beam or sector reassociation, especially when motion direction changes.
- **Proof**: [`results/research_platform/rl/l31_dynamic_variant_development_multiblock_20260715_v1/failure_diagnostics.json`, `docs/rl/43_l31_dynamic_variant_generalization_results_2026-07-15.md`]
- **Dependencies**: [C32, C35]
- **Tags**: LaserScan, scan-flow, dynamic-obstacle, observability, negative-result

## C37: High temporal hazard implies high competence of the current RL prior
- **Statement**: Increasing the RL-prior blend monotonically with scan-derived closing risk is a safe controller-selection rule for moving obstacles.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The temporal gate saturates during collision, or the identical high-risk gate selects a policy that lacks motion-direction observability and dynamic-domain training.
- **Proof**: [`results/research_platform/rl/l31_dynamic_variant_development_multiblock_20260715_v1/failure_episode_diagnostics.csv`, `docs/rl/43_l31_dynamic_variant_generalization_results_2026-07-15.md`]
- **Dependencies**: [C35, C36]
- **Tags**: hazard, competence, uncertainty-gating, RL-prior, negative-result

## C38: Multi-beam robust scan-flow removes the L31 nonphysical rate failure
- **Statement**: Requiring contiguous multi-beam support, median aggregation and physical jump bounds prevents the 29--32 m/s sector-reassociation spikes observed in L31 while retaining scan-only temporal hazard estimates.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The audited L32 development episodes exceed the configured 1.5 m/s physical bound, produce non-finite rates, or reproduce the L31 reassociation spikes despite valid support diagnostics.
- **Proof**: [`docs/rl/46_l32_temporal_safety_remediation_results_2026-07-16.md`, `results/research_platform/rl/l32_temporal_safety_remediation_development_20260715_v1/audit.json`]
- **Dependencies**: [C36]
- **Tags**: LaserScan, scan-flow, robust-estimation, development-result

## C39: Reactive temporal safety is insufficient to make a static-trained RL prior dynamically competent
- **Statement**: A shared scan-only TTC-style slow/stop layer can reduce collisions for traditional MPPI, but it does not by itself make the existing single-scan/static-trained learned prior robust across reverse, fast and large dynamic-obstacle variants.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis shows the robust learned condition with shared safety is collision-free and noninferior across all L32 motion/physics strata, or the shared layer provides lateral predictive avoidance rather than only bounded longitudinal intervention.
- **Proof**: [`docs/rl/46_l32_temporal_safety_remediation_results_2026-07-16.md`, `results/research_platform/rl/l32_temporal_safety_remediation_development_20260715_v1/paired_summary.csv`]
- **Dependencies**: [C37, C38]
- **Tags**: dynamic-obstacle, safety-layer, competence, negative-result

## C40: Full learned-prior authority is learnable under the L33 dynamic curriculum
- **Statement**: With three-frame LaserScan input and randomized dynamic-domain SAC training, completely replacing the traditional MPPI sampling prior produces a trained checkpoint that improves held-out success over its warm-start reference.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Any preregistered L33 trained checkpoint achieves positive held-out success improvement without collision regression and the replay contains successful training episodes.
- **Proof**: [`docs/rl/49_l33_l34_dynamic_bounded_correction_results_2026-07-16.md`, `results/research_platform/rl/dynamic_history_bounded_l34_seed20260731_30k_20260716_v1/checkpoint_summary.csv`]
- **Dependencies**: [C39]
- **Tags**: SAC, dynamic-training, learned-prior, negative-result

## C41: Bounded RL correction improves dynamic held-out control across training seeds
- **Statement**: With learned-prior authority fixed at 0.25, three independent SAC training seeds each improve held-out success without increasing collision relative to their own step-zero reference on two unseen motion paths in a combined-unseen MuJoCo physics domain.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The audited three-seed development data fail to reproduce positive net success in all three seeds, show a seed-level collision increase, contain any paired success loss or collision regression, or fail artifact integrity.
- **Proof**: [`docs/rl/49_l33_l34_dynamic_bounded_correction_results_2026-07-16.md`, `results/research_platform/rl/l34_bounded_dynamic_multiseed_development_20260716_v1/multiseed_summary.json`, `ara/evidence/tables/table02_l34_dynamic_multiseed.md`]
- **Dependencies**: [C40]
- **Tags**: SAC, MPPI, bounded-correction, dynamic-obstacle, multiseed-development
- **Evidence update**: L35 did not reproduce this development effect on new episode seeds; C42 records the failed independent-generalization claim. C41 remains scoped only to the frozen L34 model-selection cells.

## C42: L34 bounded-RL gains independently generalize to new episode seeds
- **Statement**: The three L34 best checkpoints retain positive pooled success, collision noninferiority and positive success change in at least two training seeds when evaluated on independently seeded episodes under the same held-out motion paths and combined-unseen physics domain.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered L35 confirmation gate fails, pooled success is nonpositive, or fewer than two training seeds show positive success change.
- **Proof**: [`docs/rl/51_l35_independent_confirmation_results_2026-07-16.md`, `results/research_platform/rl/l35_l34_independent_confirmation_20260716_v1/confirmation_summary.json`, `ara/evidence/tables/table03_l35_independent_confirmation.md`]
- **Dependencies**: [C41]
- **Tags**: SAC, bounded-correction, independent-confirmation, negative-result

## C43: Traditional-only calibration yields identifiable dynamic benchmark strata
- **Statement**: Selecting dynamic-obstacle geometries using only traditional GoalWarmStart MPPI produces easy, moderate and hard development scenes with both success and collision outcomes and at least 0.25 collision-rate span, without selecting on RL or ICODE performance.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Fewer than three candidates have both success and collision, the selected collision-rate span is below 0.25, or RL/ICODE outcomes influence scene selection.
- **Proof**: [`docs/rl/53_l36_dynamic_benchmark_calibration_results_2026-07-16.md`, `results/research_platform/rl/l36_dynamic_benchmark_calibration_20260716_v1/calibration_summary.json`, `ara/evidence/tables/table04_l36_dynamic_calibration.md`]
- **Dependencies**: [C42]
- **Tags**: benchmark-design, identifiability, dynamic-obstacle, method-blind-selection

## C44: Current bounded RL and ICODE improve calibrated dynamic closed-loop control
- **Statement**: Relative to traditional nominal MPPI, bounded RL and ICODE have nonnegative main effects, and their combination is success- and collision-noninferior across all three paired model blocks on the L37 calibrated development scenes.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Either main effect is adverse, the combined method loses successes or adds collisions, or any model block ranks the combination below the best comparator.
- **Proof**: [`docs/rl/55_l37_bounded_rl_icode_factorial_results_2026-07-16.md`, `results/research_platform/rl/l37_bounded_rl_icode_factorial_20260716_v1/factorial_summary.json`, `ara/evidence/tables/table05_l37_factorial.md`]
- **Dependencies**: [C42, C43]
- **Tags**: ICODE, bounded-RL, factorial, closed-loop, negative-result

## C45: L37 shows a positive success interaction without absolute combined benefit
- **Statement**: In the L37 development factorial, the bounded-RL by ICODE success difference-in-differences is positive even though the combined controller remains worse than traditional nominal MPPI in absolute success, collision and final-distance metrics.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Re-analysis fails to reproduce a success interaction of +0.222 with a hierarchical-bootstrap 95% interval above zero, or the combined controller is not absolutely worse on the preregistered primary comparisons.
- **Proof**: [`docs/rl/55_l37_bounded_rl_icode_factorial_results_2026-07-16.md`, `results/research_platform/rl/l37_bounded_rl_icode_factorial_20260716_v1/factorial_summary.json`]
- **Dependencies**: [C44]
- **Tags**: interaction, difference-in-differences, interpretation-boundary, development-result

## C46: Structure- and delay-aligned ICODE improves H=36 on-policy prediction
- **Statement**: Restricting residual outputs to dynamic states, using interval-average applied controls and modeling known command delay produces independently initialized ICODE checkpoints that reduce H=36 total, position and heading error on both held-out test and unseen-domain path transitions.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any of the three frozen L49 checkpoints fails to beat nominal total H=36 RMSE on either split, or fewer than two improve both position and heading on either split.
- **Proof**: [`results/research_platform/l49_icode_iterative_path_offline_gate_v1/summary.json`, `docs/rl/74_l49_l50_iterative_path_results_2026-07-16.md`]
- **Dependencies**: [C44]
- **Tags**: ICODE, multi-step, command-delay, applied-control, held-out-prediction

## C47: Delay-aligned ICODE globally improves path-tracking accuracy
- **Statement**: Always-on delay-aligned ICODE reduces pooled closed-loop cross-track RMSE relative to nominal MPPI across the frozen L47 path and physics blocks.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered L47 pooled relative cross-track reduction is below 10%, its confidence interval crosses zero, or safety/task noninferiority fails.
- **Proof**: [`docs/rl/69_l47_delay_aligned_path_tracking_results_2026-07-16.md`]
- **Dependencies**: [C46]
- **Tags**: ICODE, path-tracking, closed-loop, negative-result

## C48: ICODE reduces closed-loop control jerk under stronger delay mismatch
- **Statement**: In the frozen long-delay MuJoCo stratum, task-specific ICODE reduces both issued-control and plant-applied-control jerk without net success loss or collision increase.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: A new-seed replication makes either hierarchical-bootstrap 95% lower bound nonpositive, adds collisions, or loses net successes.
- **Proof**: [`results/research_platform/rl/l50_icode_iterative_path_confirmation_20260716_v1/efficiency_paired_effects.csv`, `docs/rl/74_l49_l50_iterative_path_results_2026-07-16.md`, `ara/evidence/figures/fig06_l49_l50.md`]
- **Dependencies**: [C46, C47]
- **Tags**: ICODE, smoothness, control-jerk, mismatch-stratification, exploratory-interaction

## C49: Task-specific residual retraining closes the global path-efficiency gate
- **Statement**: L49 task-specific retraining makes ICODE pass all inherited L48 path-length, jerk, tracking, safety and compute clauses across both physics domains.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Any frozen L50 joint-gate clause fails.
- **Proof**: [`results/research_platform/rl/l50_icode_iterative_path_confirmation_20260716_v1/efficiency_confirmation_summary.json`, `docs/rl/74_l49_l50_iterative_path_results_2026-07-16.md`]
- **Dependencies**: [C46, C48]
- **Tags**: ICODE, independent-confirmation, joint-gate, negative-result

## C50: Conditional residual authority has a useful simulator-domain oracle upper bound
- **Statement**: If residual authority is enabled only in the frozen long-delay mismatch stratum, ICODE reduces issued and applied control jerk without changing success or collision outcomes across all three model blocks.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis of L51 makes either pooled jerk interval nonpositive, any model block nonpositive, or changes success/collision.
- **Proof**: [`results/research_platform/rl/l51_icode_oracle_domain_gate_20260716_v1/oracle_gate_summary.json`]
- **Dependencies**: [C48, C49]
- **Tags**: ICODE, oracle-gate, upper-bound, non-deployable

## C51: Past relative one-step innovation alone identifies residual closed-loop utility
- **Statement**: A fail-closed causal gate using only exponentially forgotten nominal-versus-ICODE relative one-step error can separate the matched- and long-delay benefit domains strongly enough to reproduce the L51 authority pattern.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: No preregistered L52 candidate reaches the required worst-block alpha separation while satisfying cold-start and matched-domain constraints.
- **Proof**: [`results/research_platform/rl/l52_icode_online_reliability_calibration_20260716_v1/reliability_calibration_summary.json`, `docs/rl/76_l52_online_reliability_calibration_results_2026-07-16.md`]
- **Dependencies**: [C50]
- **Tags**: ICODE, online-reliability, innovation, negative-result

## C52: Horizon-matched ICODE improves fixed-plant high-dynamic prediction
- **Statement**: On the frozen L56 fixed high-dynamic MuJoCo plant, three independently initialized control-affine residual models reduce active derivative RMSE and H=36 rollout RMSE relative to nominal dynamics on both held-out test paths and the unseen reverse-S path.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Any model block fails the frozen long-horizon gate on either split, dataset hashes differ, or episode-wise split integrity fails.
- **Proof**: [`results/research_platform/l57_icode_high_dynamic_offline_gate_20260716_v1.json`, `docs/rl/82_l57_high_dynamic_icode_offline_results_2026-07-16.md`]
- **Dependencies**: [C46, C51]
- **Tags**: ICODE, fixed-plant, high-dynamic, H36, held-out-prediction

## C53: High-dynamic ICODE improves residual-only closed-loop tracking on sealed seeds
- **Statement**: On the frozen fixed high-dynamic plant and four clean path families, including an unseen reverse-S path, ICODE reduces cross-track RMSE relative to nominal MPPI across every model block and scene while preserving success, collision safety and the 50 ms mean-compute bound on independently sealed L59 seeds.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis fails to reproduce the positive paired 95% interval, any block or scene becomes nonpositive, success falls, collisions increase, seed integrity fails, or mean ICODE compute exceeds 50 ms.
- **Proof**: [`results/research_platform/rl/l59_icode_high_dynamic_confirmation_20260716_v1/high_dynamic_closed_loop_summary.json`, `docs/rl/85_l59_high_dynamic_confirmation_results_2026-07-16.md`]
- **Dependencies**: [C52]
- **Tags**: ICODE, MPPI, independent-confirmation, fixed-plant, high-dynamic, closed-loop

## C54: L59 establishes general learned-dynamics robustness beyond its fixed clean plant
- **Statement**: The L59 result establishes cross-plant, noisy-odometry, obstacle-rich, dynamic-obstacle, real-robot and RL-composed robustness.
- **Status**: untested
- **Provenance**: ai-suggested
- **Falsification criteria**: Any orthogonal factor produces nonpositive tracking benefit, unsafe outcomes or compute failure under an independently frozen protocol.
- **Proof**: [`docs/rl/85_l59_high_dynamic_confirmation_results_2026-07-16.md`]
- **Dependencies**: [C53]
- **Tags**: scope-boundary, external-validity, untested

## C55: Lower aggregate offline rollout RMSE implies better MPPI closed-loop tracking
- **Statement**: Between residual models trained on identical data with matched parameter count, the model with lower held-out H=36 state-rollout RMSE must also produce lower receding-horizon MPPI cross-track error.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: A model with worse held-out H=36 rollout RMSE achieves a positive independently confirmed paired closed-loop tracking advantage under the same MPPI contract.
- **Proof**: [`results/research_platform/l60_residual_structure_offline_20260716_v1/summary.json`, `results/research_platform/rl/l62_residual_structure_confirmation_20260716_v1/residual_structure_closed_loop_summary.json`, `docs/rl/90_l62_residual_structure_confirmation_results_2026-07-16.md`]
- **Dependencies**: [C52, C53]
- **Tags**: offline-to-control-gap, residual-model, MPPI, negative-result

## C56: Control-affine ICODE improves fixed-plant closed-loop tracking over a parameter-matched direct MLP
- **Statement**: On the frozen fixed high-dynamic clean-state plant, control-affine ICODE reduces MPPI cross-track RMSE relative to a direct MLP residual with matched data, seeds, optimizer, H=36 loss, output mask and parameter count, across every model block and the unseen path on sealed L62 seeds without success, collision or mean-compute gate regression.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis makes the paired overall or unseen confidence-interval lower bound nonpositive, any model block nonpositive, seed/hash integrity fail, success fall, collisions rise, or mean compute exceed 50 ms.
- **Proof**: [`results/research_platform/rl/l62_residual_structure_confirmation_20260716_v1/residual_structure_closed_loop_summary.json`, `results/research_platform/rl/l62_residual_structure_confirmation_20260716_v1/figures/fig_l60_l62_residual_structure_ablation.pdf`, `docs/rl/90_l62_residual_structure_confirmation_results_2026-07-16.md`]
- **Dependencies**: [C52, C53, C55]
- **Tags**: ICODE, MLP, structural-ablation, independent-confirmation, fixed-plant, closed-loop

## C57: Control-affine ICODE retains a closed-loop advantage across bounded MuJoCo plant shifts
- **Statement**: With checkpoints trained only on the fixed L56 anchor plant, control-affine ICODE reduces MPPI cross-track RMSE relative to both nominal dynamics and a parameter-matched direct MLP across frozen mass, friction, actuator, delay and combined MuJoCo parameter shifts, including an unseen path, on sealed L65 execution seeds without success, collision or mean-compute regression.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis makes the shifted-plant, unseen-shifted or combined-domain ICODE-versus-MLP interval lower bound nonpositive; fewer than four shifted domains or two training blocks are positive; calibration is shown to use learned-model outcomes; seed/hash integrity fails; success falls; collisions rise; or mean compute exceeds 50 ms.
- **Proof**: [`results/research_platform/rl/l65_residual_structure_cross_plant_confirmation_20260716_v1/residual_structure_cross_plant_summary.json`, `results/research_platform/rl/l65_residual_structure_cross_plant_confirmation_20260716_v1/figures/fig_l64_l65_cross_plant_residual_structure.pdf`, `docs/rl/95_l63_l65_cross_plant_residual_structure_results_2026-07-16.md`, `ara/evidence/tables/table07_l63_l65_cross_plant.md`, `ara/evidence/figures/fig07_l64_l65_cross_plant.md`]
- **Dependencies**: [C54, C56]
- **Tags**: ICODE, MLP, cross-plant, structural-ablation, independent-confirmation, MuJoCo, closed-loop

## C58: Control-affine ICODE retains a closed-loop advantage under bounded observation latency and noise
- **Statement**: With checkpoints trained on clean anchor-plant data, control-affine ICODE reduces MPPI cross-track RMSE relative to both nominal dynamics and a parameter-matched direct MLP under frozen clean-state, 100 ms observation-latency and moderate observation-noise-plus-latency domains, including an unseen path, on sealed L68 execution seeds without success, collision or mean-compute regression.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis makes the primary, shifted-domain or unseen-domain ICODE-versus-MLP interval lower bound nonpositive; any primary domain or model block becomes nonpositive; calibration is shown to use learned-model outcomes; seed/hash integrity fails; success falls; collisions rise; or mean compute exceeds 50 ms.
- **Proof**: [`results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/residual_structure_observation_summary.json`, `results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/figures/fig_l67_l68_observation_robustness.pdf`, `docs/rl/99_l66_l68_observation_robustness_results_2026-07-16.md`, `ara/evidence/tables/table08_l66_l68_observation.md`, `ara/evidence/figures/fig08_l67_l68_observation.md`]
- **Dependencies**: [C56, C57]
- **Tags**: ICODE, MLP, observation-latency, observation-noise, structural-ablation, independent-confirmation, MuJoCo, closed-loop

## C59: Residual dynamics alone robustly correct raw wheel-odometry drift
- **Statement**: Without a separate localization or state-estimation correction, ICODE consistently improves raw wheel-odometry task success relative to nominal and parameter-matched MLP across development and sealed observation-domain experiments.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: An independently sealed comparison reverses the development success ordering or makes ICODE worse than nominal on raw wheel-odometry task success.
- **Proof**: [`results/research_platform/rl/l68_residual_structure_observation_confirmation_20260716_v1/figures/table_l68_wheel_odometry_stress.csv`, `docs/rl/99_l66_l68_observation_robustness_results_2026-07-16.md`, `ara/evidence/tables/table08_l66_l68_observation.md`]
- **Dependencies**: [C58]
- **Tags**: ICODE, odometry, localization, state-estimation, negative-result, scope-boundary

## C60: Free covariance RL satisfies both precision noninferiority and time superiority over a tuned fixed covariance
- **Statement**: Under frozen ICODE-MPPI, unconstrained covariance-only SAC with an episode-level precision dual is noninferior in cross-track RMSE and faster than the tuned fixed `(1.75, 0.75)` covariance.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The paired RMSE 95% upper bound exceeds +0.002 m, time 95% upper bound is nonnegative, or safety regresses.
- **Proof**: [`results/research_platform/rl/l85_precision_constrained_development_eval_20260717_v1/summary.json`, `docs/rl/129_l86_anchored_covariance_results_2026-07-17.md`]
- **Dependencies**: [C53, C58]
- **Tags**: RL, MPPI, covariance, precision-constraint, negative-result

## C61: Anchoring covariance RL to a tuned traditional MPPI setting preserves tracking precision
- **Statement**: A bounded log-residual covariance policy anchored at `(1.75, 0.75)` is cross-track noninferior to that fixed anchor while preserving success and collision outcomes on the L86 development matrix.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The paired RMSE 95% upper bound exceeds +0.002 m, success falls, or collision rises on a new-seed replication.
- **Proof**: [`results/research_platform/rl/l86_anchored_covariance_development_eval_20260717_v1/summary.json`, `docs/rl/129_l86_anchored_covariance_results_2026-07-17.md`]
- **Dependencies**: [C60]
- **Tags**: RL, MPPI, covariance-anchor, noninferiority, development-only

## C62: Anchored covariance RL is faster than its tuned fixed anchor
- **Statement**: The L86 bounded log-residual policy reduces time to goal relative to fixed `(1.75, 0.75)` while retaining precision and safety.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The paired time-to-goal 95% upper bound is nonnegative.
- **Proof**: [`results/research_platform/rl/l86_anchored_covariance_development_eval_20260717_v1/summary.json`, `docs/rl/129_l86_anchored_covariance_results_2026-07-17.md`]
- **Dependencies**: [C61]
- **Tags**: RL, MPPI, covariance-anchor, time-to-goal, negative-result

## C63: Closed-loop optimal MPPI covariance is path-context dependent
- **Statement**: Under frozen ICODE-MPPI on four clean path geometries and four bounded MuJoCo physics domains, a selection-seed context oracle uses at least two covariance options and beats the strongest globally fixed covariance on held-out seeds in both time to goal and cross-track RMSE without changing success or collision outcomes.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis finds duplicate or mixed selection/evaluation keys, fewer than two selected candidates, a nonnegative upper 95% bound for time change, an RMSE upper bound above +0.002 m, lower success, or more collisions.
- **Proof**: [`results/research_platform/rl/l87_covariance_context_oracle_20260718_v1/summary.json`, `docs/rl/131_l87_closed_loop_covariance_heterogeneity_results_2026-07-18.md`]
- **Dependencies**: [C61, C62]
- **Tags**: RL, MPPI, covariance, context-oracle, learnability, held-out-seeds

## C64: A contextual-bandit covariance policy improves frozen ICODE-MPPI on held-out route geometries
- **Statement**: A LinUCB policy fitted from closed-loop rewards on straight, sweep, accel-turn and chicane routes selects between audited narrow and speed MPPI sampling covariances and, on independently executed hairpin and reverse-S routes across four MuJoCo physics domains, reduces time to goal and cross-track RMSE relative to the strongest global fixed covariance without success, collision or measurable planner-compute regression.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The optimized 80-episode artifact has nonunique keys; the paired time 95% upper bound is nonnegative; the paired RMSE 95% upper bound exceeds +0.002 m; success falls; collision increases; or the planner-compute interval is strictly positive.
- **Proof**: [`results/research_platform/rl/l89_contextual_covariance_independent_eval_cached_20260718_v2/summary.json`, `docs/rl/134_l89_contextual_covariance_bandit_results_2026-07-18.md`]
- **Dependencies**: [C53, C57, C63]
- **Tags**: RL, contextual-bandit, ICODE, MPPI, held-out-geometry, independent-confirmation

## C65: Route-local high-frequency covariance switching has material oracle headroom
- **Statement**: Within a route, observable local polyline geometry creates enough conflicting MPPI covariance optima that segment-level covariance switching materially improves frozen ICODE-MPPI over a single route-level covariance.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: A preregistered local-context oracle must select at least two candidates within multiple route-domain cells and achieve a strictly positive progress interval or a material prediction/tracking benefit without safety regression.
- **Proof**: [`results/research_platform/rl/l90_local_covariance_context_oracle_20260718_v1/summary.json`, `docs/rl/137_l90_local_covariance_context_oracle_results_2026-07-18.md`]
- **Dependencies**: [C63, C64]
- **Tags**: RL, contextual-bandit, local-context, negative-result, adaptation-timescale

## C66: Scene-label covariance selection generalizes a dynamic-obstacle safety improvement
- **Statement**: Choosing a frozen MPPI covariance from a dynamic-scene label improves success or collision safety over the best global covariance and reproduces on independent dynamic-motion geometries.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: An independent safety confirmation must produce a positive success interval or a negative collision interval while preserving the complementary safety metric and task completion.
- **Proof**: [`results/research_platform/rl/l91_dynamic_covariance_context_oracle_20260718_v1/summary.json`, `results/research_platform/rl/l92_dynamic_covariance_safety_confirmation_20260718_v1/summary.json`, `docs/rl/140_l92_dynamic_covariance_safety_confirmation_results_2026-07-18.md`]
- **Dependencies**: [C64]
- **Tags**: RL, dynamic-obstacle, covariance, independent-confirmation, negative-result

## C67: Covariance adaptation alone resolves route-crossing dynamic-obstacle failures
- **Statement**: Without adding future obstacle-motion prediction to the rollout, selecting among narrow, turn and speed MPPI sampling covariances is sufficient to produce measurable safety or completion headroom on route-crossing dynamic obstacles.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: A method-blind route-crossing oracle must improve success or collision rate relative to the strongest global covariance while retaining safe clean-route behavior.
- **Proof**: [`results/research_platform/rl/l93_route_dynamic_covariance_headroom_20260718_v1/summary.json`, `docs/rl/142_l93_route_dynamic_headroom_results_2026-07-18.md`]
- **Dependencies**: [C66]
- **Tags**: dynamic-obstacle, prediction-horizon, covariance, negative-result, scope-boundary

## C68: Contextual covariance selection preserves performance at half the fixed-policy MPPI sample budget
- **Statement**: On independently executed hairpin and reverse-S routes across four bounded MuJoCo physics domains, the frozen L89 contextual bandit with K=50 preserves cross-track precision and safety relative to the strongest fixed speed covariance at K=100, while reducing time to goal and per-step planner compute; the result reproduces under single-process, arm-order-balanced execution.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Either the L94 preregistered comparison or the independent L95 controlled confirmation has duplicate keys, lower success, more collisions, RMSE upper 95% bound above +0.002 m, nonnegative time upper bound, or nonnegative compute upper bound.
- **Proof**: [`results/research_platform/rl/l94_contextual_covariance_sample_efficiency_20260718_v1/summary.json`, `results/research_platform/rl/l95_contextual_covariance_half_budget_confirmation_20260718_v1/summary.json`, `docs/rl/145_l94_contextual_covariance_sample_efficiency_results_2026-07-18.md`, `docs/rl/146_l95_half_budget_controlled_confirmation_results_2026-07-18.md`, `ara/evidence/tables/table09_l90_l95_covariance_adaptation.md`, `ara/evidence/figures/fig09_l94_l95_sample_efficiency.md`]
- **Dependencies**: [C64, C65, C66, C67]
- **Tags**: RL, contextual-bandit, ICODE, MPPI, sample-efficiency, independent-confirmation

## C69: The confirmed half-budget contextual policy trades smoothness for efficiency
- **Statement**: In both L94 and L95, contextual K=50 increases control jerk relative to fixed speed K=100 while preserving success, collision safety and cross-track noninferiority and reducing time and compute.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis makes either paired jerk interval include zero, or a safety, precision, time or compute gate used to contextualize the tradeoff fails.
- **Proof**: [`results/research_platform/rl/l94_contextual_covariance_sample_efficiency_20260718_v1/summary.json`, `results/research_platform/rl/l95_contextual_covariance_half_budget_confirmation_20260718_v1/summary.json`, `results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/summary.json`, `docs/rl/149_l97_contextual_covariance_jerk_confirmation_results_2026-07-18.md`]
- **Dependencies**: [C68]
- **Tags**: RL, sample-efficiency, control-jerk, tradeoff

## C70: The contextual half-budget efficiency effect replicates on a third untouched seed set
- **Statement**: In the single-process L97 three-arm study, raw contextual K=50 again preserves success, collision safety and +2 mm cross-track noninferiority relative to fixed K=100 while reducing completion time and planner compute across held-out hairpin/reverse-S routes and four bounded physics domains.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis reveals duplicate or incomplete blocks, lower success, more collisions, RMSE upper 95% bound above +0.002 m, nonnegative elapsed upper bound, or nonnegative compute upper bound.
- **Proof**: [`results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/summary.json`, `docs/rl/149_l97_contextual_covariance_jerk_confirmation_results_2026-07-18.md`, `ara/evidence/tables/table10_l96_l97_jerk_remediation.md`]
- **Dependencies**: [C68, C69]
- **Tags**: RL, contextual-bandit, ICODE, MPPI, sample-efficiency, replication

## C71: A hard yaw-command slew bound reduces contextual-controller jerk without task regression
- **Statement**: The L96-selected combination of a 2.5 rad/s^2 yaw-command rate limit and 0.16 MPPI control-rate weight reduces both issued and physically applied jerk relative to raw contextual K=50, and independently reproduces in L97 while retaining success, collision safety, +2 mm tracking noninferiority and +0.5 s completion-time noninferiority.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: L96 selection integrity fails, or the independent L97 smoothed-versus-raw issued/applied jerk upper interval is nonnegative, safety regresses, RMSE upper bound exceeds +0.002 m, or elapsed upper bound exceeds +0.5 s.
- **Proof**: [`results/research_platform/rl/l96_contextual_covariance_jerk_screening_20260718_v1/summary.json`, `results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/summary.json`, `docs/rl/148_l96_jerk_screening_results_and_l97_prereg_2026-07-18.md`, `docs/rl/149_l97_contextual_covariance_jerk_confirmation_results_2026-07-18.md`]
- **Dependencies**: [C69, C70]
- **Tags**: control-smoothing, slew-limit, mechanism, independent-confirmation

## C72: The complete smoothed contextual K50 package has confirmed jerk superiority over fixed K100
- **Statement**: Smoothed contextual K50 has issued- and applied-jerk 95% confidence intervals entirely at or below zero relative to fixed K100 while retaining the confirmed half-budget performance gates.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Either jerk upper 95% bound is positive under the preregistered L97 paired hierarchical analysis.
- **Proof**: [`results/research_platform/rl/l97_contextual_covariance_jerk_confirmation_20260718_v1/summary.json`, `docs/rl/149_l97_contextual_covariance_jerk_confirmation_results_2026-07-18.md`, `ara/evidence/figures/fig10_l96_l97_jerk_remediation.md`]
- **Dependencies**: [C70, C71]
- **Tags**: jerk, strict-gate, negative-result, scope-boundary

## C73: Additional MPPI samples have selectively predictable physical value
- **Statement**: Under one frozen ICODE dynamics model, proposal mean and covariance, the physical value of extending a nested K50 pool to K100 varies across sampled route states and can be predicted from information available after the first 50 rollouts under an explicit compute constraint.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Exact-prefix or snapshot integrity fails, Oracle ADD collapses outside `[0.10, 0.75]`, Oracle relative gain falls below 1%, or both constrained L101 and L103 policies fail to improve K50 while retaining at least 30% of Oracle gain under mean K no greater than 80.
- **Proof**: [`results/research_platform/rl/l101_budget_constrained_ridge_20260718_v1/summary.json`, `results/research_platform/rl/l103_calibrated_budget_bandit_20260718_v1/bandit_summary.json`, `docs/rl/156_l103_calibrated_budget_bandit_confirmation_results_2026-07-18.md`]
- **Dependencies**: [C53, C68]
- **Tags**: ICODE, MPPI, sample-efficiency, anytime-compute, common-random-numbers

## C74: A terminal online dual price transports the discovery compute envelope to deployment
- **Statement**: The final primal-dual price reached during contextual-bandit exploration is sufficient without calibration to preserve the intended ADD rate and Oracle-utility retention on held-out contexts.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Held-out ADD remains within the intended envelope, mean K remains no greater than 80, and Oracle-gain retention is at least 30% without post-training calibration.
- **Proof**: [`results/research_platform/rl/l102_primal_dual_budget_bandit_20260718_v1/bandit_summary.json`, `docs/rl/155_l102_failure_l103_calibrated_bandit_prereg_2026-07-18.md`]
- **Dependencies**: [C73]
- **Tags**: contextual-bandit, primal-dual, deployment-calibration, negative-result

## C75: A calibrated contextual bandit learns non-random ICODE-MPPI compute allocation at sampled states
- **Statement**: A partial-feedback contextual bandit trained only on selected ADD50 rewards and calibrated only on discovery predictions improves held-out true branch cost over fixed K50 under mean K no greater than 80, retains at least 30% of hindsight-Oracle gain, and allocates compute better than matched random selection across the same route and physics strata.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The L103 integrity audit fails; its K50 cost-delta interval includes zero; mean K exceeds 80; Oracle retention is below 30%; or a matched-budget randomization reanalysis places the learned statistic inside the random 95% interval.
- **Proof**: [`results/research_platform/rl/l103_calibrated_budget_bandit_20260718_v1/bandit_summary.json`, `docs/rl/156_l103_calibrated_budget_bandit_confirmation_results_2026-07-18.md`, `ara/evidence/tables/table11_l100_l103_anytime_budget.md`]
- **Dependencies**: [C73, C74]
- **Tags**: contextual-bandit, ICODE, MPPI, constrained-RL, sample-efficiency, confirmation

## C76: L103 establishes whole-episode closed-loop and ICODE-specific interaction benefit
- **Statement**: The sampled-state L103 counterfactual result by itself proves that adaptive budgeting improves complete closed-loop episodes and that its benefit statistically interacts with ICODE rather than nominal rollout dynamics.
- **Status**: untested
- **Provenance**: ai-suggested
- **Falsification criteria**: A nominal/ICODE x fixed/adaptive closed-loop factorial yields no adaptive-budget episode benefit or no identifiable dynamics-by-budget interaction.
- **Proof**: [`docs/rl/156_l103_calibrated_budget_bandit_confirmation_results_2026-07-18.md`]
- **Dependencies**: [C75]
- **Tags**: scope-boundary, closed-loop, factorial, ICODE-interaction, untested

## C77: A multi-domain direct-control SAC Actor provides useful MPPI guidance
- **Statement**: A deterministic Actor trained across four scenes and four bounded MuJoCo physics domains provides an MPPI sampling prior that materially improves closed-loop task completion, final goal distance, minimum clearance and control jerk relative to traditional MPPI across the L176 two-scene, three-domain development matrix.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Clean-SHA training provenance fails, the selected validation checkpoint does not outperform its early checkpoints, or seed-clustered L176 comparisons show no favorable RL main effect in success, final distance, clearance or jerk.
- **Proof**: [`results/research_platform/rl/gate1_direct_control_multidomain_screen_l175/checkpoints/best.pt`, `results/research_platform/rl/gate1_multidomain_factorial_l176/summary.csv`, `results/research_platform/rl/gate1_multidomain_factorial_l176/factorial_contrasts_seed_clustered.json`, `docs/rl/170_gate1_multidomain_factorial_results_2026-07-18.md`]
- **Dependencies**: [C57]
- **Tags**: SAC, RL-driven-MPPI, sampling-prior, multi-domain, development-gate

## C78: The Gate 1 ICODE plus RL simple combination has a small consistent endpoint benefit
- **Statement**: In the L176 development block, adding the frozen ICODE rollout model to the same frozen RL prior reduces final goal distance and control jerk relative to RL-driven nominal MPPI in each of three independent seed clusters, while preserving zero collisions and adding one successful episode.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis at the independent seed-cluster level reverses the final-distance or jerk contrast, collision count increases, or the result fails to reproduce across independent Actor and ICODE training blocks.
- **Proof**: [`results/research_platform/rl/gate1_multidomain_factorial_l176/summary.csv`, `results/research_platform/rl/gate1_multidomain_factorial_l176/factorial_contrasts_seed_clustered.json`, `docs/rl/170_gate1_multidomain_factorial_results_2026-07-18.md`]
- **Dependencies**: [C57, C77]
- **Tags**: ICODE, RL-driven-MPPI, simple-combination, factorial, development-evidence

## C79: Gate 1 proves universal ICODE-RL synergy
- **Statement**: The L176 simple-combination result by itself establishes a universal positive ICODE by RL interaction across success, safety, smoothness, compute and unseen environments.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Independent model-block confirmation yields an interaction interval including zero or an unfavorable effect for any claimed primary endpoint, narrow-corridor failures persist, or hard real-time compute limits are violated.
- **Proof**: [`docs/rl/170_gate1_multidomain_factorial_results_2026-07-18.md`, `results/research_platform/rl/full_proposed_confirmation_l211/factorial_contrasts.json`, `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`, `ara/evidence/tables/table14_full_proposed_factorial.md`]
- **Dependencies**: [C78]
- **Tags**: scope-boundary, interaction, synergy, confirmation-needed

## C80: Unconditional critic consistency is insufficient for task-relevant ICODE fine-tuning
- **Statement**: Applying the frozen SAC critic value-consistency loss without a task-competence gate can improve smoothness while degrading completion performance, so observation support and twin-critic agreement alone are insufficient safeguards against negative transfer.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis of the clean-provenance L188 paired cells shows that unconditional value alignment does not worsen final goal distance or success relative to ordinary task fine-tuning.
- **Proof**: [`results/research_platform/rl/gate2_value_alignment_paired_l188/paired_comparison.json`, `docs/rl/173_gate2_competence_gated_value_alignment_results_2026-07-18.md`, `ara/evidence/tables/table12_l188_l191_value_alignment.md`]
- **Dependencies**: [C77, C78]
- **Tags**: ICODE, SAC-critic, value-consistency, negative-transfer, negative-result

## C81: Competence-gated RL value alignment improves ICODE in solvable dynamics-mismatch tasks
- **Statement**: When critic gradients are gated by observation support, twin-critic agreement and competence thresholds calibrated only from training-episode outcomes, value-aligned ICODE improves terminal value prediction and independently improves final goal distance, success and stuck steps relative to ordinary ICODE fine-tuning in solvable MuJoCo dynamics-mismatch tasks while retaining zero collisions.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Provenance validation fails, the L189 offline value-error improvement exceeds the frozen rollout-regression allowance, or independent L191 seed-cluster intervals for final distance, success or stuck steps include zero in the unfavorable direction.
- **Proof**: [`results/research_platform/gate2_competence_gated_icode_l189/training_summary.json`, `results/research_platform/rl/gate2_competence_confirmation_l191/gate2_full_analysis.json`, `docs/rl/172_gate2_competence_gated_remediation_prereg_2026-07-18.md`, `docs/rl/173_gate2_competence_gated_value_alignment_results_2026-07-18.md`, `ara/evidence/tables/table12_l188_l191_value_alignment.md`]
- **Dependencies**: [C77, C80]
- **Tags**: ICODE, SAC-critic, competence-gating, value-alignment, independent-confirmation

## C82: Gate 2 establishes a universal no-regression control improvement
- **Statement**: Competence-gated value-aligned ICODE strictly improves every closed-loop endpoint, including jerk and clearance, and resolves geometrically difficult narrow-corridor planning.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Any preregistered endpoint has an adverse point estimate, or the narrow-corridor success rate remains unchanged.
- **Proof**: [`results/research_platform/rl/gate2_competence_confirmation_l191/gate2_full_analysis.json`, `docs/rl/173_gate2_competence_gated_value_alignment_results_2026-07-18.md`]
- **Dependencies**: [C81]
- **Tags**: scope-boundary, jerk, clearance, narrow-corridor, negative-result

## C83: Cross-layer reliability ranks ICODE rollout error under graded dynamics shift
- **Statement**: With the frozen three-member value-aligned ICODE ensemble and frozen SAC Actor support statistics, the causal cross-layer authority score orders high-, medium- and low-confidence episodes by increasing ten-step prediction error on an independently collected 30-episode graded MuJoCo stress set.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis finds split leakage or threshold selection on the stress set; fewer than two occupied authority bins; a positive authority-error rank association; non-monotone high-to-low error; or missing independent episode support.
- **Proof**: [`results/research_platform/rl/gate3_reliability_stress_evaluation_l194/stress_summary.json`, `docs/rl/176_gate3_reliability_hss_results_2026-07-18.md`, `ara/evidence/tables/table13_gate3_reliability_hss.md`]
- **Dependencies**: [C81]
- **Tags**: ICODE, ensemble, uncertainty, reliability, OOD, calibration

## C84: Reliability-calibrated hybrid sampling improves fixed-budget MPPI progress
- **Statement**: Relative to a fixed 30% persistent Actor sampling ratio, reliability-adaptive 0/30/60% hybrid sampling with the same 100-rollout budget, value-aligned ICODE ensemble and frozen Actor/critic reduces final goal distance and control jerk without collision or compute regression on five sealed confirmation seeds across two MuJoCo scenes and three physics domains.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Confirmation provenance or seed separation fails; rollout budgets differ; final-distance seed-cluster bootstrap interval includes zero; collisions increase; jerk worsens by more than the frozen 1% margin; or low and non-low authority are not both exercised.
- **Proof**: [`results/research_platform/rl/gate3_hss_confirmation_l196/paired_comparison.json`, `results/research_platform/rl/gate3_hss_combined_l197/gate3_hss_analysis.json`, `docs/rl/176_gate3_reliability_hss_results_2026-07-18.md`, `ara/evidence/tables/table13_gate3_reliability_hss.md`]
- **Dependencies**: [C81, C83]
- **Tags**: RL, ICODE, MPPI, hybrid-sampling, fixed-budget, confirmation

## C85: Gate 3 proves universal complex-navigation or success-rate improvement
- **Statement**: The current reliability-adaptive HSS improves success and complex navigation generally.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: This boundary can be revised only by a preregistered independent study with nonzero success resolution and favorable effects across multiple complex task families.
- **Proof**: [`results/research_platform/rl/gate3_hss_combined_l197/stratified_effects.csv`, `docs/rl/176_gate3_reliability_hss_results_2026-07-18.md`]
- **Dependencies**: [C84]
- **Tags**: RL, ICODE, MPPI, scope-boundary, refuted

## C86: Twin-critic disagreement calibrates terminal critic error
- **Statement**: Larger frozen twin-critic disagreement or Actor-input support distance identifies larger episode-level Monte Carlo critic error and can safely attenuate terminal value authority.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered direction fails if either score has a null or negative episode-level association with absolute critic/return error.
- **Proof**: [`docs/rl/178_gate4a_critic_calibration_result_2026-07-19.md`, `docs/rl/179_gate4_conservative_terminal_results_2026-07-19.md`]
- **Dependencies**: [C77, C84]
- **Tags**: SAC-critic, disagreement, support, calibration, negative-result

## C87: Candidate-level ICODE reliability should attenuate terminal critic authority
- **Statement**: Multiplying the fixed SAC terminal value by candidate-level ICODE dynamics confidence improves equal-budget MPPI control relative to the same fixed terminal weight.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Sealed confirmation fails to improve final goal distance or produces a strictly adverse interval for a key secondary endpoint.
- **Proof**: [`results/research_platform/rl/gate4_dyn_terminal_confirmation_l202/paired_comparison.json`, `docs/rl/179_gate4_conservative_terminal_results_2026-07-19.md`]
- **Dependencies**: [C84, C86]
- **Tags**: ICODE, SAC-critic, terminal-value, reliability, negative-result

## C88: Sampling authority and terminal-value authority share one reliability mapping
- **Statement**: A reliability score validated for allocating persistent Actor samples can be reused to calibrate candidate-wise SAC terminal value authority.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The mapping independently improves fixed-budget HSS but fails or regresses sealed terminal-value confirmation.
- **Proof**: [`docs/rl/176_gate3_reliability_hss_results_2026-07-18.md`, `docs/rl/179_gate4_conservative_terminal_results_2026-07-19.md`]
- **Dependencies**: [C84, C87]
- **Tags**: cross-layer-reliability, calibration-target, scope-boundary

## C89: The frozen Full Proposed package improves completion over a strong simple combination
- **Statement**: At equal 100-rollout budget on five sealed seeds and three MuJoCo physics domains, competence-gated value-aligned ICODE plus completion-preserving reliability-adaptive HSS improves success and final goal distance relative to ordinary ICODE with fixed 30% Actor guidance, without increasing collisions and within the prospectively fixed 5% jerk noninferiority margin.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Seed-cluster inference fails to show favorable success or final-distance direction; either primary interval includes zero; collisions increase; jerk worsens by more than 5%; rollout budgets differ; or protected seeds/checkpoints were used during remediation.
- **Proof**: [`results/research_platform/rl/full_proposed_confirmation_l211/paired_comparisons.json`, `results/research_platform/rl/full_proposed_confirmation_l211/analysis_summary.json`, `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`, `ara/evidence/tables/table14_full_proposed_factorial.md`]
- **Dependencies**: [C81, C83, C84, C87]
- **Tags**: ICODE, RL-driven-MPPI, value-alignment, adaptive-HSS, independent-confirmation

## C90: Value alignment and adaptive HSS have reproducible main effects but no super-additive interaction
- **Statement**: In the sealed success-resolved factorial, value-aligned ICODE and reliability-adaptive HSS each independently improve success and final distance over ordinary fixed sampling, while their combined success interaction is adverse/sub-additive because both mechanisms repair overlapping failures.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis does not reproduce value-at-fixed success +0.3333 and HSS-at-ordinary success +0.2667 with strictly favorable seed-cluster intervals, or the success interaction is not negative with interval below zero.
- **Proof**: [`results/research_platform/rl/full_proposed_confirmation_l211/paired_comparisons.json`, `results/research_platform/rl/full_proposed_confirmation_l211/factorial_contrasts.json`, `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`]
- **Dependencies**: [C79, C81, C84, C89]
- **Tags**: factorial, main-effect, interaction, scope-boundary, confirmation

## C91: The current Full Proposed package resolves complex navigation success
- **Statement**: The frozen Full Proposed package improves goal-reaching success in `lab_complex` under the preregistered 180-step horizon.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: All four factorial cells have zero successes, or the Full Proposed progress interval is not strictly favorable.
- **Proof**: [`results/research_platform/rl/full_proposed_lab_supplement_l213/analysis_summary.json`, `results/research_platform/rl/full_proposed_lab_supplement_l213/paired_comparisons.json`, `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`]
- **Dependencies**: [C85, C89]
- **Tags**: complex-navigation, lab-complex, scope-boundary, negative-result

## C92: Adaptive HSS retains a compute benefit when complex geometry dominates
- **Statement**: In the 60-episode `lab_complex` robustness supplement, completion-preserving adaptive HSS reduces mean planner time by about 10.8 ms under both ordinary and value-aligned ICODE, with zero collisions, even though no method reaches the goal within 180 steps.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Seed-cluster intervals for HSS-at-ordinary or HSS-at-value planner-time savings include zero, collisions increase, or per-decision rollout budgets differ.
- **Proof**: [`results/research_platform/rl/full_proposed_lab_supplement_l213/paired_comparisons.json`, `results/research_platform/rl/full_proposed_lab_supplement_l213/analysis_summary.json`, `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`]
- **Dependencies**: [C84, C89, C91]
- **Tags**: adaptive-HSS, compute, complex-navigation, robustness-supplement

## C93: Frozen Full Proposed reduces polyline cross-track error but does not establish path completion
- **Statement**: On five sealed seeds, three obstacle-free polyline paths and three MuJoCo physics domains, the frozen Full Proposed package reduces cross-track RMSE relative to ordinary ICODE with fixed Actor sampling at equal K=100, without increasing collision or jerk; this does not establish successful terminal convergence.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The seed-cluster cross-track favorable interval includes zero; rollout budgets differ; collision or jerk increases beyond the preregistered margin; provenance is inconsistent; or any successful-terminal claim is made from projection completion alone.
- **Proof**: [`results/research_platform/rl/full_proposed_path_transfer_confirm_l183_analysis/paired_comparisons.json`, `docs/rl/184_full_proposed_path_tracking_confirmation_results_2026-07-19.md`, `ara/evidence/tables/table15_full_proposed_path_transfer.md`]
- **Dependencies**: [C89, C91]
- **Tags**: path-tracking, adaptive-HSS, cross-track, confirmation, scope-boundary

## C94: Value-aligned ICODE independently improves frozen path tracking
- **Statement**: Replacing ordinary ICODE with the frozen competence-gated value-aligned ICODE improves cross-track RMSE in the independent polyline factorial.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The value-alignment factorial main effect is adverse with a seed-cluster interval excluding zero.
- **Proof**: [`results/research_platform/rl/full_proposed_path_transfer_confirm_l183_analysis/factorial_contrasts.json`, `docs/rl/184_full_proposed_path_tracking_confirmation_results_2026-07-19.md`]
- **Dependencies**: [C90, C93]
- **Tags**: ICODE, value-alignment, path-tracking, negative-result

## C95: Parity-calibrated Actor competence improves the multi-physics development set
- **Statement**: Mapping Actor/Gaussian elite-yield ratios relative to parity, rather than directly to authority, improves equal-budget path tracking on the fresh L197 development set.
- **Status**: weakened
- **Provenance**: ai-suggested
- **Falsification criteria**: The frozen mapping fails independent path-and-physics confirmation or its seed-cluster interval includes material adverse effects.
- **Proof**: [`results/research_platform/rl/path_aware_multiphysics_dev_l197/multidomain_gate.json`, `results/research_platform/rl/sealed_path_physics_confirmation_l198/sealed_gate.json`, `docs/rl/202_l196_l201_reliability_and_value_alignment_results_2026-07-19.md`]
- **Dependencies**: [C93]
- **Tags**: source-competence, parity-calibration, adaptive-HSS, development-only

## C96: The L197 frozen stack confirms path-tracking superiority over ordinary ICODE
- **Statement**: Parity-calibrated Full Proposed has no greater pooled cross-track RMSE than ordinary fixed sampling on sealed paths, physics domains and seeds.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Pooled Full Proposed cross-track RMSE exceeds ordinary fixed or the preregistered sealed Gate fails.
- **Proof**: [`results/research_platform/rl/sealed_path_physics_confirmation_l198/sealed_gate.json`, `docs/rl/202_l196_l201_reliability_and_value_alignment_results_2026-07-19.md`]
- **Dependencies**: [C95]
- **Tags**: path-tracking, sealed-confirmation, negative-result

## C97: Path-policy value-consistent ICODE generalizes beyond its training routes
- **Statement**: A path-conditioned SAC value-consistency loss improves ICODE rollout and value errors on held-out and unseen routes sufficiently to qualify a fine-tuned checkpoint for closed-loop evaluation.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Held-out value error worsens, the validation selector retains epoch zero, or no ensemble member passes the frozen offline Gate.
- **Proof**: [`docs/rl/202_l196_l201_reliability_and_value_alignment_results_2026-07-19.md`, `research_artifacts/l196_l201_2026-07-19/manifest.json`]
- **Dependencies**: [C94]
- **Tags**: value-consistency, ICODE, path-policy, generalization, negative-result

## C98: Residual-conditioned SAC action correction generalizes across nominal and unseen dynamics
- **Statement**: A bounded SAC correction conditioned on causal ICODE residual, innovation, disagreement and support improves MPPI path tracking in both nominal and unseen dynamics without a smoothness regression.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Covariance-isolated or innovation-gated paired screens worsen nominal cross-track RMSE, fail unseen improvement, or exceed the preregistered jerk margin.
- **Proof**: [`docs/rl/211_two_coupling_mechanisms_frozen_2026-07-19.md`, `results/research_platform/rl/residual_conditioned_closed_loop_l207/`, `results/research_platform/rl/residual_conditioned_closed_loop_l209/`, `ara/evidence/tables/table17_l204_l210_coupling_selection.md`]
- **Dependencies**: [C83, C90]
- **Tags**: residual-conditioned-policy, SAC, ICODE, negative-result, paired-design

## C99: Pairwise critic ranking improves path-policy ICODE generalization
- **Statement**: Replacing dominant absolute-Q regression with a confidence-weighted pairwise critic-ranking loss improves held-out and unseen-route value ordering while preserving ICODE rollout accuracy.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Either held-out or unseen rank correlation decreases, rollout degradation exceeds 3%, or the offline selector retains an overfit checkpoint.
- **Proof**: [`docs/rl/210_value_ranked_icode_prereg_2026-07-19.md`, `docs/rl/211_two_coupling_mechanisms_frozen_2026-07-19.md`, `results/research_platform/value_ranked_icode_member1_l210/training_summary.json`, `ara/evidence/tables/table17_l204_l210_coupling_selection.md`]
- **Dependencies**: [C97]
- **Tags**: value-ranking, ICODE, path-policy, ceiling-effect, negative-result

## C100: Amendment 17 is a conditionally accepted dynamic-obstacle baseline
- **Statement**: On the three registered development seeds, the Amendment 17 risk-enabled controller completed all runs with zero collision, improved seed 730100003 final goal distance from 1.3383 m to 0.3334 m relative to Amendment 12, retained a +0.2437 m median-clearance delta, and met 8 of 9 automatic gates; the project lead accepted the remaining 0.004309 completion-threshold excess as a documented stage waiver.
- **Status**: supported
- **Provenance**: user-revised
- **Falsification criteria**: Artifact reanalysis does not reproduce the stated collision, distance, clearance or 8/9 gate results, or the acceptance record is changed to withdraw the project-lead waiver.
- **Proof**: [`docs/experiments/dynamic_uncertainty/POST_ESCAPE_RECOVERY_AMENDMENT17_REPORT.md`, `docs/experiments/dynamic_uncertainty/ENVIRONMENT_STAGE_ACCEPTANCE.md`, `research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment17/`]
- **Dependencies**: []
- **Tags**: dynamic-obstacle, probabilistic-prediction, MPPI, conditional-pass, development

## C101: Increasing terminal progress pressure is not a safe Amendment 17 completion remedy
- **Statement**: Raising only `goal_terminal_weight` from 25 to 35 caused a collision on the preregistered seed-730100001 Amendment 18 screen, so progress-weight escalation was rejected without opening a full three-seed run.
- **Status**: supported
- **Provenance**: ai-executed
- **Falsification criteria**: The retained Amendment 18 probe artifact does not reproduce the collision, or the recorded configurations differ in another controller/environment factor.
- **Proof**: [`docs/experiments/dynamic_uncertainty/POST_ESCAPE_RECOVERY_AMENDMENT17_REPORT.md`, `configs/research/mujoco_v3_probabilistic_crossing_smoke_amendment18_probe.yaml`, `research_artifacts/mujoco_v3_probabilistic_crossing_smoke_amendment18_probe/`]
- **Dependencies**: [C100]
- **Tags**: MPPI, terminal-cost, collision, negative-result, stopping-rule

## C102: L57 residual prediction accuracy does not transfer safely to Amendment 17 obstacle avoidance
- **Statement**: On the three Amendment 17 development obstacle seeds, all three frozen L57 ICODE checkpoints improve shared-trajectory horizon-36 velocity/yaw-rate rollout RMSE by 14.33%--16.91% over nominal dynamics, yet each model block increases collision count, aggregate median clearance falls by 0.03018 m, and residual planner p95 reaches 206.01 ms.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Artifact reanalysis fails to reproduce the three positive prediction effects, per-block collision increases `[1,1,2]`, clearance gate failure, or compute gate failure.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage1_development/gate.json`, `research_artifacts/dynamic_uncertainty_residual_stage1_development/prediction_block_summary.csv`, `research_artifacts/dynamic_uncertainty_residual_stage1_development/closed_loop_pairs.csv`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_RESULT.md`]
- **Dependencies**: [C100]
- **Tags**: ICODE, residual-dynamics, prediction-control-gap, dynamic-obstacle, negative-result

## C103: Global-coordinate canonicalization is sufficient to repair Stage 1 safety
- **Statement**: Replacing residual inputs `x,y` by checkpoint training means removes the relevant spatial OOD and restores zero-collision Amendment 17 control under the frozen L57 checkpoints.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered first residual probe collides or exceeds the 150 ms planner p95 threshold.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage1_amendment1_probe/early_stop.json`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT1_RESULT.md`]
- **Dependencies**: [C102]
- **Tags**: canonicalization, spatial-OOD, ICODE, collision, negative-result

## C104: Post-stall residual shutdown is sufficient to repair Stage 1 safety
- **Statement**: Causally latching residual authority to zero after ten completed low-risk near-zero-speed transitions prevents the seed-730100003 collision without future truth.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered first residual block latches but still collides.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage1_amendment2_probe/early_stop.json`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT2_RESULT.md`]
- **Dependencies**: [C102, C103]
- **Tags**: residual-authority, causal-gate, stall, collision, negative-result

## C105: Causal innovation routing makes direct L57 residual dynamics safe in the crossing task
- **Statement**: Starting residual authority at zero and increasing it only when completed-transition velocity/yaw-rate innovations favor the residual restores zero-collision Amendment 17 control.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: A preregistered residual block reaches nondegenerate authority and still collides.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage1_amendment3_probe/early_stop.json`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT3_RESULT.md`]
- **Dependencies**: [C102, C104]
- **Tags**: innovation, reliability, residual-authority, collision, negative-result

## C106: L57 remains predictively useful under structure preservation but is unsafe as direct crossing-task dynamics
- **Statement**: On saved nominal seed-730100003 data, masking pose-derivative residuals and retaining only velocity/yaw-rate corrections improves horizon-36 active RMSE by 15.83%--16.62% across all three checkpoints, but the preregistered first structure-preserving closed-loop block still collides.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: The masked offline replay fails to reproduce three positive improvements, or the retained Amendment 4 artifact is collision-free.
- **Proof**: [`docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT4_RESULT.md`, `research_artifacts/dynamic_uncertainty_residual_stage1_amendment4_probe/early_stop.json`]
- **Dependencies**: [C102, C105]
- **Tags**: structure-preserving, prediction-control-gap, ICODE, scope-boundary

## C107: Step-local dual-model noninterference is sufficient to repair frozen L57 safety
- **Statement**: Accepting an L57 residual plan only when nominal and residual model views are hard-safe, nominal-view probability does not exceed the exact nominal plan, progress is noninferior and predicted positions stay within 0.10 m prevents the seed-730100003 collision.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered shield accepts a nonzero fraction of residual plans and the episode still collides.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage1_amendment5_probe/`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT5_RESULT.md`]
- **Dependencies**: [C102, C106]
- **Tags**: residual-shield, noninterference, chance-constraint, negative-result

## C108: Speed-only L57 residual authority preserves safe obstacle avoidance
- **Statement**: Keeping the exact nominal yaw sequence while allowing the frozen L57 residual to alter only forward-speed timing prevents route divergence and collision.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: A preregistered speed-only residual block accepts nonzero residual authority and still collides.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage1_amendment6_probe/`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT6_PREREGISTRATION.md`]
- **Dependencies**: [C107]
- **Tags**: residual-shield, speed-authority, collision, negative-result

## C109: Task-aware conservative fine-tuning improves obstacle-region residual prediction without material forgetting
- **Statement**: Fine-tuning each L57 model on an episode-disjoint mixture with proximity-weighted dynamic-obstacle transitions and an original-data output anchor improves held-out task H36 position RMSE while limiting original test/unseen H36 position regression to at most 2%.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Fewer than two of three blocks improve task-test H36 position RMSE, fail to beat nominal active derivative error, or exceed 2% original test/unseen regression.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage2_offline/gate.json`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_RESULT.md`, `ara/evidence/tables/table18_stage2_task_aware_residual.md`]
- **Dependencies**: [C102, C108]
- **Tags**: task-aware-training, importance-weighting, anti-forgetting, ICODE

## C110: Stage 2 residuals avoid the observed closed-loop safety regression on fresh seed 730100006
- **Statement**: On the preregistered fresh development obstacle seed 730100006, nominal and all three task-aware residual blocks reach the goal with zero collisions and identical 0.39754 m minimum clearance; the shield accepts 67.6%--73.2% of residual plans.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Artifact reanalysis finds a collision, a failed goal reach, lower residual minimum clearance, permanent nominal fallback, or use of a task-data/sealed seed.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_probe/gate.json`, `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_probe/episode_summary.csv`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_RESULT.md`, `ara/evidence/tables/table18_stage2_task_aware_residual.md`]
- **Dependencies**: [C100, C109]
- **Tags**: residual-dynamics, dynamic-obstacle, fresh-seed, development-only, scope-boundary

## C111: Task-aware residuals pass the frozen three-seed development safety and noninferiority gate
- **Statement**: Across obstacle seeds 730100006, 730100008 and 730100010, nominal and all three task-aware residual blocks reach the goal without collision or safety-layer intervention; every block has zero collision-count increase, and the aggregate median completion delta remains above the frozen -0.02 limit.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Artifact reanalysis finds any residual collision or failed goal reach, a positive per-block collision increase, median completion delta below -0.02, invalid checkpoint/forecast artifacts, or use of sealed seeds.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/gate.json`, `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/episode_summary.csv`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_DEVELOPMENT_RESULT.md`, `ara/evidence/tables/table19_stage2_residual_development.md`]
- **Dependencies**: [C100, C109, C110]
- **Tags**: residual-dynamics, dynamic-obstacle, development-gate, noninferiority, scope-boundary

## C112: Offline H36 improvement does not select the fastest Stage 2 checkpoint
- **Statement**: In the frozen three-seed matrix, the checkpoint with the largest offline task H36 position improvement does not have the best closed-loop median step delta, and the three block medians of 0, -2 and +5 steps do not establish a consistent completion-time benefit.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis shows the offline-improvement ordering matches a preregistered, consistently favorable episode-level completion ordering or that a checkpoint satisfies a previously frozen selection rule.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage2_offline/gate.json`, `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/closed_loop_pairs.csv`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_DEVELOPMENT_RESULT.md`]
- **Dependencies**: [C109, C111]
- **Tags**: prediction-control-gap, checkpoint-selection, completion-time, scope-boundary

## C113: Stage 2 safety is not explained by permanent nominal fallback
- **Statement**: In the nine paired development comparisons, the residual shield has nonzero acceptance in every episode, median paired-realized effective acceptance is 29.7%, and the median fraction of executed controls differing from nominal by more than 0.01 is 52.0%.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Trace reanalysis finds permanent nominal fallback, zero effective residual participation, mismatched episode pairing, or a different executed-control threshold.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/contribution_audit.json`, `experiments/dynamic_uncertainty/analyze_task_aware_residual_contribution.py`, `ara/evidence/tables/table19_stage2_residual_development.md`]
- **Dependencies**: [C111]
- **Tags**: residual-shield, effective-authority, paired-audit, nondegeneracy

## C114: Optimized Stage 3 residual control meets the development-machine 100 ms P95 gate without behavior change
- **Statement**: On the current RTX 5060 Laptop GPU and the frozen seeds 730100006/008/010, static-buffer CUDA Graph residual rollout plus parallel nominal/residual shield planning preserves 12/12 goal reaches and zero collisions while reducing the maximum residual-controller P95 to 97.19 ms; all 12 cells exactly match the sequential CUDA-Graph controller in steps, positions, executed controls, shield acceptance and outcomes.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Artifact reanalysis finds maximum residual-controller P95 above 100 ms, any collision or failed goal reach, any sequential-equivalence difference, a changed safety/controller protocol, or use of sealed seeds.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/gate.json`, `research_artifacts/dynamic_uncertainty_residual_runtime_stage3_parallel_development/sequential_equivalence_audit.json`, `docs/experiments/dynamic_uncertainty/RESIDUAL_RUNTIME_STAGE3_RESULT.md`, `ara/evidence/tables/table20_residual_runtime_stage3.md`]
- **Dependencies**: [C111, C113]
- **Tags**: residual-dynamics, CUDA-Graph, parallel-planning, runtime, behavior-equivalence, development-only

## C115: Eager CUDA device residency alone is sufficient for residual real-time qualification
- **Statement**: Keeping the full H36/K600/RK4 residual rollout on CUDA without graph capture reduces concurrent worst-block P95 relative to the legacy-transfer CUDA implementation and is sufficient to meet the 100 ms budget.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: The preregistered Amendment 1 screen shows eager device-resident P95 below legacy-transfer P95 and at or below 100 ms while satisfying trajectory equivalence.
- **Proof**: [`research_artifacts/dynamic_uncertainty_residual_runtime_stage3_amendment1/runtime_screen.json`, `docs/experiments/dynamic_uncertainty/RESIDUAL_RUNTIME_STAGE3_RESULT.md`, `ara/evidence/tables/table20_residual_runtime_stage3.md`]
- **Dependencies**: [C111]
- **Tags**: CUDA, device-residency, kernel-launch-overhead, negative-result

## C116: Stage 4 RL/HSS treatment construction preserves the frozen controller contracts at preflight
- **Statement**: The preregistered 24-episode Stage 4 development schedule constructs RL-off cells through the exact Stage 3 path and RL-on cells with the frozen L217 Actor/HSS at an unchanged 600-candidate budget; residual cells use independent matched nominal/residual RL controllers, retain the residual shield and scan guard, and do not open sealed seeds.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Preflight or construction tests find a non-24-cell schedule, an RL-off configuration difference, more than 600 candidates per controller decision, shared mutable Actor/HSS/RNG state inside a residual shield, a missing shield or scan guard, an unregistered Actor action remap, or any sealed seed access.
- **Proof**: [`configs/research/dynamic_uncertainty_rl_hss_stage4_preregistered.yaml`, `configs/research/dynamic_uncertainty_rl_hss_stage4_amendment1.yaml`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE4_PREREGISTRATION.md`, `tests/dynamic_uncertainty/test_rl_hss_stage4.py`, `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/gate.json`, `ara/evidence/tables/table21_stage4_rl_hss_preflight.md`]
- **Dependencies**: [C114]
- **Tags**: RL, HSS, preregistration, treatment-integrity, action-subspace, development-only

## C117: Frozen L217 Actor/HSS integration preserves dynamic-obstacle navigation safety and completion
- **Statement**: Adding the frozen L217 Actor and role-aware HSS to the accepted nominal or qualified Stage 3 residual controller at equal 600-candidate budget preserves collision-free goal reaching on the registered Stage 4 development seeds.
- **Status**: refuted
- **Provenance**: ai-suggested
- **Falsification criteria**: Any completed RL/HSS-on cell collides, or paired goal-reaching and final-distance outcomes consistently regress relative to RL/HSS-off under the frozen common-random-number schedule.
- **Proof**: [`research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/gate.json`, `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/paired_analysis.json`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE4_RESULT.md`, `ara/evidence/tables/table22_stage4_rl_hss_result.md`]
- **Dependencies**: [C89, C114, C116]
- **Tags**: RL, HSS, dynamic-obstacle, negative-result, collision, development

## C118: Policy-rescue authority explains the Stage 4 RL/HSS regression
- **Statement**: The Stage 4 completion and safety regression is principally caused by `policy_rescue` retaining high Actor proposal authority when dynamics confidence is near zero and Actor-guided candidates are cost-inferior to Gaussian candidates.
- **Status**: hypothesis
- **Provenance**: ai-suggested
- **Falsification criteria**: A separately preregistered advantage-gated or policy-rescue-disabled ablation fails to improve paired completion/safety, or controlled replay shows that the regression persists with identical proposal authority and is instead attributable to another isolated mechanism.
- **Proof**: [`research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/episode_summary.csv`, `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/paired_analysis.json`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE4_RESULT.md`, `research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/paired_analysis.json`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_RESULT.md`]
- **Dependencies**: [C117]
- **Tags**: HSS, policy-rescue, proposal-authority, mechanism-hypothesis, development

## C119: Episode-latched proposal-advantage veto can recover baseline-level completion from the frozen bad Actor treatment
- **Statement**: On the two complete fresh nominal-dynamics Stage 5 development blocks, causally applying a three-cycle episode-latched veto to persistently cost-inferior Actor proposals recovers goal completion to the RL/HSS-off level while preserving the 600-rollout budget and unchanged safety stack.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis finds that shadow and active arms did not share the same evidence stream, that veto application changed the rollout budget or safety stack, that either active episode did not reach the goal, or a separately preregistered adequately powered fresh-seed matrix fails to reproduce the completion recovery.
- **Proof**: [`research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/paired_analysis.json`, `research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/integrity_audit.json`, `research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/gate.json`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_RESULT.md`, `ara/evidence/tables/table23_stage5_proposal_advantage.md`]
- **Dependencies**: [C116, C117, C118]
- **Tags**: RL, HSS, proposal-advantage, safety-veto, bounded-support, two-pairs, development-only

## C120: Zero-velocity hard stop may be unsafe against a still-closing dynamic obstacle
- **Statement**: The Stage 5 shadow collision is consistent with a stopping-feasibility defect: predicted collision probability approached one while proposed, executed and applied robot velocities were zero, after which the moving obstacle contacted the stationary robot.
- **Status**: hypothesis
- **Provenance**: ai-suggested
- **Falsification criteria**: Time-aligned replay shows that the obstacle was not closing during the zero-velocity interval, that a nonzero escape action had no lower predicted risk under observation-only information, or isolated geometric tests show that the current hard-stop rule already selects the minimum-risk feasible action in all registered closing-obstacle cases.
- **Proof**: [`research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/runs/rl_hss_shadow/seed_730100064/metrics.json`, `research_artifacts/dynamic_uncertainty_rl_hss_stage5_proposal_advantage_development/runs/rl_hss_shadow/seed_730100064/trajectory.csv`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_RESULT.md`, `ara/evidence/tables/table23_stage5_proposal_advantage.md`]
- **Dependencies**: [C117]
- **Tags**: dynamic-obstacle, stopping-feasibility, hard-stop, collision, mechanism-hypothesis

## C121: Initial V5 Amendment 2 development improvement did not independently replicate
- **Statement**: V5 Amendment 2 improved success from 6/8 to 7/8 without collisions on paired development seeds 730100157--730100164, but this early screen did not generalize to fresh seeds 730100172--730100179: the candidate introduced collisions on seeds 730100176, 730100177 and 730100179. The original improvement therefore remains a valid bounded observation but no longer supports retaining V5 Amendment 2 as the dynamic Actor candidate.
- **Status**: weakened
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis fails to reproduce either the original eight-pair improvement or the three fresh-seed candidate-only collisions, or a prospectively frozen independent replication establishes noninferior collision incidence and completion for this exact checkpoint.
- **Proof**: [`research_artifacts/dynamic_actor_v5a2_vs_v3_rollin_seed_pair_summary.json`, `research_artifacts/dynamic_actor_v5a2_vs_v3_rollin_seed_pair_730100162/result.json`, `research_artifacts/dynamic_actor_v5a2_vs_v3_fresh_shadow_seed730100176/result.json`, `research_artifacts/dynamic_actor_v5a2_vs_v3_fresh_shadow_seed730100177/result.json`, `research_artifacts/dynamic_actor_v5a2_vs_v3_fresh_shadow_seed730100179/result.json`, `ara/evidence/tables/table24_dynamic_actor_temporal_correction.md`]
- **Dependencies**: [C117, C118]
- **Tags**: RL, Actor, temporal-lidar, bidirectional-control, dynamic-obstacle, safety-regression, development-only

## C122: The initial V5 Amendment 6 same-cycle-filtered gain did not survive expanded safety qualification
- **Statement**: V5 Amendment 6 update 250 improved success from 13/16 to 14/16 with no new collision on seeds 730100188--730100203, but a separately frozen 24-pair expansion on seeds 730100204--730100227 introduced one candidate-only collision and lost one source success. The original two-batch gain therefore remains a valid bounded observation but does not qualify this checkpoint for sealed evaluation.
- **Status**: weakened
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis fails to reproduce either the initial 16-pair result or the expanded candidate-only collision/lost-success outcomes, a frozen binding mismatch is found, or an independently frozen fresh matrix for the exact same checkpoint passes all registered safety and completion gates.
- **Proof**: [`research_artifacts/dynamic_actor_v5a6_u000250_samecycle_fresh_replication_summary.json`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/summary.json`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/gate.json`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/integrity_audit.json`, `ara/evidence/tables/table24_dynamic_actor_temporal_correction.md`]
- **Dependencies**: [C117, C118, C121]
- **Tags**: RL, Actor, temporal-lidar, bidirectional-control, same-cycle-filter, safety-regression, single-obstacle, development-only

## C123: Expanded V5 Amendment 6 development improves completion and steps but fails the joint safety-distance gate
- **Statement**: Under a frozen, arm-order-balanced 24-pair single-obstacle development protocol in which both arms use proposal-only same-cycle filtering and 600 rollouts, V5 Amendment 6 update 250 improves success from 18/24 to 20/24 and reduces total steps from 7845 to 7611, but increases collisions from 2/24 to 3/24 through candidate-only collision seed 730100208, loses source success seed 730100206, worsens mean final distance from 0.77807 m to 0.81760 m, and reaches candidate planner P95 100.36 ms. The preregistered expanded gate therefore fails.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Artifact reanalysis does not reproduce the 24 paired outcomes, protocol/config/checkpoint hashes do not match, the same-cycle filter or 600-rollout budget was not active in both arms, the run order was not balanced as registered, or the recorded gate result differs from the frozen criteria.
- **Proof**: [`research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/summary.json`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/gate.json`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/paired_analysis.json`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/integrity_audit.json`, `configs/research/dynamic_actor_v5a6_samecycle_expanded_development.yaml`, `ara/evidence/tables/table24_dynamic_actor_temporal_correction.md`]
- **Dependencies**: [C117, C118, C122]
- **Tags**: RL, Actor, expanded-replication, paired-development, collision, completion-gain, gate-failure, single-obstacle

## C124: Pareto-repaired V5 Amendment 6 passes the bounded single-obstacle development gate
- **Statement**: Under the prospectively frozen Amendment 2 protocol on 24 new arm-order-balanced single-dynamic-obstacle development seeds, V5 Amendment 6 update 250 plus same-cycle filtering and Pareto active traversal improves success from 20/24 to 22/24, reduces collisions from 3/24 to 1/24, introduces zero Candidate-only collisions, loses zero Source successes, and improves preregistered outcome-aware efficiency from 7741 to 7615 while preserving exactly 600 rollouts. A separate unused seed730100255 then reaches the goal without collision with 17 causal temporal-escape cycles and Candidate planner P95 83.54 ms.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Reanalysis fails any registered Amendment 2 gate, any of the 220 bound matrix files or nine fresh-confirmation raw files fails SHA256 verification, Candidate-only collision or lost Source success is found, any valid decision does not use exactly 600 rollouts, the online LaserScan tracker/forecast is not causal and active, or seed730100255 does not reproduce the stated success/collision/runtime result.
- **Proof**: [`research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/gate.json`, `research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/summary.json`, `research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/integrity_audit.json`, `research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/formal_statistics.json`, `research_artifacts/dynamic_actor_v5a6_single_obstacle_final_confirmation_seed730100255/result.json`, `research_artifacts/dynamic_actor_v5a6_single_obstacle_final_confirmation_seed730100255/integrity_audit.json`, `ara/evidence/tables/table26_single_obstacle_final_validation.md`]
- **Dependencies**: [C122, C123]
- **Tags**: RL, Actor, Pareto-traversal, paired-development, single-obstacle, runtime, bounded-support, development-only

## C125: AR(1) Gaussian sampling passes the Chapter 1 confirmatory-screen threshold without improving success
- **Statement**: Under the frozen 12-pair Chapter 1 common-random-number screen, changing only the Paper-RL Gaussian noise basis from iid to `ar1:2.0` leaves both arms at zero successes, changes collisions from 9/12 to 6/12, produces one treatment-only collision, and improves the paired final-goal-distance median by 0.283041 m. This exceeds the preregistered 0.25 m screening threshold without reaching the two-pair safety-failure threshold, so the bounded verdict is `PASS_TO_CONFIRMATORY`, not demonstrated closed-loop efficacy.
- **Status**: supported
- **Provenance**: ai-suggested
- **Falsification criteria**: Artifact reanalysis does not reproduce the 12 paired outcomes or 0.283041 m median reduction; any episode protocol hash differs from `f1dfd4098941bb9101c0efc525295ac31df3c49a6aa7bba79fa2d5c37cc68d0e`; the resolved arms differ behaviorally beyond `planner.noise_basis`; or a missing/nonempty-error episode invalidates the complete-pair analysis.
- **Proof**: [`research_artifacts/noise_basis_ab_v1/ab_screen_result.json`, `research_artifacts/noise_basis_ab_v1/paired_results.csv`, `research_artifacts/ar1_closed_loop_screen_ch1/AUDIT.md`, `ara/evidence/tables/table27_ar1_noise_basis_screen.md`]
- **Dependencies**: []
- **Tags**: MPPI, AR1, correlated-sampling, Chapter-1, paired-screen, development-only, zero-success
