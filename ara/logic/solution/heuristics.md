# Heuristics

## H01: Pair every learned checkpoint against step zero
- **Rationale**: Matching scene/seed rows exposes success losses that an aggregate return can hide and preserves the frozen BC behavior as an explicit safe reference.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/trainer.py`, `configs/rl/sac_mppi_utrap_conservative_correction_l17.yaml`]

## H02: Separate training augmentation from the validation contract
- **Rationale**: Initial-state jitter is useful for training coverage but silently applying it during validation makes the trainer incomparable with the fixed-start benchmark evaluator.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/trainer.py`, `configs/rl/sac_mppi_utrap_conservative_correction_l17_v3.yaml`]

## H03: Treat validation episodes as repeated measures, not training replicates
- **Rationale**: Evaluation seeds are nested within each trained model; the independent replication unit is the training seed.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/13_conservative_correction_preregistered_gate_2026-07-14.md`]

## H04: Gate correction in latent action space against its frozen BC action
- **Rationale**: A rejected correction must recover the same frozen BC latent action used by the actor comparison. Reusing the outer OOD/fallback alpha would instead blend toward GoalWarmStart and would not isolate the correction effect.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/sac.py`, `src/mobile_robot_mppi/rl/prior.py`, `tests/rl/test_sac_and_checkpoint.py`]

## H05: Seal selection seeds until calibration produces an eligible method
- **Rationale**: Calibration chooses the method and selection estimates its held-out performance. Opening selection seeds after calibration has already failed would only spend independent evidence and invite post-hoc method tuning; fail closed to the frozen BC prior instead.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/calibration.py`, `experiments/rl/select_correction_advantage_margin.py`, `docs/rl/18_l19_advantage_margin_calibration_prereg_2026-07-15.md`]

## H06: Penalize critic action-advantage disagreement in a scale-invariant form
- **Rationale**: The score `mean(delta_q) - beta * half_disagreement` is unchanged in sign when both critic advantages are multiplied by the same positive scale. Beta one is exactly the smaller twin advantage, providing a tested regression anchor while beta greater than one demands increasing critic consensus.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/sac.py`, `tests/rl/test_sac_and_checkpoint.py`, `docs/rl/20_l20_twin_critic_consensus_lcb_prereg_2026-07-15.md`]

## H07: Restore counterfactual branches by full action replay, not MuJoCo pose snapshots
- **Rationale**: Copying only `qpos/qvel` omits sensor RNG, MPPI RNG, previous controls, encoder history and safety/controller state. Same-seed reset plus exact latent-action replay restores all stochastic subsystems and supplies a measurable zero-drift contract before intervention.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/collect_correction_counterfactual_branches.py`, `docs/rl/22_l21_counterfactual_risk_dataset_prereg_2026-07-15.md`]

## H08: Choose burst duration from prior closed-loop run lengths before new collection
- **Rationale**: Using only the already completed L20 gate logs placed a ten-step intervention near the q90 accepted-run length, making it representative of meaningful accumulated influence without selecting the duration from L22 outcomes.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/24_l22_counterfactual_burst_prereg_2026-07-15.md`, `configs/rl/correction_counterfactual_burst_l22.yaml`]

## H09: Preserve continuous counterfactual targets when a class gate fails
- **Rationale**: Lowering label thresholds after seeing the observed range manufactures apparent classes and invalidates confirmatory interpretation. Raw return, distance, clearance and safety deltas remain useful for diagnosing target scale and preregistering a new continuous-utility study.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/risk_dataset.py`, `experiments/rl/audit_correction_counterfactual_dataset.py`, `docs/rl/25_l22_counterfactual_burst_results_2026-07-15.md`]

## H10: Give zero-variance features unit scale
- **Rationale**: Dividing a constant training feature by an epsilon turns a harmless distribution shift into an artificial million-scale input. Standardization therefore centers near-constant features but uses scale one, and the behavior is protected by a regression test.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/utility_model.py`, `tests/rl/test_utility_model.py`, `docs/rl/27_l23_zero_variance_scaling_correction_2026-07-15.md`]

## H11: Learned gates fail closed on development eligibility
- **Rationale**: A serialized model is not deployable merely because training completed. The loader rejects checkpoints whose development gate failed unless an explicit analysis-only override is requested, preserving frozen BC as the runtime fallback.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/utility_model.py`, `tests/rl/test_utility_model.py`]

## H12: Calibrate sequence uncertainty at the episode group level
- **Rationale**: Multiple branch rows from one checkpoint and episode share history and are not independent. Group conformal calibration uses each group's worst nonconformity so nominal coverage is not inflated by treating correlated branches as independent samples.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/utility_model.py`, `experiments/rl/train_counterfactual_utility_ensemble.py`]

## H13: Preview paired stochastic candidates with common random numbers
- **Rationale**: Candidate comparisons must use the same MPPI perturbations so observed feature differences come from the prior rather than sampling noise. The preview must clone controller RNG and restore mutable prior/reference state, with an enabled-versus-disabled exact outcome regression before formal collection.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/mppi.py`, `src/mobile_robot_mppi/rl/environment.py`, `tests/platform/test_generic_mppi.py`, `tests/rl/test_trajectory_features.py`]

## H14: Compare learned representations on identical groups and training seeds
- **Rationale**: A trajectory representation is only useful if it beats the current-state representation under identical data splits, bootstrap groups, ensemble seeds and optimization. Training an explicit state-only ablation prevents extra model capacity or favorable splits from being mistaken for information gain.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/train_counterfactual_utility_ensemble.py`, `src/mobile_robot_mppi/rl/utility_model.py`, `docs/rl/29_l24_trajectory_utility_prereg_2026-07-15.md`]

## H15: Align feature horizon with the intervention horizon
- **Rationale**: A single open-loop candidate preview cannot identify the outcome of a future sequence that repeatedly re-senses, re-samples, replans, gates and passes through safety arbitration. When the feature horizon is shorter than the treatment, first test a direct representation ablation; if it fails, simplify or realign the estimand instead of only increasing network capacity.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/trajectory_features.py`, `experiments/rl/collect_correction_counterfactual_branches.py`, `docs/rl/30_l24_trajectory_utility_results_2026-07-15.md`]

## H16: Gate from observable local geometry, not global scene labels
- **Rationale**: One obstacle can create a decisive local blockage even when a scene is globally sparse. LaserScan-derived proximity, bilateral constriction and density measure what the controller currently faces and transfer without exposing simulator truth.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/scene_complexity.py`, `src/mobile_robot_mppi/rl/prior.py`, `docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md`]

## H17: Verify fallback step by step under common seeds
- **Rationale**: Equal aggregate success can hide changed controls and safety interventions. Comparing paired trajectories at every step establishes that a zero gate really restores the traditional controller rather than merely reaching the same endpoint.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/analyze_scene_complexity_gate_ablation.py`, `tests/rl/test_scene_complexity.py`]

## H18: Evaluate learned sampling priors across MPPI sample budgets
- **Rationale**: A sampling prior may primarily improve sample efficiency rather than the asymptotic optimum. A fixed `K` cannot distinguish a genuine prior benefit from an under-sampled traditional baseline, so `K=50,100,200,400` should be a blocked experimental factor.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/rl/sac_mppi_scene_complexity_l25.yaml`, `docs/rl/32_l25_scene_complexity_gate_results_2026-07-15.md`]

## H19: Build real-robot scenes from reusable modules rather than bespoke fixed obstacles
- **Rationale**: Reusing 600 mm and 300 mm wall blocks plus three cylinder diameters allows the same hardware to isolate clean dynamics, blockage, corridor, U-trap and clutter factors while preserving exact placement coordinates and reducing procurement cost.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`configs/real_robot/obstacle_kit_6p5m.yaml`, `docs/real_robot/obstacle_kit_6p5m/bill_of_materials.csv`, `docs/real_robot/obstacle_kit_6p5m/scene_placements.csv`]

## H20: Measure the swept footprint and LaserScan beam height before bulk fabrication
- **Rationale**: Simulation collision radius and chassis mesh dimensions are not manufacturing measurements. The supplier dimensions are only safe if the maximum real swept footprint, protruding sensors and LaserScan beam height are measured, and one wall and cylinder sample are detected reliably before ordering the full kit.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/real_robot/obstacle_kit_6p5m/README.md`, `docs/real_robot/obstacle_kit_6p5m/01_wall_module_fabrication.svg`, `docs/real_robot/obstacle_kit_6p5m/02_cylinder_module_fabrication.svg`]

## H21: Separate sampling efficiency from compute efficiency
- **Rationale**: A learned prior can make each MPPI sample more useful while adding actor, critic and decoding overhead. Report success-versus-K and end-to-end planner time as separate outcomes, and do not infer wall-clock acceleration from a lower sample count.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_scene_complexity_sample_efficiency.py`, `experiments/rl/analyze_sample_efficiency_results.py`, `docs/rl/34_l26_sample_efficiency_results_2026-07-15.md`]

## H22: Short-circuit learned inference only when all learned outputs are irrelevant
- **Rationale**: Skipping an actor is behavior preserving only when the outer gate mathematically fixes its mean contribution to zero and no learned covariance remains active. Enforce those conditions in code and require common-seed stepwise identity before accepting the optimization.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/prior.py`, `tests/rl/test_scene_complexity.py`, `tests/rl/test_zero_complexity_fastpath_ablation.py`, `docs/rl/36_l27_zero_complexity_fastpath_results_2026-07-15.md`]

## H23: Separate statistically detectable speedup from practically sufficient speedup
- **Rationale**: A timing confidence interval that excludes zero shows a reproducible engineering effect, but it does not establish that the effect is large enough to change the scientific compute claim. Freeze magnitude thresholds before formal profiling, preserve exact behavior checks, and record a positive near-miss as a failed practical gate rather than weakening the threshold or adding conditions after observing results.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/37_l28_inference_profile_prereg_2026-07-15.md`, `experiments/rl/summarize_inference_profile_ablation.py`, `docs/rl/38_l28_inference_profile_results_2026-07-15.md`]

## H24: Combine scan-spatial complexity with temporal sector closing risk
- **Rationale**: A sparse obstacle approaching from one side can look like a benign wall to a spatial score. Consecutive LaserScan sector minima expose positive closing velocity without simulator truth; taking the maximum of spatial and temporal activation preserves the spatial interpretation while intervening earlier on moving lateral hazards.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/prior.py`, `tests/rl/test_scene_complexity.py`, `configs/rl/cross_layer_temporal_gate_l30.yaml`]
- **Evidence update**: L31 refuted this rule as a general dynamic-controller selector; it is retained only as historical L30 logic and must not be treated as deployment-safe outside its tested anchor motion.

## H25: Bind dynamic-obstacle contact, sensing and visualization to one geometry while keeping the planner blind
- **Rationale**: A moving obstacle is a valid platform test only when MuJoCo contact, clearance, ray-cast LaserScan and viewer use the same current body pose. MPPI and the gate must still receive only LaserScan-derived information, never the motion script or body truth.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/simulation/model_factory.py`, `src/mobile_robot_mppi/simulation/mujoco_plant.py`, `configs/research/mujoco_dynamic_crossing.yaml`, `tests/platform/test_mujoco_physics.py`]

## H26: Separate hazard intensity from policy competence confidence
- **Rationale**: A high closing-risk score says that the environment is dangerous; it does not say that the learned prior is the controller most capable of handling that danger. Keep hazard estimation, policy competence and the final blend as distinct logged quantities, and never use a maximum operation that turns high risk directly into full learned-policy authority.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/prior.py`, `docs/rl/43_l31_dynamic_variant_generalization_results_2026-07-15.md`]

## H27: Dynamic direction requires temporal observation with robust association
- **Rationale**: A single LaserScan cannot distinguish two obstacles with the same instantaneous geometry but opposite velocities. Use scan history or robust scan-flow with multiple-beam support, physical rate bounds and explicit invalid-association diagnostics; sector-minimum differencing can generate arbitrarily large identity-switching spikes.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/analyze_dynamic_variant_failures.py`, `results/research_platform/rl/l31_dynamic_variant_development_multiblock_20260715_v1/failure_diagnostics.json`]

## H28: Isolate stateful policy objects by experimental condition
- **Rationale**: Caching one mutable prior by policy family can make conditions with different gates or temporal state silently share the first constructed configuration. Key stateful objects by the complete experimental condition, audit unique episode keys, and quarantine any partial run produced before the repair.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_cross_layer_factorial.py`, `tests/rl/test_cross_layer_factorial.py`, `results/research_platform/rl/l32_temporal_safety_remediation_invalid_shared_prior_20260715/INVALID_DO_NOT_USE.md`]

## H29: Learn bounded corrections before granting full prior authority
- **Rationale**: In hard closed-loop tasks, an unconstrained policy can destroy the conventional prior before replay contains successful experience. Blending a learned proposal with a trusted MPPI warm start bounds early exploration damage while preserving a gradient-bearing influence on candidate trajectories.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/rl/sac_mppi_dynamic_history_bounded_l34.yaml`, `src/mobile_robot_mppi/rl/prior.py`, `docs/rl/49_l33_l34_dynamic_bounded_correction_results_2026-07-16.md`]

## H30: Evaluate a fitted warm start before online normalization changes it
- **Rationale**: Adding one training-seed-specific reset observation before step-zero evaluation makes nominally common warm-start references differ across training seeds and can be amplified by chaotic closed-loop replanning. When a checkpoint already provides fitted normalization statistics, save and evaluate it first; only then update the normalizer with training data.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/trainer.py`, `docs/rl/49_l33_l34_dynamic_bounded_correction_results_2026-07-16.md`]

## H31: Calibrate benchmark difficulty without inspecting the learned methods
- **Rationale**: A scene that is impossible or trivial for every episode cannot identify a controller effect, while choosing scenes after looking at RL or ICODE outcomes introduces method-favoring selection bias. Screen candidate geometry using only the frozen traditional baseline, require within-scene outcome variation, then freeze easy/moderate/hard strata before learned-method comparison.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/rl/dynamic_benchmark_calibration_l36.yaml`, `experiments/rl/run_dynamic_benchmark_calibration.py`, `experiments/rl/summarize_dynamic_benchmark_calibration.py`]

## H32: Require horizon-matched closed-loop eligibility for residual models
- **Rationale**: Lower average offline state RMSE at H=20 does not guarantee that a residual model preserves MPPI candidate-cost ranking at H=36 or improves closed-loop control. Before combining ICODE with RL, collect planner-distribution transitions, evaluate the full planning horizon and cost ranking, and pass a residual-only traditional-MPPI safety/efficacy gate.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/summarize_bounded_rl_icode_factorial.py`, `docs/rl/55_l37_bounded_rl_icode_factorial_results_2026-07-16.md`]

## H33: Match residual structure and control semantics to the physical interface
- **Rationale**: A dynamic-unicycle residual should not overwrite exact pose kinematics, and a delayed plant transition should be supervised by the command actually applied over the interval rather than the newly issued command. Structural output masks plus applied-control labels reduce unidentifiable degrees of freedom and prevent delay from being mislearned as arbitrary state drift.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/models.py`, `src/mobile_robot_mppi/learning/trainer.py`, `src/mobile_robot_mppi/planning/mppi.py`, `experiments/rl/build_l38_onpolicy_residual_dataset.py`]

## H34: Treat offline prediction as an eligibility gate, not a control result
- **Rationale**: Multi-step held-out RMSE verifies that the residual can predict planner-distribution transitions, but MPPI depends on candidate ranking, feedback, costs and safety arbitration. Require an explicit nominal-versus-residual closed-loop gate after offline qualification and preserve failed primary endpoints even when secondary metrics improve.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/summarize_l49_offline_gate.py`, `experiments/rl/summarize_icode_efficiency_confirmation.py`, `docs/rl/74_l49_l50_iterative_path_results_2026-07-16.md`]

## H35: Estimate residual reliability from past innovations, never simulator domain labels
- **Rationale**: L50 shows materially different residual value across frozen mismatch strata. A deployable gate should compare nominal and residual one-step errors only after the transition is observed, accumulate evidence with forgetting and fail closed during cold start; using the configured domain name or future plant state would leak simulator truth and invalidate transfer claims.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/74_l49_l50_iterative_path_results_2026-07-16.md`]

## H36: Isolate dynamics learning before adding state-estimation and perception errors
- **Rationale**: Wheel-slip odometry can dominate tracking and make a dynamics benchmark non-identifiable. First hold the MuJoCo plant fixed and use ground-truth pose/twist with zero sensor noise and latency; add odometry, noise, obstacles and perception later as separately blocked factors.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/research/mujoco_high_dynamic_fixed_plant.yaml`, `docs/rl/81_l56_high_dynamic_benchmark_and_data_results_2026-07-16.md`]

## H37: Use an explicit terminal translation contract without disabling rotation
- **Rationale**: A path tracker can orbit or overshoot when heading correction and forward motion remain coupled near the goal. A local terminal speed limit and bearing gate should suppress translation while leaving angular control available, preserving the safety requirement that the robot may rotate in place near obstacles or the terminal pose.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/mppi.py`, `configs/research/path_tracking_high_dynamic_base.yaml`]

## H38: Bind checkpoints to immutable dataset hashes and evaluate every rollout window
- **Rationale**: First-window-only evaluation can hide trajectory-position bias, and a checkpoint without split hashes cannot prove which transitions trained it. Store split SHA-256 values in checkpoint provenance and evaluate all valid H=1,5,10,20,36 windows before admitting a residual model to closed-loop testing.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/trainer.py`, `experiments/evaluate_platform_residual.py`, `docs/rl/82_l57_high_dynamic_icode_offline_results_2026-07-16.md`]

## H39: After sealed same-domain confirmation, pivot to orthogonal validity factors
- **Rationale**: L58 and L59 reproduced nearly identical 26.62% and 27.06% tracking reductions. More seeds under the same fixed clean plant have low information value; the next experiments should change one predeclared factor at a time, starting with an MLP structural baseline and cross-plant variation before RL recomposition.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/85_l59_high_dynamic_confirmation_results_2026-07-16.md`]

## H40: Parameter-match residual structure ablations on immutable data
- **Rationale**: Comparing ICODE with a narrower direct MLP confounds structure with model capacity. Match parameter count within a frozen tolerance, reuse identical dataset hashes, masks, optimizers, horizons and training seeds, and expose each training seed as a model-level replicate.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/research/mlp_high_dynamic_h36_l60.yaml`, `experiments/rl/summarize_residual_structure_offline.py`, `docs/rl/86_l60_residual_structure_ablation_prereg_2026-07-16.md`]

## H41: Evaluate residual structure through the downstream optimizer, not RMSE alone
- **Rationale**: L60/L62 showed that the direct MLP had lower aggregate H=36 rollout RMSE while ICODE had lower sealed-seed MPPI tracking error. Offline error is an eligibility measure; structure claims require paired closed-loop evaluation because candidate cost ordering and control-dependent perturbations can matter beyond average state error.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/summarize_residual_structure_closed_loop.py`, `docs/rl/90_l62_residual_structure_confirmation_results_2026-07-16.md`]

## H42: Calibrate physical domains with the baseline and cap amendment attempts
- **Rationale**: Cross-plant tests are uninformative when a parameter change is behaviorally invisible or makes the baseline task infeasible, while selecting plant shifts after viewing learned-model outcomes introduces favorable-domain bias. Measure candidate-domain resolution, completion and safety using only the frozen nominal controller; retain failed calibrations; freeze accepted single factors; and cap method-blind amendments before opening learned or sealed results.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/rl/cross_plant_nominal_calibration_l63.yaml`, `configs/rl/cross_plant_nominal_calibration_l63_v3.yaml`, `experiments/rl/summarize_cross_plant_nominal_calibration.py`, `docs/rl/92_l63_cross_plant_nominal_calibration_v2_amendment_2026-07-16.md`]

## H43: Separate bounded observation perturbations from localization drift
- **Rationale**: Latency and measurement noise can be calibrated method-blindly as identifiable observation-domain factors, whereas raw wheel-odometry drift changes the state-estimation problem and can dominate terminal tracking. Calibrate observation factors with nominal MPPI, cap amendments when noise is below outcome resolution, treat raw odometry as a non-gating stress test, and do not expect residual dynamics to replace localization correction.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/rl/observation_domain_nominal_calibration_l66_v2.yaml`, `configs/rl/residual_structure_observation_confirmation_l68.yaml`, `experiments/rl/summarize_residual_structure_observation.py`, `docs/rl/99_l66_l68_observation_robustness_results_2026-07-16.md`]

## H44: Prove context-dependent oracle headroom before training an adaptive MPPI parameter policy
- **Rationale**: L85 traded precision for speed, while anchoring L86 repaired precision but became statistically indistinguishable from the tuned fixed anchor in time to goal. Before further RL tuning, show that different observable contexts have materially different closed-loop optimal sampling parameters and that a context oracle beats the best global fixed setting by more than inference cost and statistical noise. Otherwise RL is being asked to learn an approximately constant mapping.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/129_l86_anchored_covariance_results_2026-07-17.md`, `results/research_platform/rl/l86_anchored_covariance_development_eval_20260717_v1/summary.json`]

## H45: Move RL authority upward when the useful MPPI adaptation is coarse
- **Rationale**: Free continuous covariance SAC traded precision for speed, and a tightly anchored actor collapsed to an approximately constant mapping. Once L87 showed two stable path-dependent options, treating covariance choice as an interpretable contextual-bandit action made the credit assignment shorter, kept MPPI and safety in control of wheel commands, and allowed a direct fixed-policy comparator.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/contextual_bandit.py`, `experiments/rl/train_contextual_covariance_bandit.py`, `docs/rl/134_l89_contextual_covariance_bandit_results_2026-07-18.md`]

## H46: Cache frozen critic factorizations at inference
- **Rationale**: Recomputing a 10-by-10 inverse for each action at every control cycle added about 30 ms under parallel closed-loop evaluation even though the deployed bandit never updates. Cache inverse and coefficient vectors, invalidate only on an actual update, and rerun the complete paired matrix after the optimization; this reduced cached decision time to about 14 microseconds and removed detectable planner overhead without changing control outcomes.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`src/mobile_robot_mppi/rl/contextual_bandit.py`, `tests/rl/test_contextual_bandit.py`, `results/research_platform/rl/l89_contextual_covariance_independent_eval_cached_20260718_v2/summary.json`]

## H47: Test oracle headroom at the intended policy timescale before adding policy complexity
- **Rationale**: Route-level covariance heterogeneity was useful, but L90 found almost no conflicting local optimum and no material segment-progress benefit. A high-frequency RL switch would therefore add variance and credit-assignment difficulty without a demonstrated target. Run the oracle at the exact proposed action frequency and stop when the practical-effect gate fails.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_local_covariance_context_oracle.py`, `docs/rl/137_l90_local_covariance_context_oracle_results_2026-07-18.md`]

## H48: Do not use covariance selection as a substitute for dynamic-obstacle prediction
- **Rationale**: L91's scene-label safety effect did not reproduce on the pre-existing L33 motion geometries, and L93 exposed crossing cases where every covariance candidate collided. When future obstacle motion is absent from the rollout model, changing exploration spread cannot create the missing temporal information. Treat motion prediction and sampling adaptation as separate layers and claims.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_dynamic_covariance_context_oracle.py`, `docs/rl/140_l92_dynamic_covariance_safety_confirmation_results_2026-07-18.md`, `docs/rl/142_l93_route_dynamic_headroom_results_2026-07-18.md`]

## H49: Confirm wall-clock planner gains in one process with balanced arm order
- **Rationale**: Parallel process scheduling can contaminate small planner-time differences. L95 alternated evaluation order and executed both arms in one process on untouched seeds; it reproduced the half-budget time and compute gains. Use parallel runs for throughput, but require a controlled single-process confirmation before making compute-efficiency claims.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_contextual_covariance_half_budget_confirmation.py`, `docs/rl/146_l95_half_budget_controlled_confirmation_results_2026-07-18.md`]

## H50: Evaluate high-level sampling policies through budget-equivalent and half-budget comparisons
- **Rationale**: A route-context policy can improve sample quality without directly controlling the robot. Compare it both against the strongest fixed sampler at equal K and against a fixed sampler with twice K. L94/L95 show the latter contract can turn an interpretable high-level policy into a concrete sample-efficiency result while exposing its jerk cost.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_contextual_covariance_sample_efficiency.py`, `experiments/rl/run_contextual_covariance_half_budget_confirmation.py`, `docs/rl/145_l94_contextual_covariance_sample_efficiency_results_2026-07-18.md`]

## H51: Prefer actuator-facing slew constraints when a predictive rate cost does not change closed-loop jerk
- **Rationale**: In the L96 2x2 screen, doubling MPPI control-rate weight alone left issued and applied jerk statistically unchanged, while lowering the yaw-command rate limit reduced both by about 0.0094 and 0.0060. A hard bound directly constrains the command channel that produces the metric and is easier to transfer to hardware; retain the predictive cost only when it provides an independently measured benefit.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/rl/contextual_covariance_jerk_screening_l96.yaml`, `experiments/rl/run_contextual_covariance_jerk_screening.py`, `docs/rl/148_l96_jerk_screening_results_and_l97_prereg_2026-07-18.md`]

## H52: Separate mechanism confirmation from full-package dominance
- **Rationale**: L97 confirmed that smoothing reduces jerk relative to raw contextual K50, but the smoothed-versus-fixed jerk intervals crossed zero. A supported within-method mechanism does not imply strict system-level superiority; preregister and report both contrasts so a positive mechanism cannot hide an inconclusive final comparator.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_contextual_covariance_jerk_confirmation.py`, `docs/rl/149_l97_contextual_covariance_jerk_confirmation_results_2026-07-18.md`]

## H53: Defer CUDA until profiling shows a task-relevant bottleneck
- **Rationale**: MuJoCo and the current NumPy MPPI rollout are CPU-oriented, and moving only the small frozen ICODE inference to CUDA can add transfer overhead while invalidating established CPU timing baselines. The user explicitly chose not to add CUDA unless it benefits the experiment. Keep a future backend boundary possible, but do not introduce GPU variables before a measured real-time bottleneck.
- **Provenance**: user
- **Sensitivity**: medium
- **Code ref**: [`src/mobile_robot_mppi/planning/mppi.py`, `configs/rl/contextual_covariance_jerk_confirmation_l97.yaml`]

## H54: Use exact nested samples and restored physical snapshots for budget causality
- **Rationale**: Comparing independent K50 and K100 draws confounds budget with Monte Carlo luck. Generate one K100 pool, require a byte-identical K50 prefix, reweight each complete pool, and execute both weighted sequences from the same full MuJoCo snapshot. Average repeated anchors within episode before inference.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/rl/run_anytime_budget_oracle.py`, `tests/rl/test_anytime_budget_oracle.py`, `docs/rl/152_l100_nested_anytime_budget_oracle_prereg_2026-07-18.md`]

## H55: Separate exploration price from deployment-budget calibration
- **Rationale**: A primal-dual price controls partial-feedback exploration but its terminal value can depend on episode order and fail under context shift. Freeze the learned reward model, calibrate a deterministic deployment price from discovery prediction quantiles only, and evaluate the resulting compute envelope on disjoint episodes.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/budget_bandit.py`, `experiments/rl/evaluate_anytime_budget_bandit.py`, `docs/rl/155_l102_failure_l103_calibrated_bandit_prereg_2026-07-18.md`]

## H56: Treat sampled-state counterfactual success as an online-controller eligibility Gate
- **Rationale**: Snapshot branches identify whether added rollouts have useful state-dependent value, but they do not include recurrent state visitation, safety interventions or accumulated timing effects. Require an incremental online implementation and a nominal/ICODE x fixed/adaptive factorial before claiming closed-loop or cross-layer interaction benefit.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/156_l103_calibrated_budget_bandit_confirmation_results_2026-07-18.md`]

## H57: Cluster repeated scene-domain measurements by independent seed
- **Rationale**: A scene-by-domain-by-seed factorial produces many rows, but rows sharing one seed are repeated strata rather than independent replications. Aggregate treatment contrasts within seed before bootstrap inference; otherwise the nominal sample size is inflated and confidence intervals are too narrow.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/evaluation/factorial.py`, `experiments/rl/analyze_gate1_factorial.py`, `tests/evaluation/test_factorial.py`, `docs/rl/170_gate1_multidomain_factorial_results_2026-07-18.md`]

## H58: Batch frozen Actor rollouts and cache step-invariant perception features
- **Rationale**: Candidate Actor rollouts share the same latest LaserScan sector encoding within one planning step, while only kinematic state changes across candidates. Cache the scan feature once, vectorize kinematic encoding and normalizer operations, and evaluate deterministic and stochastic Actor branches jointly. Exact regression tests are required because a performance optimization that changes the proposal distribution invalidates the experiment.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/observation.py`, `src/mobile_robot_mppi/rl/direct_control.py`, `src/mobile_robot_mppi/rl/paper_direct_control.py`, `tests/rl/test_paper_rl_mppi.py`]

## H59: Separate critic task competence from support and twin agreement
- **Rationale**: Two critics can agree confidently on values in a state region where the frozen policy still has poor task outcomes. Multiply support and consensus confidence by a competence term calibrated only from successful versus failed training episodes before allowing value gradients to shape the residual model.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/value_alignment.py`, `src/mobile_robot_mppi/learning/residual_trainer.py`, `configs/icode/gate2_competence_gated_icode_l189.yaml`, `tests/learning/test_value_alignment.py`]

## H60: Merge parallel confirmation shards only after immutable-provenance validation
- **Rationale**: Independent result directories can reduce wall time without compromising treatment trajectories, but only when source revision, checkpoint hashes, rollout budget and optimizer iterations match and seed sets are disjoint. Recompute paired effects from merged episode rows rather than averaging shard summaries.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/evaluation/paired_checkpoint.py`, `experiments/rl/merge_paired_checkpoint_runs.py`, `tests/evaluation/test_paired_checkpoint.py`]

## H61: Pre-register practical equivalence margins for secondary controller endpoints
- **Rationale**: An exact zero-regression rule turns a negligible 0.066% jerk point estimate into a formal failure despite an interval spanning both directions and large progress gains. Engineering noninferiority margins must be chosen prospectively from physical relevance before the final experiment, never retrofitted to make an observed run pass.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/evaluation/paired_checkpoint.py`, `docs/rl/173_gate2_competence_gated_value_alignment_results_2026-07-18.md`]

## H62: Keep ensemble-disagreement and innovation normalization dimensionally separate
- **Rationale**: ICODE ensemble disagreement measures derivative spread, while completed-transition innovation measures state prediction error. Reusing one scale vector gives physically inconsistent authority. Normalize disagreement by training residual-derivative scales and innovation by task-relevant one-step state-error scales, and version both in the calibration config.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/models.py`, `configs/rl/gate3_reliability_calibration_l193.yaml`, `tests/learning/test_residual_ensemble.py`]

## H63: Validate reliability on a severity spectrum rather than a severe-only OOD split
- **Rationale**: The original four-episode unseen set correctly saturated at low authority but could not test ordering across discrete authority bins. Retain that failed Gate, freeze thresholds, then use new seeds and a preregistered mix of nominal, single-factor and combined shifts with episode as the independent unit. A severe-only split is useful for fallback activation, not for calibration resolution.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/175_gate3a2_graded_stress_prereg_2026-07-18.md`, `experiments/rl/evaluate_gate3_reliability_stress.py`, `configs/icode/gate3_reliability_stress_data_l194.yaml`]

## H64: Let reliability allocate sampling authority without attenuating learned dynamics
- **Rationale**: Falling back toward nominal dynamics when uncertainty grows can discard a residual model that remains better than nominal. Keep the ensemble mean as the rollout model, use only online-available disagreement/support/innovation signals to choose 0/30/60% persistent Actor samples, and apply current-rollout authority on the next control cycle to preserve causal joint batching.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/models.py`, `src/mobile_robot_mppi/rl/reliability.py`, `src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `docs/rl/174_gate3_reliability_hss_prereg_2026-07-18.md`]

## H65: Calibrate each authority against its own downstream error target
- **Rationale**: ICODE rollout reliability can rank model error and improve sampling allocation while still being unsuitable for attenuating a frozen critic. Sampling authority should be calibrated against proposal utility; terminal-value authority requires critic-return or candidate-ranking calibration. Do not reuse a confidence mapping merely because both consumers sit inside MPPI.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/reliability.py`, `docs/rl/178_gate4a_critic_calibration_result_2026-07-19.md`, `docs/rl/179_gate4_conservative_terminal_results_2026-07-19.md`]

## H66: Preserve an explicit geometric fallback when attenuating learned terminal cost
- **Rationale**: Multiplying a learned terminal cost by low confidence can otherwise make an unknown candidate artificially cheap. Keep the ordinary MPPI geometric terminal in the base cost, attenuate only the incremental learned value and make any uncertainty penalty non-negative. Even a safe algebraic construction still requires independent outcome confirmation.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `tests/planners/test_paper_rl_driven_mppi.py`, `docs/rl/177_gate4_conservative_terminal_prereg_2026-07-19.md`]

## H67: Give reliability degradation a task-completion boundary condition
- **Rationale**: A globally low model-reliability score can suppress all Actor candidates just as the controller approaches the goal, even when fixed Actor guidance is what closes the final 30--50 cm. Outside the terminal region, retain the calibrated 0/30/60% authority; inside one maximum-horizon travel distance, impose only the already qualified 30% paper baseline as a floor. Derive the radius from \(v_{\max}H\Delta t\), not from a trajectory-tuned threshold, and keep total MPPI rollouts unchanged.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `tests/planners/test_paper_rl_driven_mppi.py`, `docs/rl/181_completion_preserving_hss_amendment_2026-07-19.md`]

## H68: Use one preregistered remediation and keep confirmation seeds sealed
- **Rationale**: When a development Gate exposes a concrete failure, inspect trajectory-level mechanism evidence, freeze one physically derived remedy, rerun only development seeds and stop if it fails. Do not search multiple mappings against the same outcomes or inspect confirmation seeds before the corrected method passes its development Gate.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/rl/180_full_proposed_factorial_prereg_2026-07-19.md`, `docs/rl/181_completion_preserving_hss_amendment_2026-07-19.md`, `experiments/rl/analyze_full_proposed_factorial.py`]

## H69: Separate package superiority from factorial synergy
- **Rationale**: A complete method may strongly outperform a simple combination while still showing a null or adverse interaction because two modules repair overlapping failures or saturate success. Report the full-versus-simple contrast, both factorial main effects and the interaction separately; never relabel package superiority as super-additive synergy.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/evaluation/factorial.py`, `experiments/rl/analyze_full_proposed_factorial.py`, `docs/rl/182_full_proposed_factorial_confirmation_results_2026-07-19.md`]

## H70: Bind completion safeguards to reference phase, not lookahead distance alone
- **Rationale**: A polyline controller's current target is a nearby lookahead point throughout the route. A distance-only terminal floor therefore activates globally and silently destroys adaptive sampling. Require an explicit `terminal_approach` or `terminal` reference phase, log both raw and floor-adjusted authority, and test tracking and terminal phases separately.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `src/mobile_robot_mppi/evaluation/metrics.py`, `tests/planners/test_paper_rl_driven_mppi.py`, `docs/rl/183_full_proposed_path_tracking_prereg_2026-07-19.md`]

## H71: Calibrate relative proposal yield around parity
- **Rationale**: A guided/Gaussian elite-yield ratio is a relative source comparison, not an absolute confidence probability. Map a preregistered below-parity region to zero authority and parity to full competence before smoothing; then require independent confirmation because the mapping can remain path dependent.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/reliability.py`, `configs/rl/source_relative_actor_competence_l197.yaml`, `tests/rl/test_reliability.py`]

## H72: Require held-out value consistency before using critic gradients to fine-tune dynamics
- **Rationale**: A value loss can be active and improve its training-route objective while worsening control-relevant generalization. Match the critic's observation encoding, split by route and physics, select by held-out rollout/value metrics, and reject the fine-tuned model if epoch zero remains optimal.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/value_alignment.py`, `experiments/icode/train_value_aligned_icode.py`, `docs/rl/202_l196_l201_reliability_and_value_alignment_results_2026-07-19.md`]

## H73: Isolate learned-prior mean from proposal covariance before attribution
- **Rationale**: A bounded correction Actor can silently change both deterministic prior mean and post-tanh sampling spread. Preserve the frozen base Actor's first-order post-tanh spread when testing residual-conditioned mean correction; otherwise an apparent policy effect is confounded by covariance adaptation.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/sac.py`, `tests/rl/test_sac_and_checkpoint.py`, `docs/rl/207_mean_only_residual_policy_screen_prereg_2026-07-19.md`]

## H74: Do not optimize a critic-ranking objective at a held-out ceiling
- **Rationale**: When the baseline dynamics already yields critic rank correlations near 0.98 on test and unseen routes, a pairwise ranking loss has little headroom and readily overfits the weaker validation split. Measure baseline rank first and require disjoint-route improvement plus rollout noninferiority before expanding to an ensemble.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/value_alignment.py`, `experiments/icode/train_value_aligned_icode.py`, `configs/icode/value_ranked_icode_member1_l210.yaml`, `docs/rl/211_two_coupling_mechanisms_frozen_2026-07-19.md`]

## H75: Preserve automatic gate truth when authorizing a stage waiver
- **Rationale**: A project decision may accept a narrowly missed development gate, but the machine-readable result and failed threshold must remain unchanged. Record the waiver separately, state its scope, and freeze the accepted baseline so later gains cannot be attributed to hidden environment retuning.
- **Provenance**: user-revised
- **Sensitivity**: high
- **Code ref**: [`docs/experiments/dynamic_uncertainty/ENVIRONMENT_STAGE_ACCEPTANCE.md`, `docs/experiments/dynamic_uncertainty/POST_ESCAPE_RECOVERY_AMENDMENT17_REPORT.md`]

## H76: Block residual comparisons by checkpoint seed and obstacle seed
- **Rationale**: A frozen residual checkpoint is a trained-model replicate, while a complete obstacle episode is the closed-loop stochastic replicate. Compare nominal and ICODE prediction with common obstacle seeds inside each model block, summarize effects across the three model blocks, and keep sealed seeds closed until the development gate passes.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/research/dynamic_uncertainty_residual_stage1_protocol.yaml`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_PREREGISTRATION.md`]

## H77: Gate learned dynamics on closed-loop safety, not average rollout RMSE alone
- **Rationale**: A residual model can reduce average multi-step velocity/yaw-rate error while changing early MPPI actions enough to enter a different obstacle-avoidance mode many seconds later. Require checkpoint-blocked collision and clearance gates in addition to offline prediction metrics, especially when hard chance constraints create discontinuous candidate selection.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/dynamic_uncertainty/run_residual_dynamics_stage1.py`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_RESULT.md`]

## H78: Prediction-only residual wrappers must expose unity confidence
- **Rationale**: Canonicalization adds a diagnostic wrapper around a residual model, but a plain single-model residual has no uncertainty gate. When an MPPI consumer queries confidence, the mathematically neutral and backward-compatible value is one with the state's leading shape; assuming a `support_confidence` method exists crashes valid single-checkpoint inference.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`src/mobile_robot_mppi/learning/models.py`, `tests/learning/test_residual_state_canonicalization.py`]

## H79: Residual stall guards must be causal and preview-side-effect free
- **Rationale**: A stall counter may consume measured state and the previous completed plan's risk, but preview planning must not advance it. Episode latching can fail closed after a reproducible trigger, yet a late latch cannot undo state displacement accumulated earlier in the closed loop.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/models.py`, `src/mobile_robot_mppi/planning/mppi.py`, `tests/platform/test_mppi_residual_stall_guard.py`]

## H80: Preserve known kinematics without treating the mask as a safety certificate
- **Rationale**: Dynamic-unicycle residuals should normally leave `x_dot`, `y_dot` and `theta_dot` to their analytic equations and correct only `v_dot` and `omega_dot`. This removes an avoidable structural degree of freedom, but Amendment 4 shows that structurally valid acceleration corrections can still alter receding-horizon decisions enough to cause collision.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`src/mobile_robot_mppi/learning/models.py`, `configs/research/dynamic_uncertainty_residual_stage1_amendment4_probe.yaml`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT4_RESULT.md`]

## H81: Separate predictive utility from direct closed-loop authority
- **Rationale**: A residual can reduce blocked multi-step prediction error and still be unsafe when recursively inserted into a chance-constrained controller. Retain it as an auxiliary predictor or representation until task-aware bounded integration is independently qualified; use the frozen nominal probabilistic MPPI for testing an RL sampling prior so residual and policy effects remain identifiable.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE1_AMENDMENT4_RESULT.md`]

## H82: Weight hazard-region transitions without breaking episode splits
- **Rationale**: Rare near-obstacle transitions are otherwise diluted by thousands of ordinary path samples. Increase their positive loss weight using only causal saved quantities, but append complete episodes to fixed train/validation/test partitions so rollout windows and model selection remain leakage-free.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/dynamic_uncertainty/build_task_aware_residual_dataset.py`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_PREREGISTRATION.md`]

## H83: Anchor task-aware fine-tuning on the original residual function
- **Rationale**: A small task dataset can improve its local operating region while catastrophically forgetting previously qualified dynamics. Initialize from the matching checkpoint and penalize output deviation on original-source rows; then impose an explicit original test/unseen regression gate.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/trainer.py`, `configs/research/icode_dynamic_uncertainty_task_aware_stage2.yaml`]

## H84: Propagate the accepted sequence's conservative risk diagnostics
- **Rationale**: A dual-controller shield may accept a residual or hybrid control sequence whose downstream safety diagnostics cannot be copied from either source plan. Recompute both model views and publish the conservative probability mass, union bound, maximum step probability and hard-violation flag for the actual selected sequence.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/residual_shield.py`, `tests/platform/test_residual_safety_shield.py`]

## H85: Require a fresh development pilot after offline residual qualification
- **Rationale**: Task-weighted validation can establish prediction and retention but not chance-constrained closed-loop safety. Register a new obstacle-process seed before execution, pair nominal with every independently trained residual block, keep sealed seeds closed, and require nondegenerate residual acceptance before interpreting zero collisions as more than permanent fallback.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/research/dynamic_uncertainty_residual_stage2_task_aware_probe.yaml`, `experiments/dynamic_uncertainty/run_residual_dynamics_stage1.py`]

## H86: Audit where the global minimum clearance occurs
- **Rationale**: A shared minimum clearance can be dominated by the initial pose rather than the obstacle crossing. Record its time index and report clearance quantiles or time-below-threshold fractions so an initialization artifact is not mistaken for equivalent crossing safety.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`experiments/dynamic_uncertainty/analyze_task_aware_residual_contribution.py`, `docs/experiments/dynamic_uncertainty/RESIDUAL_DYNAMICS_STAGE2_TASK_AWARE_DEVELOPMENT_RESULT.md`]

## H87: Separate shield acceptance from realized control contribution
- **Rationale**: A proposal can pass a model gate yet equal the nominal action, while paired closed-loop traces cease to be same-state counterfactuals after their first action difference. Report both shield acceptance and an explicit executed-action difference, label the latter paired-realized, and do not interpret later stepwise differences as isolated causal effects.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/dynamic_uncertainty/analyze_task_aware_residual_contribution.py`, `research_artifacts/dynamic_uncertainty_residual_stage2_task_aware_development/contribution_audit.json`]

## H88: A rollout microbenchmark does not replace full-controller runtime qualification
- **Rationale**: Component profiling identifies the dominant kernel and cheaply rejects bad backends, but the deployed latency also includes nominal planning, cost evaluation, shield comparison and scheduling. Require both a fixed-shape rollout screen and the complete blocked closed-loop matrix before claiming the control-period target.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/dynamic_uncertainty/benchmark_residual_runtime_stage3.py`, `experiments/dynamic_uncertainty/run_residual_dynamics_stage1.py`, `docs/experiments/dynamic_uncertainty/RESIDUAL_RUNTIME_STAGE3_RESULT.md`]

## H89: Capture fixed-horizon small-network residual integration as one CUDA Graph
- **Rationale**: The H36/K600 RK4 rollout invokes the same small residual network 144 times, so eager CUDA is dominated by CPU launch dispatch. Reuse static state/control/output buffers, capture the complete fixed-shape computation once, replay it, and copy only the inputs and final trajectory while retaining a numerical-equivalence gate.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/learning/models.py`, `src/mobile_robot_mppi/planning/dynamics.py`, `src/mobile_robot_mppi/planning/mppi.py`, `tests/learning/test_residual_device_rollout.py`]

## H90: Parallelize independent dual-model planning and preserve the shield join
- **Rationale**: Nominal and residual MPPI consume the same immutable observation/reference but maintain independent RNGs, dynamics, warm starts and diagnostics. Running them on persistent workers removes their serial sum; comparing the completed plans with the unchanged shield preserves decision semantics and makes exact sequential-equivalence auditing possible.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/residual_shield.py`, `src/mobile_robot_mppi/runtime/factories.py`, `tests/platform/test_residual_safety_shield.py`]

## H91: Treat a frozen Actor's action range as an explicit proposal subspace
- **Rationale**: A checkpoint trained on forward speed `[0.0, 0.35]` must not be silently stretched to a controller that also permits reverse speed. Validate that the checkpoint bounds are a strict subset, preserve them for Actor proposals, retain the controller's full bounds for Gaussian MPPI and safety recovery, and require an explicit opt-in contract. In a dual-model shield, deep-copy the Actor, HSS sidecar and optimizer state so matched controllers share frozen parameters and seeds but not mutable objects.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/paper_policy.py`, `src/mobile_robot_mppi/runtime/factories.py`, `tests/dynamic_uncertainty/test_rl_hss_stage4.py`]

## H92: Exercise one real planning call for every newly wired safety interface
- **Rationale**: Component construction, hashes and configuration equality cannot prove that a custom optimizer forwards every runtime observation auxiliary. Before a formal run, execute the real `plan()` path with each enabled safety contract, including present and absent probabilistic forecasts, and assert candidate filtering, final action guarding and diagnostics. This would have caught the original Stage 4 failure before launch.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `tests/dynamic_uncertainty/test_rl_hss_stage4.py`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE4_AMENDMENT1_PREREGISTRATION.md`]

## H93: Do not rescue proposal authority without observed proposal advantage
- **Rationale**: In all ten completed Stage 4 RL/HSS-on episodes, the best guided subset was worse than the Gaussian subset under the controller cost, yet mean proposal authority remained 0.611--0.922 while mean dynamics confidence was 0--0.016. A transferred Actor should lose proposal share when its causal cost or progress advantage is absent, even if checkpoint competence was high in its source domain; any new gate requires a separately frozen development study.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `research_artifacts/dynamic_uncertainty_rl_hss_stage4_amendment1_development/paired_analysis.json`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE4_RESULT.md`]

## H94: Couple an episode-latched bad-proposal veto to both candidate quota and distribution authority
- **Rationale**: When completed current-planner evidence shows the best guided proposal is more costly than the best Gaussian proposal for three consecutive comparable cycles, apply the rejection only from the next cycle and latch it for the episode. Set guided count and Actor mean/variance authority to zero together, make fallback fraction one, and preserve the fixed rollout budget; otherwise a nominally vetoed Actor can still bias either candidate allocation or the sampling distribution.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/reliability.py`, `src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `tests/rl/test_cross_layer_reliability.py`, `tests/planners/test_paper_rl_driven_mppi.py`]

## H95: Use unequal-source minimum cost only as a one-way rejection signal
- **Rationale**: Guided and Gaussian candidate pools can have different unique sample counts and reuse patterns, so their minima have unequal order-statistic bias. A persistent guided-minus-Gaussian minimum-cost disadvantage is acceptable for a conservative one-way safety veto whose false rejection falls back to Gaussian search, but it is not a calibrated competence probability and must not by itself restore Actor authority.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`experiments/dynamic_uncertainty/analyze_stage4_proposal_advantage.py`, `src/mobile_robot_mppi/rl/reliability.py`, `docs/experiments/dynamic_uncertainty/RL_HSS_STAGE5_PROPOSAL_ADVANTAGE_RESULT.md`]

## H96: Preserve the frozen Actor prefix and append causal scan deltas
- **Rationale**: A single LaserScan cannot distinguish an obstacle approaching from one receding. Appending per-sector causal range deltas gives the correction head motion evidence while `correction_base_observation_dim` keeps the frozen 48-dimensional Actor input and its physical action exactly unchanged. Reverse demonstrations require a separate stratum, retention anchor and false-reverse penalty because naive reverse oversampling can produce prolonged unnecessary backing.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/rl/observation.py`, `src/mobile_robot_mppi/rl/sac.py`, `experiments/dynamic_uncertainty/train_dynamic_actor_correction.py`, `configs/research/dynamic_actor_temporal_bidirectional_correction_development_amendment2.yaml`]

## H97: Gate guided elites with same-cycle proposal advantage
- **Rationale**: An episode-latched veto waits for several bad cycles and then permanently removes the Actor, while unrestricted proposal use lets a currently inferior guided subset bias the rolling MPPI warm start. Compare the best guided and Gaussian costs after both are evaluated in the current cycle; if the guided best exceeds the Gaussian best by the frozen relative margin, exclude guided candidates from that cycle's elite/update only. This preserves useful active-passage proposals on good cycles, fails closed on clearly bad cycles, and does not change the fixed rollout budget. Because unequal pool sizes affect minima, the comparison remains a one-way rejection rule rather than a calibrated competence score.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`src/mobile_robot_mppi/planning/rl_driven_mppi.py`, `experiments/dynamic_uncertainty/run_dynamic_actor_mujoco_probe.py`, `tests/planners/test_paper_rl_driven_mppi.py`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_fresh_replication_summary.json`]

## H98: Require an arm-order-balanced expanded safety gate before sealing an adapted Actor
- **Rationale**: Two favorable eight-pair batches did not reveal the candidate-only collision and lost source success found in the next 24 frozen pairs. Freeze checkpoint, controller budget, filter and safety stack; pair treatments on the same episode seed; balance source-first and candidate-first execution within batches; prohibit tuning during the matrix; and require zero new collisions and zero lost baseline successes in addition to pooled completion/efficiency gains. Completion gains cannot compensate for a candidate-only collision in a safety-critical navigation qualification.
- **Provenance**: ai-suggested
- **Sensitivity**: high
- **Code ref**: [`configs/research/dynamic_actor_v5a6_samecycle_expanded_development.yaml`, `experiments/dynamic_uncertainty/run_dynamic_actor_expanded_development.py`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/gate.json`, `research_artifacts/dynamic_actor_v5a6_u000250_samecycle_expanded_development/integrity_audit.json`]

## H99: Penalize failed episodes before comparing navigation efficiency
- **Rationale**: Uncensored raw step sums can make an early collision look efficient because the failed episode terminates before a successful controller finishes the route. Freeze a failure-step penalty before execution, compare this outcome-aware total alongside raw steps, and still keep collision and lost-success gates lexicographically mandatory. This preserves the real raw-step trade-off while preventing success from being penalized solely for continuing after the baseline failed.
- **Provenance**: ai-suggested
- **Sensitivity**: medium
- **Code ref**: [`configs/research/dynamic_actor_v5a6_samecycle_expanded_development_amendment2.yaml`, `experiments/dynamic_uncertainty/run_dynamic_actor_expanded_development.py`, `research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/summary.json`, `research_artifacts/dynamic_actor_v5a6_u000250_pareto_forward_commit_expanded_development_amendment2/formal_statistics.json`]
