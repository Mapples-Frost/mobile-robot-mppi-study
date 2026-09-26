# Bohn 2021 full-grid validation diagnosis and H-range probe

The paper-configuration grid is complete: `delivery_complete.json` has `passed=true`, the cost audit has `passed=true` for 26 models and 1,560 validation/holdout episode conditions, and no continuation process remains. These checks do not establish the paper's adaptive-H advantage. The existing holdout comparison remains negative, and its scenes were already exposed.

## Old validation: descriptive decomposition

`paper_grid_validation_diagnosis.py` independently sums each saved validation trace and matches its episode summary. It reads only the ten-scene `eval_value` traces for the validation-selected H25 vehicle and H30 pendulum comparators and the three RL seeds. Input paths and SHA256 hashes are in `validation_diagnosis.json`; no holdout trace is read.

| Task | RL seed | RL minus fixed mean cost | Physical component | H component | Constraint component |
|---|---:|---:|---:|---:|---:|
| Vehicle | 0 | +187.724 | +187.308 | +0.416 | 0 |
| Vehicle | 1 | -0.948 | -0.904 | -0.044 | 0 |
| Vehicle | 2 | +30.535 | +30.135 | +0.400 | 0 |
| Pendulum | 0 | +14.368 | +13.681 | +0.687 | 0 |
| Pendulum | 1 | +9.462 | +9.733 | -0.271 | 0 |
| Pendulum | 2 | +96.329 | +98.080 | -1.751 | 0 |

Vehicle seed0 has no solver failures over 903 validation steps, yet its worst paired scene excesses are +504.245, +329.474 and +303.803. Its H choices fall in 1-9 for 159 steps and 40-50 for 286 steps; vehicle seed1 has 24 and 26 steps in those bins. Pendulum seed2 uses H1-9 for 159/909 steps, while seed0 and seed1 never use that bin. This motivates a decision-range hypothesis, not a causal diagnosis: different terminal values and reset warmups also differ between fixed and RL models.

## Frozen H-range extension: stopped after validation

Before generating new banks, `horizon_range_probe.py` registered a single [10,35] execution range for both tasks, eight new validation scenes per task and twenty independent test scenes per task. It retains the original trained model and terminal value. The vehicle seed0 original and clipped policies were evaluated on the same eight new validation scenes, with identical model-weight and bank hashes. All eight saved traces match their summaries. The test bank has been generated and hashed but has **not** been evaluated.

| Vehicle seed0 | Mean total cost | Goals / 8 | Solver failures |
|---|---:|---:|---:|
| Original actor | 238.300 | 8 | 1 / 691 steps |
| H clipped to [10,35] | 3419.829 | 7 | 151 / 703 steps |

The clipped policy improves seven paired scenes, but scene 6 changes from cost 654.49 with goal arrival and zero solver failures to 26,738.42 with 150-step timeout and 150 solver-failed steps. On its first scored action, H50 solves successfully while the clipped H35 solve reports failure; later trajectories diverge. This is sufficient to reject the proposed uniform clipping as a reliable repair on this validation sample. It does not prove that H50 is always necessary or identify a unique training defect; later solver failures could also be downstream of the changed trajectory.

The registered protocol expected a full fixed-H validation grid and all three RL seeds before test. We stopped early after a decisive adverse validation result, so that planned grid was **not completed** and no test claim is made. This is a disclosed futility deviation, not a passing confirmation. The independent test bank remains unused; do not tune the [10,35] limits against it or present the seven improved scenes alone as success.

The next intervention should target branch-value calibration or training coverage with its own new protocol. It must keep the original author-code reconstruction separate from any modified training and give fixed H matched training seeds, selection opportunities and compute accounting before an algorithm-superiority claim. The present negative probe needs no further evaluation on its held-out bank.
