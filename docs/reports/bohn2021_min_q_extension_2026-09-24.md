# Bohn 2021 min-Q actor training extension

This is a one-factor method extension, not a strict reproduction of the author's SAC. The actor uses min(Q1,Q2) where the pinned author implementation uses Q1. The terminal value, task configuration and training budget match the paper-grid reconstruction. The original experiment configuration and test files remain unavailable.

## Verdict

Data audit: **passed** for 180 validation and 360 independent-test model-episode conditions. The registered robust adaptive-H advantage: **not observed** across both tasks. A negative or mixed result does not refute the original paper because the task configuration was reconstructed.

All three training seeds are reported. Fixed H was chosen on the older validation set from the complete ten-H grid at seed0; selected H was then trained at seeds1/2 with the same 15,000-step settings. Neither new bank was used to choose H, seed or checkpoint. The historical holdout was already exposed and is not used for this claim.

## Validation

| Task | Method | Seed | Total | Physical | H proxy | Constraint | Goals | Constraint stops | Solver failures | Mean H |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pendulum | author_rl | 0 | 350.128 | 167.499 | 7.629 | 175.000 | 0 | 2 | 25/825 | 30.57 |
| pendulum | author_rl | 1 | 337.336 | 155.689 | 6.648 | 175.000 | 0 | 2 | 25/825 | 27.84 |
| pendulum | author_rl | 2 | 413.862 | 234.139 | 4.723 | 175.000 | 0 | 2 | 23/825 | 19.19 |
| pendulum | fixed | 0 | 317.714 | 134.298 | 7.416 | 176.000 | 0 | 2 | 24/824 | 30.00 |
| pendulum | fixed | 1 | 317.610 | 134.194 | 7.416 | 176.000 | 0 | 2 | 24/824 | 30.00 |
| pendulum | fixed | 2 | 317.734 | 134.318 | 7.416 | 176.000 | 0 | 2 | 24/824 | 30.00 |
| pendulum | min_q | 0 | 319.128 | 135.637 | 7.491 | 176.000 | 0 | 2 | 24/824 | 31.20 |
| pendulum | min_q | 1 | 340.645 | 159.753 | 5.892 | 175.000 | 0 | 2 | 25/825 | 24.69 |
| pendulum | min_q | 2 | 470.415 | 288.449 | 5.966 | 176.000 | 0 | 2 | 24/824 | 23.08 |
| vehicle | author_rl | 0 | 227.068 | 224.979 | 2.088 | 0.000 | 10 | 0 | 0/836 | 25.45 |
| vehicle | author_rl | 1 | 20.368 | 18.552 | 1.816 | 0.000 | 10 | 0 | 0/780 | 23.40 |
| vehicle | author_rl | 2 | 13.166 | 11.028 | 2.138 | 0.000 | 10 | 0 | 0/779 | 27.40 |
| vehicle | fixed | 0 | 20.410 | 18.462 | 1.947 | 0.000 | 10 | 0 | 0/779 | 25.00 |
| vehicle | fixed | 1 | 17.413 | 15.465 | 1.947 | 0.000 | 10 | 0 | 0/779 | 25.00 |
| vehicle | fixed | 2 | 14.943 | 12.995 | 1.947 | 0.000 | 10 | 0 | 0/779 | 25.00 |
| vehicle | min_q | 0 | 79.396 | 77.617 | 1.779 | 0.000 | 9 | 0 | 0/854 | 21.45 |
| vehicle | min_q | 1 | 26.380 | 25.420 | 0.960 | 0.000 | 10 | 0 | 0/779 | 12.38 |
| vehicle | min_q | 2 | 15.647 | 13.202 | 2.445 | 0.000 | 10 | 0 | 0/779 | 31.30 |

| Task | Fixed H | Mean author RL | Mean min-Q | Mean fixed H | min-Q minus author by seed | min-Q minus fixed by seed | Registered robust advantage |
|---|---:|---:|---:|---:|---|---|---|
| vehicle | 25 | 86.867 | 40.474 | 17.589 | -147.672, +6.012, +2.481 | +58.986, +8.967, +0.704 | no |
| pendulum | 30 | 367.109 | 376.729 | 317.686 | -30.999, +3.309, +56.553 | +1.415, +23.035, +152.681 | no |

## Test

| Task | Method | Seed | Total | Physical | H proxy | Constraint | Goals | Constraint stops | Solver failures | Mean H |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pendulum | author_rl | 0 | 710.314 | 189.278 | 4.536 | 516.500 | 0 | 12 | 167/967 | 31.48 |
| pendulum | author_rl | 1 | 711.970 | 191.318 | 4.152 | 516.500 | 0 | 12 | 165/967 | 31.84 |
| pendulum | author_rl | 2 | 750.906 | 230.065 | 3.342 | 517.500 | 0 | 12 | 160/965 | 23.09 |
| pendulum | fixed | 0 | 691.481 | 170.630 | 4.351 | 516.500 | 0 | 12 | 159/967 | 30.00 |
| pendulum | fixed | 1 | 691.448 | 170.597 | 4.351 | 516.500 | 0 | 12 | 159/967 | 30.00 |
| pendulum | fixed | 2 | 691.481 | 170.630 | 4.351 | 516.500 | 0 | 12 | 159/967 | 30.00 |
| pendulum | min_q | 0 | 692.807 | 171.690 | 4.617 | 516.500 | 0 | 12 | 167/967 | 32.22 |
| pendulum | min_q | 1 | 712.785 | 191.671 | 3.614 | 517.500 | 0 | 12 | 160/965 | 25.75 |
| pendulum | min_q | 2 | 762.786 | 242.800 | 3.486 | 516.500 | 0 | 12 | 157/967 | 26.14 |
| vehicle | author_rl | 0 | 413.339 | 410.920 | 2.420 | 0.000 | 17 | 0 | 1/1878 | 26.61 |
| vehicle | author_rl | 1 | 33.240 | 19.724 | 2.016 | 11.500 | 19 | 1 | 20/1558 | 25.62 |
| vehicle | author_rl | 2 | 345.732 | 343.082 | 2.650 | 0.000 | 18 | 0 | 80/1823 | 28.95 |
| vehicle | fixed | 0 | 1212.144 | 1210.053 | 2.091 | 0.000 | 19 | 0 | 129/1673 | 25.00 |
| vehicle | fixed | 1 | 1457.967 | 1455.778 | 2.189 | 0.000 | 18 | 0 | 134/1751 | 25.00 |
| vehicle | fixed | 2 | 555.406 | 553.217 | 2.189 | 0.000 | 18 | 0 | 41/1751 | 25.00 |
| vehicle | min_q | 0 | 2428.807 | 2426.973 | 1.834 | 0.000 | 19 | 0 | 12/1663 | 22.97 |
| vehicle | min_q | 1 | 18.461 | 17.302 | 1.159 | 0.000 | 20 | 0 | 0/1592 | 14.47 |
| vehicle | min_q | 2 | 1486.521 | 1483.620 | 2.902 | 0.000 | 18 | 0 | 137/1752 | 32.41 |

| Task | Fixed H | Mean author RL | Mean min-Q | Mean fixed H | min-Q minus author by seed | min-Q minus fixed by seed | Registered robust advantage |
|---|---:|---:|---:|---:|---|---|---|
| vehicle | 25 | 264.104 | 1311.263 | 1075.172 | +2015.468, -14.779, +1140.790 | +1216.663, -1439.506, +931.116 | no |
| pendulum | 30 | 724.397 | 722.792 | 691.470 | -17.507, +0.815, +11.879 | +1.326, +21.337, +71.304 | no |

## Failure and interpretation checks

The vehicle test mean is sensitive to single long-running scenes. The following is a post-test diagnostic, not a replacement for the registered mean-cost endpoint. Scene indices are zero-based.

| Vehicle method | Seed | Highest-cost scene | Scene cost | Share of seed's 20-scene total |
|---|---:|---:|---:|---:|
| author_rl | 0 | 16 | 1745.711 | 21.1% |
| author_rl | 1 | 3 | 331.294 | 49.8% |
| author_rl | 2 | 10 | 4923.732 | 71.2% |
| fixed | 0 | 3 | 23870.244 | 98.5% |
| fixed | 1 | 3 | 23879.488 | 81.9% |
| fixed | 2 | 3 | 5875.504 | 52.9% |
| min_q | 0 | 14 | 48131.870 | 99.1% |
| min_q | 1 | 8 | 82.230 | 22.3% |
| min_q | 2 | 3 | 24443.736 | 82.2% |

Vehicle min-Q seed0, scene 14 used H1-3 throughout the 150-step timeout; its 48131.870 cost is almost entirely physical tracking error, with zero reported solver-failed steps. The fixed-H25 seed0 controller reached the goal on this scene at cost 19.147. This associates short H with the observed failure; different trained terminal functions and the reset warmup prevent attributing the full difference to H alone.

On the pendulum test, all nine evaluated models stop on constraint in 12 of 20 scenes and none has a goal termination. These are matched scene conditions, not 108 independent failures. Solver-failed steps also occur and are reported separately above; a passed cost audit does not establish controller reliability or the paper's adaptive-H claim.

## Audit and budget

The independent audit recomputes physical, H and constraint costs from every saved trace step, checks episode and summary totals, solver failures, goal/constraint counts, model hashes, bank hashes and frozen inference. Input-bound excess and any pendulum step-100 state violation labeled as time limit are recorded separately, not erased by a passing cost audit.

The inherited paper grid contains 26 models and 390,000 retained training steps, including 300,000 fixed-H search steps and 90,000 original RL steps. This extension adds six min-Q and four fixed-H seed-matched models, 150,000 steps total (90,000 adaptive training and 60,000 extra fixed-H training); the retained total is 540,000 steps across 36 models. New training updates: 147450. Each split evaluates all 18 models, with 10 validation and 20 test episodes per model. These counts exclude interrupted historical work and the reset warmup; H cost is the paper's computation proxy, not measured latency. The fixed-H search was more expensive than the adaptive training and must be included when comparing development budgets.

The frozen protocol and bank hashes are under `research_artifacts/bohn2021_reproduction_2026-09-17/results/min_q_training_2026-09-24/`. The per-seed and paired-scene values, failure cases and audit details are in `validation_audit.json` and `test_audit.json` there. Even a local advantage would not establish exact numerical reproduction of the original paper.
