# Bohn 2021 terminal-value transplant probe

This is a mechanism/method extension, not a reproduction of the author's
training method. It reuses the three-seed min-Q and selected fixed-H models
without new training. At inference, the intervention replaces each min-Q
model's terminal polynomial with the matched-seed fixed-H model's polynomial;
the min-Q actor and its H choices stay unchanged. Fixed H=25 (vehicle) and
H=30 (pendulum) were selected on an older validation bank, before this probe.

## Registered result

The new validation bank contains ten scenes per task. All 18 model conditions
(180 model-episodes) completed. The independent audit recomputed each saved
step's physical, H-proxy and constraint costs, checked terminal labels, goal
arrival, solver failures and model/bank hashes, and passed. It also replayed
60 matched reset warmups and verified identical post-warmup states for the
transplant and fixed-H arms. No test evaluation was run.

| Task | Seed | min-Q own value | min-Q fixed value | fixed H | Transplant vs own |
|---|---:|---:|---:|---:|---:|
| vehicle | 0 | 45.930 | 34.490 | 17.350 | -24.9% |
| vehicle | 1 | 22.080 | 55.170 | 17.810 | +149.9% |
| vehicle | 2 | 14.800 | 18.000 | 17.700 | +21.6% |
| pendulum | 0 | 429.490 | 429.560 | 428.300 | +0.02% |
| pendulum | 1 | 452.910 | 451.050 | 428.310 | -0.41% |
| pendulum | 2 | 522.950 | 522.080 | 428.300 | -0.17% |

Values are mean total cost over the same ten scenes, lower is better;
displayed values are rounded. The registered gate required at least a 2%
reduction in two of three seeds *on each task*, no additional constraint stops
or solver-failed steps by seed, and a transplant mean below fixed H on each
task. Vehicle passed the 2% criterion in only one seed, pendulum in none;
neither task's transplant mean beat fixed H. Constraint stops and solver-failed
steps did not increase. The gate therefore failed and the independent test bank
remains unopened for evaluation. This probe gives no support for terminal-value
transplant as a robust remedy for the adaptive-H gap.

All vehicle validation episodes reached the goal without solver failures. The
pendulum bank produced three constraint stops and 30 solver-failed steps per
seed in each arm, so the small pendulum differences do not establish robust
control. The H cost is the paper's linear proxy, not a measured timing gain.

## Audit correction and limits

The first audit attempt incorrectly compared `trace_00[0].state` between the
transplant and fixed-H arms. Traces record state *after* the first policy
action, when their H choices may differ; that assertion failed. The audit was
corrected to replay `reset()` with the same frozen case and donor terminal
weights and compare the state *before* the policy action. No trajectory or
registered gate was changed. The original 18 evaluation outputs were retained.

Both arms use the same donor terminal value for reset warmup, but subsequent
actions and trajectories differ. The transplant-minus-own contrast changes
terminal value during both warmup and the episode, so this experiment cannot
isolate which part caused an outcome. The fixed-H arm also changes H policy;
its comparison is a performance benchmark, not a terminal-value causal effect.
The original paper's exact experiment configuration and test files remain
unavailable, and this negative extension result does not refute its claim.

The inherited fixed-H grid trained ten H values at seed 0 per task (300,000
steps); selected H seeds 1/2 added 60,000. The three-seed min-Q models add
90,000 steps. This probe adds zero training steps and 180 validation
model-episodes. Prior interrupted work and other reconstruction experiments
remain part of the broader project budget; they are not erased here.

Frozen protocol, scenario banks, hashes, trajectories, validation audit and
gate are under
`research_artifacts/bohn2021_reproduction_2026-09-17/results/terminal_transplant_2026-09-24/`.
