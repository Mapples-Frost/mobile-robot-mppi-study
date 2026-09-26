# Continuous compute allocation — completion report, 2026-09-06

The categorical prototype has been replaced by **continuous Beta-PPO with integer
planner execution**. This round performed bounded engineering verification only.
There was no formal training, reward tuning, multi-seed qualification, residual
training/checkpoint modification, cost change, or learned proposal distribution.

## Architecture and changed files

The unchanged causal scene/reliability/previous-planner builder feeds a shared
64/64 Tanh actor-critic. Two independent Beta heads generate continuous a_K/a_H.
The mapper quantizes them into actual K/H; the same standard I=1 MPPI produces all
robot controls, passed through the existing guard to MuJoCo. The environment
retains the old task reward and adds actual latency cost.

Modified files:

* `compute_allocation/budget.py`: continuous bounds, mapping, raw/effective/quantized
  action contract and Fixed/Random policy interfaces; rejects obsolete discrete configs.
* `compute_allocation/ppo.py`: Beta actor, active-axis log densities/entropy, mean
  evaluation, raw-action rollout storage, v2 checkpoint schema. Existing standard
  GAE/update/normalization/resume infrastructure retained; no SAC added.
* `compute_allocation/controller.py`: timed mapping/configuration and preallocated
  sample scratch setup around the existing planner.
* `planning/mppi.py`: opt-in max-size sample-control buffer with in-place add/clip.
  RandomState noise generation and public rollout ownership remain unchanged.
* `compute_allocation/context.py`: adds scene+planner group; previous K/H scale by
  configured bounds. Actual ICODE provider/physical a/B APIs from the previous
  implementation remain in use.
* `compute_allocation/environment.py` and `logging.py`: continuous-action logging,
  separate computation/deadline costs, normalized absolute switching cost, scene
  snapshots, minimum-clearance/deadline-count/K*H/correlation summaries.
* `experiments/compute_allocation/run.py`: continuous smoke/manifest/action schema
  and scene+planner CLI option.
* `configs/compute_allocation/base.yaml`: continuous limits/deltas.
* Existing compute-allocation tests and method documentation updated. Historical
  categorical report explicitly labelled; old logs/results preserved.

New files: `profile_compute.py`, eight common-backbone ablation YAMLs
(`fixed_nominal`, `fixed_residual`, `scene_only`, `scene_dynamics`, `scene_planner`,
`full`, `adaptive_k`, `adaptive_h`), this report and short-run artifacts.
All source paths above are within `src/mobile_robot_mppi/` unless otherwise shown.

## Exact execution contract

Default K:128..1024 step32; H:10..40 step1. Round half up:
`K=128+32*floor(28*a_K+.5)`, `H=10+floor(30*a_H+.5)`.
At a=0/.5/1, K=128/576/1024 and H=10/25/40. All values are config-driven;
the policy is continuous over 29*31 execution cells, not a classifier.
Inactive axes are filled with fixed values and excluded from sampling, log-prob
and entropy. Deterministic learned evaluation uses Beta means. PPO scores stored
raw continuous actions, never rounded K/H. Old categorical checkpoints fail
explicitly rather than loading under changed semantics.

The default state is 41 channels; scene-only=17, scene+dynamics=33,
scene+planner=25. Physical a/B norms, historical-command residual norm, normalization
support and completed-observation innovation are available with explicit masks.
Single-model disagreement is unavailable; the optional online ensemble supplies
residual/a/B disagreement. Reset masks exclude unavailable history. No calibrated
probability or horizon-specific reliability is claimed. The nominal ablation can
retain a separately configured frozen context checkpoint if needed for matched
information access. No future truth/current-cycle planner diagnostics enter the actor.

Warm start is shift-before-truncate or last-control tail extension, with the
existing prior blend retained. K changes reuse a 655,360-byte sample-control buffer.
Noise and independent rollout results still allocate as before. Fixed default
(600,36) direct-versus-wrapped outputs are exactly equivalent under the test seed.
Setup allocation metadata, horizon allocation and normal sampling time are distinct;
setup is not falsely charged every step.

Reward is exactly `task_scale*r_task-beta*rho-eta*max(0,rho-1)-switch_cost`,
rho=T_plan/T_control. Defaults 1/.1/1/0; optional switching is absolute K/H changes
normalized by their configured ranges. Raw/normalized task reward, both latency
penalties and RL total are saved. T_plan includes mapping through action production,
excludes actor/context/safety/physics. Full-cycle realtime behavior is not qualified.

## Tests and smoke

**104 tests passed, no failures/skips**: 25 compute-allocation cases and 79 relevant
existing regressions, including generic/delayed/reliability MPPI, ICODE models,
task reward/direct-control execution and anytime-MPPI compatibility. Final run:
`research_artifacts/continuous_compute_2026-09-06/tests.xml`. `git diff --check` passed.

| Requirement | Verification |
|---|---|
| T1 Beta bounds | 1,000 sampled two-axis actions strictly inside (0,1), positive concentrations |
| T2 K mapping | 0/.5/1 endpoints and all 29 grid values round-trip |
| T3 H mapping | all 31 grid values and both sides of every threshold; half-up ties and delta_H=2 checked |
| T4 dynamic K | 128->1024->352->640->128, valid plans and retained sequence |
| T5 dynamic H | 10->40->17->31->10, exact shifted prefix/tail plus H=1 edge |
| T6 joint switching | six random continuous control cycles per nominal/residual mode, at least four distinct pairs |
| T7 fixed equivalence | three cycles at old (600,36), exact controls/trajectories with preallocation enabled |
| T8 ICODE on/off | same controller runs both; affine physical output equivalence retained |
| T9 causal reliability | auxiliary/future truth/current diagnostics ignored; reset masks and online ensemble checks |
| T10 reward | rho=.5/1/1.5, hinge penalty, task scaling and normalized switching checked |
| T11 PPO math | analytic Beta(2,2) density/entropy, inactive-axis zero gradients, positive/negative clipped surrogates, value loss, GAE terminal/truncation targets |
| T12 checkpoint | exact deterministic action and next seeded stochastic sample after reload; optimizer restored and mismatched contract rejected |
| T13 MuJoCo smoke | frequent random continuous switching, stable buffer identity and no returned-plan aliasing; CSV/JSON summary checks |
| T14 tiny PPO | collect8->update->save->resume4->evaluate, finite losses and unchanged evaluation normalization |

Standalone CLI checks use full default K/H bounds, seed0, the frozen existing
ICODE model, four-step episodes: `train_validated` collected8/update1;
`resume` collected4 more/total12/update2; `eval` executed4 deterministic cycles;
`ensemble_random` executed4 random-continuous cycles with online ensemble context.
Profile ran four cycles each nominal/residual, no PPO updates.

Initial `train` and `profile` attempts failed before collecting any transition
because a new file named profile.py shadowed Python's standard module during Adam
initialization. It was renamed profile_compute.py, and all CLI pipelines were
rerun successfully. Those zero-transition artifact directories are retained;
they must not be interpreted as successful episodes or quietly included in metrics.

## Short instrumented profile

Source: `profile_validated/profile.json`. Four cycles per mode, including cold
starts. These are diagnostic proportions, not steady-state benchmarks or matched
performance comparisons. Derivative times are **nested inside rollout**; actor
inference is outside T_plan. Columns are total milliseconds and percentage of
each mode's summed planner time, so rows must not be added together.

| Component | Nominal ms (%) | Residual ms (%) |
|---|---:|---:|
| Total planner | 16.844 (100) | 184.737 (100) |
| Nominal derivative | 6.207 (36.85) | 9.224 (4.99) |
| Residual inference | 0 (0) | 160.470 (86.86) |
| Sampling | 1.963 (11.65) | 2.051 (1.11) |
| Batch rollout | 8.367 (49.67) | 161.818 (87.59) |
| Cost evaluation | 0.696 (4.13) | 0.733 (0.40) |
| Weighting/update | 0.609 (3.62) | 0.672 (0.36) |
| Final rollout | 2.905 (17.25) | 17.142 (9.28) |
| Horizon resize | 0.0117 (0.070) | 0.0111 (0.006) |
| Action mapping | 0.206 (1.23) | 0.174 (0.094) |
| RL inference, outside planner | 2.980 (17.69) | 1.930 (1.04) |

The safe in-place sample-buffer change is fixed-equivalence tested, but no speedup
claim is made. ICODE is already batched across K; dependent H/RK4 propagation
remains the bottleneck. No residual core rewrite was performed.

## Commands, status and next screen

Commands from WSL project root:

```bash
.venv/bin/python experiments/compute_allocation/run.py train --steps 8 --max-episode-steps 4 --seed 0 --output results/continuous/train
.venv/bin/python experiments/compute_allocation/run.py evaluate --policy learned --checkpoint results/continuous/train/checkpoints/latest.pt --max-episode-steps 4 --seed 0 --output results/continuous/eval
.venv/bin/python experiments/compute_allocation/run.py train --config configs/compute_allocation/scene_planner.yaml --steps 8 --max-episode-steps 4 --seed 0 --output results/continuous/scene_planner
```

No known unresolved failing contract in the tested CPU path. Unqualified areas:
long-run learning, GPU/hardware, calibrated reliability, partial-FOV lidar,
full-cycle realtime effects and bitwise simulator resume. Resume intentionally
starts a fresh episode with restored learning/RNG state. The scientific reliability
to foresight hypothesis remains unresolved, and pilot Complex success remains zero.

Branch: codex/post-l233-five-direction-experiments; base HEAD f0e5c89. The workspace
contains pre-existing unrelated dirty files, so the user's clean-workspace condition
for a new commit is not met. This implementation is left uncommitted; nothing was
pushed and unrelated changes were preserved.

Next proposed screen, **not executed**: one fixed development setup, seed0,
2,048 transitions, unchanged beta/eta and bounds. Inspect concentrations/entropy,
continuous versus quantized actions, availability and actual latency before
matched fixed-budget comparisons. Separate reliability manipulation and later
held-out qualification are still needed; no outcome-driven reward/scene adjustment.
