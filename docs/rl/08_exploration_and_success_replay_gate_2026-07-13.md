# RL Gate L9--L12: exploration, forgetting, and success-aware replay

Date: 2026-07-13

Status updated 2026-07-14: **the single-training-seed result was invalidated by
the required multi-training-seed replication**.  Safety remained intact, but
method efficacy did not pass.  See
[`09_multitraining_seed_replication_2026-07-14.md`](09_multitraining_seed_replication_2026-07-14.md).

> Historical-result notice: the 10/10 result below is a correct record for
> training seed 20260718, not a fabricated or corrupted run.  It must no longer
> be presented as the expected performance of L12 because seeds 20260719 and
> 20260720 failed to reproduce it.

This note records executed experiments rather than projected paper results.
All reported values below come from the listed result directories.  ICODE and
Memory were disabled throughout this gate so the sampling-prior contribution
is identifiable.

## 1. Question and protected control stack

The question was whether SAC can provide a useful MPPI sampling prior in the
deceptive `u_trap_long_board` scene without replacing MPPI or bypassing safety.

```text
Odometry + LaserScan + goal
             |
             v
      SAC local-subgoal prior
             |
             v
  MPPI sampled control sequences
             |
             v
 LaserScan -> scan_guard -> safety arbitration
             |
             v
       MuJoCo true plant
```

RL never emits final wheel/body commands.  The final command is still selected
by MPPI and then filtered by the unchanged safety arbiter.  Planner obstacles
still come from `LaserScan -> local_obstacle_layer`; no simulator obstacle
truth is supplied to the policy or planner.

Fixed controlled variables in the main ablation:

- MuJoCo differential-drive plant and contact parameters;
- nominal MPPI prediction, horizon 36 and `K=100` samples;
- kinematic local-subgoal decoder;
- ground-truth pose/twist sensor mode, used as a mocap-like controlled
  localization condition;
- near-goal handoff: full conventional MPPI at 0.35 m and full RL influence
  beyond 0.80 m;
- no ICODE residual, no Memory-Augmented MPPI, no dynamic obstacle;
- the same SAC architecture, seed, reverse-reset curriculum and reward;
- zero collision was required; success alone was not sufficient.

## 2. L10: scene balance did not remove interference

L9 balanced every update batch by scene.  L10 additionally removed abrupt
scene-curriculum changes, used a stationary multi-start distribution and
lowered the learning rates.  Every L10 update batch was exactly 50%/50% by
scene, so replay imbalance was not a possible explanation.

Executed run:

```text
results/research_platform/rl/l10_stationary_40k_20260713
```

Training validation:

| step | clean success | U-trap success | clean final distance | U-trap final distance |
|---:|---:|---:|---:|---:|
| 5k | 5/5 | 0/5 | 0.286 m | 2.254 m |
| 10k | 2/5 | 0/5 | 0.893 m | 2.432 m |
| 15k | 4/5 | 2/5 | 0.447 m | 0.704 m |
| 20k | 5/5 | 1/5 | 0.292 m | 0.999 m |
| 25k | 5/5 | 0/5 | 0.286 m | 1.592 m |
| 40k | 5/5 | 0/5 | 0.293 m | 1.773 m |

At approximately 25k steps, successful training episodes began to come almost
entirely from the clean scene.  High-TD-error spikes then increased and the
policy became a clean-scene specialist.  Scene-balanced replay therefore did
not balance the rare outcome distribution.

The 15k best checkpoint was tested with held-out seeds 81--85:

- clean: 3/5 without near-goal handoff and 5/5 with the handoff;
- U-trap: 0/5 in both cases;
- collision: 0/10 in each paired condition.

This rejected the hypothesis that a single stationary multi-scene SAC policy
was already sufficient.

## 3. L11: uninformed count novelty was rejected

`src/mobile_robot_mppi/rl/intrinsic.py` implements a default-off episodic pose
count bonus.  It uses odometry pose bins only, resets at every episode, and
never reads obstacle truth or a collision-free route.  Validation forcibly
disables the bonus, so checkpoint selection cannot be inflated by intrinsic
reward.

Paired 20k runs used identical configurations and seeds:

```text
results/research_platform/rl/l11_intrinsic_ablation_20k_20260713
results/research_platform/rl/l11_no_intrinsic_ablation_20k_20260713
```

The count bonus produced more novelty reward but worse navigation:

| method | best training validation | held-out seeds 91--100 | mean held-out final distance |
|---|---:|---:|---:|
| pose-count intrinsic | 0/5 | 0/10 | 3.311 m |
| no intrinsic | 2/5 at 10k | 2/10 | 1.357 m |

Interpretation: visiting a new `(x, y, heading)` bin is not the same as making
a useful detour.  The bonus can reward turning or locally novel wandering.
This module is retained as a negative ablation and reusable interface, but it
is disabled in the selected method.

## 4. L12: outcome-balanced replay

The L11 control occasionally discovered successful trajectories and then lost
them.  L12 therefore labels replay transitions after the episode outcome is
known and samples a configured fraction from self-generated successful
episodes.

Implementation:

- `ReplayBuffer.add()` returns overwrite-safe `(slot, transition_id)` handles;
- `mark_episode_outcome()` labels a completed episode without relabelling a
  ring-buffer slot that has since been overwritten;
- `outcome_balanced` replay uses ordinary uniform replay before the first
  success, then uses `replay_success_fraction` (0.25 in L12);
- old replay checkpoints without outcome labels load as non-success samples;
- batch success and labelled fractions are written to `updates.csv`;
- outcome counts are written to `training_summary.json`;
- the feature is configuration-controlled and default behavior remains
  `uniform`.

Configuration:

```text
configs/rl/sac_mppi_utrap_success_replay_l12.yaml
```

The first 20k covers terminal bootstrap and route expansion; both L11 and L12
were then resumed from their own checkpoints for the 20k--40k original-start
phase.  The split preserves all intermediate evidence and replay state.

Training validation during the original-start phase:

| step | outcome-balanced success | uniform success | outcome final distance | uniform final distance |
|---:|---:|---:|---:|---:|
| 25k | 0/5 | 0/5 | 1.050 m | 1.838 m |
| 30k | 0/5 | 0/5 | 1.077 m | 1.180 m |
| 35k | 3/5 | 2/5 | 0.329 m | 0.543 m |
| 40k | 5/5 | 0/5 | 0.296 m | 0.695 m |

Every L12 update after successful experience became available contained
exactly 25% transitions from successful episodes.  No additional MPPI samples
or policy parameters were introduced.

## 5. Held-out result

Fresh evaluation seeds 101--110 used each method's independently selected best
checkpoint:

```text
results/research_platform/rl/l12_success_replay_best_heldout_seeds101_110_20260713
results/research_platform/rl/l11_uniform_best_heldout_seeds101_110_20260713
```

| metric | L12 outcome-balanced | uniform replay |
|---|---:|---:|
| success | **10/10** | 5/10 |
| collision | 0/10 | 0/10 |
| final goal distance | **0.291 m** | 0.515 m |
| minimum clearance | 0.214 m | 0.224 m |
| safety interventions | **16.2** | 42.0 |
| control jerk | 0.176 | 0.176 |
| mean planner compute | 5.33 ms | 5.31 ms |
| trajectory length | 5.94 m | 6.04 m |

The compute-time and jerk differences are negligible in this small sample.
The supported claim is therefore narrower: under the controlled localization
condition, outcome-balanced replay substantially improved success retention
for this U-trap specialist without increasing collisions or planner sample
count.

This is not yet a paper-level aggregate because it uses one training seed and
one complex static scene.

## 6. Localization boundary

The selected L12 checkpoint was evaluated with the same seeds and settings but
wheel odometry instead of ground-truth localization:

```text
results/research_platform/rl/l12_success_replay_best_wheel_odom_seeds101_110_20260713
```

Result:

- success: 0/10;
- collision: 0/10;
- mean final distance: 2.179 m;
- mean safety interventions: 157.3.

This fails the localization robustness gate.  It agrees with the earlier L8
audit that wheel slip can make the policy and planner believe the robot is in
a different place from the true MuJoCo body.  Ground-truth pose is not obstacle
truth, but it is still a controlled mocap-like assumption.  The 10/10 result
must not be described as wheel-odometry or real-robot robustness.

## 7. What is and is not established

Established in this gate:

1. simple episodic count novelty is not a useful exploration objective here;
2. rare successful trajectories were genuinely forgotten by uniform replay;
3. the selected outcome-balanced replay improved one-seed U-trap held-out
   success from 5/10 to 10/10 under controlled localization;
4. all compared runs retained zero measured collisions and the full safety
   chain;
5. the selected policy fails under the current wheel-odometry drift.

Not established:

- statistical significance over independent training seeds;
- generalization to other difficult static or dynamic-obstacle scenes;
- robustness to physics-domain changes, sensing dropout or real odometry;
- superiority over a tuned classical/global planning system;
- benefit from ICODE, Memory or a learned uncertainty gate;
- any stability, convergence or safety theorem.

## 8. Reproduction commands

Smoke:

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_success_replay_l12.yaml \
  --output-dir results/research_platform/rl/l12_smoke --smoke
```

Full training can be executed in one 40k run:

```bash
.venv/bin/python experiments/rl/train_rl_sampling_prior.py \
  --config configs/rl/sac_mppi_utrap_success_replay_l12.yaml \
  --output-dir results/research_platform/rl/l12_full_40k
```

Held-out U-trap evaluation:

```bash
.venv/bin/python experiments/rl/evaluate_rl_sampling_prior.py \
  --config configs/research/mujoco_u_trap_long_board.yaml \
  --checkpoint results/research_platform/rl/l12_full_40k/checkpoints/best.pt \
  --output-dir results/research_platform/rl/l12_eval \
  --seeds 101,102,103,104,105,106,107,108,109,110 \
  --gate-mode none --near-goal-fallback \
  --near-goal-full-fallback-distance 0.35 \
  --near-goal-full-rl-distance 0.80 \
  --num-samples 100 --pose-source ground_truth --twist-source ground_truth
```

## 9. Next gate

Before integrating ICODE with this RL result:

1. run at least three independent L12 training seeds, then expand to five;
2. add at least two new difficult static scenes and a held-out layout family;
3. compare against conventional MPPI and a tuned non-learning planning prior;
4. design a task-relevant exploration/competence signal rather than raw pose
   novelty;
5. separate the **need for exploration** from **trust in a learned policy** in
   the gate instead of forcing one uncertainty scalar to represent both;
6. treat localization robustness as a separate experiment axis;
7. only after those gates pass, test nominal versus ICODE rollout dynamics with
   the RL prior fixed.

## 10. Verification executed

```bash
find src experiments tests -type f -name '*.py' -print0 \
  | xargs -0 .venv/bin/python -m py_compile
git diff --check
.venv/bin/python -m pytest -q
```

Result: all Python files compiled, `git diff --check` passed, and **187 tests
passed**.  Invoking `.venv/bin/pytest` directly is not the repository test
entry point because it omits the repository root needed by the legacy
`src.*`/`experiments.*` imports; use `python -m pytest` as shown above.
