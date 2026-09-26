> Historical 2026-09-05 categorical prototype report; superseded by the [2026-09-06 continuous implementation](model_reliability_aware_continuous_implementation.md). Original results below are preserved.

# Compute-allocation implementation report — 2026-09-05

Implementation and bounded verification are complete. No formal RL run, reward
tuning, multi-seed qualification, ICODE training/checkpoint modification, or
external-baseline experiment was performed. The method remains an unqualified
research candidate. Full definitions and pseudocode are in the
[method document](model_reliability_aware_compute_allocation.md).

## Architecture and files

| File | Responsibility |
|---|---|
| `src/mobile_robot_mppi/compute_allocation/budget.py` | Stable joint action mapping, dimension masks, BudgetPolicy/Fixed/Random contracts |
| `.../context.py` | Sensor-only scene builder, dynamics availability masks, causal online ModelReliabilityProvider |
| `.../controller.py` | Budget configuration and actual planning timer around standard MPPI |
| `.../environment.py` | ComputeAllocationEnv, unchanged task execution/reward reuse, latency reward |
| `.../ppo.py` | LearnedBudgetPolicy, shared MLP categorical actor-critic, rollout buffer, GAE/PPO, normalization, atomic checkpoints/resume/evaluation |
| `.../logging.py` | Streaming cycle CSV/JSON, episode summaries and safe missing-value serialization |
| `.../__init__.py` | Public lightweight package interfaces |
| `configs/compute_allocation/base.yaml` | Full 16-action default and explicit observation/reward/PPO settings |
| `configs/compute_allocation/online_ensemble.yaml` | Optional causal online three-member diagnostics using frozen existing checkpoints |
| `experiments/compute_allocation/run.py` | Bounded train/evaluate/smoke CLI, config overrides and provenance |
| `tests/compute_allocation/test_budget_stack.py` | Mapping, warm start, fixed equivalence, real ICODE/MuJoCo, causality and reward tests |
| `tests/compute_allocation/test_ppo.py` | GAE boundaries, normalization, short PPO/checkpoint/resume/evaluation tests |

Minimal modifications to existing code:

* `planning/mppi.py`: public `configure_budget`, shape-aware shift-then-truncate/hold
  warm start and measured resize allocation branch. Old equal-H behavior retained.
* `learning/models.py`: physical masked `affine_components` on actual ICODE and
  member/mean APIs on its ensemble; derivative/forward algorithms unchanged.
* `rl/environment.py`: optional `external_prior_enabled=False` initialization
  seam. It builds the ordinary MPPI and shares task execution/reward without a
  legacy proposal actor. Default prior-enabled construction stays unchanged.

No dependency or pyproject changes were required. Existing unrelated dirty files,
pilot source/artifacts, visual modules and old experiment logs were left as found.

## State/action/reward and reliability

Default state: **41 dimensions = scene 17 + dynamics 8 + availability 8 + previous
planner 8**. Scene-only is 17; scene+dynamics is 33. Only previous-cycle planner
K/H/rho/ESS and guard state enter the actor. No raw cross-H cost is consumed.

Action: K-major categorical mapping of config K/H candidates; 16 defaults. One
mask supports fixed, adaptive-K, adaptive-H, joint modes. Fixed `(600,36)` can be
selected by a config with those candidate values, reproducing old defaults.

Reward: `task_scale*r_task - beta*rho - eta*max(0,rho-1) - switch_cost`,
rho=`actual_plan_seconds/control_seconds`; defaults 1/.1/1/0. Task reward is the
existing RewardConfig implementation, including its discounted-potential shaping.
Full coefficient values and normalization are saved; no tuning occurred.

Default single-model input actually provides physical masked drift norm,
control-map norm, historical-command residual norm, explicit normalization-based
support, and completed-observation innovation EMA. The three disagreement
channels are unavailable (zero + false). Optional online ensemble enables
residual/a/B disagreement and ensemble support. At reset, historical-command
features and innovation remain unavailable. No calibrated model-reliability
probability or horizon-specific reliability is claimed. Nominal mode can disable
all dynamics context or explicitly retain a frozen context checkpoint independently
of rollout correction.

## Verification

Final targeted test command:

```bash
.venv/bin/python -m pytest tests/compute_allocation \
  tests/platform/test_generic_mppi.py \
  tests/platform/test_mppi_command_delay.py \
  tests/platform/test_mppi_residual_reliability.py \
  tests/learning/test_icode_shapes.py \
  tests/learning/test_structured_residual_network.py \
  tests/learning/test_residual_ensemble.py \
  tests/rl/test_residual_context.py tests/rl/test_reward.py \
  tests/rl/test_direct_control_environment.py \
  tests/learning/test_residual_component_mask.py \
  tests/learning/test_residual_state_canonicalization.py \
  tests/learning/test_residual_support_gate.py \
  tests/learning/test_residual_reliability_gate.py -q \
  --junitxml=research_artifacts/compute_allocation_implementation_2026-09-05/tests.xml
```

**93 passed in 2.95 s**: 18 new compute-allocation cases and 75 relevant existing
regressions; no skips/failures. This is a targeted suite, not a whole-repository
audit or exhaustive dynamics/backend qualification. `compileall` and targeted
`git diff --check` also passed.

| Requested test | Result |
|---|---|
| T1 action mapping | All 16 unique pairs round-trip, all four action masks checked, invalid actions rejected |
| T2 dynamic H | 10 -> 20 -> 40 -> 10; correct shifted prefix and last-control tail; H=1 edge case also checked |
| T3 dynamic K | 128 -> 1024 -> 256; nominal sequence object retained across configuration; valid plans/ESS |
| T4 fixed equivalence | `(600,36)` controls and predicted trajectories exactly equal between direct and wrapped planner for three cycles |
| T5 ICODE on/off | Same wrapper/controller for nominal and actual frozen residual; physical affine decomposition equivalent under multiple controls/batched states |
| T6 causality | No auxiliary/future-truth/local-geometry leakage; current diagnostics ignored; truth replacement cannot change budget input; availability/reset checks; online ensemble does not replace active model |
| T7 reward | rho=.5/1/1.5, deadline hinge, fixed task scaling and optional K/H switch cost match exact formula |
| T8 MuJoCo | Four steps each nominal/residual with `(128,10)->(1024,20)->(256,40)->(128,10)`; shapes, previous diagnostics, termination and CSV/JSON verified |
| T9 PPO | 8 nominal steps, 2 small updates, checkpoint/reload with identical deterministic probabilities and RNG sample, 4 resumed steps, 3-step evaluation; GAE truncation bootstrap tested separately |

Additional command-line end-to-end smoke records use the full default candidate
set and the existing ICODE checkpoint, not a toy model:

| Artifact subdirectory | Work performed |
|---|---|
| `nominal` | Four real MuJoCo cycles, rotating candidate extremes, no RL update |
| `residual` | Four real MuJoCo cycles, same schedule, frozen single ICODE |
| `ensemble_smoke` | Four cycles, active single ICODE with online shadow ensemble context |
| `ppo_smoke` | Eight residual-MPPI transitions, one PPO update |
| `resume_smoke` | Restored learning state, four additional transitions, total_steps=12/update=2 |
| `evaluation_smoke` | Loaded learned policy, four deterministic argmax control cycles |

Artifacts: [verification directory](../../research_artifacts/compute_allocation_implementation_2026-09-05/),
[JUnit XML](../../research_artifacts/compute_allocation_implementation_2026-09-05/tests.xml),
[machine-readable summary](../../research_artifacts/compute_allocation_implementation_2026-09-05/verification.json).
Binary PPO smoke checkpoints remain local in `ppo_smoke/checkpoints/latest.pt` and
`resume_smoke/checkpoints/latest.pt`; they are excluded from the source commit.
The CLI smoke actor losses/returns establish finite computation only, not progress
towards a good allocation policy.

## Performance observations and unresolved limitations

The four-cycle standalone profile includes cold starts and uses different K*H;
it is **not a performance benchmark**. Raw timing is retained without filtering.
For the residual run, the `(1024,40)` cycle took **142.49 ms**, with **129.78 ms**
in batch rollout and **8.43 ms** in final rollout. The nominal corresponding
single cycle took **17.39 ms**. Default single-model context construction took
approximately 0.49-1.21 ms in this short run; online ensemble context ranged
approximately 1.14-7.54 ms, including first-use overhead. There are too few samples
to estimate steady-state tails or compare methods.

Observed horizon-resize allocation/copy/fill was approximately **0.0035-0.0044 ms**;
configuration/validation approximately **0.026-0.041 ms**. Ordinary sampling and
rollout allocations remain within the existing stage profiles, not isolated as
incremental K-allocation cost. The shared planner is never reconstructed per step.

K is already batched in residual inference. The main remaining cost is dependent
RK4/H model propagation plus final rollout; flattening these into one batch would
change semantics. No core acceleration or speedup claim was made. Candidate
pruning requires later platform measurements, not this smoke's maxima.

No known failing contract or unresolved bug was observed in the tested CPU path.
GPU, hardware execution, partial-FOV lidar, other residual wrappers' affine APIs,
long-run numerical stability and learning quality have not been qualified.
Checkpoint resume starts a new episode instead of restoring MuJoCo hidden state.
Reward charges planning latency only; context/actor/safety/logging add cycle cost.
The simulation does not model extra actuation delay from a slow host cycle.

Scientifically, support remains a heuristic, ensemble disagreement uncalibrated,
and ICODE reliability -> H value unresolved. The pilot showed substantially better
Complex goal progress for balanced allocation but **no Complex success**; Easy
arms were essentially equivalent. No result is hard-coded into K/H selection.

## Next step (proposed, not executed)

Use the method document's single 2,048-transition development command after
freezing the platform/config/scales/reward. Inspect losses, entropy, masks, action
counts and actual latency first. A matched fixed-policy comparison can then check
the trade-off; establishing a model-reliability contrast is a separate requirement
before claiming the proposed mechanism. Formal training and qualification remain
outside this implementation round.
