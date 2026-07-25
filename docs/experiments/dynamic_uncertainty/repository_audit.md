# Dynamic-Uncertainty Repository Audit

Audit date: 2026-07-23  
Execution platform: native Windows only  
Repository: `D:\Projects\mobile-robot-mppi-study`

## Git snapshot before this program

- Source branch: `codex/post-l233-five-direction-experiments`
- Source HEAD: `73f48724b3dd3ee2f7e0db66adbafa4bee9c8fca`
- Source branch state: clean working tree, one local commit ahead of its remote
- New working branch: `codex/change-aware-probabilistic-mppi`
- Destructive Git operations: none
- Commit, stage, and push: not performed

The WSL copy at `/home/mapples/projects/mobile-robot-mppi-study` is stale and is
not an execution source. All code, tests, generated data, and simulation work
for this program must use the Windows repository above.

## Latest experiment state

The latest registered experiment number is L285. L280--L284 form a complete
Tracking diagnostic chain. L285 was amended before outcome inspection from a
45-episode confirmation into a 15-episode, single-seed direction screen. The
current `progress.csv` contains 9/15 rows and no matching active Python process
was found during this audit. It is therefore incomplete and cannot support a
cross-seed or confirmatory claim.

Relevant evidence:

- `docs/experiments/post_l217/l276_gate_status.md`
- `docs/experiments/post_l217/l277_gate_status.md`
- `docs/experiments/post_l217/l278_gate_status.md`
- `docs/experiments/post_l217/l279_gate_status.md`
- `docs/experiments/post_l217/l280_gate_status.md`
- `docs/experiments/post_l217/l281_gate_status.md`
- `docs/experiments/post_l217/l282_gate_status.md`
- `docs/experiments/post_l217/l283_gate_status.md`
- `docs/experiments/post_l217/l284_gate_status.md`
- `docs/experiments/post_l217/l285_frozen_actor_proposal_gate_protocol.md`
- `results/research_platform/rl/l285_frozen_actor_proposal_gate/`

## Reusable interfaces

| Role | Stable interface | Intended later reuse |
|---|---|---|
| Actor | `PaperDirectControlPolicy.propose` in `src/mobile_robot_mppi/rl/paper_policy.py` | Frozen action-sequence proposals |
| Actor adapter | `ExternalActionPrior.propose` and `TorchSACPrior.propose` in `src/mobile_robot_mppi/rl/prior.py` | Convert a frozen Actor into MPPI prior/proposal data |
| ICODE | `CombinedDynamics.derivative` in `src/dynamics/combined_dynamics.py` and `ICODEResidual` in `src/dynamics/residual/icode_residual.py` | Robot rollout prediction with additive residual dynamics |
| HSS | `HybridSamplingReliability.evaluate` in `src/mobile_robot_mppi/rl/reliability.py` | Allocate fixed candidate budget by dynamics and source reliability |
| MPPI | `MppiController.plan`, `rollout`, and `evaluate_control_sequences` in `src/mobile_robot_mppi/planning/mppi.py` | Candidate generation, rollout, and unified cost evaluation |
| RL/ICODE MPPI | `PaperRLDrivenMppiController` in `src/mobile_robot_mppi/planning/rl_driven_mppi.py` | Later Stage E integration point |
| Safety | `ScanGuardArbiter.arbitrate` in `src/mobile_robot_mppi/safety/arbiter.py` and `MppiController.observe_safety_decision` | Final control arbitration; cannot be bypassed by probability risk |
| Artifacts | `ArtifactWriter` in `src/mobile_robot_mppi/evaluation/artifacts.py` | Existing resolved-config/provenance conventions |
| Plant | `MujocoDiffDrivePlant` in `src/mobile_robot_mppi/simulation/mujoco_plant.py` | Later dynamic-obstacle closed-loop scene integration |

The new obstacle predictor will remain independent of HSS in the first
integration. It will produce candidate-level risk inputs for MPPI; it will not
change Actor proposal authority.

## Test baseline

Windows environment: `.venv-cuda\Scripts\python.exe`, Python 3.10.11.

The first collection attempt failed because the environment lacked the
repository-declared `matplotlib==3.7.5`. After installing that dependency into
the Windows virtual environment, the baseline was:

```text
977 passed, 5 failed, 11 warnings
```

All five failures require historical L34/L70/L72 checkpoints absent from this
Windows checkout:

- `tests/rl/test_checkpoint_rule_validation.py` (1)
- `tests/rl/test_l34_independent_confirmation.py` (1)
- `tests/rl/test_static_multigeometry_training.py` (3)

These are artifact-availability failures, not failures introduced by the
dynamic-obstacle program. They are retained and will not be weakened.

## Compatibility and protected boundaries

- Do not modify the ROS/Python-2 hardware bridge for this probability sandbox.
- Do not bypass LaserScan, `scan_guard`, the local obstacle layer, or safety
  arbitration in later closed-loop work.
- Development may use MuJoCo obstacle truth plus synthetic observation noise
  only to isolate predictor behavior. Truth fields are offline labels and must
  never enter a predictor update.
- Do not read or use `configs/seeds/sealed_dynamic_seeds.yaml` in smoke,
  tuning, or qualification code.
- Do not activate RL, ICODE, HSS, or closed-loop MPPI in this first stage.
- Preserve all Tracking code, raw results, incomplete L285 artifacts, and
  negative Gate reports.

