# Gate 1: Paper-Faithful Simple Combination Audit

Date: 2026-07-18  
Branch: `codex/value-consistent-residual-aware-rl-mppi`

## Purpose

This document freezes the implementation contract for Gate 1 of the research
plan:

> Build and validate a faithful simple combination of ICODE-MPPI and
> RL-Driven MPPI before adding reliability gates or value-aligned residual
> fine-tuning.

Passing Gate 1 establishes an interpretable baseline. It does not by itself
constitute the proposed method.

## Source contracts

The implementation is grounded in the public descriptions in:

1. *Robust Path Tracking for Vehicles via Continuous-Time Residual Learning:
   An ICODE-MPPI Approach*.
2. *RL-Driven MPPI: Accelerating Online Control Laws Calculation With Offline
   Policy*.

For ICODE, the required prediction model is

$$
\dot{x}_{\mathrm{pred}}
= f_{\mathrm{nom}}(x,u)
+ f_{\mathrm{ICODE}}(x,u),
$$

with the disclosed control-affine residual

$$
f_{\mathrm{ICODE}}(x,u)
= f_\theta(x)+G_\theta(x)u.
$$

The current project uses the dynamic-unicycle state

$$
x=[p_x,p_y,\theta,v,\omega],
\qquad
u=[v_{\mathrm{cmd}},\omega_{\mathrm{cmd}}],
$$

not the paper's five-state vehicle bicycle model. This Gate reproduces the
published residual-learning and controller-integration structure on the
project's differential-drive platform. It does not claim reproduction of
undisclosed ICODE theory, stability, contraction, or convergence guarantees.

## Fidelity audit

| Requirement | Existing implementation before Gate 1 | Gate 1 action |
|---|---|---|
| Actor action is physical low-level control | Actor selected MPPI prior parameters such as control knots, a local subgoal, or covariance scale | Add an explicit `direct_control` training/inference contract for `[v_cmd, omega_cmd]` |
| Actor is trained against the true plant | Prior-policy SAC acted through an MPPI controller before MuJoCo | Train the direct-control actor by applying its bounded command through the unchanged safety arbiter to MuJoCo |
| Actor rollout initializes MPPI mean | A high-level prior decoder generated the mean | Autoregressively query the low-level actor along model-predicted states |
| Actor output initializes covariance | Covariance could be a separately decoded high-level parameter | Estimate per-step covariance from the stochastic actor distribution in physical action units |
| Guided actor samples persist across MPPI iterations | Samples were mixed from several heuristic sources at every iteration | Generate the actor-guided set once per control decision and reuse it at every MPPI iteration |
| MPPI samples around and updates its Gaussian | Present approximately in the existing RL-driven controller | Preserve and test mean and covariance updates explicitly |
| Terminal cost uses a distributional critic | Quantile critics existed, but their action was a high-level prior parameter | Evaluate terminal `Q(x_H, u_H)` with a physical actor action and an explicit reward-to-cost sign conversion |
| All candidate rollouts use one declared model | Supported by the current dynamics adapter | Test nominal and ICODE variants without allowing the planner to inspect the true plant |
| RL never directly publishes the final command | Satisfied in the MPPI prior route | Preserve `Actor -> MPPI -> scan_guard -> safety -> plant` for online control |
| Reward and MPPI cost have compatible physical semantics | Partially aligned | Record a term-by-term sign and unit audit before Gate 1 benchmark claims |

## Compatibility boundaries

- Legacy `MppiPriorEnv` and all existing high-level prior checkpoints remain
  valid. The new direct-control route is selected only by
  `rl.training.action_mode: direct_control`.
- `scan_guard`, `local_obstacle_layer`, safety arbitration, ROS bridge, and
  Memory-Augmented MPPI are not weakened or removed.
- Direct-control Actor training still uses odometry and LaserScan-derived
  observations. Simulator obstacle truth remains restricted to reward and
  evaluation metrics.
- PyTorch remains outside `mppi_hardware_bridge/scripts/`; real-robot learned
  control is not deployed in this Gate.
- Memory is disabled in the formal ICODE/RL factorial unless explicitly tested
  as a separate later ablation.

## Gate 1 acceptance tests

Gate 1 is considered integrated correctly only when all of the following hold:

1. `mppi_prior` remains the default action mode and legacy regression tests
   pass unchanged.
2. `direct_control` maps the normalized Actor action to the configured physical
   action bounds, applies rate limits, and then passes through safety
   arbitration.
3. Direct Actor training never receives privileged obstacle geometry.
4. Actor mean rollout and stochastic guided rollout are deterministic under a
   fixed random seed.
5. Guided sequences are generated once and reused during every optimization
   iteration.
6. The reported rollout count separates Actor-guided candidates from current
   Gaussian candidates.
7. Terminal critic cost is evaluated on a physical low-level action and has a
   tested reward/cost sign convention.
8. With RL guidance and terminal cost disabled, the controller reproduces the
   standard MPPI result within numerical tolerance.
9. The four paired baselines run under the same plant domains, seeds, horizon,
   sample budget, and safety configuration:
   Traditional MPPI, ICODE-MPPI, RL-Driven MPPI, and Simple Combination.
10. Only after these integration conditions pass may Gate 2 add value-aligned
    ICODE training.

## Experimental unit and run order

The independent replicate is a simulation seed cluster, not an individual
controller timestep. Scene and physics-domain conditions are repeated strata
within each seed. All four controller treatments form a randomized complete
block within each scene/seed/domain:

```text
block = scene x seed x physics domain
treatments = Traditional / ICODE / RL-Driven / Simple Combination
```

Method execution order is permuted inside every block with a recorded schedule
seed. This prevents wall-clock order, CPU temperature, or simulator warm-up
from being confounded with a method's latency. Controller outcomes use paired
comparisons within the same block. Inference resamples seed clusters so that
reusing common random numbers across scene/domain strata does not create
pseudoreplication. Timestep-level observations remain nested within an episode
and are never counted as independent replicates.

The 15k-step single-scene Actor run is explicitly a learnability screen. It may
select architecture and optimization settings, but it is not part of the
confirmatory multi-domain result.

## Explicitly out of scope for Gate 1

- Reliability-calibrated Hybrid Sampling.
- Conservative terminal fallback.
- Value-aligned ICODE fine-tuning.
- Adaptive sample count, contextual covariance, and contextual bandits.
- Online residual adaptation and real-robot learned-control deployment.

These remain useful baselines or later extensions but are not allowed to blur
the Simple Combination baseline.
