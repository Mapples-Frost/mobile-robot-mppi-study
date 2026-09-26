# L38 ICODE command-delay identifiability diagnostic — preregistration

## Question

L37 showed that ICODE lowers offline state-prediction RMSE but worsens closed-loop MPPI. The current five-state residual receives the current command, while the MuJoCo plant executes delayed commands. L38 tests whether the increase from the dominant 40 ms training delay to the 100 ms L37 delay explains the closed-loop reversal.

This is a mechanism diagnostic, not an efficacy confirmation and not permission to tune on sealed seeds.

## Frozen design

- Policy: traditional GoalWarmStart MPPI only; RL and memory disabled.
- Residual factor: nominal versus each paired ICODE checkpoint.
- Delay factor: 40 ms versus 100 ms.
- All other MuJoCo mass, inertia, actuator and contact parameters are identical.
- Blocks: three independently initialized ICODE checkpoints.
- Scenes: the L36 easy, moderate and hard dynamic scenes, frozen before inspecting learned-method outcomes.
- Episode seeds: five new development seeds, repeated across every condition.
- Experimental unit: model-block × scene × episode-seed. Control steps are nested technical measurements.
- Run order: seeded shuffle within model block.
- Safety: identical robust temporal scan and scan-guard chain in every condition.

The 180 planned episodes are

\[
3\ \text{model blocks}\times3\ \text{scenes}\times5\ \text{seeds}
\times2\ \text{delays}\times2\ \text{residual levels}.
\]

## Primary estimands

Within each delay stratum, paired ICODE-minus-nominal effects are computed for success, collision and final goal distance. The delay interaction is

\[
\Delta_{\mathrm{delay}}
=
(Y_{\mathrm{ICODE}}-Y_{\mathrm{nominal}})_{100\,\mathrm{ms}}
-
(Y_{\mathrm{ICODE}}-Y_{\mathrm{nominal}})_{40\,\mathrm{ms}}.
\]

Uncertainty uses a hierarchical bootstrap that resamples model blocks and then experimental units within block. Binary gain/loss counts remain visible.

## Frozen interpretation gate

“Delay plausibly explains the L37 reversal” requires all of:

1. complete artifacts and no protected/sealed seed use;
2. at 40 ms, ICODE has nonnegative net success, no collision increase and nonnegative mean final-distance improvement;
3. the 100 ms minus 40 ms success interaction is at least -0.05 or the corresponding final-distance interaction is no greater than -0.05 m, demonstrating a material delay-dependent loss.

If ICODE is already adverse at 40 ms, delay mismatch is not a sufficient explanation and the next diagnostic must examine H=36 rollout and MPPI cost-ranking preservation. No threshold will be changed after observing results.

## Seed firewall

Development seeds are `20760731`–`20760735`. Seeds `20760736`–`20760745` remain sealed. L34–L37 seeds are protected and rejected by the runner.
