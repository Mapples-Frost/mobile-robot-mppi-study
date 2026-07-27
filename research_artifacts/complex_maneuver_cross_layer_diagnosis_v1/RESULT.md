# Complex Maneuver Cross-Layer Diagnosis v1

## Status

Complete. This diagnosis uses only already-opened development evidence. It does not authorize Actor retraining, threshold changes, seed replacement, or formal execution.

## v3 expansion failure

The pre-oracle failure is an encounter-coverage failure in the fixed 120-step source probes, not a systematic forecasting failure.

- All six sources established forecasts during most of the episode (73.3%–93.3% of steps).
- Chapter 2 seeds `790202031` and `790202037` never entered hard risk and remained at least 2.6495 m and 2.3180 m from a dynamic obstacle, respectively.
- Chapter 1 seed `791101031` also established forecasts but did not create a qualifying hard-risk interaction.
- Only three training sources and zero validation sources had valid anchors, below the frozen 4/2 scenario requirements.

The failed Gate remains binding. Seeds are not replaced, the anchor threshold is not lowered, and no training is started.

## Bootstrap Gate D failure

The bootstrap Actor repaired candidate coverage but did not translate that coverage into successful execution.

- 4,971 supervised proposals were generated.
- 1,016 entered the elite set (20.44%).
- Only 16 were selected (0.322%).
- Success gain was zero on all three maps.
- Safety intervention pressure was 49.3% on Chapter 1, 29.6% on Chapter 2, and 35.1% on Chapter 3.

This localizes the current closed-loop bottleneck after proposal generation: MPPI arbitration and downstream execution do not reliably preserve useful maneuver influence.

## Next bounded mechanism

The next mechanism family is `proposal_to_execution_consistency`. It may inspect and minimally adjust only shared proposal weighting/selection behavior while keeping the Actor checkpoint, Risk, ICODE, Safety, maps, and the total 600-rollout budget frozen.

The hypothesis is falsifiable: selection or measurable influence of qualified supervised maneuvers must increase on all three opened maps, without a collision regression, and terminal goal distance or success must improve on all three maps. No map-specific logic is allowed.

Stop this family after three rounds without core improvement or two consecutive safety regressions.
