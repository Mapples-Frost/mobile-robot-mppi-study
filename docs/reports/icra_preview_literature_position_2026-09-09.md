# Preview model selection: bounded prior-art check

Accessed September9,2026. This is a targeted, non-exhaustive web-index search,
followed by primary-source inspection. It is not a systematic review or a
claim that no similar method exists. No confirmation outcomes were used here.

Queries: `model predictive control adaptive model fidelity switching curvature
path tracking computational cost`; `MPPI residual dynamics model switching multi
fidelity planning`. Search results included vehicle horizon/weight adaptation,
multi-model uncertainty adaptation and multi-fidelity motion planning.

## Closest conceptual overlap

Breelyn Melissa Kane Styler, *Using Multiple Fidelity Models in Motion Planning*,
CMU PhD thesis, April2018, CMU-RI-TR-18-15. The institutional abstract explicitly
describes choosing when and which model to use within a trajectory, reducing
planning effort while maintaining execution success, evaluated on a mobile robot
with trailer. Therefore, broad claims that selective model fidelity or invoking
complex dynamics only when needed are new are unsupported.
[Primary institutional record](https://publications.ri.cmu.edu/using-multiple-fidelity-models-in-motion-planning).

Liang, Li, Khajepour and Zheng, *Multi-model adaptive predictive control for path
following of autonomous vehicles*, DOI10.1049/iet-its.2020.0357. Publisher record
shows volume14(14),2092–2101 and first-online February19,2021; search labels2020,
so resolve bibliographic year before final BibTeX. The method uses multiple
models with an RLS adaptive law for tire cornering-stiffness uncertainty, with
path curvature included in the predictive formulation. It addresses parameter
uncertainty rather than specifically allocating learned-residual inference.
[Primary publisher text](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-its.2020.0357).

*Adaptive Dynamics Planning for Robot Navigation*, arXiv2510.05330v1, already
inspected in the earlier pivot: adaptive temporal discretization for navigation.
Keep it as adjacent prior art, not an exact reproduction or a measured baseline
without implementing its method and validating the port.
[Primary manuscript](https://arxiv.org/html/2510.05330v1).

## Implications for the current experiment

The defensible narrow hypothesis is that known-path preview can schedule an
existing learned residual within MPPI and preserve tracking accuracy at reduced
inference cost. Current implementation is a threshold heuristic, with no learned
selector, model-error guarantee, safety theorem, or new residual architecture.
Even a positive confirmation only supports this empirical hypothesis; it does
not automatically establish an ICRA-level method contribution.

Required evidence beyond pooled accuracy: demonstrate anticipatory scheduling
versus reactive turn and invocation-matched periodic controls; isolate preview
lead distance from merely increasing residual usage; measure overhead and missed
deadlines; show failure boundaries under speed/curvature/plant mismatch without
claiming transformed paths are independent environments. Measured planning cost
in fixed-step simulation is not evidence of physical delay robustness. Hardware
or a validated real-time closed-loop test remains absent.

Do not alter the active135 protocol based on this note. Finish and report all
confirmation results first. If it fails, retain the failure; if it passes, the
next bounded study should test the mechanism and strongest missing simple
control, rather than expanding a success-rate grid.
