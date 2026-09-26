# V2 development-to-landscape freeze

This is an explicit pre-grid development decision within the new v2 scientific
question. It changes no v1 result, no Phase A–D threshold, and no recorded outcome.

The initial v2 document proposed H={8,12,20,40} via evenly spaced candidate indices.
Phase A instead identified pooled best H=16 and robust Medium-high success at
H16/H20, with failure at both H8/H40. Before generating any joint-grid outcome,
replace the index rule with **H={8,16,20,40}**: keep the two successful middle
points and both failure endpoints. K remains {16,64,128,256}. This choice is based
on development evidence as requested, not on final-grid outcomes. H12 is excluded;
no claim covers all integer horizons or the full continuous action space.

Use fresh paired final-grid seeds **7092861–7092864**, distinct from development
7092801–7092804. Each treatment receives all four seeds. Same benchmark geometry,
model, controls, sensors, real physical delay, costs and lambda=.1 secondary price.
Do not inspect incremental grid outcomes to retune anything.

The executable freeze_landscape.py requires completed A/D, computes development
noise from A/D only, and records a JSON freeze before phase_e exists:

- Noise = median of per-cell sample SD/abs(mean J), with all 24 A and 16 D cells
  retained. Robust median avoids one unstable failed regime setting the practical
  threshold for every regime; individual CVs remain reported.
- Net headroom threshold = max(10%, 2 × median development CV).
- Probe an untrained 32/32 network, current causal context construction and alternating
  budget configuration 256 times after warmup; no RL algorithm/training is implemented.
  Common context cost is already included in all fixed-grid latency; extra actor and
  switching P99 are priced for the oracle. Report context timing separately.
- Subtract a fixed 0.5 percentage point sensitivity reserve from headroom for physical
  consequences of unmeasured actor/switch overhead. This is a conservative judgment
  parameter, not a measured causal bound. Report gross results and reserve sensitivity.
- Train selection on three seeds, evaluate selected global and context pairs on the
  remaining seed; rotate all four. Equal context weight, mean real task cost.
- Strong GO requires zero-price net benefit >= threshold, positive benefit in >=3/4
  held-out folds, seed-block percentile bootstrap lower bound >0 after reserve,
  and all context modes selected in >=3/4 folds with at least two distinct modes.
  Secondary priced outcomes cannot rescue a failed natural-performance gate.

Phase B closed-loop speed–delay evidence was null at its meaningful-task threshold;
therefore any continuation concerns sampling/foresight, with no staleness-based core
claim. Medium-low stagnation and the limited four seed clusters remain explicit
limitations. If the oracle gate fails, stop before SAC; do not add cells or seeds
after looking at the oracle to manufacture a pass.
