# L216 Role-Aware HSS Development Protocol

Date: 2026-07-20  
Status: development-only protocol; no sealed seed may be opened.

## Motivation fixed before L216 execution

The complete L215 qualification matrix contains 126/126 unique cells across
seven methods, three scenes, two physics domains and seeds 48--50.  All cells
are retained.  Simple Combination and Value-only reached 18/18 without a
collision, while Full Proposed reached 11/18 and HSS-only reached 9/18.  Failed
adaptive trajectories had already navigated the obstacles and ended at
0.306--0.393 m for a frozen 0.300 m success radius.  Their ICODE dynamics
confidence was commonly 0--0.04 even when Actor support confidence was high.

This is a development failure of the authority mapping, not evidence from a
confirmatory test and not permission to change the paper's core components.

## Frozen candidates

- C1 `completion_only`: disable the 0.40--0.80 m Actor completion handover;
  retain the original trust-weighted HSS mapping.
- C2 `policy_rescue`: make the same completion correction, and use role-aware
  uncertainty routing.  If the Actor is in support and its causal elite yield
  remains competitive, low ICODE confidence allocates more Actor proposals;
  fully confident ICODE retains a medium Actor allocation.  Actor OOD or poor
  causal competence still removes Actor authority.

For C2, with calibrated dynamics confidence `c_d`, Actor factor `c_pi`, and
rescue floor `rho=0.5`, the online HSS authority is

$$
\alpha_{\mathrm{HSS}}
=c_{\pi}\left[\rho+(1-\rho)(1-c_d)\right].
$$

This does not make ICODE uncertainty a probability or safety certificate.

## Qualification screen

Use only development seeds 48, 49 and 50, both `nominal_seen` and
`combined_unseen`, and the hardest observed scene `narrow_corridor`.  Evaluate
both adaptive arms (`ordinary_adaptive`, `full_proposed`) for both candidates.
No seed, failed episode or collision may be removed.

Selection is lexicographic and fixed before execution:

1. zero collisions;
2. higher success count across the six seed-domain blocks;
3. lower mean steps, counting failure as the frozen 600-step budget;
4. if tied, retain the simpler C1 candidate.

Only the selected candidate may be rerun on the full 126-cell development
matrix.  Sealed seeds remain forbidden until the original L215 Development
Gate is met by that complete rerun.

