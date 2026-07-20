# L217 sealed complex-navigation evidence index

This tracked bundle contains compact evidence from the immutable 420-episode
L217 benchmark.  The complete raw trajectories and resolved configurations are
kept out of Git under
`results/research_platform/rl/complex_navigation_sealed_l217/` and copied to
the Windows Desktop delivery package.

Formal experiment identity:

- Git SHA used for every episode: `9ec885007a56b3c3689422a45e450262fb9a4dc5`;
- manifest SHA-256: `b4e93d2b383b8cc96e5c6b673dd039f70c0617b51f38b6f8af1909422cf674d2`;
- 420 episodes, 60 complete blocks, 10 independent seed clusters;
- zero duplicate and zero qualification rows;
- bootstrap: 10,000 seed-cluster resamples, seed `20260724`.

Read `docs/rl/218_l217_complex_navigation_sealed_results_2026-07-20.md`
before using the tables.  In particular, L217 has partial mechanism coverage:
role-aware HSS was active, but residual-policy context and reliability-weighted
terminal-value authority were not active.  This bundle must not be cited as a
confirmation of those inactive mechanisms.

