# L89 quadratic feature amendment

Date: 2026-07-18

This amendment was made **before** any independent L89 confirmation episode
was run.

The first linear contextual critic failed the internal learnability gate: it
selected `speed` on the support-set chicane, even though the logged training
and validation return preferred `narrow`.  The four support geometries reveal
a non-monotonic relationship (straight narrow, medium-turn paths speed,
high-turn chicane narrow), which a single linear boundary need not represent.

The route encoder is therefore expanded with the squared terms and pairwise
interactions of its three original observable geometric quantities.  This is
a fixed degree-two basis, not a scene-name feature or a test-outcome lookup.
Ridge candidates `0.001` and `0.01` are added and remain selected solely by
the support-geometry validation seeds.

The independent seeds, primary comparator, 2 mm precision margin, safety gate
and time-superiority gate remain unchanged.  The already inspected L87 rows
may be reported only as a development check; the final L89 claim still
depends on the untouched 20269721--20269725 MuJoCo confirmation seeds.

